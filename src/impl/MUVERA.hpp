//
// Created by 13172 on 2024/8/7.
//

#ifndef VECTORSETSEARCH_MUVERA_HPP
#define VECTORSETSEARCH_MUVERA_HPP

#include <spdlog/spdlog.h>
#include <fstream>

#include "include/alg/Distance.hpp"
#include "include/alg/MatrixMulBLAS.hpp"

#include "include/struct/MethodBase.hpp"
#include "include/struct/PositionHeap.hpp"

#include "include/util/TimeMemory.hpp"
#include "include/util/util.hpp"

#include "include/muvera/TransformIP.hpp"

#include "include/muvera/src/utils/mips_point.h"
#include "include/muvera/src/utils/pq_ip_point.hpp"
#include "include/muvera/src/utils/pq_point_range.hpp"
#include "include/muvera/src/utils/point_range.h"
#include "include/muvera/src/utils/graph.h"
#include "include/muvera/src/vamana/neighbors.h"

#include "include/alg/query_coreset/QueryCoreset.hpp"


namespace VectorSetSearch::Method
{
    class MUVERA
    {
    public:
        uint32_t _n_item{}, _vec_dim{}, max_item_n_vec_{};
        size_t _n_vecs{};
        std::vector<uint32_t> _item_n_vecs_l; // n_item
        std::vector<size_t> _item_n_vecs_offset_l; // n_item

        pyarray_float _item_vec_l_py; // n_vec * vec_dim
        const float* _item_vec_l; // n_vec * vec_dim

        uint32_t _k_sim, _n_cluster, _d_proj, _r_reps, _ip_vec_dim;
        std::vector<float> _partition_vec_l; // r_reps * k_sim * vec_dim
        std::vector<float> _random_matrix_l; // r_reps * d_proj * vec_dim

        BuildParams _build_graph_para;
        Graph<uint32_t> _graph_ins;

        uint32_t _n_centroid_subspace, _dim_subspace;
        PQPointRange<uint32_t, PQ_IP_Point<uint32_t>> _quantized_vec_l;

        TransformQuery _transform_query_ins;

        QueryCorset query_corset_;
        constexpr static float query_score_thres_ = 0.7;

        MUVERA() = default;

        MUVERA(const std::vector<uint32_t>& item_n_vecs_l,
               const uint32_t n_item, const uint32_t vec_dim,
               const uint32_t k_sim, const uint32_t d_proj, const uint32_t r_reps,
               const uint32_t R, const uint32_t L, const double alpha,
               const uint32_t n_centroid_subspace, const uint32_t dim_subspace)
        {
            _n_item = n_item;
            _vec_dim = vec_dim;
            auto max_ptr = std::max_element(item_n_vecs_l.begin(), item_n_vecs_l.end());
            max_item_n_vec_ = *max_ptr;

            _item_n_vecs_l = item_n_vecs_l;
            assert(item_n_vecs_l.size() == _n_item);
            _item_n_vecs_offset_l.resize(_n_item);

            size_t n_vecs = _item_n_vecs_l[0];
            _item_n_vecs_offset_l[0] = 0;
            for (uint32_t itemID = 1; itemID < _n_item; itemID++)
            {
                n_vecs += _item_n_vecs_l[itemID];
                _item_n_vecs_offset_l[itemID] = _item_n_vecs_offset_l[itemID - 1] + _item_n_vecs_l[itemID - 1];
            }
            _n_vecs = n_vecs;

            _k_sim = k_sim;
            _n_cluster = 1 << _k_sim;
            _d_proj = d_proj;
            _r_reps = r_reps;
            _ip_vec_dim = _n_cluster * _d_proj * _r_reps;

            _partition_vec_l.resize(_r_reps * _k_sim * _vec_dim);
            _random_matrix_l.resize(_r_reps * _d_proj * _vec_dim);

            const int num_passes = 1;
            const int single_batch = 0;
            _build_graph_para = BuildParams(R, L, alpha, num_passes, single_batch);

            const uint32_t max_degree = R;
            _graph_ins = Graph<uint32_t>(_build_graph_para.max_degree(), _n_item);

            this->_n_centroid_subspace = n_centroid_subspace;
            this->_dim_subspace = dim_subspace;

            _transform_query_ins = TransformQuery(_partition_vec_l.data(), _random_matrix_l.data(),
                                                  _k_sim, _n_cluster, _d_proj, _r_reps, _vec_dim);
        }

        void add_projection(const pyarray_float& partition_vec_l_py, const pyarray_float& random_matrix_l_py)
        {
            // partition_vec_l_py: r_reps * k_sim * vec_dim, random_matrix_l_py: r_reps * d_proj * vec_dim
            const float* partition_vec_l = partition_vec_l_py.data();
            assert(partition_vec_l_py.ndim() == 3);
            assert(partition_vec_l_py.shape(0) == _r_reps);
            assert(partition_vec_l_py.shape(1) == _k_sim);
            assert(partition_vec_l_py.shape(2) == _vec_dim);

            const float* random_matrix_l = random_matrix_l_py.data();
            assert(random_matrix_l_py.ndim() == 3);
            assert(random_matrix_l_py.shape(0) == _r_reps);
            assert(random_matrix_l_py.shape(1) == _d_proj);
            assert(random_matrix_l_py.shape(2) == _vec_dim);

            std::memcpy(_partition_vec_l.data(), partition_vec_l, sizeof(float) * _r_reps * _k_sim * _vec_dim);
            std::memcpy(_random_matrix_l.data(), random_matrix_l, sizeof(float) * _r_reps * _k_sim * _vec_dim);
            //            std::memcpy(_random_matrix_l.data(), random_matrix_l, sizeof(float) * _r_reps * _d_proj * _vec_dim);
            spdlog::info("finish add projection");
        }

        void build_graph_index(const pyarray_float& ip_vector_l_py)
        {
            const float* ip_vector_l = ip_vector_l_py.data();
            assert(ip_vector_l_py.ndim() == 2);
            assert(ip_vector_l_py.shape(0) == _n_item);
            assert(ip_vector_l_py.shape(1) == _ip_vec_dim);

            using Point = Mips_Point<float>;
            using PR = PointRange<float, Point>;
            using indexType = uint32_t;

            const uint32_t cluster_dim = _ip_vec_dim;
            PointRange<float, Mips_Point<float>> Points = PointRange<float, Mips_Point<float>>(
                ip_vector_l, _n_item, cluster_dim);

            ANN_build_index<Point, PR, indexType>(_graph_ins, _build_graph_para, Points);

            spdlog::info("finish build_graph_index");
        }

        void add_item_vector_l(pyarray_float& vec_l)
        {
            spdlog::info("start add_item_vector_l");
            assert(vec_l.ndim() == 2);
            assert(vec_l.shape(0) == _n_vecs);
            assert(vec_l.shape(1) == _vec_dim);
            this->_item_vec_l_py = std::move(vec_l);
            this->_item_vec_l = _item_vec_l_py.data();
            spdlog::info("finish add_item_vector_l");
        }

        // add the code for every subspace
        void add_pq_code_l(const pyarray_float& sub_centroid_l_l_py, const pyarray_uint32& sub_code_l_l_py)
        {
            assert(sub_centroid_l_l_py.ndim() == 3);
            const float* sub_centroid_l_l = sub_centroid_l_l_py.data();
            const uint32_t n_subspace = sub_centroid_l_l_py.shape(0);
            assert(n_subspace == _ip_vec_dim / _dim_subspace + (_ip_vec_dim % _dim_subspace == 0 ? 0 : 1));
            assert(sub_centroid_l_l_py.shape(1) == _n_centroid_subspace);
            assert(sub_centroid_l_l_py.shape(2) == _dim_subspace);

            assert(sub_code_l_l_py.ndim() == 2);
            const uint32_t* sub_code_l_l = sub_code_l_l_py.data();
            assert(sub_code_l_l_py.shape(0) == n_subspace);
            assert(sub_code_l_l_py.shape(1) == _n_item);

            _quantized_vec_l = PQPointRange<uint32_t, PQ_IP_Point<uint32_t>>(sub_centroid_l_l, sub_code_l_l,
                                                                             n_subspace, _n_centroid_subspace,
                                                                             _dim_subspace,
                                                                             _n_item);

            spdlog::info("finish add_pq_code_l");
        }

        py::tuple search(const vector_set_list& qry_embeddings, const uint32_t& topk,
                         const uint32_t n_candidate)
        {
            if (qry_embeddings.ndim() != 3)
                throw std::runtime_error("the vector set list dimension should be 3");
            if (qry_embeddings.shape(2) != _vec_dim)
                throw std::runtime_error("the embedding dimension should equal to the pre-assignment");

            const uint32_t n_query = qry_embeddings.shape(0);
            const uint32_t query_n_vecs = qry_embeddings.shape(1);
            const float* query_start_ptr = qry_embeddings.data();

            float* result_score_l = new float[(int64_t)n_query * topk];
            uint32_t* result_ID_l = new uint32_t[(int64_t)n_query * topk];
            double* compute_time_l = new double[n_query];

            double* transform_time_l = new double[n_query];
            double* ip_time_l = new double[n_query];
            double* decode_time_l = new double[n_query];
            double* refine_time_l = new double[n_query];

            uint32_t* n_search_candidate_l = new uint32_t[n_query];

            query_corset_ = QueryCorset(query_n_vecs, _vec_dim, query_score_thres_);
            std::vector<float> query_corset_l_(query_n_vecs * _vec_dim);

            _transform_query_ins.initQueryInfo(query_n_vecs);

            BatchMaxSim batch_max_sim(query_n_vecs, n_candidate, max_item_n_vec_);
            std::vector<float> fine_item_l(n_candidate * max_item_n_vec_ * _vec_dim);
            std::vector<uint32_t> fine_item_n_vec_accu_l(n_candidate);
            std::vector<uint32_t> fine_item_n_vec_l(n_candidate);

            std::vector<std::pair<float, uint32_t>> item_candidate_cache_l(_n_item); // n_item

            TimeRecord record, part_record;
            for (uint32_t queryID = 0; queryID < n_query; queryID++)
            {
                if (queryID % 100 == 0)
                {
                    spdlog::info("start processing queryID {}", queryID);
                }
                record.reset();

                part_record.reset();
                const float* query = query_start_ptr + queryID * query_n_vecs * _vec_dim;

                uint32_t query_n_vec_actual = query_n_vecs;
                query_corset_.compute_query_corset(query,
                                                   query_corset_l_.data(),
                                                   query_n_vec_actual);

                const float* query_ip_vector = _transform_query_ins.transformQuery2IP(
                    query); // n_cluster * r_reps * d_proj
                const double transform_time = part_record.get_elapsed_time_second();
                part_record.reset();

                _quantized_vec_l.ComputeCentroidQueryIP(query_ip_vector, _ip_vec_dim);


                uint32_t start_point = 0;
                stats<uint32_t> QueryStats(1);

                PointRange<float, Mips_Point<float>> Query_Points = PointRange<float, Mips_Point<float>>(
                    query_ip_vector, 1, _ip_vec_dim);
                QueryParams QP = QueryParams((long)n_candidate, (long)n_candidate, (double)1.35,
                                             (long)_graph_ins.size(),
                                             (long)_graph_ins.max_degree());
                auto [pairElts, dist_cmps] = beam_search(Query_Points[0], _graph_ins, _quantized_vec_l, start_point,
                                                         QP);
                auto [ip_result_l, visited_ele_l] = pairElts;
                const double ip_time = part_record.get_elapsed_time_second();


                //refine by score
                part_record.reset();
                const uint32_t n_search_candidate = ip_result_l.size();
                assert(n_search_candidate <= n_candidate);
                uint32_t refine_vec_offset = 0;
                for (uint32_t candID = 0; candID < n_search_candidate; candID++)
                {
                    const float ip_score = ip_result_l[candID].second;
                    const uint32_t itemID = (uint32_t)ip_result_l[candID].first;
                    item_candidate_cache_l[candID] = std::make_pair(ip_score, itemID);

                    const uint32_t item_n_vec = _item_n_vecs_l[itemID];
                    const size_t vec_offset = _item_n_vecs_offset_l[itemID];
                    const float* item = _item_vec_l + vec_offset * _vec_dim;

                    std::memcpy(fine_item_l.data() + refine_vec_offset * _vec_dim, item,
                                sizeof(float) * item_n_vec * _vec_dim);
                    fine_item_n_vec_accu_l[candID] = refine_vec_offset;
                    fine_item_n_vec_l[candID] = item_n_vec;
                    refine_vec_offset += item_n_vec;
                }
                const double decode_time = part_record.get_elapsed_time_second();

                part_record.reset();
                batch_max_sim.compute(query_corset_l_.data(), query_n_vec_actual,
                                      fine_item_l.data(), fine_item_n_vec_l.data(), fine_item_n_vec_accu_l.data(),
                                      _vec_dim, n_search_candidate,
                                      item_candidate_cache_l.data());

                std::sort(item_candidate_cache_l.begin(), item_candidate_cache_l.begin() + n_search_candidate,
                          [](const std::pair<float, uint32_t>& l, const std::pair<float, uint32_t>& r)
                          {
                              return l.first > r.first;
                          });
                const double refine_score_time = part_record.get_elapsed_time_second();

                for (uint32_t candID = 0; candID < topk; candID++)
                {
                    const int64_t insert_offset = (int64_t)queryID * topk;
                    result_score_l[insert_offset + (int64_t)candID] = item_candidate_cache_l[candID].first;
                    result_ID_l[insert_offset + (int64_t)candID] = item_candidate_cache_l[candID].second;
                }

                compute_time_l[queryID] = record.get_elapsed_time_second();

                transform_time_l[queryID] = transform_time;
                ip_time_l[queryID] = ip_time;
                decode_time_l[queryID] = decode_time;
                refine_time_l[queryID] = refine_score_time;

                n_search_candidate_l[queryID] = n_search_candidate;
            }

            py::capsule handle_result_score_ptr(result_score_l, Method::PtrDelete<float>);
            py::capsule handle_result_ID_ptr(result_ID_l, Method::PtrDelete<uint32_t>);
            py::capsule handle_compute_time_ptr(compute_time_l, Method::PtrDelete<double>);

            py::capsule handle_transform_time_ptr(transform_time_l, Method::PtrDelete<double>);
            py::capsule handle_ip_time_ptr(ip_time_l, Method::PtrDelete<double>);
            py::capsule handle_decode_time_ptr(decode_time_l, Method::PtrDelete<double>);
            py::capsule handle_refine_time_ptr(refine_time_l, Method::PtrDelete<double>);

            py::capsule handle_n_search_candidate_ptr(n_search_candidate_l, Method::PtrDelete<uint32_t>);

            return py::make_tuple(
                py::array_t<float>(
                    {n_query, topk}, {topk * sizeof(float), sizeof(float)},
                    result_score_l, handle_result_score_ptr
                ),
                py::array_t<uint32_t>(
                    {n_query, topk}, {topk * sizeof(uint32_t), sizeof(uint32_t)},
                    result_ID_l, handle_result_ID_ptr
                ),
                py::array_t<double>(
                    {n_query}, {sizeof(double)},
                    compute_time_l, handle_compute_time_ptr
                ),

                py::array_t<double>(
                    {n_query}, {sizeof(double)},
                    transform_time_l, handle_transform_time_ptr
                ),
                py::array_t<double>(
                    {n_query}, {sizeof(double)},
                    ip_time_l, handle_ip_time_ptr
                ),
                py::array_t<double>(
                    {n_query}, {sizeof(double)},
                    decode_time_l, handle_decode_time_ptr
                ),
                py::array_t<double>(
                    {n_query}, {sizeof(double)},
                    refine_time_l, handle_refine_time_ptr
                ),

                py::array_t<uint32_t>(
                    {n_query}, {sizeof(uint32_t)},
                    n_search_candidate_l, handle_n_search_candidate_ptr
                )
            );
        }
    };
}
#endif //VECTORSETSEARCH_MUVERA_HPP

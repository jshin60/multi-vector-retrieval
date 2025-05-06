//
// Created by Administrator on 2025/1/17.
//

#ifndef IGPPROBECONTROLLER_HPP
#define IGPPROBECONTROLLER_HPP

#include <queue>
#include <numeric>
#include <set>
#include <complex>

#include "include/struct/BoolArray.hpp"
#include "include/alg/probe/IGP/IGPStruct.hpp"
#include "include/alg/probe/IGP/IGPMatrixMultProbe.hpp"
#include "include/alg/probe/IGP/IGPProximityGraphProbe.hpp"
#include "include/alg/probe/ProbeHeap.hpp"
#include "include/alg/MatrixMulBLAS.hpp"

#include "include/util/TimeMemory.hpp"
#include "include/util/util.hpp"

namespace VectorSetSearch
{
    class IGPProbeController
    {
        // the input of this class is the centroid and the query
        // the output is the probe element when the function are called
    public:
        IGPProximityGraphProbe _pg_probe;
        IGPMatrixMultProbe _mm_probe;
        uint32_t query_n_vec_;

        uint32_t _n_centroid;

        bool _first_time_matrix_mult = false;

        constexpr static uint32_t _threshold_nprobe = 1024;
        std::vector<uint32_t> _current_nprobe_l; // query_n_vec, # element we have probed for each query vector, default value: 0

        uint32_t _n_computation = 0;

        IGPProbeController() = default;

        IGPProbeController(const float* centroid_l,
                                const uint32_t n_centroid,
                                const uint32_t vec_dim
        )
        {
            this->_pg_probe = IGPProximityGraphProbe(centroid_l, n_centroid, vec_dim);
            this->_mm_probe = IGPMatrixMultProbe(centroid_l, n_centroid, vec_dim);

            this->_n_centroid = n_centroid;
        }

        void set_query_info(const uint32_t query_n_vecs)
        {
            this->_pg_probe.set_query_info(query_n_vecs);
            this->_mm_probe.set_query_info(query_n_vecs);
            this->query_n_vec_ = query_n_vecs;
            this->_current_nprobe_l.resize(query_n_vecs, 0);
        }

        void set_retrieval_parameter(const uint32_t topk_per_batch, const uint32_t addition_search_size)
        {
            _pg_probe.set_retrieval_parameter(topk_per_batch, addition_search_size);
        }

        void reset()
        {
            this->_pg_probe.reset();
            this->_mm_probe.reset();
            _current_nprobe_l.assign(query_n_vec_, 0);
            _n_computation = 0;
            _first_time_matrix_mult = true;
        }

        void set_query(const float* query)
        {
            this->_pg_probe.set_query(query);
            this->_mm_probe.set_query(query);
        }

        // ele_l is the output of this function
        // qvec_upper_bound means the upper bound of each query vector
        // it should be the upper bound of all query score in the last probe
        void next_probe_element(const uint32_t nprobe_require, const uint32_t qvecID,
                                IGPProbeEle* ele_l, uint32_t& nprobe_return)
        {
            // add the probe element, may perform the argsort if necessary
            assert(qvecID < query_n_vec_);

            if (_current_nprobe_l[qvecID] < _threshold_nprobe)
            {
                this->_pg_probe.next_probe_element(nprobe_require, qvecID,
                                                   ele_l, nprobe_return,
                                                   _n_computation);
            }
            else
            {
                if (_first_time_matrix_mult)
                {
                    this->_mm_probe.init_search();
                    _first_time_matrix_mult = false;
                    _n_computation += _mm_probe._query_n_vecs * _mm_probe._n_centroid;
                }

                this->_mm_probe.next_probe_element(nprobe_require, qvecID,
                                                   ele_l, nprobe_return);
            }

            _current_nprobe_l[qvecID] = std::min(_current_nprobe_l[qvecID] + nprobe_require, _n_centroid);
        }
    };
}
#endif //IGPPROBECONTROLLER_HPP

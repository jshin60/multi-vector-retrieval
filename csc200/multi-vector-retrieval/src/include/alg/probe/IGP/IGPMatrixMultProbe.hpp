//
// Created by Administrator on 2025/1/17.
//

#ifndef IGPMATRIXMULTPROBE_HPP
#define IGPMATRIXMULTPROBE_HPP

#include <queue>
#include <numeric>
#include <set>
#include <complex>

#include "include/struct/BoolArray.hpp"
#include "include/alg/probe/IGP/IGPStruct.hpp"
#include "include/alg/probe/ProbeHeap.hpp"
#include "include/alg/MatrixMulBLAS.hpp"

#include "include/util/TimeMemory.hpp"
#include "include/util/util.hpp"

namespace VectorSetSearch
{
    class IGPMatrixMultProbe
    {
        // the input of this class is the centroid and the query
        // the output is the probe element when the function are called
    public:
        uint32_t _query_n_vecs, _n_centroid, _vec_dim;

        const float* _centroid_l; // n_centroid * vec_dim
        const float* _query; // query_n_vecs * vec_dim

        constexpr static uint32_t _n_sort_per_batch = 16;
        std::vector<float> _centroid_score_l; // query_n_vecs * _n_centroid
        std::vector<uint32_t> _centroid_score_idx_l; // query_n_vecs * _n_centroid
        uint32_t _n_sort; // # sorted element for each query vector, default value: 0
        uint32_t _current_nprobe; // # element we have probed for each query vector, default value: 0

        IGPMatrixMultProbe() = default;

        IGPMatrixMultProbe(const float* centroid_l,
                           const uint32_t n_centroid,
                           const uint32_t vec_dim
        )
        {
            this->_centroid_l = centroid_l;
            this->_n_centroid = n_centroid;
            this->_vec_dim = vec_dim;
        }

        void set_query_info(const uint32_t query_n_vecs)
        {
            this->_query_n_vecs = query_n_vecs;
            this->_centroid_score_l.resize(query_n_vecs * _n_centroid);
            this->_centroid_score_idx_l.resize(query_n_vecs * _n_centroid);
        }

        void reset()
        {
            for (uint32_t query_vecID = 0; query_vecID < _query_n_vecs; query_vecID++)
            {
                uint32_t* idx_l = _centroid_score_idx_l.data() + query_vecID * _n_centroid;
                std::iota(idx_l, idx_l + _n_centroid, 0);
            }
            _n_sort = 0;
            _current_nprobe = 0;
        }

        void set_query(const float* query)
        {
            this->_query = query;
        }

        void init_search()
        {
            MatrixMultiply(_query, _centroid_l,
                           _query_n_vecs, _n_centroid, _vec_dim,
                           _centroid_score_l.data());
        }

        void qvec_centroid_argsort(const float* score_l, const uint32_t n_finish_sort, uint32_t* idx_l) const
        {
            const uint32_t n_sort = n_finish_sort;
            const uint32_t n_sort_after = std::min(n_sort + _n_sort_per_batch, _n_centroid);

            std::partial_sort(idx_l + n_sort, idx_l + n_sort_after, idx_l + _n_centroid,
                              [&score_l](const uint32_t& l, const uint32_t& r)
                              {
                                  return score_l[l] > score_l[r];
                              });

#ifndef NDEBUG
            for (uint32_t candID = 0; candID < n_sort_after - 1; candID++)
            {
                const uint32_t prev_centID = idx_l[candID];
                const uint32_t this_centID = idx_l[candID + 1];
                assert(score_l[prev_centID] >= score_l[this_centID]);
            }
            for (uint32_t candID = n_sort_after; candID < _n_centroid; candID++)
            {
                const uint32_t prev_centID = idx_l[n_sort_after - 1];
                const uint32_t this_centID = idx_l[candID];
                assert(score_l[prev_centID] >= score_l[this_centID]);
            }
            std::set s(idx_l, idx_l + _n_centroid);
            assert(s.size() == _n_centroid);
#endif
        }

        // ele_l is the output of this function
        // qvec_upper_bound means the upper bound of each query vector
        // it should be the upper bound of all query score in the last probe
        void next_probe_element(const uint32_t nprobe_require, const uint32_t qvecID,
                                IGPProbeEle* ele_l, uint32_t& nprobe_return)
        {
            // add the probe element, may perform the argsort if necessary
            const uint32_t next_first_nprobe = _current_nprobe + nprobe_require;
            const uint32_t n_sort_target = std::min(next_first_nprobe, _n_centroid);
            while (_n_sort < n_sort_target)
            {
                const uint32_t score_offset = qvecID * _n_centroid;
                qvec_centroid_argsort(_centroid_score_l.data() + score_offset, _n_sort,
                                      _centroid_score_idx_l.data() + score_offset);
                _n_sort += _n_sort_per_batch;
            }
            _n_sort = std::min(_n_sort, _n_centroid);
            nprobe_return = std::min(nprobe_require, _n_centroid - _current_nprobe);

            for (uint32_t probeID = 0; probeID < nprobe_return; probeID++)
            {
                const uint32_t score_offset = qvecID * _n_centroid;
                const uint32_t centID = _centroid_score_idx_l[score_offset + _current_nprobe + probeID];
                const float centroid_score = _centroid_score_l[score_offset + centID];

                ele_l[probeID] = IGPProbeEle(qvecID, centID, centroid_score);
            }

#ifndef NDEBUG
            for (uint32_t probeID = 0;
                 probeID < std::min(nprobe_require, _n_centroid - _current_nprobe) - 1; probeID++)
            {
                const IGPProbeEle probe = ele_l[probeID];
                assert(probe.qvecID == qvecID);
                const uint32_t centID = probe.centroidID;
                const float centroid_score = probe.score;
                assert(centroid_score == _centroid_score_l[qvecID * _n_centroid + centID]);

                assert(ele_l[probeID].score >=
                    ele_l[(probeID + 1)].score);
            }
#endif
            _current_nprobe = std::min(_current_nprobe + nprobe_require, _n_centroid);
        }
    };
}
#endif //IGPMATRIXMULTPROBE_HPP

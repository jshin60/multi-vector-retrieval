//
// Created by 13172 on 2024/8/14.
//

#ifndef VECTORSETSEARCH_PQPOINTRANGE_HPP
#define VECTORSETSEARCH_PQPOINTRANGE_HPP

#include <sys/mman.h>
#include <algorithm>
#include <iostream>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"
#include "types.h"
#include "include/alg/MatrixMulBLAS.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

//tp_size must divide 64 evenly--no weird/large types!
long pq_ip_dim_round_up(long dim, long tp_size) {
    long qt = (dim * tp_size) / 64;
    long remainder = (dim * tp_size) % 64;
    if (remainder == 0) return dim;
    else return ((qt + 1) * 64) / tp_size;
}


template<typename T_, class Point_>
struct PQPointRange {
    using T = T_;
    using Point = Point_;
    using parameters = typename Point::parameters;

    long dimension() const { return _vec_dim; }

    long aligned_dimension() const { return _aligned_dim; }

    PQPointRange() : _data(std::shared_ptr<T[]>(nullptr, std::free)) { _n_item = 0; }

    PQPointRange(const float *sub_centroid_l_l, const uint32_t *sub_code_l_l,
                 const uint32_t n_subspace, const uint32_t n_centroid_subspace, const uint32_t dim_subspace,
                 const uint32_t n_item) {

        this->_n_item = n_item;
        this->_vec_dim = n_subspace;
        this->_para = parameters(sub_centroid_l_l,
                                 n_centroid_subspace, n_subspace, dim_subspace);
        std::cout << "Detected " << n_item << " points with dimension " << _vec_dim << std::endl;

        _aligned_dim = pq_ip_dim_round_up(_vec_dim, sizeof(T));
        std::cout << "Aligning dimension to " << _aligned_dim << std::endl;
        int64_t num_bytes = _n_item * _aligned_dim * sizeof(T);

        T *ptr = (T *) aligned_alloc(1l << 21, num_bytes);
        madvise(ptr, num_bytes, MADV_HUGEPAGE);
        _data = std::shared_ptr<T[]>(ptr, std::free);

        size_t BLOCK_SIZE = 1000000;
        size_t index = 0;
        while (index < _n_item) {
            size_t floor = index;
            size_t ceiling = index + BLOCK_SIZE <= _n_item ? index + BLOCK_SIZE : _n_item;
            int data_bytes = _vec_dim * sizeof(T);
            parlay::parallel_for(floor, ceiling, [&](size_t itemID) {
                for (int subspaceID = 0; subspaceID < _vec_dim; subspaceID++)
                    _data.get()[itemID * _aligned_dim + subspaceID] = sub_code_l_l[subspaceID * n_item + itemID];
                //std::memmove(_data.get() + i*_aligned_dim, data.begin() + (i-floor)*_vec_dim, data_bytes);
            });
            index = ceiling;
        }

    }

    void ComputeCentroidQueryIP(const float *query_ip_vec, const uint32_t ip_vec_dim) {
        assert(ip_vec_dim <= _para._dim_per_subspace * _para._n_subspace);

        std::vector<float> query_ip_vec_pad(_para._dim_per_subspace * _para._n_subspace);
        std::memcpy(query_ip_vec_pad.data(), query_ip_vec, sizeof(float) * ip_vec_dim);

        for (uint32_t subspaceID = 0; subspaceID < _para._n_subspace; subspaceID++) {
            const uint32_t subspace_dim = _para._dim_per_subspace;
            const float *query_sub_vec = query_ip_vec_pad.data() + subspaceID * subspace_dim;

            const float *subspace_codebook = _para._codebook_l.get() + subspaceID * _para._n_cluster * subspace_dim;
            float *precompute_score = _para._precompute_score_l.get() + subspaceID * _para._n_cluster;

            VectorSetSearch::MatrixTimesVector(subspace_codebook, query_sub_vec, _para._n_cluster, subspace_dim,
                                               precompute_score);

        }
    }

    size_t size() const { return _n_item; }

    uint32_t get_dims() const { return _vec_dim; }

    Point operator[](long i) const {
        if (i > _n_item) {
            std::cout << "ERROR: point index out of range: " << i << " from range " << _n_item << ", " << std::endl;
            abort();
        }
        return Point(_data.get() + i * _aligned_dim, i, &_para);
    }

    parameters _para;

private:
    std::shared_ptr<T[]> _data;
    uint32_t _vec_dim;
    uint32_t _aligned_dim;
    size_t _n_item;
};

#endif //VECTORSETSEARCH_PQPOINTRANGE_HPP

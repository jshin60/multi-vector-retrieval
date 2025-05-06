//
// Created by 13172 on 2024/8/14.
//

#ifndef VECTORSETSEARCH_PQ_IP_POINT_HPP
#define VECTORSETSEARCH_PQ_IP_POINT_HPP

#include <algorithm>
#include <iostream>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"
#include "types.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


template<typename T>
float pq_mips_distance(const T *code, const float *precompute_score_l,
                       const uint32_t n_subspace, const uint32_t n_cluster) {
    float result = 0;
    for (uint32_t subspaceID = 0; subspaceID < n_subspace; subspaceID++) {
        result += precompute_score_l[subspaceID * n_cluster + code[subspaceID]];
    }

    return -result;
}

template<typename T>
struct PQ_IP_Point {
    using distanceType = float;
    //template<typename C, typename range> friend struct Quantized_Mips_Point;

    struct parameters {
        std::unique_ptr<float[]> _codebook_l; // n_subspace * _n_cluster * _dim_per_subspace
        std::unique_ptr<float[]> _precompute_score_l; // n_subspace * _n_cluster
        uint32_t _n_cluster, _n_subspace, _dim_per_subspace;

        parameters() : _n_subspace(0) {}

        parameters(const float *codebook_l,
                   const uint32_t n_cluster, const uint32_t n_subspace, const uint32_t dim_per_subspace) {
            _n_cluster = n_cluster;
            _n_subspace = n_subspace;
            _dim_per_subspace = dim_per_subspace;

            _codebook_l = std::make_unique<float[]>(n_subspace * n_cluster * dim_per_subspace);
            std::memcpy(_codebook_l.get(), codebook_l, sizeof(float) * n_subspace * n_cluster * dim_per_subspace);

            _precompute_score_l = std::make_unique<float[]>(n_subspace * n_cluster);

        }
    };

    static distanceType d_min() { return -std::numeric_limits<float>::max(); }

    static bool is_metric() { return false; }

    T operator[](long i) const { return *(_data + i); }

    float distance(const Mips_Point<float> &x) const {
        return pq_mips_distance(this->_data, _para->_precompute_score_l.get(),
                                _para->_n_subspace, _para->_n_cluster);
    }

    void prefetch() const {
        int l = (_para->_n_subspace * sizeof(T) - 1) / 64 + 1;
        for (int i = 0; i < l; i++)
            __builtin_prefetch((char *) _data + i * 64);
    }

    long id() const { return _id; }

    PQ_IP_Point() : _data(nullptr), _id(-1), _para(0) {}

    PQ_IP_Point(T *_data, long id, const parameters* _para)
            : _data(_data), _id(id), _para(_para) {}

    bool operator==(const PQ_IP_Point<T> &q) const {
        for (int i = 0; i < _para->_n_subspace; i++) {
            if (_data[i] != q._data[i]) {
                return false;
            }
        }
        return true;
    }

    bool same_as(const Mips_Point<float> &x) const {
        return false;
    }

private:
    T *_data;
    long _id;
    const parameters* _para;
};

#endif //VECTORSETSEARCH_PQ_IP_POINT_HPP

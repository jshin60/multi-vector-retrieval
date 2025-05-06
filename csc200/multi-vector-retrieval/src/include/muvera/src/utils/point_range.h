// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <sys/mman.h>
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

//tp_size must divide 64 evenly--no weird/large types!
long dim_round_up(long dim, long tp_size) {
    long qt = (dim * tp_size) / 64;
    long remainder = (dim * tp_size) % 64;
    if (remainder == 0) return dim;
    else return ((qt + 1) * 64) / tp_size;
}


template<typename T_, class Point_>
struct PointRange {
    using T = T_;
    using Point = Point_;
    using parameters = typename Point::parameters;

    long dimension() const { return _vec_dim; }

    long aligned_dimension() const { return _aligned_dim; }

    PointRange() : values(std::shared_ptr<T[]>(nullptr, std::free)) { _n_item = 0; }

    template<typename PR>
    PointRange(PR &pr) : PointRange(pr, Point::generate_parameters(pr)) {}

    PointRange(const T *arr, const uint32_t n_item, const uint32_t dim) {

        this->_n_item = n_item;
        this->_vec_dim = dim;
        this->_para = parameters(dim);
//        std::cout << "Detected " << n_item << " points with dimension " << _vec_dim << std::endl;

        _aligned_dim = dim_round_up(_vec_dim, sizeof(T));
//        std::cout << "Aligning dimension to " << _aligned_dim << std::endl;

        int64_t num_bytes = _n_item * _aligned_dim * sizeof(T);

        T *ptr = (T *) aligned_alloc(1l << 21, num_bytes);
        madvise(ptr, num_bytes, MADV_HUGEPAGE);
        values = std::shared_ptr<T[]>(ptr, std::free);

        size_t BLOCK_SIZE = 1000000;
        size_t index = 0;
        while (index < _n_item) {
            size_t floor = index;
            size_t ceiling = index + BLOCK_SIZE <= _n_item ? index + BLOCK_SIZE : _n_item;
            const T *data_start = arr + floor * _vec_dim;
            const T *data_end = arr + ceiling * _vec_dim;
            parlay::slice<const T *, const T *> data = parlay::make_slice(data_start, data_end);
            int data_bytes = _vec_dim * sizeof(T);
            parlay::parallel_for(floor, ceiling, [&](size_t i) {
                for (int j = 0; j < _vec_dim; j++)
                    values.get()[i * _aligned_dim + j] = data[(i - floor) * _vec_dim + j];
                //std::memmove(values.get() + i*_aligned_dim, data.begin() + (i-floor)*_vec_dim, data_bytes);
            });
            index = ceiling;
        }

    }

    size_t size() const { return _n_item; }

    uint32_t get_dims() const { return _vec_dim; }

    Point operator[](long i) const {
        if (i > _n_item) {
            std::cout << "ERROR: point index out of range: " << i << " from range " << _n_item << ", " << std::endl;
            abort();
        }
        return Point(values.get() + i * _aligned_dim, i, _para);
    }

    parameters _para;

private:
    std::shared_ptr<T[]> values;
    uint32_t _vec_dim;
    uint32_t _aligned_dim;
    size_t _n_item;
};

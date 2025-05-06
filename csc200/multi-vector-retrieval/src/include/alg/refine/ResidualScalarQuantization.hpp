//
// Created by 13172 on 2024/2/23.
//

#ifndef VECTORSETSEARCH_RESIDUALSCALARQUANTIZATION_HPP
#define VECTORSETSEARCH_RESIDUALSCALARQUANTIZATION_HPP

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "include/struct/TypeDef.hpp"

namespace VectorSetSearch {

    py::tuple ComputeQuantizedScalar(const pyarray_float &item_vec_l_py, const pyarray_float &centroid_l_py,
                                     const pyarray_uint32 &code_l_py, const uint32_t &n_bit) {
        const size_t n_vec = item_vec_l_py.shape(0);
        const uint32_t vec_dim = item_vec_l_py.shape(1);

        assert(vec_dim == centroid_l_py.shape(1));
        assert(code_l_py.ndim() == 1 && code_l_py.shape(0) == n_vec);

        const float *item_vec_l = item_vec_l_py.data();
        const float *centroid_l = centroid_l_py.data();
        const uint32_t *code_l = code_l_py.data();

        std::vector<float> residual_error_l((size_t) n_vec * vec_dim);
#pragma omp parallel for default(none) shared(n_vec, code_l, item_vec_l, vec_dim, centroid_l, residual_error_l, centroid_l_py)
        for (size_t vecID = 0; vecID < n_vec; vecID++) {
            const uint32_t code = code_l[vecID];
            assert(code < centroid_l_py.shape(0));
            const float *vec = item_vec_l + (size_t) vecID * vec_dim;

            std::vector<float> centroid(centroid_l + code * vec_dim, centroid_l + (code + 1) * vec_dim);

            float *vec_error_l = residual_error_l.data() + (size_t) vecID * vec_dim;
            for (uint32_t dim = 0; dim < vec_dim; dim++) {
                vec_error_l[dim] = vec[dim] - centroid[dim];
            }
        }

        const uint32_t n_quantile = 1 << n_bit;
        std::vector<double> quantile_l(n_quantile);
        for (uint32_t quanID = 0; quanID < n_quantile; quanID++) {
            const double quantile = 1.0 * quanID / n_quantile;
            quantile_l[quanID] = quantile;
        }

        const uint32_t n_cutoff = n_quantile - 1;
        std::vector<float> cutoff_l(n_cutoff);
        for (uint32_t cutID = 0; cutID < n_cutoff; cutID++) {
            const double quantile = quantile_l[cutID + 1];
            assert(0 < quantile && quantile < 1.0);
            const size_t quantile_idx = std::floor(quantile * (double) residual_error_l.size());
            std::nth_element(residual_error_l.begin(), residual_error_l.begin() + quantile_idx, residual_error_l.end());

            const float weight = residual_error_l[quantile_idx];
            cutoff_l[cutID] = weight;
        }

        const uint32_t n_weight = n_quantile;
        std::vector<float> weight_l(n_weight);
        for (uint32_t wID = 0; wID < n_weight; wID++) {
            const double quantile = quantile_l[wID] + 0.5 / n_quantile;
            assert(0 < quantile && quantile < 1.0);
            const size_t quantile_idx = std::floor(quantile * (double) residual_error_l.size());
            std::nth_element(residual_error_l.begin(), residual_error_l.begin() + quantile_idx, residual_error_l.end());

            const float weight = residual_error_l[quantile_idx];
            weight_l[wID] = weight;
        }

#ifndef NDEBUG
        assert(cutoff_l.size() + 1 == weight_l.size());
        for (uint32_t cutID = 0; cutID < n_cutoff - 1; cutID++) {
            assert(cutoff_l[cutID] < cutoff_l[cutID + 1]);
        }
        for (uint32_t wID = 0; wID < n_weight - 1; wID++) {
            assert(weight_l[wID] < weight_l[wID + 1]);
        }
        for (uint32_t ID = 0; ID < n_cutoff; ID++) {
            assert(weight_l[ID] < cutoff_l[ID]);
        }
        assert(cutoff_l[n_cutoff - 1] < weight_l[n_weight - 1]);
#endif

        return py::make_tuple(cutoff_l, weight_l);
    }

    py::tuple ComputeResidualCode(const pyarray_float &vec_l_py, const pyarray_float &centroid_l_py,
                                  const pyarray_uint32 &code_l_py,
                                  const pyarray_float &cutoff_l_py, const pyarray_float &weight_l_py,
                                  const uint32_t &n_bit) {
        const size_t n_vec = vec_l_py.shape(0);
        const uint32_t vec_dim = vec_l_py.shape(1);
        const uint32_t n_centroid = centroid_l_py.shape(0);
        assert(vec_dim == centroid_l_py.shape(1));
        assert(n_vec == code_l_py.shape(0));
        const uint32_t n_weight = weight_l_py.shape(0);
        assert(n_weight == (1 << n_bit));
        const uint32_t n_cutoff = n_weight - 1;
        assert(cutoff_l_py.shape(0) == n_cutoff);

        const float *vec_l = vec_l_py.data();
        const float *centroid_l = centroid_l_py.data();
        const uint32_t *code_l = code_l_py.data();
        const float *weight_l = weight_l_py.data();
        const float *cutoff_l = cutoff_l_py.data();
        spdlog::info("ComputeResidualCode");

        std::vector<uint8_t> residual_code_l((size_t) n_vec * vec_dim);
        std::vector<float> residual_norm_l((size_t) n_vec);
#pragma omp parallel for default(none) shared(n_vec, code_l, vec_l, vec_dim, centroid_l, residual_code_l, n_weight, weight_l, cutoff_l, n_centroid, n_cutoff, residual_norm_l)
        for (size_t vecID = 0; vecID < n_vec; vecID++) {
            const uint32_t code = code_l[vecID];
            assert(code < n_centroid);
            const float *vec = vec_l + (size_t) vecID * vec_dim;

            std::vector<float> centroid(centroid_l + code * vec_dim, centroid_l + (code + 1) * vec_dim);

            uint8_t *residual_code = residual_code_l.data() + vecID * vec_dim;
            float norm = 0.0f;
            for (uint32_t dim = 0; dim < vec_dim; dim++) {
                const float error = vec[dim] - centroid[dim];
                norm += error * error;

                const float *ptr = std::upper_bound(cutoff_l, cutoff_l + n_cutoff, error,
                                                    [](const float &ele, const float &error) {
                                                        return ele < error;
                                                    });
                const uint8_t code_dist = ptr - cutoff_l;
#ifndef NDEBUG
                if (code_dist == n_cutoff) {
                    assert(cutoff_l[n_cutoff - 1] <= error);
                } else {
                    assert(error < cutoff_l[code_dist]);
                }
                assert(code_dist <= n_weight);
#endif
                residual_code[dim] = code_dist;
            }

            residual_norm_l[vecID] = std::sqrt(norm);

        }

#ifndef NDEBUG
        for (size_t ID = 0; ID < n_vec * vec_dim; ID++) {
            assert(0 <= residual_code_l[ID] && residual_code_l[ID] < n_weight);
        }
#endif

        return py::make_tuple(residual_code_l, residual_norm_l);
    }

    class ResidualCode {
    public:

        pyarray_uint8 residual_code_l_py_;
        const uint8_t *residual_code_l_;
        std::vector<float> weight_l_;
        uint32_t n_weight_;

        const uint32_t *item_n_vec_l_;
        const size_t *item_n_vec_accu_l_;
        uint32_t n_item_;
        uint32_t vec_dim_;

        const float *centroid_l_;
        const uint32_t *vq_code_l_;

        ResidualCode() = default;

        ResidualCode(pyarray_uint8 &residual_code_l_py,
                     const pyarray_float &weight_l_py,
                     const uint32_t *item_n_vec_l, const size_t *item_n_vec_accu_l,
                     const float *centroid_l, const uint32_t *vq_code_l,
                     const uint32_t &n_item, const size_t &n_vec,
                     const uint32_t &vec_dim) {

            assert(residual_code_l_py.ndim() == 1);
            assert(residual_code_l_py.shape(0) == (size_t) n_vec * vec_dim);
//            const uint8_t *residual_code_l = residual_code_l_py.data();
//            residual_code_l_ = residual_code_l;

            residual_code_l_py_ = std::move(residual_code_l_py);
            residual_code_l_ = residual_code_l_py_.data();

//            residual_code_l_.resize((size_t) n_vec * vec_dim);
//            std::memcpy(residual_code_l_.data(), residual_code_l_py.data(), sizeof(uint8_t) * (size_t) n_vec * vec_dim);

            assert(weight_l_py.ndim() == 1);
            const uint32_t n_weight = weight_l_py.shape(0);
//            weight_l_ = weight_l_py.data();
            weight_l_.resize(n_weight);
            std::memcpy(weight_l_.data(), weight_l_py.data(), sizeof(float) * n_weight);
            n_weight_ = n_weight;

            item_n_vec_l_ = item_n_vec_l;
            item_n_vec_accu_l_ = item_n_vec_accu_l;
            n_item_ = n_item;
            vec_dim_ = vec_dim;

            centroid_l_ = centroid_l;
            vq_code_l_ = vq_code_l;
        }

        void Decode(const uint32_t &itemID, float *item_vec) const {
            assert(itemID < n_item_);
            const size_t item_offset = item_n_vec_accu_l_[itemID];
            const uint32_t item_n_vec = item_n_vec_l_[itemID];

            const uint32_t *item_vq_code = vq_code_l_ + item_offset;
//            const uint8_t *item_residual_code = residual_code_l_.data() + item_offset * vec_dim_;
            const uint8_t *item_residual_code = residual_code_l_ + item_offset * vec_dim_;

            for (uint32_t vecID = 0; vecID < item_n_vec; vecID++) {
                const uint32_t vq_code = item_vq_code[vecID];
                std::memcpy(item_vec + (size_t) vecID * vec_dim_, centroid_l_ + (size_t) vq_code * vec_dim_,
                            sizeof(float) * vec_dim_);

                for (uint32_t dim = 0; dim < vec_dim_; dim++) {
                    const uint8_t residual_code = item_residual_code[vecID * vec_dim_ + dim];
                    assert(residual_code < n_weight_);
                    const float weight = weight_l_[residual_code];
                    item_vec[vecID * vec_dim_ + dim] += weight;
                }

                // normalize
//                const float ip = std::inner_product(item_vec + vecID * vec_dim_, item_vec + (vecID + 1) * vec_dim_,
//                                                    item_vec + vecID * vec_dim_, 0.0f);
//                const float norm = std::sqrt(ip);
//                for (uint32_t dim = 0; dim < vec_dim_; dim++) {
//                    item_vec[vecID * vec_dim_ + dim] /= norm;
//                }

            }

        }


    };


};
#endif //VECTORSETSEARCH_RESIDUALSCALARQUANTIZATION_HPP

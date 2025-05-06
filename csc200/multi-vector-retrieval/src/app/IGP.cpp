//
// Created by Administrator on 2025/1/17.
//
#include <pybind11/pybind11.h>
#include <pybind11/cast.h>

#include "include/struct/TypeDef.hpp"
#include "impl/IGP.hpp"
#include "include/alg/refine/ResidualScalarQuantization.hpp"

namespace VectorSetSearch::Method {

    PYBIND11_MODULE(IGP, m) {  // NOLINT
        m.def("compute_quantized_scalar", &ComputeQuantizedScalar,
              py::arg("item_vec_l"), py::arg("centroid_l"), py::arg("code_l"), py::arg("n_bit"));
        m.def("compute_residual_code", &ComputeResidualCode,
              py::arg("vec_l"), py::arg("centroid_l"), py::arg("code_l"),
              py::arg("cutoff_l"), py::arg("weight_l"),
              py::arg("n_bit"));

        py::class_<IGP>(m, "DocRetrieval",
                                      "The DocRetrieval module allows you to build, query, save, and load a "
                                      "semantic document search index.")
                .def(py::init<const std::vector<uint32_t> &,
                             const uint32_t &, const uint32_t &, const uint32_t &>(),
                     py::arg("item_n_vec_l"),
                     py::arg("n_item"), py::arg("vec_dim"), py::arg("n_centroid"))
                .def("build_index", &IGP::buildIndex,
                     py::arg("centroid_l"), py::arg("vq_code_l"),
                     py::arg("weight_l"), py::arg("residual_code_l"))
                .def("search", &IGP::search,
                     py::arg("query_l"), py::arg("topk"),
                     py::arg("nprobe"), py::arg("probe_topk"));

    }

}  // namespace VectorSetSearch::python
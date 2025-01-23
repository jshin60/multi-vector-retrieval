//
// Created by 13172 on 2024/8/7.
//
#include <pybind11/pybind11.h>
#include <pybind11/cast.h>

#include "include/struct/TypeDef.hpp"
#include "impl/MUVERA.hpp"

namespace VectorSetSearch::Method {

    PYBIND11_MODULE(MUVERA, m) {  // NOLINT

        m.def("assign_cluster_vector", &AssignClusterVector,
              py::arg("vec_cluster_bit_l"), py::arg("item_vecs_l_chunk"), py::arg("item_n_vec_l_chunk"),
              py::arg("batch_n_vec"), py::arg("batch_n_item"),
              py::arg("r_reps"), py::arg("k_sim"), py::arg("vec_dim"));

        py::class_<MUVERA>(m, "DocRetrieval",
                           "The DocRetrieval module allows you to build, query, save, and load a "
                           "semantic document search index.")
                .def(py::init<const std::vector<uint32_t> &,
                             const uint32_t, const uint32_t,
                             const uint32_t, const uint32_t, const uint32_t,
                             const uint32_t, const uint32_t , const double,
                             const uint32_t, const uint32_t>(),
                     py::arg("item_n_vec_l"),
                     py::arg("n_item"), py::arg("vec_dim"),
                     py::arg("k_sim"), py::arg("d_proj"), py::arg("r_reps"),
                     py::arg("R"), py::arg("L"), py::arg("alpha"),
                     py::arg("n_centroid_per_subspace"), py::arg("dim_per_subspace")
                )
                .def("add_projection", &MUVERA::add_projection,
                     py::arg("partition_vec_l"), py::arg("random_matrix_l")
                )
                .def("build_graph_index", &MUVERA::build_graph_index,
                     py::arg("ip_vector_l")
                )
                .def("add_item_vector_l", &MUVERA::add_item_vector_l,
                     py::arg("vec_l")
                )
                .def("add_pq_code_l", &MUVERA::add_pq_code_l,
                     py::arg("sub_centroid_l_l"), py::arg("sub_code_l_l")
                )
                .def("search", &MUVERA::search,
                     py::arg("query_l"), py::arg("topk"),
                     py::arg("n_candidate")
                );

    }

}  // namespace VectorSetSearch::python

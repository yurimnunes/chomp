// src/foretree_pybind.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring> // memcpy

// headers (updated paths)
#include "../include/tree/data_binner.hpp"
#include "../include/tree/gradient_hist_system.hpp"
#include "../include/tree/unified_tree.hpp"

namespace py = pybind11;

using foretree::DataBinner;
using foretree::EdgeSet;
using foretree::FeatureBins;
using foretree::GradientHistogramSystem;
using foretree::HistogramConfig;
using foretree::TreeConfig;
using foretree::UnifiedTree;

template <typename T>
static std::vector<T> arr_to_vec_1d(
    const py::array_t<T, py::array::c_style | py::array::forcecast> &a) {
    auto buf = a.request();
    if (buf.ndim != 1)
        throw std::invalid_argument("Expected a 1D array");
    const T *ptr = static_cast<const T *>(buf.ptr);
    return std::vector<T>(ptr, ptr + buf.shape[0]);
}

template <typename T>
static std::vector<T> arr_to_vec_any(
    const py::array_t<T, py::array::c_style | py::array::forcecast> &a) {
    auto buf = a.request();
    size_t n = 1;
    for (auto s : buf.shape)
        n *= static_cast<size_t>(s);
    const T *ptr = static_cast<const T *>(buf.ptr);
    return std::vector<T>(ptr, ptr + n);
}

PYBIND11_MODULE(foretree, m) {
    m.doc() = "Foretree: gradient-aware binning + unified tree (pybind11)";

    // ---------------- EdgeSet ----------------
    py::class_<EdgeSet>(m, "EdgeSet")
        .def(py::init<>())
        .def_readwrite("edges_per_feat", &EdgeSet::edges_per_feat)
        .def_readwrite("finite_bins", &EdgeSet::finite_bins)
        .def_readwrite("missing_bin_id", &EdgeSet::missing_bin_id);

    // ---------------- DataBinner (power-user API) ----------------
    py::class_<DataBinner>(m, "DataBinner")
        .def(py::init<int>(), py::arg("P"))
        .def(
            "register_edges",
            [](DataBinner &db, const std::string &mode,
               const std::vector<std::vector<double>> &edges_per_feat) {
                EdgeSet es;
                es.edges_per_feat = edges_per_feat;
                db.register_edges(mode, std::move(es));
            },
            py::arg("mode"), py::arg("edges_per_feat"))
        .def("register_edges_es", &DataBinner::register_edges, py::arg("mode"),
             py::arg("edge_set"))
        .def("set_node_override", &DataBinner::set_node_override,
             py::arg("mode"), py::arg("node_id"), py::arg("feature"),
             py::arg("edges"))
        .def(
            "prebin",
            [](const DataBinner &db,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast> &X,
               const std::string &mode, int node_id) {
                auto buf = X.request();
                if (buf.ndim != 2)
                    throw std::invalid_argument("X must be 2D (N, P)");
                const int N = static_cast<int>(buf.shape[0]);
                const int P = static_cast<int>(buf.shape[1]);
                const double *Xptr = static_cast<const double *>(buf.ptr);

                auto res = db.prebin(Xptr, N, P, mode, node_id);
                const int miss = res.second;
                const std::vector<uint16_t> &host = *res.first;

                py::array_t<uint16_t> out({N, P});
                auto out_buf = out.request();
                std::memcpy(out_buf.ptr, host.data(),
                            host.size() * sizeof(uint16_t));
                return py::make_tuple(out, miss);
            },
            py::arg("X"), py::arg("mode") = "hist", py::arg("node_id") = -1)
        .def("finite_bins", &DataBinner::finite_bins, py::arg("mode"))
        .def("missing_bin_id", &DataBinner::missing_bin_id, py::arg("mode"))
        .def("total_bins", &DataBinner::total_bins, py::arg("mode"));

    // ---------------- HistogramConfig ----------------
    py::class_<HistogramConfig>(m, "HistogramConfig")
        .def(py::init<>())
        .def_readwrite("method", &HistogramConfig::method)
        .def_readwrite("max_bins", &HistogramConfig::max_bins)
        .def_readwrite("use_missing_bin", &HistogramConfig::use_missing_bin)
        .def_readwrite("coarse_bins", &HistogramConfig::coarse_bins)
        .def_readwrite("density_aware", &HistogramConfig::density_aware)
        .def_readwrite("subsample_ratio", &HistogramConfig::subsample_ratio)
        .def_readwrite("min_sketch_size", &HistogramConfig::min_sketch_size)
        .def_readwrite("use_parallel", &HistogramConfig::use_parallel)
        .def_readwrite("max_workers", &HistogramConfig::max_workers)
        .def_readwrite("rng_seed", &HistogramConfig::rng_seed)
        .def_readwrite("eps", &HistogramConfig::eps)
        .def("total_bins", &HistogramConfig::total_bins)
        .def("missing_bin_id", &HistogramConfig::missing_bin_id);

    // ---------------- FeatureBins (introspection) ----------------
    py::class_<FeatureBins>(m, "FeatureBins")
        .def(py::init<>())
        .def_readwrite("edges", &FeatureBins::edges)
        .def_readwrite("is_uniform", &FeatureBins::is_uniform)
        .def_readwrite("strategy", &FeatureBins::strategy)
        .def_readwrite("lo", &FeatureBins::lo)
        .def_readwrite("width", &FeatureBins::width)
        .def("n_bins", &FeatureBins::n_bins);

    // ---------------- GradientHistogramSystem ----------------
    // Bind with shared_ptr so Python owns lifetime; we'll tie UnifiedTree to
    // it.
    py::class_<GradientHistogramSystem,
               std::shared_ptr<GradientHistogramSystem>>(
        m, "GradientHistogramSystem")
        .def(py::init<HistogramConfig>(), py::arg("config"))
        .def(
            "fit_bins",
            [](GradientHistogramSystem &self,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast> &X,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast> &g,
               const py::array_t<double, py::array::c_style |
                                             py::array::forcecast> &h) {
                auto xb = X.request();
                if (xb.ndim != 2)
                    throw std::invalid_argument("X must be 2D (N, P)");
                const int N = static_cast<int>(xb.shape[0]);
                const int P = static_cast<int>(xb.shape[1]);
                const double *Xptr = static_cast<const double *>(xb.ptr);

                std::vector<double> gv = arr_to_vec_1d<double>(g);
                std::vector<double> hv = arr_to_vec_1d<double>(h);
                if ((int)gv.size() != N || (int)hv.size() != N)
                    throw std::invalid_argument(
                        "g and h must have length N = X.shape[0]");

                self.fit_bins(Xptr, N, P, gv.data(), hv.data());
            },
            py::arg("X"), py::arg("g"), py::arg("h"))
        .def(
            "prebin_dataset",
            [](GradientHistogramSystem &self,
               const py::array_t<double, py::array::c_style |
                                             py::array::forcecast> &X) {
                auto xb = X.request();
                if (xb.ndim != 2)
                    throw std::invalid_argument("X must be 2D (N, P)");
                const int N = static_cast<int>(xb.shape[0]);
                const int P = static_cast<int>(xb.shape[1]);
                const double *Xptr = static_cast<const double *>(xb.ptr);

                auto pr = self.prebin_dataset(Xptr, N, P);
                const int miss = pr.second;
                const std::vector<uint16_t> &host = *pr.first;

                py::array_t<uint16_t> out({N, P});
                auto out_buf = out.request();
                std::memcpy(out_buf.ptr, host.data(),
                            host.size() * sizeof(uint16_t));
                return py::make_tuple(out, miss);
            },
            py::arg("X"))
        .def(
            "prebin_matrix",
            [](GradientHistogramSystem &self,
               const py::array_t<double, py::array::c_style |
                                             py::array::forcecast> &X) {
                auto xb = X.request();
                if (xb.ndim != 2)
                    throw std::invalid_argument("X must be 2D (N, P)");
                const int N = static_cast<int>(xb.shape[0]);
                const int P = static_cast<int>(xb.shape[1]);
                const double *Xptr = static_cast<const double *>(xb.ptr);

                auto pr = self.prebin_matrix(Xptr, N, P);
                const int miss = pr.second;
                const std::vector<uint16_t> &host = *pr.first;

                py::array_t<uint16_t> out({N, P});
                auto out_buf = out.request();
                std::memcpy(out_buf.ptr, host.data(),
                            host.size() * sizeof(uint16_t));
                return py::make_tuple(out, miss);
            },
            py::arg("X"))

        .def("finite_bins", &GradientHistogramSystem::finite_bins)
        .def("missing_bin_id", &GradientHistogramSystem::missing_bin_id)
        .def("total_bins", &GradientHistogramSystem::total_bins)
        .def("P", &GradientHistogramSystem::P)
        .def("N", &GradientHistogramSystem::N);

    // ---------------- TreeConfig ----------------
    py::class_<TreeConfig>(m, "TreeConfig")
        .def(py::init<>())
        .def_readwrite("max_depth", &TreeConfig::max_depth)
        .def_readwrite("max_leaves", &TreeConfig::max_leaves)
        .def_readwrite("min_samples_split", &TreeConfig::min_samples_split)
        .def_readwrite("min_samples_leaf", &TreeConfig::min_samples_leaf)
        .def_readwrite("min_child_weight", &TreeConfig::min_child_weight)
        .def_readwrite("lambda_", &TreeConfig::lambda_)
        .def_readwrite("alpha_", &TreeConfig::alpha_)
        .def_readwrite("gamma_", &TreeConfig::gamma_)
        .def_readwrite("max_delta_step", &TreeConfig::max_delta_step)
        .def_readwrite("n_bins", &TreeConfig::n_bins)
        .def_readwrite("leaf_gain_eps", &TreeConfig::leaf_gain_eps)
        .def_readwrite("allow_zero_gain", &TreeConfig::allow_zero_gain)
        .def_readwrite("leaf_depth_penalty", &TreeConfig::leaf_depth_penalty)
        .def_readwrite("leaf_hess_boost", &TreeConfig::leaf_hess_boost)
        .def_readwrite("colsample_bytree_percent",
                       &TreeConfig::colsample_bytree_percent)
        .def_readwrite("colsample_bylevel_percent",
                       &TreeConfig::colsample_bylevel_percent)
        .def_readwrite("colsample_bynode_percent",
                       &TreeConfig::colsample_bynode_percent)
        .def_readwrite("use_sibling_subtract",
                       &TreeConfig::use_sibling_subtract)
        .def_readwrite("monotone_constraints",
                       &TreeConfig::monotone_constraints);
    // ---------------- UnifiedTree (binned API) ----------------
    py::class_<UnifiedTree>(m, "UnifiedTree")
        .def(py::init<>())
        // Attach a GradientHistogramSystem; keep_alive so GHS outlives the tree
        .def(py::init([](const TreeConfig &cfg,
                         std::shared_ptr<GradientHistogramSystem> ghs) {
                 return new UnifiedTree(cfg, ghs.get());
             }),
             py::arg("config"), py::arg("ghs"), py::keep_alive<1, 2>())

        // --- training / inference ---
        .def(
            "fit_binned",
            [](UnifiedTree &self,
               const py::array_t<uint16_t,
                                 py::array::c_style | py::array::forcecast> &Xb,
               const py::array_t<float,
                                 py::array::c_style | py::array::forcecast> &g,
               const py::array_t<float, py::array::c_style |
                                            py::array::forcecast> &h) {
                auto xb = Xb.request();
                if (xb.ndim != 2)
                    throw std::invalid_argument("Xb must be 2D (N, P)");
                const int N = static_cast<int>(xb.shape[0]);
                const int P = static_cast<int>(xb.shape[1]);

                std::vector<uint16_t> Xv = arr_to_vec_any<uint16_t>(Xb);
                std::vector<float> gv = arr_to_vec_1d<float>(g);
                std::vector<float> hv = arr_to_vec_1d<float>(h);
                if ((int)gv.size() != N || (int)hv.size() != N)
                    throw std::invalid_argument(
                        "g and h must have length N = Xb.shape[0]");

                self.fit(Xv, N, P, gv, hv);
            },
            py::arg("Xb"), py::arg("g"), py::arg("h"))
        .def(
            "predict_binned",
            [](const UnifiedTree &self,
               const py::array_t<uint16_t, py::array::c_style |
                                               py::array::forcecast> &Xb) {
                auto xb = Xb.request();
                if (xb.ndim != 2)
                    throw std::invalid_argument("Xb must be 2D (N, P)");
                const int N = static_cast<int>(xb.shape[0]);
                const int P = static_cast<int>(xb.shape[1]);

                std::vector<uint16_t> Xv = arr_to_vec_any<uint16_t>(Xb);
                std::vector<double> pred = self.predict(Xv, N, P);

                py::array_t<double> out(pred.size());
                std::memcpy(out.request().ptr, pred.data(),
                            pred.size() * sizeof(double));
                return out;
            },
            py::arg("Xb"))

        // --- introspection: properties ---
        .def_property_readonly("n_nodes", &UnifiedTree::n_nodes)
        .def_property_readonly("n_leaves", &UnifiedTree::n_leaves)
        .def_property_readonly("depth", &UnifiedTree::depth)

        // --- feature importance (method returning a Python list) ---
        .def("feature_importance_gain",
             [](const UnifiedTree &self) {
                 // return a copy as a Python list (simple & safe)
                 return self.feature_importance_gain();
             })
        .def("post_prune_ccp", &UnifiedTree::post_prune_ccp,
             py::arg("ccp_alpha"));
}

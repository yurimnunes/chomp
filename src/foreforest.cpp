// foreforest_bindings.cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>

#include "../include/tree/gradient_hist_system.hpp"
#include "../include/tree/unified_tree.hpp"
#include "../include/tree/forest.hpp"   // the header from the previous turn (ForeForest + ForeForestConfig)

namespace nb = nanobind;
using nb::ndarray;
using nb::dtype;

using foretree::HistogramConfig;
using foretree::TreeConfig;
using foretree::ForeForestConfig;
using foretree::ForeForest;

template <class T>
static inline void ensure_contig_2d(const ndarray<T>& a, const char* name) {
    if (a.ndim() != 2)
        throw std::invalid_argument(std::string(name) + ": expected 2D array");
    if (!a.is_c_contiguous())
        throw std::invalid_argument(std::string(name) + ": expected C-contiguous row-major");
}

template <class T>
static inline void ensure_1d(const ndarray<T>& a, const char* name) {
    if (a.ndim() != 1)
        throw std::invalid_argument(std::string(name) + ": expected 1D array");
    if (!a.is_c_contiguous())
        throw std::invalid_argument(std::string(name) + ": expected contiguous array");
}

NB_MODULE(foreforest, m) {
    m.doc() = "ForeForest — bagging/GBDT with DART, nanobind bindings";

    // --------------------- HistogramConfig ---------------------
    nb::class_<HistogramConfig>(m, "HistogramConfig")
        .def(nb::init<>())
        .def_readwrite("method", &HistogramConfig::method)
        .def_readwrite("max_bins", &HistogramConfig::max_bins)
        .def_readwrite("use_missing_bin", &HistogramConfig::use_missing_bin)
        .def_readwrite("coarse_bins", &HistogramConfig::coarse_bins)
        .def_readwrite("density_aware", &HistogramConfig::density_aware)
        .def_readwrite("min_bins", &HistogramConfig::min_bins)
        .def_readwrite("target_bins", &HistogramConfig::target_bins)
        .def_readwrite("adaptive_binning", &HistogramConfig::adaptive_binning)
        .def_readwrite("importance_threshold", &HistogramConfig::importance_threshold)
        .def_readwrite("complexity_threshold", &HistogramConfig::complexity_threshold)
        .def_readwrite("use_feature_importance", &HistogramConfig::use_feature_importance)
        .def_readwrite("feature_importance_weights", &HistogramConfig::feature_importance_weights)
        .def_readwrite("subsample_ratio", &HistogramConfig::subsample_ratio)
        .def_readwrite("min_sketch_size", &HistogramConfig::min_sketch_size)
        .def_readwrite("use_parallel", &HistogramConfig::use_parallel)
        .def_readwrite("max_workers", &HistogramConfig::max_workers)
        .def_readwrite("rng_seed", &HistogramConfig::rng_seed)
        .def_readwrite("eps", &HistogramConfig::eps)
        .def("total_bins", &HistogramConfig::total_bins)
        .def("missing_bin_id", &HistogramConfig::missing_bin_id);

    // --------------------- TreeConfig --------------------------
    nb::class_<TreeConfig>(m, "TreeConfig")
        .def(nb::init<>())
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
        .def_readwrite("feature_bagging_k", &TreeConfig::feature_bagging_k)
        .def_readwrite("feature_bagging_with_replacement", &TreeConfig::feature_bagging_with_replacement)
        .def_readwrite("colsample_bytree_percent", &TreeConfig::colsample_bytree_percent)
        .def_readwrite("colsample_bylevel_percent", &TreeConfig::colsample_bylevel_percent)
        .def_readwrite("colsample_bynode_percent", &TreeConfig::colsample_bynode_percent)
        .def_readwrite("use_sibling_subtract", &TreeConfig::use_sibling_subtract)
        .def_readwrite("monotone_constraints", &TreeConfig::monotone_constraints)
        .def_readwrite("exact_cutover", &TreeConfig::exact_cutover)
        .def_readwrite("subsample_bytree", &TreeConfig::subsample_bytree)
        .def_readwrite("subsample_bylevel", &TreeConfig::subsample_bylevel)
        .def_readwrite("subsample_bynode", &TreeConfig::subsample_bynode)
        .def_readwrite("subsample_with_replacement", &TreeConfig::subsample_with_replacement)
        .def_readwrite("subsample_importance_scale", &TreeConfig::subsample_importance_scale)
        // enums
        .def_readwrite("growth", &TreeConfig::growth)
        .def_readwrite("missing_policy", &TreeConfig::missing_policy)
        .def_readwrite("split_mode", &TreeConfig::split_mode);

    nb::enum_<TreeConfig::Growth>(m, "Growth")
        .value("LeafWise", TreeConfig::Growth::LeafWise)
        .value("LevelWise", TreeConfig::Growth::LevelWise);

    nb::enum_<TreeConfig::MissingPolicy>(m, "MissingPolicy")
        .value("Learn", TreeConfig::MissingPolicy::Learn)
        .value("AlwaysLeft", TreeConfig::MissingPolicy::AlwaysLeft)
        .value("AlwaysRight", TreeConfig::MissingPolicy::AlwaysRight);

    nb::enum_<TreeConfig::SplitMode>(m, "SplitMode")
        .value("Histogram", TreeConfig::SplitMode::Histogram)
        .value("Exact", TreeConfig::SplitMode::Exact)
        .value("Hybrid", TreeConfig::SplitMode::Hybrid);

    // GOSS nested struct
    nb::class_<TreeConfig::GOSS>(m, "GOSS")
        .def(nb::init<>())
        .def_readwrite("enabled", &TreeConfig::GOSS::enabled)
        .def_readwrite("top_rate", &TreeConfig::GOSS::top_rate)
        .def_readwrite("other_rate", &TreeConfig::GOSS::other_rate)
        .def_readwrite("scale_hessian", &TreeConfig::GOSS::scale_hessian)
        .def_readwrite("min_node_size", &TreeConfig::GOSS::min_node_size);
    // Allow access via TreeConfig.goss
    nb::detail::get_class<TreeConfig>(m, "TreeConfig")
        .def_readwrite("goss", &TreeConfig::goss);

    // --------------------- ForeForestConfig --------------------
    nb::class_<ForeForestConfig>(m, "ForeForestConfig")
        .def(nb::init<>())
        .def_readwrite("mode", &ForeForestConfig::mode)
        .def_readwrite("n_estimators", &ForeForestConfig::n_estimators)
        .def_readwrite("learning_rate", &ForeForestConfig::learning_rate)
        .def_readwrite("rng_seed", &ForeForestConfig::rng_seed)
        .def_readwrite("use_raw_matrix_for_exact", &ForeForestConfig::use_raw_matrix_for_exact)
        .def_readwrite("hist_cfg", &ForeForestConfig::hist_cfg)
        .def_readwrite("tree_cfg", &ForeForestConfig::tree_cfg)
        .def_readwrite("rf_row_subsample", &ForeForestConfig::rf_row_subsample)
        .def_readwrite("rf_bootstrap", &ForeForestConfig::rf_bootstrap)
        .def_readwrite("rf_parallel", &ForeForestConfig::rf_parallel)
        .def_readwrite("gbdt_row_subsample", &ForeForestConfig::gbdt_row_subsample)
        .def_readwrite("gbdt_use_subsample", &ForeForestConfig::gbdt_use_subsample)
        .def_readwrite("dart_enabled", &ForeForestConfig::dart_enabled)
        .def_readwrite("dart_drop_rate", &ForeForestConfig::dart_drop_rate)
        .def_readwrite("dart_max_drop", &ForeForestConfig::dart_max_drop)
        .def_readwrite("dart_normalize", &ForeForestConfig::dart_normalize);

    nb::enum_<ForeForestConfig::Mode>(m, "Mode")
        .value("Bagging", ForeForestConfig::Mode::Bagging)
        .value("GBDT", ForeForestConfig::Mode::GBDT);

    // --------------------- ForeForest (Python-facing) -----------------
    nb::class_<ForeForest>(m, "ForeForest")
        .def(nb::init<ForeForestConfig>(), nb::arg("config"))

        // set_raw_matrix: float32 (N x P) + optional uint8 mask (N x P)
        .def("set_raw_matrix",
            [](ForeForest& self,
               ndarray<float> Xraw,
               nb::object miss_mask /* None or ndarray<uint8> */) {
                nb::scoped_interpreter_guard gil; // ensure GIL
                ensure_contig_2d(Xraw, "Xraw");
                const float* ptr = Xraw.data();
                const uint8_t* mptr = nullptr;

                ndarray<uint8_t> mask;
                if (!miss_mask.is_none()) {
                    mask = nb::cast<ndarray<uint8_t>>(miss_mask);
                    ensure_contig_2d(mask, "miss_mask");
                    if (mask.shape(0) != Xraw.shape(0) || mask.shape(1) != Xraw.shape(1))
                        throw std::invalid_argument("miss_mask: shape must match Xraw");
                    mptr = mask.data();
                }
                self.set_raw_matrix(ptr, mptr);
            },
            nb::arg("Xraw"), nb::arg("miss_mask") = nb::none())

        // fit_complete: X float64 (N x P), y float64 (N)
        .def("fit_complete",
            [](ForeForest& self, ndarray<double> X, ndarray<double> y) {
                nb::scoped_interpreter_guard gil;
                ensure_contig_2d(X, "X");
                ensure_1d(y, "y");
                const int N = (int) X.shape(0);
                const int P = (int) X.shape(1);
                if ((int) y.shape(0) != N)
                    throw std::invalid_argument("y length must equal X.shape[0]");
                self.fit_complete(X.data(), N, P, y.data());
            },
            nb::arg("X"), nb::arg("y"))

        // predict: X float64 (N x P) -> float64 (N)
        .def("predict",
            [](const ForeForest& self, ndarray<double> X) {
                nb::scoped_interpreter_guard gil;
                ensure_contig_2d(X, "X");
                const int N = (int) X.shape(0);
                const int P = (int) X.shape(1);
                std::vector<double> out = self.predict(X.data(), N, P);

                // Return as a NumPy array without extra copy
                auto arr = ndarray<double>::from_shape({(size_t)N});
                std::memcpy(arr.data(), out.data(), sizeof(double) * (size_t)N);
                return arr;
            },
            nb::arg("X"))

        .def("feature_importance_gain",
            [](const ForeForest& self) {
                std::vector<double> v = self.feature_importance_gain();
                auto arr = ndarray<double>::from_shape({(size_t)v.size()});
                if (!v.empty())
                    std::memcpy(arr.data(), v.data(), sizeof(double) * v.size());
                return arr;
            })

        .def("size", &ForeForest::size)
        .def("clear", &ForeForest::clear)
        ;
}

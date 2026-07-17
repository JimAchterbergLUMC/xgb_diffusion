/**
 * Copyright 2015-2025, XGBoost Contributors
 * \file simple_dmatrix.h
 * \brief In-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SIMPLE_DMATRIX_H_
#define XGBOOST_DATA_SIMPLE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>

#include <memory>
#include <string>
#include <vector>

#include "array_interface.h"
#include "gradient_index.h"

namespace xgboost::data {
// Used for single batch data.
class SimpleDMatrix : public DMatrix {
 public:
  SimpleDMatrix() = default;
  template <typename AdapterT>
  explicit SimpleDMatrix(AdapterT* adapter, float missing, std::int32_t nthread,
                         DataSplitMode data_split_mode = DataSplitMode::kRow);

  explicit SimpleDMatrix(dmlc::Stream* in_stream);
  ~SimpleDMatrix() override = default;

  void SaveToLocalFile(const std::string& fname);
  void XGBDDPMRefresh(ArrayInterface<2> const& x0, ArrayInterface<1> const& y0,
                      ArrayInterface<1> const& alpha_bars, std::int32_t noise_samples_per_row,
                      std::int32_t timestep, bst_feature_t target_index,
                      std::int32_t objective, std::uint64_t seed,
                      std::vector<std::int64_t> const& n_classes);
  void XGBDiffusionRefresh(ArrayInterface<2> const& x0, ArrayInterface<1> const& y0,
                           ArrayInterface<1> const& alpha_bars, ArrayInterface<1> const& times,
                           std::int32_t noise_samples_per_row, std::int32_t timestep,
                           bst_feature_t target_index, std::int32_t diffusion_type,
                           std::uint64_t seed);

  MetaInfo& Info() override;
  const MetaInfo& Info() const override;
  Context const* Ctx() const override { return &fmat_ctx_; }

  DMatrix* Slice(common::Span<int32_t const> ridxs) override;
  DMatrix* SliceCol(int num_slices, int slice_id) override;

  /*! \brief magic number used to identify SimpleDMatrix binary files */
  static const int kMagic = 0xffffab01;

 protected:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches(Context const* ctx) override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches(Context const* ctx) override;
  BatchSet<EllpackPage> GetEllpackBatches(Context const* ctx, const BatchParam& param) override;
  BatchSet<GHistIndexMatrix> GetGradientIndex(Context const* ctx, const BatchParam& param) override;
  BatchSet<ExtSparsePage> GetExtBatches(Context const* ctx, BatchParam const& param) override;

  MetaInfo info_;
  // Primary storage type
  std::shared_ptr<SparsePage> sparse_page_ = std::make_shared<SparsePage>();
  std::shared_ptr<CSCPage> column_page_{nullptr};
  std::shared_ptr<SortedCSCPage> sorted_column_page_{nullptr};
  std::shared_ptr<EllpackPage> ellpack_page_{nullptr};
  std::shared_ptr<GHistIndexMatrix> gradient_index_{nullptr};
  std::shared_ptr<common::HistogramCuts> xgbddpm_cuts_{nullptr};
  bool xgbddpm_enabled_{false};
  BatchParam batch_param_;

  bool EllpackExists() const override { return static_cast<bool>(ellpack_page_); }
  bool GHistIndexExists() const override { return static_cast<bool>(gradient_index_); }
  bool SparsePageExists() const override { return true; }
  std::shared_ptr<GHistIndexMatrix> MakeXGBDDFastGHist();
  void FinishXGBDDFastGHist(std::shared_ptr<GHistIndexMatrix> const& gidx,
                            std::vector<std::size_t> const& hit_count_tloc);

  /**
   * @brief Reindex the features based on a global view.
   *
   * In some cases (e.g. column-wise data split and vertical federated learning), features are
   * loaded locally with indices starting from 0. However, all the algorithms assume the features
   * are globally indexed, so we reindex the features based on the offset needed to obtain the
   * global view.
   */
  void ReindexFeatures(Context const* ctx, DataSplitMode split_mode);

 private:
  // Context used only for DMatrix initialization.
  Context fmat_ctx_;
};
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_H_

/*!
 * Copyright (c) 2016 by Contributors
 * \file mask_identity-inl.h
 * \brief Mask identity operator
 */
#ifndef MXNET_OPERATOR_MASK_IDENTITY_INL_H_
#define MXNET_OPERATOR_MASK_IDENTITY_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace mi_enum {
enum MaskIdentityOpInputs {kBBoxRg, kLandmarkRg, kBBoxRgGt, kLandmarkRgGt, kMask};
enum MaskIdentityOpOutputs {kBBoxRgGtOut, kLandmarkRgGtOut};
}  // namespace mi_enum

struct MaskIdentityParam : public dmlc::Parameter<MaskIdentityParam> {
};

template<typename xpu, typename DType>
class MaskIdentityOp : public Operator {
 public:
  explicit MaskIdentityOp(MaskIdentityParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // Do nothing
  }

 private:
  MaskIdentityParam param_;
};  // class MaskIdentityOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(MaskIdentityParam param, int dtype);

#if DMLC_USE_CXX11
class MaskIdentityProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"bbox_rg", "landmark_rg", "bbox_rg_gt", "landmark_rg_gt", "mask"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"bbox_rg_gt_refined", "landmark_rg_gt_refined"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 5) << "Input:[bbox_rg, landmark_rg, bbox_rg_gt, landmark_rg_gt, mask]";
    const TShape &shape1 = in_shape->at(0);
    const TShape &shape2 = in_shape->at(1);
    if (shape1.ndim() == 0 || shape2.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(shape1);
    out_shape->push_back(shape2);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MaskIdentityProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MaskIdentity";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return std::vector<int>();
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return std::vector<std::pair<int, void*> >();
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[kBBoxRgGt], out_data[kBBoxRgGtOut]},
            {in_data[kLandmarkRgGt], out_data[kLandmarkRgGtOut]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  MaskIdentityParam param_;
};  // class MaskIdentityProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MASK_IDENTITY_INL_H_

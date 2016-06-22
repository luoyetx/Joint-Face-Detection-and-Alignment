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

namespace mshadow {
namespace expr {
/*!
 * \breif Special Multiply Exp
 *  lhs is mask with shape [n, 1], rhs is data with shape [n, m]
 *  result with shape [n, m], where result[i, j] = lhs[i, 0] * data[i, j]
 *
 * \tparam xpu device type
 * \tparam DType data type
 */
template<typename xpu, typename DType>
struct SpecialMutExp : public Exp<SpecialMutExp<xpu, DType>, DType, type::kChainer> {
  const Tensor<xpu, 2, DType> &lhs_;
  const Tensor<xpu, 2, DType> &rhs_;
  SpecialMutExp(const Tensor<xpu, 2, DType> &lhs, const Tensor<xpu, 2, DType> &rhs)
    : lhs_(lhs), rhs_(rhs) {}
};  // struct SpecialMutExp

template<typename xpu, typename DType>
inline SpecialMutExp<xpu, DType>
spmt(const Tensor<xpu, 2, DType> &mask,
     const Tensor<xpu, 2, DType> &data) {
  return SpecialMutExp<xpu, DType>(mask, data);
}

template<typename xpu, typename DType>
struct Plan<SpecialMutExp<xpu, DType>, DType> {
 public:
  explicit Plan(const SpecialMutExp<xpu, DType> &e)
    : lhs_(e.lhs_), rhs_(e.rhs_) {}

  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return lhs_[y][0] * rhs_[y][x];
  }

 private:
  const Tensor<xpu, 2, DType> &lhs_;
  const Tensor<xpu, 2, DType> &rhs_;
};  // struct Plan

template<typename xpu, typename DType>
inline Plan<SpecialMutExp<xpu, DType>, DType>
MakePlan(const SpecialMutExp<xpu, DType> &e) {
  return Plan<SpecialMutExp<xpu, DType>, DType>(e);
}

template<int dim, typename xpu, typename DType>
struct ShapeCheck<dim, SpecialMutExp<xpu, DType> > {
  inline static Shape<dim> Check(const SpecialMutExp<xpu, DType> &e) {
    CHECK_EQ(dim, 2) << "SpecialMutExp only support 2D";
    Shape<dim> shape(e.rhs_.shape_);
    return shape;
  }
};  // struct ShapeCheck

template<typename xpu, typename DType>
struct ExpInfo<SpecialMutExp<xpu, DType> > {
  static const int kDim = 2;
  static const int kDevMask = xpu::kDevMask;
};  // struct ExpInfo

}  // namespace expr
}  // namespace mshadow

namespace mxnet {
namespace op {

namespace mi_enum {
enum MaskIdentityOpInputs {kBBox, kLandmark, kBBoxGt, kLandmarkGt, kBBoxMask, kLandmarkMask};
enum MaskIdentityOpOutputs {kBBoxGtOut, kLandmarkGtOut};
}  // namespace mi_enum

struct MaskIdentityParam : public dmlc::Parameter<MaskIdentityParam> {
  DMLC_DECLARE_PARAMETER(MaskIdentityParam) {
  }
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
    CHECK_EQ(in_data.size(), 6);
    CHECK_EQ(out_data.size(), 2);
    Stream<xpu> *s =  ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> bbox = in_data[mi_enum::kBBox].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> landmark = in_data[mi_enum::kLandmark].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> bbox_gt = in_data[mi_enum::kBBoxGt].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> landmark_gt = in_data[mi_enum::kLandmarkGt].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> bbox_mask = in_data[mi_enum::kBBoxMask].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> landmark_mask = in_data[mi_enum::kLandmarkMask].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> bbox_gt_refined = out_data[mi_enum::kBBoxGtOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> landmark_gt_refined = out_data[mi_enum::kLandmarkGtOut].FlatTo2D<xpu, DType>(s);
    // refined = mask*rg + gt
    bbox_gt_refined = spmt<xpu, DType>(bbox_mask, bbox) + bbox_gt;
    landmark_gt_refined = spmt<xpu, DType>(landmark_mask, landmark) + landmark_gt;
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
    return {"bbox", "landmark", "bbox_gt", "landmark_gt", "bbox_mask", "landmark_mask"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"bbox_gt_refined", "landmark_gt_refined"};
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
    CHECK_EQ(in_shape->size(), 6) << "Input:[bbox, landmark, bbox_gt, landmark_gt, bbox_mask, landmark_mask]";
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
    return {{in_data[mi_enum::kBBoxGt], out_data[mi_enum::kBBoxGtOut]},
            {in_data[mi_enum::kLandmarkGt], out_data[mi_enum::kLandmarkGtOut]}};
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

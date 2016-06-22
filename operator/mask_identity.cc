/*!
 * Copyright (c) 2016 by Contributors
 * \file mask_identity.cc
 * \brief
 */
#include "./mask_identity-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MaskIdentityParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MaskIdentityOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *MaskIdentityProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                             std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(MaskIdentityParam);

MXNET_REGISTER_OP_PROPERTY(MaskIdentity, MaskIdentityProp)
.describe("Dynamicly fill missing gt labels by mask")
.add_argument("bbox_rg", "Symbol", "BBox regression output from network")
.add_argument("landmark_rg", "Symbol", "Landmark regression output from network")
.add_argument("bbox_rg_gt", "Symbol", "BBox regression ground truth")
.add_argument("landmark_rg_gt", "Symbol", "Landmark regression ground truth")
.add_argument("mask", "Symbol", "Mask to fill or not")
.add_arguments(MaskIdentityParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

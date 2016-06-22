/*!
 * Copyright (c) 2016 by Contributors
 * \file mask_identity.cu
 * \brief
 */
#include "./mask_identity-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MaskIdentityParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MaskIdentityOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

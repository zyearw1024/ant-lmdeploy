// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T>
void invokeFusedAddBiasResidualRMSNorm(
    T* residual, T* inout, const T* bias, const T* scale, float eps, int batch_size, int n_dims, cudaStream_t stream);

}  // namespace fastertransformer
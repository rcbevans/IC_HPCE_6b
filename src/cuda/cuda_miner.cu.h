#include <stdint.h>
#include <stdlib.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <curand_kernel.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand_kernel.h>

#include "cuPrintf.cuh"
#include "cuPrintf.cu"

#include "cuda_wide_maths.cu.h"

#define NLIMBS 8

namespace bitecoin
{
	struct bigint_t
	{
	    uint32_t limbs[NLIMBS];
	};

__device__ void CudaFastPoolHashStep(bigint_t &x, const uint32_t *d_hashConstant)
{
    bigint_t tmp;
    cuda_wide_mul(4, tmp.limbs + 4, tmp.limbs, x.limbs, d_hashConstant);
    uint32_t carry = cuda_wide_add(4, x.limbs, tmp.limbs, x.limbs + 4);
    cuda_wide_add(4, x.limbs + 4, tmp.limbs + 4, carry);
}

// Given the various round parameters, this calculates the hash for a particular index value.
// Multiple hashes of different indices will be combined to produce the overall result.
__device__ bigint_t CudaFastPoolHash(const uint32_t *d_hashConstant, const uint32_t hashSteps, bigint_t &x)
{
    for (unsigned j = 0; j < hashSteps; j++)
    {
    	CudaFastPoolHashStep(x, d_hashConstant);
    }
    return x;
}

}//Close namespace
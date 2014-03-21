#include "cuda_miner.cu.h"

namespace bitecoin
{

__global__ void cudaInitial(uint32_t *d_ParallelBestProofs)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    bigint_t ones;
    cuda_wide_ones(8, ones.limbs);
    cuda_wide_copy(8, &d_ParallelBestProofs[globalID * 8], ones.limbs);
}

__global__ void cudaGenProof(uint32_t *d_ParallelIndices, uint32_t *d_ParallelProofs, const bigint_t x, const uint32_t *d_hashConstant, const uint32_t hashSteps, const unsigned baseNum, const unsigned offset)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t index = offset + globalID;

    bigint_t proof = cudaHash(d_hashConstant,
                              hashSteps,
                              index,
                              x);

    cuda_wide_copy(8, &d_ParallelProofs[globalID * 8], proof.limbs);
    d_ParallelIndices[globalID] = index;
}

__global__ void cudaWideCrossHash(uint32_t *d_ParallelIndices, uint32_t *d_ParallelProofs, uint32_t *d_ParallelBestProofs, uint32_t *d_ParallelBestSolutions, const unsigned cudaTotalSize, const uint32_t maxIndices)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    bigint_t candidateBestProof;
    cuda_wide_copy(8, candidateBestProof.limbs, &d_ParallelProofs[globalID * 8]);

    for (int offset = 1; offset < maxIndices; offset++)
    {
        cuda_wide_xor(8, candidateBestProof.limbs, candidateBestProof.limbs, &d_ParallelProofs[(globalID * 8) + offset]);
    }

    if(cuda_wide_compare(8, candidateBestProof.limbs, &d_ParallelBestProofs[globalID * 8]) < 0)
    {
        cuda_wide_copy(8, &d_ParallelBestProofs[globalID * 8], candidateBestProof.limbs);
        cuda_wide_copy(maxIndices, &d_ParallelBestSolutions[globalID * maxIndices], &d_ParallelIndices[globalID]);
    }
}

__global__ void cudaReduce(const unsigned cudaDim, const uint32_t maxIndices, uint32_t *d_ParallelBestProofs, uint32_t *d_ParallelBestSolutions)
{

    int threadID = threadIdx.x;
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 1; i < cudaDim; i++)
    {
        if (cuda_wide_compare(8, &d_ParallelBestProofs[(globalID * 8) + (i * cudaDim * 8)], &d_ParallelBestProofs[globalID * 8]) < 0)
        {
            cuda_wide_copy(8, &d_ParallelBestProofs[globalID * 8], &d_ParallelBestProofs[(globalID + i * cudaDim) * 8]);
            cuda_wide_copy(maxIndices, &d_ParallelBestSolutions[globalID * maxIndices], &d_ParallelBestSolutions[(globalID + i * cudaDim) * maxIndices]);
        }
    }

    for (unsigned toDo = cudaDim>>1; toDo <= 1; toDo >>= 1)
    {
        if (threadID < toDo)
        {
            if (cuda_wide_compare(8, &d_ParallelBestProofs[(threadID + toDo) * 8)], &d_ParallelBestProofs[threadID * 8]) < 0)
            {
                cuda_wide_copy(8, &d_ParallelBestProofs[threadID * 8], &d_ParallelBestProofs[(threadID  + toDo) * 8]);
                cuda_wide_copy(maxIndices, &d_ParallelBestSolutions[threadID * maxIndices], &d_ParallelBestSolutions[(threadID + toDo) * maxIndices]);
            }
        }
        __syncthreads();
    }
}

void cudaInit(const unsigned cudaDim, uint32_t *d_ParallelBestProofs)
{
    dim3 grid(cudaDim);
    dim3 threads(cudaDim);

    cudaInitial <<< grid, threads >>> (d_ParallelBestProofs);

    getLastCudaError("Kernel execution failed");

    cudaDeviceSynchronize();
}

void cudaIteration(uint32_t *d_ParallelIndices, uint32_t *d_ParallelProofs, uint32_t *d_ParallelBestProofs, uint32_t *d_ParallelBestSolutions, const bigint_t x, const uint32_t *d_hashConstant, const uint32_t hashSteps, const unsigned baseNum, const unsigned cudaDim, const unsigned offset, const unsigned cudaTotalSize, const uint32_t maxIndices)
{
    dim3 grid(cudaDim);
    dim3 threads(cudaDim);

    cudaGenProof <<< grid, threads >>> (d_ParallelIndices, d_ParallelProofs, x, d_hashConstant, hashSteps, baseNum, offset);

    getLastCudaError("Kernel execution failed");

    cudaDeviceSynchronize();

    cudaWideCrossHash <<< grid, threads >>> (d_ParallelIndices, d_ParallelProofs, d_ParallelBestProofs, d_ParallelBestSolutions, cudaTotalSize, maxIndices);

    getLastCudaError("Kernel execution failed");

    cudaDeviceSynchronize();
}

void cudaParallelReduce(const unsigned cudaDim, const uint32_t maxIndices, uint32_t *d_ParallelBestProofs, uint32_t *d_ParallelBestSolutions, uint32_t *gpuBestSolution, uint32_t *gpuBestProof)
{
    dim3 grid(1);
    dim3 threads(cudaDim);

    cudaReduce <<< grid, threads >>>(cudaDim, maxIndices, d_ParallelBestProofs, d_ParallelBestSolutions);

    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaMemcpyAsync(gpuBestProof, d_ParallelBestProofs, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpyAsync(gpuBestSolution, d_ParallelBestSolutions, sizeof(uint32_t) * maxIndices, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
}

}//End of Bitecoin Namespace
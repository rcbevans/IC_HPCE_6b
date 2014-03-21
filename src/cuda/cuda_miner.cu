#include "cuda_miner.cu.h"

namespace bitecoin
{

__global__ void cudaInitial(uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, const bigint_t x, const bigint_t nLessOne, const uint32_t *d_hashConstant, const uint32_t hashSteps, const unsigned baseNum)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t index = baseNum + globalID;

    d_ParallelSolutions[globalID] = index;

    bigint_t proof = oneHashReference(d_hashConstant,
                                      hashSteps,
                                      index,
                                      x,
                                      nLessOne);

    cuda_wide_copy(8, &d_ParallelProofs[globalID * 8], proof.limbs);
}

__global__ void cudaIteration(uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, const bigint_t x, const bigint_t nLessOne, const uint32_t *d_hashConstant, const uint32_t hashSteps, const unsigned baseNum, const unsigned cudaDim, const unsigned offset)
{

    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t index = baseNum + offset + globalID;

    // uint32_t index = offset + (threadIdx.x * (blockIdx.x+1)) << blockIdx.y;

    bigint_t proof = oneHashReference(d_hashConstant,
                                      hashSteps,
                                      index,
                                      x,
                                      nLessOne);

    if (cuda_wide_compare(8, proof.limbs, &d_ParallelProofs[globalID * 8]) < 0)
    {
        // if (threadIdx.x == 0 && blockIdx.x == 0)
        // {
        //     cuPrintf("from %lg to %lg\n", cuda_wide_as_double(8, &d_ParallelProofs[globalID * 8]), cuda_wide_as_double(8, proof.limbs));
        // }
        cuda_wide_copy(8, &d_ParallelProofs[globalID * 8], proof.limbs);
        cuda_wide_copy(1, &d_ParallelSolutions[globalID], &index);
    }
}

__global__ void cudaReduce(const unsigned cudaDim, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs)
{

    int threadID = threadIdx.x;
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 1; i < cudaDim; i++)
    {
        if (cuda_wide_compare(8, &d_ParallelProofs[(globalID + i * cudaDim) * 8], &d_ParallelProofs[globalID * 8]) < 0)
        {
            // if (threadIdx.x == 0 && blockIdx.x == 0)
            // {
            //     cuPrintf("Reduce from %lg to %lg\n", cuda_wide_as_double(8, &d_ParallelProofs[globalID * 8]), cuda_wide_as_double(8, &d_ParallelProofs[(globalID + i * cudaDim) * 8]));
            // }
            cuda_wide_copy(8, &d_ParallelProofs[globalID * 8], &d_ParallelProofs[(globalID + i * cudaDim) * 8]);
            cuda_wide_copy(1, &d_ParallelSolutions[globalID * 8], &d_ParallelSolutions[globalID + (i * cudaDim)]);
        }
    }

    __syncthreads();

    for (unsigned toDo = cudaDim >> 1; toDo <= 1; toDo >>= 1)
    {
        if (threadID < toDo)
        {
            if (cuda_wide_compare(8, &d_ParallelProofs[(threadID + toDo) * 8], &d_ParallelProofs[threadID * 8]) < 0)
            {
                // if (threadIdx.x == 0 && blockIdx.x == 0)
                // {
                //     cuPrintf("Reduce 2 from %lg to %lg\n", cuda_wide_as_double(8, &d_ParallelProofs[globalID * 8]), cuda_wide_as_double(8, &d_ParallelProofs[(threadID + toDo) * 8]));
                // }
                cuda_wide_copy(8, &d_ParallelProofs[threadID * 8], &d_ParallelProofs[(threadID + toDo) * 8]);
                cuda_wide_copy(1, &d_ParallelSolutions[threadID], &d_ParallelSolutions[threadID + toDo]);
            }
        }
        __syncthreads();
    }
}

void cudaInit(const unsigned cudaDim, const uint32_t hashSteps, const bigint_t &x, const bigint_t &nLessOne, const uint32_t *d_hashConstant, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, const unsigned baseNum)
{
    dim3 grid(cudaDim);
    dim3 threads(cudaDim);

    cudaInitial <<< grid, threads >>> (d_ParallelSolutions, d_ParallelProofs, x, nLessOne, d_hashConstant, hashSteps, baseNum);

    getLastCudaError("Kernel execution failed");

    cudaDeviceSynchronize();
}

void cudaIteration(const unsigned cudaDim, const unsigned baseNum, const unsigned offset, const uint32_t hashSteps, const bigint_t &x, const bigint_t &nLessOne, const uint32_t *d_hashConstant, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs)
{
    dim3 grid(cudaDim);
    dim3 threads(cudaDim);

    // cudaPrintfInit();

    cudaIteration <<< grid, threads >>> (d_ParallelSolutions, d_ParallelProofs, x, nLessOne, d_hashConstant, hashSteps, baseNum, cudaDim, offset);

    getLastCudaError("Kernel execution failed");

    // cudaPrintfDisplay(stdout, true);
    // cudaPrintfEnd();

    getLastCudaError("Kernel execution failed");

    cudaDeviceSynchronize();
}

void cudaParallelReduce(const unsigned cudaDim, const uint32_t maxIndices, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, uint32_t *gpuBestSolution, uint32_t *gpuBestProof)
{
    dim3 grid(1);
    dim3 threads(cudaDim);

    // cudaPrintfInit();

    cudaReduce <<< grid, threads >>>(cudaDim, d_ParallelSolutions, d_ParallelProofs);

    getLastCudaError("Kernel execution failed");

    // cudaPrintfDisplay(stdout, true);
    // cudaPrintfEnd();

    checkCudaErrors(cudaMemcpyAsync(gpuBestProof, d_ParallelProofs, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpyAsync(&gpuBestSolution[maxIndices - 1], d_ParallelSolutions, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
}

}//End of Bitecoin Namespace
#include "cuda_miner.cu.h"

namespace bitecoin
{

// __global__ void firstCudaRun(const bigint_t x, const uint32_t *d_hashConstant, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, unsigned long long time, curandState *d_state, uint32_t randomizer)
// {
//     extern __shared__ uint32_t sharedMem[];
//     uint32_t *localIndices = &sharedMem[0];
//     uint32_t *localProofs = &sharedMem[maxIndices];

//     int threadID = threadIdx.x;
//     int blockID = blockIdx.x;
//     int globalID = (blockID * blockDim.x) + threadID;

//     if (threadID == 0)
//     {
//      curand_init (time, globalID, threadID, d_state);

//         uint32_t curr = (4*blockID) + (uint32_t(curand(d_state)) & randomizer);
//         for (unsigned j = 0; j < maxIndices; j++)
//         {
//             curr += 1 + (4*blockID) + (uint32_t(curand(d_state)) & randomizer);
//             localIndices[j] = curr;
//         }
//     }

//     __syncthreads();

//     bigint_t fph = x;
//     fph.limbs[0] = localIndices[threadID];
//     bigint_t point = CudaFastPoolHash(d_hashConstant, hashSteps, fph);

//     cuda_wide_copy(8, &localProofs[threadID * 8], point.limbs);

//     __syncthreads();

//     if (threadID == 0)
//     {
//         for (unsigned j = 1; j < maxIndices; j++)
//         {
//             cuda_wide_xor(8, &localProofs[0], &localProofs[0], &localProofs[j * 8]);
//         }

//         cuda_wide_copy(8, &d_ParallelProofs[blockID * 8], &localProofs[0]);
//     }
// }

// __global__ void cudaIteration(const bigint_t x, const uint32_t *d_hashConstant, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, curandState *d_state, uint32_t randomizer)
// {
//     extern __shared__ uint32_t sharedMem[];
//     uint32_t *localIndices = &sharedMem[0];
//     uint32_t *localProofs = &sharedMem[maxIndices];

//     int threadID = threadIdx.x;
//     int blockID = blockIdx.x;

//     if (threadID == 0)
//     {
//         uint32_t curr = (4*blockID) + (uint32_t(curand(d_state)) & randomizer);
//         for (unsigned j = 0; j < maxIndices; j++)
//         {
//             curr += 1 + (uint32_t(curand(d_state)) & randomizer);
//             localIndices[j] = curr;
//         }
//     }

//     __syncthreads();

//     bigint_t fph = x;
//     fph.limbs[0] = localIndices[threadID];
//     bigint_t point = CudaFastPoolHash(d_hashConstant, hashSteps, fph);

//     cuda_wide_copy(8, &localProofs[threadID * 8], point.limbs);

//     __syncthreads();

//     if (threadID == 0)
//     {
//         for (unsigned j = 1; j < maxIndices; j++)
//         {
//             cuda_wide_xor(8, &localProofs[0], &localProofs[0], &localProofs[j * 8]);
//         }

//         if (cuda_wide_compare(8, &localProofs[0], &d_ParallelProofs[blockID * 8]) < 0)
//         {
//             cuda_wide_copy(maxIndices, &d_ParallelSolutions[blockID * maxIndices], &localIndices[0]);
//             cuda_wide_copy(8, &d_ParallelProofs[blockID * 8], &localProofs[0]);
//         }
//     }
// }

// __global__ void cudaReduce(const uint32_t maxIndices, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs)
// {
//     int threadID = threadIdx.x;
//     int blockWidth = blockDim.x;

//     for (unsigned toDo = blockWidth; blockWidth <= 1; blockWidth >>= 1)
//     {
//         if (threadID < toDo)
//         {
//             if (cuda_wide_compare(8, &d_ParallelProofs[(threadID * 8) + (toDo * 8)], &d_ParallelProofs[threadID * 8]) < 0)
//             {
//                 cuda_wide_copy(8, &d_ParallelProofs[threadID * 8], &d_ParallelProofs[(threadID * 8) + (toDo * 8)]);
//                 cuda_wide_copy(maxIndices, &d_ParallelSolutions[threadID * maxIndices], &d_ParallelSolutions[(threadID * maxIndices) + (toDo * maxIndices)]);
//             }
//         }
//         __syncthreads();
//     }
// }

__global__ void cudaInitial(uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, const bigint_t x, const bigint_t nLessOne, const uint32_t *d_hashConstant, const uint32_t hashSteps, const unsigned baseNum)
{
    // int threadID = threadIdx.x;
    // int blockID = blockIdx.x;
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
    // int threadID = threadIdx.x;
    // int blockID = blockIdx.x;
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t index = baseNum + (offset) + (globalID);

    // if (blockIdx.x == 0 && threadIdx.x == 0)
    // {
    //     // cuPrintf("%d %d %d %d %d %d %d %d\n", nLessOne.limbs[0], nLessOne.limbs[1], nLessOne.limbs[2], nLessOne.limbs[3], nLessOne.limbs[4], nLessOne.limbs[5], nLessOne.limbs[6], nLessOne.limbs[7]);
    //     cuPrintf("best Score: %g, index: %d\n", cuda_wide_as_double(8, &d_ParallelProofs[globalID * 8]), index);
    // }

    bigint_t proof = oneHashReference(d_hashConstant,
                                      hashSteps,
                                      index,
                                      x,
                                      nLessOne);

    if (cuda_wide_compare(8, proof.limbs, &d_ParallelProofs[globalID * 8]) < 0)
    {
        cuda_wide_copy(8, &d_ParallelProofs[globalID * 8], proof.limbs);
        cuda_wide_copy(1, &d_ParallelSolutions[globalID], &index);
    }
}

__global__ void cudaReduce(const unsigned cudaDim, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs)
{

    int threadID = threadIdx.x;
    // int blockID = blockIdx.x;
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < cudaDim; i++)
    {
        if (cuda_wide_compare(8, &d_ParallelProofs[(globalID * 8) + (i * cudaDim * 8)], &d_ParallelProofs[globalID * 8]) < 0)
        {
            cuda_wide_copy(8, &d_ParallelProofs[globalID * 8], &d_ParallelProofs[(globalID * 8) + (i * cudaDim * 8)]);
            cuda_wide_copy(1, &d_ParallelSolutions[globalID * 8], &d_ParallelSolutions[globalID + (i * cudaDim)]);
        }
    }

    for (unsigned toDo = cudaDim; toDo <= 1; toDo >>= 1)
    {
        if (threadID < toDo)
        {
            if (cuda_wide_compare(8, &d_ParallelProofs[(threadID * 8) + (toDo * 8)], &d_ParallelProofs[threadID * 8]) < 0)
            {
                cuda_wide_copy(8, &d_ParallelProofs[threadID * 8], &d_ParallelProofs[(threadID * 8) + (toDo * 8)]);
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

    // cudaPrintfInit ();

    cudaIteration <<< grid, threads >>> (d_ParallelSolutions, d_ParallelProofs, x, nLessOne, d_hashConstant, hashSteps, baseNum, cudaDim, offset);

    getLastCudaError("Kernel execution failed");

    // cudaPrintfDisplay (stdout, true);

    cudaDeviceSynchronize();
}

void cudaParallelReduce(const unsigned cudaDim, const uint32_t maxIndices, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, uint32_t *gpuBestSolution, uint32_t *gpuBestProof)
{
    dim3 grid(1);
    dim3 threads(cudaDim);

    cudaReduce <<< grid, threads >>>(cudaDim, d_ParallelSolutions, d_ParallelProofs);

    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaMemcpyAsync(gpuBestProof, d_ParallelProofs, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpyAsync(&gpuBestSolution[maxIndices - 1], d_ParallelSolutions, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
}

}//End of Bitecoin Namespace
#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint_client.hpp"

// Provies a basic (non-cryptographic) hash function
#include "contrib/fnv.hpp"

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>

#include <thread>

// CUDA runtime
#include <cuda_runtime.h>
// #include <curand_kernel.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

namespace bitecoin
{

extern void cudaMiningRun(unsigned cudaBlockCount, const bigint_t &bestProof, bigint_t &gpuBestProof, uint32_t *gpuBestSolution, const bigint_t &x, const uint32_t *d_hashConstant, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, uint32_t *GPUparallelProofs);

void threadTask(unsigned cudaBlockCount, const bigint_t &bestProof, bigint_t &gpuBestProof, uint32_t *gpuBestSolution, const bigint_t &x, const uint32_t *d_hashConstant, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, uint32_t *GPUparallelProofs)
{
    cudaMiningRun(cudaBlockCount, bestProof, gpuBestProof, &gpuBestSolution[0], x, d_hashConstant, maxIndices, hashSteps, d_ParallelSolutions, d_ParallelProofs, GPUparallelProofs);
}

class cudaEndpointClient : public EndpointClient
{
    //using EndpointClient::EndpointClient; //C++11 only!

protected:
    uint32_t *d_hashConstant, *d_ParallelProofs, *gpuParallelProofs;
    unsigned CUDA_PARALLEL_COUNT;

public:

    explicit cudaEndpointClient(std::string clientId,
                                std::string minerId,
                                std::unique_ptr<Connection> &conn,
                                std::shared_ptr<ILog> &log,
                                uint32_t *d_hashConstant,
                                uint32_t *d_ParallelProofs,
                                uint32_t *gpuParallelProofs,
                                unsigned cuda_jobs) : EndpointClient(clientId,
                                            minerId,
                                            conn,
                                            log)
    {
        this->d_hashConstant = d_hashConstant;
        this->d_ParallelProofs = d_ParallelProofs;
        this->gpuParallelProofs = gpuParallelProofs;
        this->CUDA_PARALLEL_COUNT = cuda_jobs;
    };

    void MakeBid(
        const std::shared_ptr<Packet_ServerBeginRound> roundInfo,   // Information about this particular round
        const std::shared_ptr<Packet_ServerRequestBid> request,     // The specific request we received
        double period,                                                                          // How long this bidding period will last
        double skewEstimate,                                                                // An estimate of the time difference between us and the server (positive -> we are ahead)
        std::vector<uint32_t> &solution,                                                // Our vector of indices describing the solution
        uint32_t *pProof                                                                        // Will contain the "proof", which is just the value
    )
    {
        double tSafetyMargin = 0.5; // accounts for uncertainty in network conditions
        /* This is when the server has said all bids must be produced by, plus the
            adjustment for clock skew, and the safety margin
        */
        double tFinish = request->timeStampReceiveBids * 1e-9 + skewEstimate - tSafetyMargin;

        Log(Log_Verbose, "MakeBid - start, total period=%lg.", period);

        /*
            We will use this to track the best solution we have created so far.
        */
        std::vector<uint32_t> bestSolution(roundInfo->maxIndices);
        std::vector<uint32_t> gpuBestSolution(roundInfo->maxIndices);
        bigint_t bestProof, gpuBestProof; //uint32_t [8];
        //set bestProof.limbs = 1's
        wide_ones(BIGINT_WORDS, bestProof.limbs);

        double worst = pow(2.0, BIGINT_LENGTH * 8); // This is the worst possible score

        // Incorporate the existing block chain data - in a real system this is the
        // list of transactions we are signing. This is the FNV hash:
        // http://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        hash::fnv<64> hasher;
        uint64_t chainHash = hasher((const char *)&roundInfo->chainData[0], roundInfo->chainData.size());

        bigint_t x;
        wide_x_init(&x.limbs[0], uint32_t(0), roundInfo->roundId, roundInfo->roundSalt, chainHash);

        std::vector<uint32_t> indices(roundInfo->maxIndices);

        // unsigned PARALLEL_COUNT = 16;

        // uint32_t *parallel_Indices = (uint32_t *)malloc(sizeof(uint32_t) * roundInfo->maxIndices * PARALLEL_COUNT);
        // uint32_t *parallel_Proofs = (uint32_t *)malloc(sizeof(uint32_t) * 8 * PARALLEL_COUNT);

        //Define GPU shit

        uint32_t *d_ParallelSolutions;

        checkCudaErrors(cudaMalloc((void **)&d_ParallelSolutions, sizeof(uint32_t)*CUDA_PARALLEL_COUNT * roundInfo->maxIndices));

        checkCudaErrors(cudaMemcpy(d_hashConstant, &roundInfo->c[0], sizeof(uint32_t) * 4, cudaMemcpyHostToDevice));

        // srand(now());

        // bestSolution[0] = 3;

        unsigned nTrials = 0;
        do
        {
            // ++nTrials;

            // nTrials += PARALLEL_COUNT;

            nTrials += CUDA_PARALLEL_COUNT;

            gpuBestProof = bestProof;

            std::thread<> gpuThread(&threadTask, CUDA_PARALLEL_COUNT, bestProof, gpuBestProof, &gpuBestSolution[0], x, d_hashConstant, roundInfo->maxIndices, roundInfo->hashSteps, d_ParallelSolutions, d_ParallelProofs, gpuParallelProofs);

            // auto parallelExecute = [ = ](unsigned i)
            // {
            //     uint32_t curr = (rand() & 8191);
            //     for (unsigned j = 0; j < roundInfo->maxIndices; j++)
            //     {
            //         // curr += 1 + (rand() % 10);
            //         curr += 1 + (rand() & 524287);
            //         parallel_Indices[(i * roundInfo->maxIndices) + j] = curr;
            //     }

            //     bigint_t proof = FastHashReference(roundInfo.get(), roundInfo->maxIndices, &parallel_Indices[i * roundInfo->maxIndices], x);

            //     wide_copy(8, &parallel_Proofs[i * 8], proof.limbs);
            // };

            // tbb::parallel_for<unsigned>(0, PARALLEL_COUNT, parallelExecute);

            // for (unsigned i = 0; i < PARALLEL_COUNT; i++)
            // {
            //     if (wide_compare(BIGINT_WORDS, &parallel_Proofs[i * 8], bestProof.limbs) < 0)
            //     {
            //         double score = wide_as_double(BIGINT_WORDS, &parallel_Proofs[i * 8]);
            //         Log(Log_Verbose, "    Found new best, nTrials=%d, score=%lg, ratio=%lg.", nTrials, score, worst / score);
            //         wide_copy(roundInfo->maxIndices, &bestSolution[0], &parallel_Indices[i * roundInfo->maxIndices]);
            //         wide_copy(8, bestProof.limbs, &parallel_Proofs[i * 8]);
            //     }
            // };

            // Log(Log_Debug, "Trial %d.", nTrials);

            // uint32_t curr = (rand() & 8191);
            // for (unsigned j = 0; j < indices.size(); j++)
            // {
            //     curr += 1 + (rand() & 524287);
            //     indices[j] = curr;
            // }

            // bigint_t proof = FastHashReference(roundInfo.get(), indices.size(), &indices[0], x);
            // // double score = wide_as_double(BIGINT_WORDS, proof.limbs);
            // // Log(Log_Debug, "    Score=%lg", score);

            // if (wide_compare(BIGINT_WORDS, proof.limbs, bestProof.limbs) < 0)
            // {
            //     double score = wide_as_double(BIGINT_WORDS, proof.limbs);
            //     Log(Log_Verbose, "    Found new best, nTrials=%d, score=%lg, ratio=%lg.", nTrials, score, worst / score);
            //     bestSolution = indices;
            //     bestProof = proof;
            // }

            // cudaMiningRun(CUDA_PARALLEL_COUNT, bestProof, gpuBestProof, &gpuBestSolution[0], x, d_hashConstant, roundInfo->maxIndices, roundInfo->hashSteps, d_ParallelSolutions, d_ParallelProofs, gpuParallelProofs);

            gpuThread.join();

            if (wide_compare(BIGINT_WORDS, gpuBestProof.limbs, bestProof.limbs) < 0)
            {
                double score = wide_as_double(BIGINT_WORDS, gpuBestProof.limbs);
                Log(Log_Verbose, "    Found new best GPU, score=%lg, ratio=%lg.", score, worst / score);
                bestSolution = gpuBestSolution;
                bestProof = gpuBestProof;
            }

        }
        while ((tFinish - now() * 1e-9) > 0);

        // free(parallel_Proofs);
        // free(parallel_Indices);

        checkCudaErrors(cudaFree(d_ParallelSolutions));

        solution = bestSolution;
        wide_copy(BIGINT_WORDS, pProof, bestProof.limbs);

        Log(Log_Verbose, "MakeBid - finish. Total trials %d", nTrials);

    }

};
};//End namespace bitecoin;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "bitecoin_client client_id logLevel connectionType [arg1 [arg2 ...]]\n");
        exit(1);
    }

    // We handle errors at the point of read/write
    signal(SIGPIPE, SIG_IGN);   // Just look at error codes

    try
    {
        std::string clientId = argv[1];
        std::string minerId = "Spunkmonkey's Miner";

        // Control how much is being output.
        // Higher numbers give you more info
        int logLevel = atoi(argv[2]);
        fprintf(stderr, "LogLevel = %s -> %d\n", argv[2], logLevel);

        std::vector<std::string> spec;
        for (int i = 3; i < argc; i++)
        {
            spec.push_back(argv[i]);
        }

        std::shared_ptr<bitecoin::ILog> logDest = std::make_shared<bitecoin::LogDest>(clientId, logLevel);
        logDest->Log(bitecoin::Log_Info, "Created log.");

        std::unique_ptr<bitecoin::Connection> connection {bitecoin::OpenConnection(spec)};

        unsigned CUDA_PARALLEL_COUNT = 128;

        uint32_t *d_hashConstant, *d_ParallelProofs, *gpuParallelProofs;

        checkCudaErrors(cudaMalloc((void **)&d_hashConstant, sizeof(uint32_t) * 4));
        checkCudaErrors(cudaMalloc((void **)&d_ParallelProofs, sizeof(uint32_t)*CUDA_PARALLEL_COUNT * 8));

        gpuParallelProofs = (uint32_t *)malloc(sizeof(uint32_t) * CUDA_PARALLEL_COUNT * 8);

        bitecoin::cudaEndpointClient endpoint(clientId, minerId, connection, logDest, d_hashConstant, d_ParallelProofs, gpuParallelProofs, CUDA_PARALLEL_COUNT);
        endpoint.Run();

    }
    catch (std::string &msg)
    {
        std::cerr << "Caught error string : " << msg << std::endl;
        return 1;
    }
    catch (std::exception &e)
    {
        std::cerr << "Caught exception : " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception." << std::endl;
        return 1;
    }

    return 0;
}
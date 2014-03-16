#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint_client.hpp"

// // Provies a basic (non-cryptographic) hash function
// #include "contrib/fnv.hpp"

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>

namespace bitecoin
{

class cudaEndpointClient : public EndpointClient
{
    //using EndpointClient::EndpointClient; //C++11 only!

public:

    explicit cudaEndpointClient(std::string clientId,
                                std::string minerId,
                                std::unique_ptr<Connection> &conn,
                                std::shared_ptr<ILog> &log) : EndpointClient(clientId,
                                            minerId,
                                            conn,
                                            log) {};

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
        bigint_t bestProof; //uint32_t [8];
        //set bestproof.limbs = 1's
        wide_ones(BIGINT_WORDS, bestProof.limbs);

        double worst = pow(2.0, BIGINT_LENGTH * 8); // This is the worst possible score

        // // Incorporate the existing block chain data - in a real system this is the
        // // list of transactions we are signing. This is the FNV hash:
        // // http://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        // hash::fnv<64> hasher;
        // uint64_t chainHash=hasher((const char*)&roundInfo->chainData[0], roundInfo->chainData.size());

        unsigned nTrials = 0;
        while (1)
        {
            ++nTrials;

            Log(Log_Debug, "Trial %d.", nTrials);
            std::vector<uint32_t> indices(roundInfo->maxIndices);
            uint32_t curr = 0;
            for (unsigned j = 0; j < indices.size(); j++)
            {
                curr = curr + 1 + (rand() % 10);
                indices[j] = curr;
            }

            bigint_t proof = HashReference(roundInfo.get(), indices.size(), &indices[0]);
            double score = wide_as_double(BIGINT_WORDS, proof.limbs);
            Log(Log_Debug, "    Score=%lg", score);

            if (wide_compare(BIGINT_WORDS, proof.limbs, bestProof.limbs) < 0)
            {
                Log(Log_Verbose, "    Found new best, nTrials=%d, score=%lg, ratio=%lg.", nTrials, score, worst / score);
                bestSolution = indices;
                bestProof = proof;
            }

            double t = now() * 1e-9; // Work out where we are against the deadline
            double timeBudget = tFinish - t;
            Log(Log_Debug, "Finish trial %d, time remaining =%lg seconds.", nTrials, timeBudget);

            if (timeBudget <= 0)
                break;  // We have run out of time, send what we have
        }

        solution = bestSolution;
        wide_copy(BIGINT_WORDS, pProof, bestProof.limbs);

        Log(Log_Verbose, "MakeBid - finish. Total %d trials.", nTrials);
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

        bitecoin::cudaEndpointClient endpoint(clientId, minerId, connection, logDest);
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
#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint.hpp"
#include "bitecoin_endpoint_server.hpp"

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h> 


namespace bitecoin{

class TestEndpointServer : public EndpointServer
{
public:
	explicit TestEndpointServer(
			std::string exchangeId,
			std::string serverId,
			std::unique_ptr<Connection> &conn,
			int logLevel=1) : EndpointServer(exchangeId, serverId, conn, logLevel){};

void Run()
{
		try{
			Log(Log_Info, "Waiting for client, exchangeId=%s, serverId=%s\n", m_exchangeId.c_str(), m_serverId.c_str());
			auto beginConnect=RecvPacket<Packet_ClientBeginConnect>();
			m_clientId=beginConnect->clientId;
			m_minerId=beginConnect->minerId;
			
			Log(Log_Info, "Received connection from clientId=%s, minerId=%s\n", m_clientId.c_str(), m_minerId.c_str());		
			
			auto completeConnect = std::make_shared<Packet_ServerCompleteConnect>(m_exchangeId, m_serverId);
			SendPacket(completeConnect);
			
			Log(Log_Verbose, "Connected to client.");
			
			uint64_t roundId=1;
			
			while(1){
				Log(Log_Info, "Starting round %llu.", roundId);
				
				auto beginRound=std::make_shared<Packet_ServerBeginRound>();
				beginRound->roundId=roundId;
				beginRound->roundSalt=100;
				beginRound->chainData.resize(500);
				beginRound->maxIndices=16;
				memset(beginRound->c, 0, BIGINT_LENGTH/2);
				// These are just arbitrary values. The real exchange may choose
				// different ones
				beginRound->c[0]=4294964621;
				beginRound->c[1]=4294967295;
				beginRound->c[2]=3418534911;
				beginRound->c[3]=2138916474;
				// Again exchange might choose differently
				beginRound->hashSteps=20;

				Log(Log_Verbose, "Sending chain data.\n");
				SendPacket(beginRound);
				
				auto requestBid=std::make_shared<Packet_ServerRequestBid>();

				// double roundLength=(rand()+1.0)/RAND_MAX;
				// roundLength=-log(roundLength)*2.75+0.25;
				// roundLength=std::max(0.25, std::min(60.0, roundLength));
				double roundLength = 3;

				timestamp_t start=now();
				timestamp_t finish=uint64_t(start+roundLength*1e9);
				
				assert(roundLength>=0.0);
				assert(roundLength<=60.0);
				
				requestBid->timeStampRequestBids=start;
				requestBid->timeStampReceiveBids=finish;
				
				SendPacket(requestBid);
				Log(Log_Verbose, "Requested bids.\n");

				auto bid=RecvPacket<Packet_ClientSendBid>();
				timestamp_t timeRecv=now();
				Log(Log_Verbose, "Received bid.\n");
				
				if(timeRecv > finish){
					Log(Log_Info, "Client bid too late.\n");
				}
				
				submission_t subClient;
				subClient.clientId=m_clientId.c_str();
				subClient.solution = bid->solution;
				memcpy(subClient.proof, bid->proof, BIGINT_LENGTH);
				subClient.timeSent=bid->timeSent;
				subClient.timeRecv=timeRecv;
				
				CheckSubmission(beginRound.get(), subClient);
				
				auto summary=std::make_shared<Packet_ServerCompleteRound>();
				summary->roundId=roundId;
				summary->winner=subClient;
				summary->submissions.push_back(subClient);
				
				SendPacket(summary);
				Log(Log_Info, "Round complete.\n");
					
				roundId++;
			}
		}catch(std::exception &e){
			Log(Log_Fatal, "Exception : %s.\n", e.what());
			throw;
		}
	}
};
};
int main(int argc, char *argv[])
{
	if(argc<2){
		fprintf(stderr, "bitecoin_client client_id logLevel connectionType [arg1 [arg2 ...]]\n");
		exit(1);
	}
	
	try{
		std::string clientId=argv[1];
		std::string minerId="David's Server";
		
		int logLevel=atoi(argv[2]);
		
		std::vector<std::string> spec;
		for(int i=3;i<argc;i++){
			spec.push_back(argv[i]);
		}		
		
		std::unique_ptr<bitecoin::Connection> connection{bitecoin::OpenConnection(spec)};
		
		bitecoin::TestEndpointServer endpoint(clientId, minerId, connection, logLevel);
		endpoint.Run();

	}catch(std::string &msg){
		std::cerr<<"Caught error string : "<<msg<<std::endl;
		return 1;
	}catch(std::exception &e){
		std::cerr<<"Caught exception : "<<e.what()<<std::endl;
		return 1;
	}catch(...){
		std::cerr<<"Caught unknown exception."<<std::endl;
		return 1;
	}
	
	return 0;
}


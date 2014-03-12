#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint_client.hpp"

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h> 
#include <csignal>

int main(int argc, char *argv[])
{
	if(argc<2){
		fprintf(stderr, "bitecoin_client client_id logLevel connectionType [arg1 [arg2 ...]]\n");
		exit(1);
	}
	
	// We handle errors at the point of read/write
	signal(SIGPIPE, SIG_IGN);	// Just look at error codes

	
	try{		
		std::string clientId=argv[1];
		std::string minerId="David's Miner";
		
		// Control how much is being output.
		// Higher numbers give you more info
		int logLevel=atoi(argv[2]);
		fprintf(stderr, "LogLevel = %s -> %d\n", argv[2], logLevel);
		
		std::vector<std::string> spec;
		for(int i=3;i<argc;i++){
			spec.push_back(argv[i]);
		}
		
		std::shared_ptr<bitecoin::ILog> logDest=std::make_shared<bitecoin::LogDest>(clientId, logLevel);
		logDest->Log(bitecoin::Log_Info, "Created log.");
		
		std::unique_ptr<bitecoin::Connection> connection{bitecoin::OpenConnection(spec)};
		
		bitecoin::EndpointClient endpoint(clientId, minerId, connection, logDest);
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


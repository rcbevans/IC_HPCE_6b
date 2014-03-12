#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint.hpp"
#include "bitecoin_endpoint_server.hpp"

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h> 

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
		
		bitecoin::EndpointServer endpoint(clientId, minerId, connection, logLevel);
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


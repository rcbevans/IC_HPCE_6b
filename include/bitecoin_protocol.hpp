#ifndef bitecoin_protocol_hpp
#define bitecoin_protocol_hpp

// Sigh. You stay classy, windows.
#define NOMINMAX

#include <cstdint>

#include "bitecoin_connection.hpp"

#include <time.h>

#include <sstream>

namespace bitecoin{
	
	/* All packets use the following protocol:
		8 byte packet length (including header).
		4 byte command id.
		4 byte sentinel. Can be any value
		[length-20] bytes of packet data
		4 byte sentinel. Must match the original sentinel
	
		The server will indicate any sort of error with a ServerError packet,
		which can be returned for any client packet.
	
		Times are represented using a 64 bit number representing nano-seconds
		from unix epoch (see time() for epoch definition).
	*/
	typedef uint64_t timestamp_t;
	
	enum{ BIGINT_LENGTH = 32 };
	enum{ BIGINT_WORDS = BIGINT_LENGTH/4 };
	
#if defined(_WIN32) || defined(_WIN64)
	timestamp_t now()
	{
		FILETIME ft;
		GetSystemTimeAsFileTime(&ft);
		uint64_t tt = ft.dwHighDateTime;
		tt = (tt<<32) + ft.dwLowDateTime;
		tt *=100;
		      //11636784000
		tt -= 11644473600000000000ULL;
		return tt;
	}
#elif defined(__MACH__)
	// http://stackoverflow.com/a/9781275
	#include <sys/time.h>
	//clock_gettime is not implemented on OSX
	uint64_t now()
	{
		struct timeval nn;
		if(gettimeofday(&nn, NULL))
			throw std::runtime_error("bitecoin::now() - Couldn't read time."); 
		return (nn.tv_sec*1000000ULL+nn.tv_usec)*1000;
	}
#else
	timestamp_t now()
	{
		struct timespec ts;
		if(0!=clock_gettime(CLOCK_REALTIME, &ts))
			throw std::runtime_error("bitecoin::now() - Couldn't read time."); 
		return uint64_t(1e9*ts.tv_sec+ts.tv_nsec);
	}
#endif

	
	class Packet{
	private:
		struct send_context_t{
			uint64_t length;
			uint32_t sentinel;
			uint64_t beginOffset;
		};
	
		send_context_t BeginSend(Connection *pConnection) const
		{
			send_context_t context={
				Length(),
				uint32_t(clock()+rand()),
				pConnection->SendOffset()
			}	;
			while(context.sentinel==0){
			  context.sentinel=uint32_t(clock()+rand());
			}
			
			if(context.length<20)
				throw std::logic_error("Packet::BeginSend - Cannot have a length of less than 20 bytes.");
			
			uint32_t command=CommandId();
			pConnection->Send(context.length);
			pConnection->Send(command);
			pConnection->Send(context.sentinel);
			
			//fprintf(stderr, "Sent[length=%llu, command=%u, sentinel=%u\n", context.length, command, context.sentinel);
			
			return context;
		}
		
		void EndSend(Connection *pConnection, const send_context_t &ctxt) const
		{
			pConnection->Send(ctxt.sentinel);
			
			uint64_t endOffset=pConnection->SendOffset();
			if(endOffset<ctxt.beginOffset)
				throw std::runtime_error("Packet::EndSend - 64-bit offset has wrapped, cannot handle more than 2^64 bytes of data.");
			
			uint64_t sent=endOffset-ctxt.beginOffset;
			if(sent != ctxt.length){
				std::stringstream acc;
				acc<<"Packet::EndSend - Sent data count ("<<sent<<") does not match what we said in header("<<ctxt.length<<")";
				throw std::logic_error(acc.str());
			}
		}
		
		static std::shared_ptr<Packet> CreatePacket(uint32_t command);
		
		void operator=(const Packet&); // = delete; // no implementation
		Packet(const Packet &); // = delete;
	protected:
		Packet()
		{}
			
		virtual void SendPayload(Connection *pConnection) const =0;
		virtual void RecvPayload(Connection *pConnection) =0;
		
		virtual uint64_t PayloadLength() const =0;
	public:
		virtual uint32_t CommandId() const=0;
		
		uint64_t Length() const
		{ return PayloadLength()+20; }
	
		void Send(Connection *pConnection) const
		{
			send_context_t ctxt=BeginSend(pConnection);
			SendPayload(pConnection);
			EndSend(pConnection, ctxt);
		}
	
		static std::shared_ptr<Packet> Recv(Connection *pConnection)
		{
			uint64_t length=0;
			uint32_t command=0, sentinelHeader=0, sentinelFooter=0;
			
			uint64_t beginOffset=pConnection->RecvOffset();
			
			pConnection->Recv(length);
			pConnection->Recv(command);
			pConnection->Recv(sentinelHeader);
			
			//fprintf(stderr, "Recvd[length=%llu, command=%u, sentinel=%u\n", length, command, sentinelHeader);
			
			if(length<20)
				throw std::runtime_error("Packet::Recv - Received packet length of < 20 bytes, which is not possible.");
			
			std::shared_ptr<Packet> res=CreatePacket(command);
			
			res->RecvPayload(pConnection);
			
			pConnection->Recv(sentinelFooter);
			if(sentinelHeader!=sentinelFooter)
				throw std::runtime_error("Packet::Recv - trailing sentinel does not match leading sentinel.");
			
			uint64_t endOffset=pConnection->RecvOffset();
			
			if(endOffset < beginOffset)
				throw std::runtime_error("Packet::Recv - Offset has wrapped, we don't support more than 2^64 bytes sent.");
			if(endOffset-beginOffset != length)
				throw std::runtime_error("Packet::Recv - Sent bytes does not match what we said in the header.");
			
			return res;
		}
	};
	
	enum{
		Command_any=0,	// Not a valid packet, used as a wildcard
		
		Command_ServerError=1,
		
		Command_ClientBeginConnect=2,
		Command_ServerCompleteConnect=3,
		
		Command_ServerBeginRound=4,	// Sent to all clients to start a mining round
		Command_ServerRequestBid=5,	// Sent to all clients to tell them the server wants  a bid
		Command_ClientSendBid=6,	// Sent by a client to indicate their response from a mining round
		Command_ServerCompleteRound=7		// Sent to all clients once round has finished
	};
	
	/*! After this packet is sent the connection is effectively shut, no further traffic is possible */
	class Packet_ServerError
		: public Packet
	{
	protected:
		virtual void RecvPayload(Connection *pConnection) override
		{
			pConnection->Recv(errorMessage);
		}

		virtual void SendPayload(Connection *pConnection) const override
		{
			pConnection->Send(errorMessage);
		}		
		
		virtual uint64_t PayloadLength() const override
		{ return 4+errorMessage.size(); }
	public:
		virtual uint32_t CommandId() const override
		{ return Command_ServerError; }
	
		std::string errorMessage;	// ASCII string identifying the error
	};
	
	class Packet_ClientBeginConnect
		: public Packet
	{
	protected:
		virtual void RecvPayload(Connection *pConnection)
		{
			pConnection->Recv(clientId);
			pConnection->Recv(minerId);
		}	
	
		virtual void SendPayload(Connection *pConnection) const
		{
			pConnection->Send(clientId);
			pConnection->Send(minerId);
		}
		
		virtual uint64_t PayloadLength() const override
		{ return 8+clientId.size()+minerId.size(); }
	
	public:
		Packet_ClientBeginConnect()
		{}
	
		Packet_ClientBeginConnect(std::string _clientId, std::string _minerId)
			: protocolVersion(0)
			, clientId(_clientId)
			, minerId(_minerId)
		{	
		}
	
		virtual uint32_t CommandId() const override
		{ return Command_ClientBeginConnect; }	
	
		uint32_t protocolVersion;	// Indicates the level of protocol supported by the client
		std::string clientId;	// ASCII string identifying the client program
		std::string minerId;		// ASCII string identifying the person doing the mining
	};
	
	class Packet_ServerCompleteConnect
		: public Packet
	{
	protected:
		virtual void RecvPayload(Connection *pConnection)
		{
			pConnection->Recv(protocolVersion);
			pConnection->Recv(serverId);
			pConnection->Recv(exchangeId);
		}
	
		virtual void SendPayload(Connection *pConnection) const
		{
			pConnection->Send(protocolVersion);
			pConnection->Send(serverId);
			pConnection->Send(exchangeId);
		}
		
		virtual uint64_t PayloadLength() const override
		{ return 4+8+serverId.size()+exchangeId.size(); }
	public:
		Packet_ServerCompleteConnect(std::string _exchangeId="<invalid>", std::string _serverId="<invalid>")
			: protocolVersion(0)
			, serverId(_serverId)
			, exchangeId(_exchangeId)
		{}
	
		virtual uint32_t CommandId() const override
		{ return Command_ServerCompleteConnect; }
		
		uint32_t protocolVersion;	// Indicates the protocol level being used (wil be <= level indicated by client). Network order.
		std::string serverId;		// ASCII string identifying server software
		std::string exchangeId;	// ASCII string identifying the exchange
	};
	
	class Packet_ServerBeginRound
		: public Packet
	{
	protected:
		virtual void RecvPayload(Connection *pConnection) override
		{
			pConnection->Recv(roundId);
			pConnection->Recv(roundSalt);
			pConnection->Recv(chainData);
			pConnection->Recv(maxIndices);
			for(unsigned i=0;i<BIGINT_WORDS/2;i++){
				pConnection->Recv(c[i]);
			}
			pConnection->Recv(hashSteps);
		}	
	
		virtual void SendPayload(Connection *pConnection) const override
		{
			pConnection->Send(roundId);
			pConnection->Send(roundSalt);
			pConnection->Send(chainData);
			pConnection->Send(maxIndices);
			for(unsigned i=0;i<BIGINT_WORDS/2;i++){
				pConnection->Send(c[i]);
			}
			pConnection->Send(hashSteps);
		}
		
		virtual uint64_t PayloadLength() const override
		{ return 8+8+4+chainData.size()+4+BIGINT_LENGTH/2+4; }
	public:
		virtual uint32_t CommandId() const override
		{ return Command_ServerBeginRound; }	
	
		uint64_t roundId;				// unique id associated with this round.
		uint64_t roundSalt;			// Random value chosen by the server
		std::vector<uint8_t> chainData;	// Chain data. On the wire consists of 64-bit length, followed by bytes of chain data
		uint32_t maxIndices;			// Maximum indices to return 
		uint32_t c[BIGINT_WORDS/2];		// Constant to use during hashing
		uint32_t hashSteps;				// Number of times to hash per point
	};
	
	class Packet_ServerRequestBid
		: public Packet
	{
	protected:
		virtual void RecvPayload(Connection *pConnection) override
		{
			pConnection->Recv(timeStampRequestBids);
			pConnection->Recv(timeStampReceiveBids);
		}	
	
		virtual void SendPayload(Connection *pConnection) const override
		{
			pConnection->Send(timeStampRequestBids);
			pConnection->Send(timeStampReceiveBids);
		}
		
		virtual uint64_t PayloadLength() const override
		{ return 2*sizeof(timestamp_t); }
	public:
		virtual uint32_t CommandId() const override
		{ return Command_ServerRequestBid; }	
	
		timestamp_t timeStampRequestBids;	// When the server considers the round to have started
		timestamp_t timeStampReceiveBids;		// When the server expects all responses by
	};
	
	class Packet_ClientSendBid
		: public Packet
	{
	protected:
		virtual void RecvPayload(Connection *pConnection) override
		{
			pConnection->Recv(roundId);
			pConnection->Recv(solution);
			for(unsigned i=0;i<BIGINT_WORDS;i++){
				pConnection->Recv(proof[i]);
			}
			pConnection->Recv(timeSent);
		}		
	
		virtual void SendPayload(Connection *pConnection) const override
		{
			pConnection->Send(roundId);
			pConnection->Send(solution);
			for(unsigned i=0;i<BIGINT_WORDS;i++){
				pConnection->Send(proof[i]);
			}
			pConnection->Send(timeSent);
		}	
		
		virtual uint64_t PayloadLength() const override
		{ return 8+4+solution.size()*4+BIGINT_LENGTH+sizeof(timestamp_t); }
	public:
		virtual uint32_t CommandId() const override
		{ return Command_ClientSendBid; }	
	
		uint64_t roundId;
		std::vector<uint32_t> solution;
		uint32_t proof[BIGINT_WORDS];
		timestamp_t timeSent;	// When the client purports to have sent this
	};
	
	struct submission_t
	{
		std::string clientId;
		std::vector<uint32_t> solution;
		uint32_t proof[BIGINT_WORDS];
		timestamp_t timeSent;	// When the client claims it was sent
		timestamp_t timeRecv;	// When the server thinks it recieved it
		
		void Send(Connection *pConnection) const
		{
			pConnection->Send(clientId);
			pConnection->Send(solution);
			for(unsigned i=0;i<BIGINT_WORDS;i++){
				pConnection->Send(proof[i]);
			}
			pConnection->Send(timeSent);
			pConnection->Send(timeRecv);
		}
		
		void Recv(Connection *pConnection)
		{
			pConnection->Recv(clientId);
			pConnection->Recv(solution);
			for(unsigned i=0;i<BIGINT_WORDS;i++){
				pConnection->Recv(proof[i]);
			}
			pConnection->Recv(timeSent);
			pConnection->Recv(timeRecv);
		}
		
		uint64_t PayloadLength() const
		{ return 4+clientId.size()+4+solution.size()*4+BIGINT_LENGTH+sizeof(timestamp_t)*2; }
	};
	
	class Packet_ServerCompleteRound
		: public Packet
	{
	protected:	
		void RecvPayload(Connection *pConnection) override
		{
			pConnection->Recv(roundId);
			winner.Recv(pConnection);
			pConnection->Recv(submissions);
		}	
	
		void SendPayload(Connection *pConnection) const override
		{
			pConnection->Send(roundId);
			winner.Send(pConnection);
			pConnection->Send(submissions);
		}
		
		virtual uint64_t PayloadLength() const override
		{
			uint64_t acc=8+winner.PayloadLength();
			acc+=4;
			for(const auto &sub : submissions){
				acc += sub.PayloadLength();
			}
			return acc;
		}
	public:
		virtual uint32_t CommandId() const override
		{ return Command_ServerCompleteRound; }	
	
		uint64_t roundId;
		submission_t winner;
		std::vector<submission_t> submissions;	// On the wire is a 64-bit length, followed by submissions
	};
	
	std::shared_ptr<Packet> Packet::CreatePacket(uint32_t command)
	{
		switch(command){
		case Command_ServerError:
			return std::make_shared<Packet_ServerError>();
		case	Command_ClientBeginConnect:
			return std::make_shared<Packet_ClientBeginConnect>();
		case Command_ServerCompleteConnect:
			return std::make_shared<Packet_ServerCompleteConnect>();
		case Command_ServerBeginRound:
			return std::make_shared<Packet_ServerBeginRound>();
		case Command_ServerRequestBid:
			return std::make_shared<Packet_ServerRequestBid>();
		case Command_ClientSendBid:
			return std::make_shared<Packet_ClientSendBid>();
		case Command_ServerCompleteRound:
			return std::make_shared<Packet_ServerCompleteRound>();
		default:
			{
				std::stringstream acc;
				acc<<"Packet::CreatePacket - Received a packet command id ("<<command<<") that isn't understood.";
				throw std::runtime_error(acc.str());
			}
		};
	}


}; // bitecoin

#endif

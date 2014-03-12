#ifndef bitecoin_log_hpp
#define bitecoin_log_hpp

#include <cstdio>
#include <cstdarg>

#include <stdarg.h>
#include <stdio.h>

namespace bitecoin
{

enum{
	Log_Fatal,
	Log_Error,
	Log_Info,
	Log_Verbose,
	Log_Debug
};
	
class ILog
{
protected:
	int m_logLevel;
	
	ILog(int logLevel)
		: m_logLevel(logLevel)
	{}
		
	ILog()
		: m_logLevel(4)
	{}
public:
	virtual ~ILog()
	{}
		
	void Log(int level, const char *str, ...)
	{
		if(level <= m_logLevel){
			va_list args;
			va_start(args,str);
			vLog(level, str, args);
			va_end(args);
		}
	}
	
	virtual void vLog(int level, const char *str, va_list args)=0;
};
	
class LogDest
	: public ILog
{
private:
	std::string m_prefix;

	std::string render(const char *str, va_list args)
	{
		std::vector<char> tmp(2000, 0);
			
		// TODO : This should be vsnprintf, but I'm having compiler problems.
		// using vsprintf opens up a potential overflow.
		unsigned n=vsprintf(&tmp[0], str, args);
		if(n>tmp.size()){
			tmp.resize(n);
			vsprintf(&tmp[0], str, args);
		}
		
		return std::string(&tmp[0]);
	}
public:
	LogDest(std::string prefix, int logLevel)
		: ILog(logLevel)
		, m_prefix(prefix)
	{}
		
	virtual void vLog(int level, const char *str, va_list args) override
	{
		if(level<=m_logLevel){
			double t=now()*1e-9;
			std::string msg=render(str, args);
			fprintf(stderr, "[%s], %.2f, %u, %s\n", m_prefix.c_str(), t, level, msg.c_str());
		}
	}
};

}; // bitecoin

#endif

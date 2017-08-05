#pragma once

#include <WinSock2.h>

#pragma comment(lib,"ws2_32.lib")

class testServer
{
public:
	testServer();
	virtual ~testServer();

	bool StartServer();
	bool CloseServer();
private:
	WSADATA wsaData;
	SOCKET sockServer;
	SOCKADDR_IN addrServer;
	SOCKET sockClient;
	SOCKADDR_IN addrClient;
};


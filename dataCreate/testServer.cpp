#include "testServer.h"
#include <iostream>


testServer::testServer()
{
}


testServer::~testServer()
{
}

bool testServer::StartServer()
{

	WSAStartup(MAKEWORD(2, 2), &wsaData);
	sockServer = socket(AF_INET, SOCK_STREAM, 0);
	addrServer.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(8234);
	bind(sockServer, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));

	//listen

	listen(sockServer, 1);//wait for one client;
	std::cout << "listenning..." << std::endl;
	int len = sizeof(SOCKADDR);
	char recvBuf[1024 * 10] = {0};

	sockClient = accept(sockServer, (SOCKADDR*)&addrClient, &len);

	recv(sockClient, recvBuf, sizeof(recvBuf), 0);
	
	return false;
}

bool testServer::CloseServer()
{
	closesocket(sockClient);
	WSACleanup();
	return false;
}

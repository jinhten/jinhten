#ifndef __km7Net_H_INCLUDED_2021_05_31__
#define __km7Net_H_INCLUDED_2021_05_31__

/* Note ----------------------
* kmMat has been created by Choi, Kiwan
* This is version 7
* kmMat v7 is including the following
*   - km7Define.h
*   - km7Define.h -> km7Mat.h
*   - km7Define.h -> km7Mat.h -> km7Wnd.h
*   - km7Define.h -> km7Mat.h -> kc7Mat.h -> km7Dnn.h
*   - km7Define.h -> km7Mat.h -> kc7Mat.h
*   - km7Define.h -> km7Mat.h -> km7Net.h
*/

// reserved port list
//
// FTP   : 21
// HTTP  : 80
// HTTPS : 8080

// base header
#include "km7Mat.h"

// general network header
#include <winsock2.h>
#include <ws2tcpip.h> // for inet_pton
#include <iphlpapi.h> // for GetIpAddrTable

#pragma comment(lib,"ws2_32")
#pragma comment(lib,"iphlpapi.lib")
///////////////////////////////////////////////////////////////
// enum and definition for net

#undef SetPort

// * Note that TCP's MTU is from ethernet v2's limit, 1500byte.
#define TCP_MTU_BYTE       1500
#define TCP_MSS_BYTE       1460  // 1500 - 20 (ip header) - 20 (tcp header)

// * Note that UDP's MTU and MSS are only for efficient transmit.
#define UDP_MTU_BYTE       1500
#define UDP_MSS_BYTE       (UDP_MTU_BYTE - 28) // 1500 - 20 (ip header) - 8 (udp header)

#define UDP_MAX_PKT_BYTE   65535
#define UDP_MAX_DATA_BYTE  65507 // 65525 - 20 (ip header) - 8 (udp header)

#define UDP_BLK_BYTE       1322  // safe size that doesn't split, 1340 - sizeof(kmNetHd) - 10 

#ifndef KMNETDPORT
#define KMNETDPORT 60165 // kmNet's default port
#endif

// socket type... tcp, udp
enum class kmSockType : int { tcp = 0, udp = 1 };

// nat type... full(full cone), rstr(restricted), prst(port-restricted), symm (symmetric)
enum class kmNatType { full, rstr, prst, symm };

///////////////////////////////////////////////////////////////////
// transfer functions between network and host
// network : big endian
// host    : big endian (linux ... ) or little endian (windows)

inline uchar  ntoh(uchar  a) { return        a ; };
inline wchar  ntoh(wchar  a) { return ntohs (a); };
inline ushort ntoh(ushort a) { return ntohs (a); };
inline uint   ntoh(uint   a) { return ntohl (a); };
inline uint64 ntoh(uint64 a) { return ntohll(a); };

inline char   ntoh(char   a) { return        a ; };
inline short  ntoh(short  a) { return ntohs (a); };
inline int    ntoh(int    a) { return ntohl (a); };
inline int64  ntoh(int64  a) { return ntohll(a); };

inline float  ntoh(float  a) { return ntohf(*((uint*  )&a)); };
inline double ntoh(double a) { return ntohd(*((uint64*)&a)); };

inline uchar  hton(uchar  a) { return        a ; };
inline wchar  hton(wchar  a) { return htons (a); };
inline ushort hton(ushort a) { return htons (a); };
inline uint   hton(uint   a) { return htonl (a); };
inline uint64 hton(uint64 a) { return htonll(a); };

inline char   hton(char   a) { return        a ; };
inline short  hton(short  a) { return htons (a); };
inline int    hton(int    a) { return htonl (a); };
inline int64  hton(int64  a) { return htonll(a); };

inline float  hton(float  a) { uint   b = htonf(a); return *((float* )&b); };
inline double hton(double a) { uint64 b = htond(a); return *((double*)&b); };

template<typename T> T& ntoh(T& a) { return a; }
template<typename T> T& hton(T& a) { return a; }

//////////////////////////////////////////////////////////////
// class for address

// mac address... 8byte
class kmMacAddr
{
public:
	union { uint64 i64; uchar c[8]; };

	// constructor
	kmMacAddr()         { i64 = 0; };
	kmMacAddr( int   a) { i64 = a; };
	kmMacAddr( int64 a) { i64 = a; };
	kmMacAddr(uchar* a) { Set(        a); };
	kmMacAddr( char* a) { Set((uchar*)a); };

	kmMacAddr(const kmMacAddr& a) { i64 = a.i64; };

	// set from char[6] or uchar[6]
	void Set(uchar* addr) { for(int i = 0; i < 8; ++i) c[i] = *(addr + i); };

	// assignment operator
	         kmMacAddr& operator=(const          kmMacAddr& a)          { i64 = a.i64; return *this; };	
	volatile kmMacAddr& operator=(const volatile kmMacAddr& a) volatile { i64 = a.i64; return *this; };

	// operator
	bool operator==(const kmMacAddr& b) const { return i64 == b.i64; };
	bool operator!=(const kmMacAddr& b) const { return i64 != b.i64; };

	// get string
	kmStra GetStr () const { return kmStra( "%02x-%02x-%02x-%02x-%02x-%02x-%02x-%02x", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]); };
	kmStrw GetStrw() const { return kmStrw(L"%02x-%02x-%02x-%02x-%02x-%02x-%02x-%02x", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]); };

	// print
	void Print() { print("* mac  : %s\n", GetStr().P()); };
};
typedef kmMat1<kmMacAddr> kmMacAddrs;

// ipv4 address's state... valid, invalid, pending
enum class kmAddr4State : uchar { valid = 0, invalid = 1, pending = 2 };

// ipv4 address's flag... 1 byte
class kmAddr4Flg
{
public:	
	uchar lan     : 1 = 0;  // 0 : unknown, 1 : lan
	uchar wan     : 1 = 0;  // 0 : unknown, 1 : wan
};

// ipv4 address... compatible with sockaddr_in (4 + 2 + 2 byte)
class kmAddr4
{
public:
	union { uint ip; uchar c[4];};   // compatible with sockaddr_in.sin_addr.s_addr
	ushort       port{};             // compatible with sockaddr_in.sin_port
	kmAddr4State state{};            // optional * Note that it's 8 byte even without this
	kmAddr4Flg   flg{};              // optional * Note that it's 8 byte even without this

	// constructor
	kmAddr4()                        :                ip(0)        {};
	kmAddr4(kmAddr4State state)      : state(state) , ip(0)        {};
	kmAddr4(int     a, ushort p = 0) : port(hton(p)), ip(a)        {};
	kmAddr4(uint    a, ushort p = 0) : port(hton(p)), ip(a)        {};
	kmAddr4(ulong   a, ushort p = 0) : port(hton(p)), ip(a)        {};
	kmAddr4(IN_ADDR a, ushort p = 0) : port(hton(p)), ip(a.s_addr) {};
	kmAddr4(LPCSTR  s, ushort p = 0) : port(hton(p)) { inet_pton(AF_INET, s, &ip); };
	kmAddr4(LPCWSTR s, ushort p = 0) : port(hton(p)) { InetPtonW(AF_INET, s, &ip); };
	kmAddr4(sockaddr_in saddr)
	{
		ip = saddr.sin_addr.s_addr; port = saddr.sin_port;
	};
	kmAddr4(uchar a0, uchar a1, uchar a2, uchar a3, ushort p = 0)
	{
		c[0] = a0; c[1] = a1; c[2] = a2; c[3] = a3; port = htons(p);
	};	

	// operator
	uchar operator()(int i) { return c[i]; };

	bool operator==(const kmAddr4& b) { return (ip == b.ip && port == b.port); };
	bool operator!=(const kmAddr4& b) { return (ip != b.ip || port != b.port); };

	// get sockaddr_in
	// * note that if ip is 0, it measns that INADDR_ANY
	sockaddr_in GetSckAddr()
	{
		sockaddr_in saddr;
		saddr.sin_family      = AF_INET;  // ipv4
		saddr.sin_addr.s_addr = ip;       // address (4 byte)
		saddr.sin_port        = port;     // port (2 byte)
		return saddr;
	};

	// get string
	kmStra GetStr   () const { return kmStra( "%d.%d.%d.%d:%d",c[0], c[1], c[2],c[3], ntohs(port)); };	
	kmStrw GetStrw  () const { return kmStrw(L"%d.%d.%d.%d:%d",c[0], c[1], c[2],c[3], ntohs(port)); };	
	kmStra GetIpStr () const { return kmStra( "%d.%d.%d.%d"   ,c[0], c[1], c[2],c[3]); };
	kmStrw GetIpStrw() const { return kmStrw(L"%d.%d.%d.%d"   ,c[0], c[1], c[2],c[3]); };

	// get state or flag
	bool  IsValid  () const { return state == kmAddr4State::valid;   };
	bool  IsInvalid() const { return state == kmAddr4State::invalid; };
	bool  IsPending() const { return state == kmAddr4State::pending; };
	bool  IsFlgLan () const { return flg.lan == 1; };
	bool  IsFlgWan () const { return flg.wan == 1; };

	// set state or flag
	void SetValid  () { state = kmAddr4State::valid;   };
	void SetInvalid() { state = kmAddr4State::invalid; };
	void SetPending() { state = kmAddr4State::pending; };
	void SetFlgLan () { flg.lan = 1; flg.wan = 0;  };
	void SetFlgWan () { flg.lan = 0; flg.wan = 1;  };

	// get port
	ushort GetPort() const { return ntoh(port); };

	// set port
	void SetPort(ushort p) { port = hton(p); };

	// print
	void Print() { print("* addr : %s\n", GetStr().P()); };
};
typedef kmMat1<kmAddr4> kmAddr4s;

// network adapter info class
class kmNetAdapter
{
public:
	int      n = 0;        // number of adapter

	kmStras  name;     // name of adapter
	kmAddr4s addr;     // ip address
	kmAddr4s mask;     // subnet mask
	
	// set adpter info
	void Set()
	{
		ulong len = sizeof(IP_ADAPTER_ADDRESSES); kmMat1i8 buf(len);

		ulong ret = GetAdaptersAddresses(AF_INET, 0, 0, (PIP_ADAPTER_ADDRESSES)buf.P(), &len);

		if(ret == ERROR_BUFFER_OVERFLOW)
		{	
			buf.Recreate(len);

			ret = GetAdaptersAddresses(AF_INET, 0, 0, (PIP_ADAPTER_ADDRESSES)buf.P(), &len);
		}

		if(ret != ERROR_SUCCESS) print("* GetSubnetMask failed\n");
		else
		{
			PIP_ADAPTER_ADDRESSES cur = (PIP_ADAPTER_ADDRESSES)buf.P();

			n = 0;
			name.Recreate(0,4);
			addr.Recreate(0,4);
			mask.Recreate(0,4);

			for(;cur != NULL; cur = cur->Next, n++)
			{
				// set adapter name
				kmStra cur_name(cur->AdapterName);

				// set address
				PIP_ADAPTER_UNICAST_ADDRESS address = cur->FirstUnicastAddress;
				
				ulong sockaddr = *((ulong*)(address->Address.lpSockaddr) + 1);

				kmAddr4 cur_addr(sockaddr);

				// set mask
				kmAddr4 cur_mask;
				ConvertLengthToIpv4Mask(address->OnLinkPrefixLength, (ulong*) &(cur_mask.ip));

				mask.PushBack(cur_mask);
				addr.PushBack(cur_addr);
				name.PushBack(cur_name);
			}
		}
	};
};

///////////////////////////////////////////////////////////////
// socket class with winsock2 and tcp/ip
//
// * Note that you should not define a destructor which will close the socket.
//
class kmSock
{
public:
	SOCKET _sck   = INVALID_SOCKET;
	int    _state = 0; // 0 : free, 1 : socket, 2 : bind, 3: listen, 4: connected

	void Init() { _sck = INVALID_SOCKET; _state = 0; };

	/////////////////////////////////
	// static functions

	// start winsock	
	static int Startup() try
	{
		WSADATA wsadata; int ret = WSAStartup(MAKEWORD(2,0), &wsadata);

		if(ret == 0) print("* kmSock::Startup\n");
		else         throw kmException::KE_NET_ERROR;
		
		print("* version       : %d, %d\n", wsadata.wVersion, wsadata.wHighVersion);
		print("* description   : %s\n", wsadata.szDescription);
		print("* system status : %s\n", wsadata.szSystemStatus);

		return ret;
	}
	catch(kmException e)
	{
		print("* kmSock::Startup failed with exception : %s\n", GetErrStr().P());
		Cleanup();
		throw e;
	};

	// clean up winsock
	static void Cleanup() { print("* kmSock::Cleanup\n"); WSACleanup(); };

	// get own local ip address
	//  if idx is -1, it'll choose the reasonable first one of addresses 
	static kmAddr4 GetLocalAddr(ushort port = 0, int idx = -1, int display_on = 0)
	{
		// get addr table
		MIB_IPADDRTABLE tbl[16]; ulong size = sizeof(tbl);
		
		if(GetIpAddrTable(tbl, &size, 0) != 0) return 0;

		// get addresses
		const int n = tbl->dwNumEntries;

		if(display_on > 0)
		{
			print("* GetLocalAddr()\n");
			for(int i = 0; i < n; ++i)
			{
				print("  [%d] ip   :%s\n", i, kmAddr4(tbl->table[i].dwAddr     ).GetIpStr().P());
				print("  [%d] brdc :%s\n", i, kmAddr4(tbl->table[i].dwBCastAddr).GetIpStr().P());
				print("  [%d] mask :%s\n", i, kmAddr4(tbl->table[i].dwMask     ).GetIpStr().P());
				print("  [%d] idx  :%lld\n", i, tbl->table[i].dwIndex);
			}
		}
		// chooese the reasonable first one of addresses
		if(idx < 0)	for(int i = 0; i < n; ++i)
		{
			kmAddr4 addr = tbl->table[i].dwAddr;

			if(addr.c[3] > 1) { idx = i; break; }
		}		
		return (idx >= n || idx < 0)? 0 : kmAddr4(tbl->table[idx].dwAddr, port);
	};	

	// get mac address... (6byte)
	static kmMacAddr GetMacAddr(int idx = 0, int display_on = 0)
	{
		// get mac address list
		IP_ADAPTER_INFO info[16]; ulong size = sizeof(info);

		int ret = GetAdaptersInfo(info, &size);

		kmMacAddr mac = 0;

		if(ret == ERROR_SUCCESS)
		{
			if(display_on > 0) print(" * GetMacAddr()\n");
			PIP_ADAPTER_INFO p = info; for(int i = 0; i < 16; ++i)
			{
				if(display_on > 0) print("  [%d] %s\n", i, kmMacAddr((uchar*)p->Address).GetStr().P());
				if(p->Next == 0) break; else p = p->Next;
			}
			mac.Set((uchar*)info[idx].Address);
		}
		return mac;
	};

	// get udp broadcasting address
	static kmAddr4 GetBrdcAddr(ushort port = 0, int idx = -1)
	{
		// get addr table
		MIB_IPADDRTABLE tbl[16]; ulong size = sizeof(tbl);

		if(GetIpAddrTable(tbl, &size, 0) != 0) return 0;

		// get addresses
		const int n = tbl->dwNumEntries;

		// chooese the reasonable first one of addresses
		if(idx < 0)	for(int i = 0; i < n; ++i)
		{
			kmAddr4 addr = tbl->table[i].dwAddr;

			if(addr.c[3] > 1) { idx = i; break; }
		}
		// get brdc addr
		kmAddr4 addr, mask, brdc_addr;

		if(idx >= 0 && idx < n)
		{
			addr = tbl->table[idx].dwAddr;
			mask = tbl->table[idx].dwMask;

			brdc_addr.ip = (addr.ip & mask.ip) | ~mask.ip;
			brdc_addr.SetPort(port);
		}
		else brdc_addr.SetInvalid();

		return brdc_addr;
	};

	// get arp table or subnet table
	//   req 0 : every ip, req 1 : only effective ip
	static int GetArpTable(kmAddr4s& ips, kmMacAddrs& macs, int req = 1)
	{
		// init parameters
		PMIB_IPNET_TABLE2 table = nullptr;

		// get arp table
		int ret = GetIpNetTable2(AF_INET, &table);

		if(ret == NO_ERROR)
		{
			const int n               = table->NumEntries;
			const PMIB_IPNET_ROW2 tbl = table->Table;

			ips.Recreate(0,n); macs.Recreate(0,n);

			for(int i = 0; i < n; ++i)
			{
				MIB_IPNET_ROW2& info = tbl[i];

				if(req == 0 || (0 < info.State && info.State < 6))
				{
					ips .PushBack(info.Address.Ipv4.sin_addr);
					macs.PushBack(info.PhysicalAddress);
				}
			}
			// display result
			for(int i = 0; i < ips.N(); ++i)
				print("[%03d] ip: %s, mac: %s\n",i, ips(i).GetIpStr().P(), macs(i).GetStr().P());
		}
		return ret;
	};

	// get WSA error
	static kmStra GetErrStr(int err_code = 0)
	{
		const char* str = nullptr;
		const int   err = (err_code == 0) ? WSAGetLastError():err_code;

		switch(err)
		{
		case WSANOTINITIALISED  : str = "not initialized"; break;
		case WSAENETDOWN        : str = "the network has failed"; break;
		case WSAEFAULT          : str = "the buf is not completely contained"; break;
		case WSAENOTCONN        : str = "the socket is not connected"; break;
		case WSAENETRESET       : str = "the connection has been broken"; break;
		case WSAESHUTDOWN       : str = "the socket has been shut down"; break;
		case WSAECONNABORTED    : str = "the connection was aborted by your host"; break;
		case WSAECONNRESET      : str = "the existing connection was forcibly closed"; break;
		case WSASYSNOTREADY     : str = "wsastartup cannot function"; break;
		case WSAVERNOTSUPPORTED : str = "the version requested is not supported"; break;
		case WSAEINPROGRESS     : str = "a blocking is currently executing"; break;
		case WSAENOTSOCK        : str = "the socket is not valid"; break;
		case WSAEADDRINUSE      : str = "the address is already in use"; break;
		case WSAEISCONN         : str = "the socket is already connected"; break;
		default                 : str = "default";
		}
		return kmStra("wsa err %d (%s)", err, str);
	};

	/////////////////////////////////
	// interface member functions

	// get socket
	// network address type : IPv4(AF_INET), IPv6(AF_INET6)
	// socket type          : TCP(SOCK_STREAM), UDP(SOCK_DGRAM)
	// protocol             : TCP(IPPROTO_TCP), UDP(IPPROTO_UDP)	
	void GetSocket(kmSockType type = kmSockType::tcp)
	{
		assertfa(_state == 0, "kmSock::GetSocket in 172");

		if(type == kmSockType::tcp) _sck = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		else                        _sck = socket(AF_INET, SOCK_DGRAM , IPPROTO_UDP);

		if(_sck != INVALID_SOCKET) _state = 1;

		assertfa(_sck != INVALID_SOCKET, "kmSock::GetSocket in 178");
	};

	// get rcv buffer size
	int GetRcvBufByte()
	{
		int byte = 0, len = sizeof(byte);

		getsockopt(_sck, SOL_SOCKET, SO_RCVBUF, (char*)&byte, &len);

		return byte;
	};	

	// get snd buffer size
	int GetSndBufByte()
	{
		int byte = 0, len = sizeof(byte);

		getsockopt(_sck, SOL_SOCKET, SO_SNDBUF, (char*)&byte, &len);

		return byte;
	};

	// set rcv and snd buffer size
	int SetRcvBufByte(int byte) { return setsockopt(_sck, SOL_SOCKET, SO_RCVBUF, (char*)&byte, sizeof(byte)); };
	int SetSndBufByte(int byte) { return setsockopt(_sck, SOL_SOCKET, SO_SNDBUF, (char*)&byte, sizeof(byte)); };

	// get max size to send
	int GetSndMaxByte()
	{
		int byte = 0, len = sizeof(byte);

		getsockopt(_sck, SOL_SOCKET, SO_MAX_MSG_SIZE, (char*)&byte, &len);

		return byte;
	};

	// set socket option
	void SetSckOptBroadcast(char on = true)
	{	
		int ret = setsockopt(_sck, SOL_SOCKET, SO_BROADCAST, &on, 1);

		assertfa(ret == 0, "kmSock::SetSckOptBroadcast in 320");
	};
	void SetSckOptReuseAddr(char on = true)
	{
		int ret = setsockopt(_sck, SOL_SOCKET, SO_REUSEADDR, &on, 1);

		assertfa(ret == 0, "kmSock::SetSckOptReuseaddr in 504");
	};

	// get socket type
	kmSockType GetSckType() const
	{
		int type, len = sizeof(int);

		getsockopt(_sck, SOL_SOCKET, SO_TYPE, (char*)&type, &len);

		return (type == SOCK_STREAM) ? kmSockType::tcp : kmSockType::udp;
	};

	// set nonblocking mode
	// * Note that in non-blockin mode, receiving procedure must be modified to wait until receiving a packet
	int SetNonblocking(uint on = 1) { return ioctlsocket(_sck, FIONBIO, (u_long*)&on); };
	
	// bind
	//  [return] 0                : if no error occurs
	//           SOCKET_ERROR(-1) : otherwise
	int Bind(kmAddr4 addr, kmSockType type = kmSockType::tcp)
	{
		// check state
		if(_state > 1) Close();

		// get socket
		if(_state == 0) GetSocket(type);

		// bind
		sockaddr_in sckaddr = addr.GetSckAddr();

		int ret = ::bind(_sck, (LPSOCKADDR)&sckaddr, sizeof(sckaddr));

		if(ret == 0) _state = 2;

		return ret;
	};

	// listen (for server)
	//  [return] 0                : if no error occurs
	//           SOCKET_ERROR(-1) : otherwise
	int Listen()
	{
		const int ret = ::listen(_sck, SOMAXCONN);

		if(ret == 0) _state = 3;

		return ret;
	};

	// accept client (for server)
	kmSock Accept()
	{
		kmSock client; sockaddr_in saddr; int len = sizeof(saddr);
	
		client._sck   = ::accept(_sck, (sockaddr*)&saddr, &len);

		if(client._sck != INVALID_SOCKET) client._state = 4;

		return client;
	};

	// connect to server (for client)
	//  [return] 0                : if no error occurs
	//           SOCKET_ERROR(-1) : otherwise
	int Connect(kmAddr4 addr, kmSockType type = kmSockType::tcp)
	{
		// get socket
		if(_state == 0) GetSocket(type);

		// connect
		sockaddr_in saddr = addr.GetSckAddr();

		int ret = ::connect(_sck, (LPSOCKADDR)&saddr, sizeof(saddr));

		if(ret == 0) _state = 4;

		return ret;
	};

	// receive and send	
	//   [return] 0 < : the number of bytes received or sent.
	//            0   : end of receiving process (Recv only)
	//            SOCKET_ERROR(-1) : you should call WSAGetLastError() to get error code.
	int Send(const kmStra& str) { return ::send(_sck, str.P(), (int)str.GetLen(), 0); };
	int Recv(      kmStra& str) { return ::recv(_sck, str.P(), (int)str.Size()  , 0); };

	// recive from (for udp)
	template<typename T> int Recvfrom(kmMat1<T>& data, kmAddr4& addr)
	{
		sockaddr_in saddr; int len = sizeof(saddr);

		int ret = ::recvfrom(_sck, data.P(), (int) data.Byte(), 0, (sockaddr*)&saddr, &len);

		data.SetN1(MAX(0, ret/sizeof(T)));

		addr = kmAddr4(saddr);

		return ret;
	}

	// send to (for udp)
	template<typename T> int Sendto(const kmMat1<T>& data, kmAddr4 addr)
	{
		sockaddr_in saddr = addr.GetSckAddr();

		// * Note that LGU's blocking range : 1349 - 1452 byte
		//             SKT's blocking range : 1389 - 1412 byte
		int byte = (int)data.N1()*sizeof(T), buf_byte = (int)data.Size()*sizeof(T);

		if(1340 < byte && byte < 1460 && 1460 <= buf_byte)
		{
			print("**** sendto byte was changed %d to 1460\n", byte);
			byte = 1460; 
		};
		/* only for debug
		int ret = 0;
		KM_CHECK_TIME_START
		ret = ::sendto(_sck, data.P(), byte, 0, (sockaddr*)&saddr, sizeof(saddr));		
		KM_CHECK_TIME_END("sendto");
		return ret;
		//*/
		return ::sendto(_sck, data.P(), byte, 0, (sockaddr*)&saddr, sizeof(saddr));
	}

	// send to (for udp broadcast) using global broadcast address (255.255.255.255)
	template<typename T> int SendtoBroadcast(const kmMat1<T>& data, ushort port = KMNETDPORT)
	{
		sockaddr_in saddr;
		saddr.sin_family      = AF_INET;
		saddr.sin_addr.s_addr = 0xffffffff; 
		saddr.sin_port        = htons(port);

		return ::sendto(_sck, data.P(), (int)data.N1()*sizeof(T), 0, (sockaddr*)&saddr, sizeof(saddr));
	};

	// send to (for udp broadcast) using local broadcast address (ex. 10.114.75.255)
	template<typename T> int SendtoBroadcastLocal(const kmMat1<T>& data, ushort port = KMNETDPORT)
	{
		kmAddr4      addr = kmSock::GetBrdcAddr(port);
		sockaddr_in saddr = addr.GetSckAddr();

		print("* broadcasting to %s\n", addr.GetStr().P());

		return ::sendto(_sck, data.P(), (int)data.N1()*sizeof(T), 0, (sockaddr*)&saddr, sizeof(saddr));
	};

	// shutdown
	//   [ how ] SD_RECEIVE (0) : shutdown receive operations
	//           SD_SEND    (1) : shutdown send operations
	//           SD_BOTH    (2) : shutdown both send and receive operations
	int Shutdown(int how = SD_BOTH)
	{
		if(_state == 0) return 0;

		const int ret = ::shutdown(_sck, how);

		if(ret == 0) _state = 1;

		return ret;
	};

	// close socket
	int Close()
	{
		if(_state == 0) return 0;

		const int ret = ::closesocket(_sck); Init();

		if(ret != 0) print("** [close] error occurs : %s\n", GetErrStr().P());
		
		return ret;
	};

	// get state
	//   0 : free, 1 : socket, 2 : bind, 3: listen, 4: connected
	int GetState() const { return _state; };

	// get source address
	kmAddr4 GetSrcAddr() const 
	{
		sockaddr_in saddr; int len = sizeof(saddr);

		getsockname(_sck, (sockaddr*)&saddr, &len);

		return kmAddr4(saddr);
	};

	// get destination address
	kmAddr4 GetDstAddr() const
	{
		sockaddr_in saddr; int len = sizeof(saddr);

		getpeername(_sck, (sockaddr*)&saddr, &len);

		return kmAddr4(saddr);
	};

	// print info of socket
	void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) print("%s\n", str);
		else               print("[socket info]\n");
		print("  handle      : %p\n", (void*)_sck);
		print("  state       : %d\n"  , _state); if(_state == 0) return;
		print("  type        : %s\n"  , (GetSckType() == kmSockType::tcp) ? "TCP":"UDP");
		print("  src address : %s\n"  , GetSrcAddr().GetStr().P());
		print("  dst address : %s\n"  , GetDstAddr().GetStr().P());
	};
};

//////////////////////////////////////////////////////////////////////
// network class for tcp
//
class kmTcp
{
protected:
	kmSock           _svr_sck;   // server socket
	kmThread         _svr_thrd;  // server thread
	kmMat1<kmSock>   _sck;       // client sockets
	kmMat1<kmThread> _thrd;      // receiving threads
	int              _n     = 0; // number of connected socket
	int              _n_max = 8; // max number of socket

public:
	// constructor
	kmTcp() { Init(); };
	
	// destructor
	virtual ~kmTcp() { CloseAll(); };

	// init
	void Init() { _sck.Recreate(_n_max); _thrd.Recreate(_n_max); _n = 0; };

	///////////////////////////////////////////////
	// interface functions

	// create server using thread	
	void CreateSvrThrd(kmAddr4 addr = 0)
	{
		// set address and port
		if(addr.ip   == 0) addr.ip   = kmSock::GetLocalAddr().ip; // or INADDR_ANY
		if(addr.port == 0) addr.port = htons(KMNETDPORT);

		// bind
		_svr_sck.Bind(addr);
						
		// create thread
		_svr_thrd.Begin([](kmTcp* net) { net->WaitForClt(); }, this);
	};

	// connect
	//   [return]  -1 : the connection failed
	//            0 < : index of socket (_sck)
	int Connect(kmAddr4 addr)
	{
		if(addr.port == 0) addr.port = htons(KMNETDPORT);

		int isck = FindFreeSck();
		int ret  = _sck(isck).Connect(addr);

		PRINTFA("[connect] find sck (%d)\n", isck);
		
		if(ret == 0) ++_n; else isck = -1;

		return isck;
	};

	// receiving process for _sck(isck)
	//   [return] 0   : end of receiving process
	//            0 < : the number of bytes received or sent.
	//            0 > : error of the socket
	//
	// * Note that this is an example. So you should re-define this virtual function.
	virtual int RcvProc(int isck)
	{
		kmStra str(32);
	
		const int n = _sck(isck).Recv(str);
	
		if(n > 0) print("-> [%d] receiving : (%d) %s\n", isck, n, str.P());

		return n;
	};

	// send data to every connected peer
	void Send(const kmStra& str)
	{
		for(int i = 0, n = _n; i < _n_max && n > 0; ++i)
		{
			if(_sck(i).GetState() == 4) { _sck(i).Send(str); n--; }
		}
	};

	// close isck socket and thread
	void Close(int isck) { _sck(isck).Close(); };

	// get socket
	kmSock& operator()(int i = -1) { return (i == -1)? _svr_sck : _sck(i); };

	///////////////////////////////////////////////
	// inner functions

	// find free socket
	int FindFreeSck()
	{
		for(int i = 0; i < _n_max; ++i)
		{
			if(_sck(i).GetState() == 0) return i;
		}
		return -1;
	};

	// wait for client (by server)
	void WaitForClt()
	{
		// set listening state
		if(_svr_sck.Listen() != 0) print("**[WaitForclt] listen : %s\n", kmSock::GetErrStr().P());

		while(1)
		{
			// wait for client and accept
			kmSock sck = _svr_sck.Accept();

			if(sck._sck == INVALID_SOCKET) break;
			else
			{
				// set sck
				int isck = FindFreeSck(); _sck(isck) = sck; ++_n; 

				// create receiving thread
				print("* [accept] sck(%d) from %s\n", isck, _sck(isck).GetDstAddr().GetStr().P());

				CreateRcvThrd(isck);
			}
		}
		print("* end of listening\n");
	};

	// create receiving thread
	void CreateRcvThrd(int isck)
	{
		_thrd(isck).Begin([](kmTcp* net, int isck)
		{
			print("* receiving [%d]\n", isck);
			int ret;

			while(1) { if((ret = net->RcvProc(isck)) < 1) break; }

			// check error
			if(ret < 0) print("**[sck %d] %s\n",isck, kmSock::GetErrStr().P());

			// close socket
			net->Close(isck);

			// decrease count
			--(net->_n);

			print("* end of receiving [%d]\n", isck);
		}, this, isck);
	};	

	// close all socket and thread
	virtual void CloseAll()
	{
		_svr_sck.Close();
		
		for(int i = 0; i < _n_max; ++i)
		{
			_sck(i).Close();
			if(_thrd(i).IsRunning()) _thrd(i).Wait();
		}
		_n = 0;
	};

	// print info
	void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) print("%s", str);
		print("=========================================\n");
		print(" n / n_max : %d / %d\n", _n, _n_max);

		_svr_sck.PrintInfo("svr_sck");
		for(int i = 0; i < _n_max; ++i)
		{
			if(_sck(i)._state > 0) _sck(i).PrintInfo(kmStra("sck(%d)",i).P());
		}
		print("=========================================\n");
	}
};

//////////////////////////////////////////////////////////////////////
// network class for udp
//
class kmUdp
{
protected:	
	kmSock   _sck;   // client sockets
	kmThread _thrd;  // receiving threads

public:
	// constructor
	kmUdp() { Init(); };
	
	// destructor
	virtual ~kmUdp() { CloseAll(); };

	// init
	void Init() {};

	///////////////////////////////////////////////
	// interface functions

	// bind
	int Bind(kmAddr4 addr) { return _sck.Bind(addr, kmSockType::udp); };

	// bind to own ip
	int Bind(ushort port = KMNETDPORT)
	{
		return _sck.Bind(kmSock::GetLocalAddr(port), kmSockType::udp);
	};
	
	// connect	
	int Connect(kmAddr4 addr = INADDR_ANY)
	{
		if(addr.port == 0) addr.port = htons(KMNETDPORT);

		return _sck.Connect(addr);
	};

	// receiving process for _sck(isck)
	//   [return] 0   : end of receiving process
	//            0 < : the number of bytes received or sent.
	//            0 > : error of the socket
	//
	// * Note that this is an example. So you should re-define this virtual function.
	virtual int RcvProc()
	{
		kmStra str(32); kmAddr4 addr;
	
		const int n = _sck.Recvfrom(str, addr);
	
		if(n > 0) print("-> receive from (%s) : %s\n", addr.GetStr().P(), str.P());

		return n;
	};

	// send data to every connected peer
	void Send(const kmStra& str, kmAddr4 addr)
	{
		_sck.Sendto(str, addr);
	};

	// close isck socket and thread
	void Close() { _sck.Close(); };

	// get socket
	kmSock& operator()() { return _sck; };

	///////////////////////////////////////////////
	// inner functions
	
	// create receiving thread
	void CreateRcvThrd()
	{
		_thrd.Begin([](kmUdp* net)
		{
			print("* receiving\n");
			int ret;

			while(1) { if((ret = net->RcvProc()) < 1) break; }

			// check error
			if(ret < 0) print("** %s\n", kmSock::GetErrStr().P());

			// close socket
			net->Close();

			print("* end of receiving\n");
		}, this);
	};	

	// close all socket and thread
	virtual void CloseAll()
	{	
		_sck.Close();
		if(_thrd.IsRunning()) _thrd.Wait();
	};

	// print info
	void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) print("%s", str);
		print("=====================================UDP\n");

		_sck.PrintInfo();

		print("=========================================\n");
	}
};

/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
// kmNet new version since 2021.12.16

// kmNet header flag
union kmNetHdFlg
{
	uchar uc; struct
	{
		uchar reqack   : 1;  // request for acknoledgement
		uchar reject   : 1;  // reject request
		uchar rsv2     : 1;
		uchar rsv3     : 1;
		uchar rsv4     : 1;
		uchar rsv5     : 1;
		uchar rsv6     : 1;
		uchar rsv7     : 1;
	};
	kmNetHdFlg()          { uc = 0;   };
	kmNetHdFlg(uchar flg) { uc = flg; };

	operator uchar() { return  uc; };
};

// kmNet header (8byte)
//  { src_id, des_id, ptc_id, cmd_id, opt, flg }
class kmNetHd
{
public:
	ushort      src_id = 0xffff;   // source ID as the sender's point of view
	ushort      des_id = 0xffff;   // destination ID as the sender's point of view
	uchar       ptc_id = 0;        // protocol ID
	 char       cmd_id = 0;        // command ID... minus means sending
	uchar       opt    = 0;        // option
	kmNetHdFlg  flg    = 0;        // flag

	// get flag
	inline bool IsReqAck() { return flg.reqack; };
	inline bool IsReject() { return flg.reject; };

	// set flag
	inline void SetReqAck() { flg.reqack = 1; };
	inline void SetReject() { flg.reject = 1; };
};

// kmNet data buffer
class kmNetBuf : public kmMat1i8
{
public:
	// constructor
	kmNetBuf() {};
	kmNetBuf(           int64 size) { Create(     0, size); };
	kmNetBuf(char* ptr, int64 size) { Set   (ptr, 0, size); };

	// set buf as starting position (in front of hd)
	void SetPos0() { SetN1(sizeof(kmNetHd)); };

	// move the current position ptr as step
	void MoveCurPtr(int stp_byte) { IncN1(stp_byte); };

	// get the current position ptr 
	char* GetCurPtr() const { return End1(); };

	// get header
	inline kmNetHd& GetHd() const { return *(kmNetHd*)P(); };

	//////////////////////////////////////////
	// buf control operator

	// operator to get data
	template<typename T> kmNetBuf& operator>>(T& data) { GetData(data); return *this; };

	// operator to put data
	template<typename T> kmNetBuf& operator<<(      T& data) { PutData(data); return *this; };
	template<typename T> kmNetBuf& operator<<(const T& data) { PutData(data); return *this; };

	// operator to set header
	kmNetBuf& operator>>(      kmNetHd& hd) { hd      = GetHd(); SetPos0(); return *this; };
	kmNetBuf& operator<<(      kmNetHd& hd) { GetHd() = hd;      SetPos0(); return *this; };	
	kmNetBuf& operator<<(const kmNetHd& hd) { GetHd() = hd;      SetPos0(); return *this; };

	//////////////////////////////////////////
	// get data from buffer

	// get n size char without reading n
	void GetDataOnly(char* data, ushort n) { memcpy(data, End1(), n); IncN1(n); };

	// get n size data without reading n
	template<typename T> void GetDataOnly(T* data, ushort n)
	{
		T* ptr = (T*)End1();

		for(ushort i = 0; i < n; ++i) *(data + i) = ntoh(*(ptr++));
		IncN1(sizeof(T)*n);
	}

	// get one size data
	template<typename T> void GetData(T& data)
	{
		data = ntoh(*((T*)End1())); IncN1(sizeof(T));
	}

	// get n size data
	template<typename T> void GetData(T* data, ushort& n)
	{
		GetData(n); GetDataOnly(data, n);
	}

	// get n size char
	void GetData( char* data, ushort& n) { GetData(n); GetDataOnly(data, n); };
	void GetData(wchar* data, ushort& n) { GetData(n); GetDataOnly(data, n); };
	
	// get kmStr or kmMat1
	template<typename T> void GetData(kmStr <T>& data) { ushort n; GetData(n); data.Recreate(n); GetDataOnly(data.P(), n); }
	template<typename T> void GetData(kmMat1<T>& data) { ushort n; GetData(n); data.Recreate(n); GetDataOnly(data.P(), n); }

	//////////////////////////////////////////
	// put data into buffer

	// put n size char without inserting n
	void PutDataOnly(const char* data, ushort n) { memcpy(End1(), data, n); IncN1(n); };

	// put n size data without inserting n
	template<typename T> void PutDataOnly(const T* data, ushort n)
	{
		T* ptr = (T*)End1();

		for(ushort i = 0; i < n; ++i) *(ptr++) = hton(*(data + i));
		IncN1(sizeof(T)*n);
	}

	// put one size data
	template<typename T> void PutData(const T& data)
	{
		*((T*)End1()) = hton(data); IncN1(sizeof(T));
	}

	// put n size data
	template<typename T> void PutData(const T* data, ushort n)
	{
		PutData((ushort) n); PutDataOnly(data, n);
	}

	// put n size char
	void PutData(const char* data, ushort n) { PutData(n); PutDataOnly(data, n); };

	// put kmStr or kmMat1
	template<typename T> void PutData(const kmStr <T>& data) { PutData(data.P(), (ushort)data.GetLen()); }
	template<typename T> void PutData(const kmMat1<T>& data) { PutData(data.P(), (ushort)data.N1()    ); }
};

// kmNet's time-out monitoring class
class kmNetTom
{
protected:
	double  _tout_sec = 2;
	kmTimer _timer;
	kmLock  _lck;

public:
	void Set(double tout_sec) { _tout_sec = tout_sec; };

	double GetToutSec() { return _tout_sec; };

	void On()  { kmLockGuard grd = _lck.Lock(); _timer.Start(); };
	void Off() { kmLockGuard grd = _lck.Lock(); _timer.Stop (); };

	bool IsOut() { kmLockGuard grd = _lck.Enter(); return _timer.IsStarted() && (_timer.sec() > _tout_sec); };
	bool IsOff() { kmLockGuard grd = _lck.Enter(); return _timer.IsNotStarted(); };
	bool IsOn () { kmLockGuard grd = _lck.Enter(); return _timer.IsStarted(); };
};

// network condition class
class kmNetCnd
{
public:
	float tkb_ms = 1.0f;    // time to send 1KB (msec)
	float toh_ms = 0.05f;   // time for over head (msec)
	float tlt_ms = 100.f;   // time for latency (round trip, msec)
	int   isdone = 0;       // 0 : test has not been done, 1: it's done

	// data member to calculate net condition
	kmMat1f32 _tec_ms; // echo time (mesc)
	kmMat1f32 _snd_kb; // data size (kbyte)
	kmMat1f32 _snd_n;  // number of sending 

	// estimate snd_t_msec for sending file or large data
	float GetSndTime(int snd_byte)
	{
		if(tkb_ms <= 0) return snd_byte/1e3f; // 1 MB/s = 1 KB/ms = 1 byte/us

		return tkb_ms*float(snd_byte)/1024.f + toh_ms;
	};

	// estimate tout_msec for sending file or large data
	// * Note that time out is from end of sending till receiving ack
	float GetTimeOut(int snd_byte, int snd_n)
	{
		if(tkb_ms <= 0) return 1e2f;

		return GetSndTime(snd_byte)*(snd_n*0.2f) + tlt_ms*3.f + 2.f;
	};

	kmStra GetStr() const
	{
		if(tkb_ms == 0) return kmStra("mps - MB/s, toh - ms, tlt - ms");

		return kmStra("mps %.1f MB/s, toh %.3f ms, tlt %.1f ms", 1.f/tkb_ms, toh_ms, tlt_ms);
	};

	void Print() const
	{
		print("* net condition (%d)\n", _tec_ms.N1());
		print("*  %s\n", GetStr().P());
	};

	void AddTest(float tec_msec, int snd_kb, int snd_n)
	{
		if(_tec_ms.N1() == 0) _tec_ms.Recreate(0,8);
		if(_snd_kb.N1() == 0) _snd_kb.Recreate(0,8);
		if(_snd_n .N1() == 0) _snd_n .Recreate(0,8);

		_tec_ms.PushBack(       tec_msec);
		_snd_kb.PushBack((float)snd_kb);
		_snd_n .PushBack((float)snd_n);
	};

	void CalcTest()
	{
		const int n = (int)_tec_ms.N1();

		if(n == 0 || _snd_kb.N1() != n || _snd_n.N1() != n)
		{
			print("* [kmNetCnd::CalcTest] cannot calculate\n"); return;
		}
		kmMat2f32 y(n,1), m(n,3);

		for(int i = 0; i < n; ++i)
		{
			y(i,0) = _tec_ms(i);
			m(i,0) = _snd_n(i)*_snd_kb(i); 
			m(i,1) = _snd_n(i); 
			m(i,2) = 1;
		}
		kmMat2f32 x = DivideLeft(m.Tp()*m, m.Tp()*y);

		tkb_ms = MIN(MAX(1e-3f, x(0)),   1.f);
		toh_ms = MIN(MAX(    0, x(1)),  0.1f);
		tlt_ms = MIN(MAX( 0.1f, x(2)), 500.f);
		isdone = 1;
	};

	void ClearTest()
	{
		_tec_ms.SetN1(0); _snd_kb.SetN1(0); _snd_n.SetN1(0); isdone = 0;
	};
};

// kmNetBase class for protocol
class kmNetBase
{
private:
	kmSock    _sck;        // * Note that it is private to prevent 
	                       // * directly using _sck without lock or unlock
public:	
	kmMacAddr _mac;        // mac address
	kmAddr4   _addr;       // source address (private)
	kmNetBuf  _rcv_buf{};  // receiving buffer including kmNetHd
	kmNetBuf  _snd_buf{};  // sending buffer including kmNetHd
	kmStra    _name{};     // net user's name... utf8
	kmLock    _lck;        // mutex for _snd_buf
	kmNetTom  _tom;        // rcv timeout monitoring

	int Bind()
	{
		// * Note that INADDR_ANY will bind to every available ip addr.
		kmAddr4 bind_addr(htonl(INADDR_ANY), _addr.GetPort());

		return _sck.Bind(bind_addr, kmSockType::udp);
	};
	int Close() { return _sck.Close();  };

	int Recvfrom(      kmAddr4& addr) { return _sck.Recvfrom(_rcv_buf, addr); };
	int Sendto  (const kmAddr4& addr)
	{
		//if(kmfrand(0,9) == 9) { UnlockSnd(); return 1; } // only for test

		const int ret = _sck.Sendto(_snd_buf, addr); 
		UnlockSnd();  return ret; ///////////////////////////// unlock
	};
	int SendtoBroadcast(ushort port = KMNETDPORT)
	{
		_sck.SetSckOptBroadcast(1);
		
		//const int ret = _sck.SendtoBroadcast(_snd_buf, port);
		const int ret = _sck.SendtoBroadcastLocal(_snd_buf, port);

		_sck.SetSckOptBroadcast(0);
		UnlockSnd();  return ret; //////////////////////////////// unlock	
	}

	template<typename T> kmNetBuf& operator<<(T& data) { return _snd_buf << data; }
	template<typename T> kmNetBuf& operator>>(T& data) { return _rcv_buf >> data; }

	kmNetBuf& operator<<(kmNetHd& hd)
	{
		LockSnd();  return _snd_buf << hd; /////////////////////////// lock
	};

	// thread lock functions
	kmLock* LockSnd  () { return _lck.Lock  (); };
	void    UnlockSnd() {        _lck.Unlock(); };
	kmLock& EnterSnd () { return _lck.Enter (); };
	void    LeaveSnd () {        _lck.Leave (); };

	// timeout monitoring functions
	inline void SetTom(float tout_sec) { _tom.Set(tout_sec); };
	inline void SetTomOn ()            { _tom.On (); };
	inline void SetTomOff()            { _tom.Off(); };
	inline bool IsTomOn ()             { return _tom.IsOn (); };
	inline bool IsTomOff()             { return _tom.IsOff(); };
	inline bool IsTomOut()             { return _tom.IsOut(); };

	// only for debug
	kmSock& GetSock() { return _sck; };
};

/////////////////////////////////////////////////////////////////////////////////////////
// kmNet protocol class 

// typedef for protocol callback
using kmNetPtcCb = int(*)(kmNetBase* net, char cmd_id, void* arg);

// base class of kmNet protocol
class kmNetPtc
{
public:
	uchar      _ptc_id;    // own ptorcol id	
	kmNetPtcCb _ptc_cb;    // protocol callback function
	kmNetBase* _net;

	// init
	kmNetPtc* Init(uchar ptc_id, kmNetBase* net, kmNetPtcCb ptc_cb = nullptr)
	{
		_ptc_id = ptc_id;
		_net    = net;
		_ptc_cb = ptc_cb;

		return this; 
	};

	// receiving procedure	
	virtual void RcvProc(char cmd_id, const kmAddr4& addr) = 0;
};

// net key type enum class... pkey, vkey, tkey
enum class kmNetKeyType : uchar { invalid = 0, pkey = 1, vkey = 2, tkey = 3 };

// net key class... 8byte
class kmNetKey
{
protected: // * Note that key is saved with big-endian for network transfer
	       // * so, you should not directly access this members
	kmNetKeyType _type{};  // key type (1byte)
	uchar        _svri{};  // server id for the future
	ushort       _idx0{};  // 1st dim's index
	ushort       _idx1{};  // 2nd dim's index
	ushort       _pswd{};  // password

public:
	// constructor
	kmNetKey() {};
	kmNetKey(kmNetKeyType type, ushort idx0, ushort idx1, ushort pswd = 0, uchar svri = 0)
	{
		Set(type, idx0, idx1, pswd, svri); 
	};

	// operator
	bool operator==(const kmNetKey& b) const { return *(int64*)this == *(int64*)&b; };

	///////////////////////////////////////
	// member functions
	void Set(kmNetKeyType type, ushort idx0, ushort idx1, ushort pswd = 0, uchar svri = 0)
	{
		_type = type; 
		_svri = svri;
		_idx0 = hton(idx0);
		_idx1 = hton(idx1); 
		_pswd = hton(pswd);
	};
	void SetPkey(ushort idx0, ushort idx1, uint pswd = 0, uchar srvi = 0) { Set(kmNetKeyType::pkey,idx0,idx1,pswd,srvi); };
	void SetVkey(ushort idx0, ushort idx1, uint pswd = 0, uchar srvi = 0) { Set(kmNetKeyType::vkey,idx0,idx1,pswd,srvi); };
	void SetTkey(ushort idx0, ushort idx1, uint pswd = 0, uchar srvi = 0) { Set(kmNetKeyType::tkey,idx0,idx1,pswd,srvi); };
	void SetType(kmNetKeyType type) { _type = type; };

	kmNetKeyType GetType() const { return _type;       };
	uchar        GetSvri() const { return _svri;       };
	ushort       GetIdx0() const { return ntoh(_idx0); };
	uint         GetIdx1() const { return ntoh(_idx1); };
	ushort       GetPswd() const { return ntoh(_pswd); };
	uint64       i64()     const { return *(int64*)this; };

	bool IsInvalid() const { return _type == kmNetKeyType::invalid; };
	bool IsValid()   const { return _type != kmNetKeyType::invalid; };
	bool IsPkey()    const { return _type == kmNetKeyType::pkey;    };
	bool IsVkey()    const { return _type == kmNetKeyType::vkey;    };
	bool IsTkey()    const { return _type == kmNetKeyType::tkey;    };

	const char* GetTypeStr() const
	{
		switch(_type)
		{
		case kmNetKeyType::invalid : return "invalid"; 
		case kmNetKeyType::pkey    : return "pkey"; 
		case kmNetKeyType::vkey    : return "vkey";
		case kmNetKeyType::tkey    : return "tkey";
		}
		return "";
	};
	kmStra GetStr() const
	{
		return kmStra("[%s] %d %d %d %d", 
			          GetTypeStr(), GetIdx0(), GetIdx1(), GetPswd(), GetSvri());
	};
	void Print() const { print("* key : %s\n", GetStr().P()); };
};

// net key renewal algorithm class
class kmNetKeyRnw
{
public:
	kmThread _thrd;	
	uint     _rnw_sec     = 30;         // renewal time in sec
	uint     _rnw_min_sec = 10;
	uint     _rnw_max_sec = (60*60*24); // 24 hour

	void Start(kmNetKey* key)
	{
		_thrd.Begin([](kmNetKeyRnw* rnw, kmNetKey* key)
		{
			kmTimer timer(1);

			while(1)
			{
			}
		}, this, key);
	};
};

// net key element class for nks server... 32 byte
class kmNetKeyElm
{
public:
	kmNetKey  key{};       // key (8byte)
	kmMacAddr mac{};       // mac of key owner (8byte)
	kmDate    date{};      // pkey : last accessed date, vkey : expiration date (8byte)
	union                  // (8byte)
	{
		kmAddr4   addr{};  // ip address for pkey ... only for pkey
		kmNetKey  pkey;    // pkey linked to vkey ... only for vkey
	};	

	// constructor
	kmNetKeyElm() {};
	kmNetKeyElm(kmNetKey key, kmMacAddr mac, kmDate date, kmAddr4 addr) :
		key(key), mac(mac), date(date), addr(addr) {};
	kmNetKeyElm(kmNetKey key, kmMacAddr mac, kmDate date, kmNetKey pkey) :
		key(key), mac(mac), date(date), pkey(pkey) {};

	// member functions
	bool IsValid  () const { return key.IsValid  (); };
	bool IsInvalid() const { return key.IsInvalid(); };
	bool IsExpired() const { return key.IsVkey() && (date.GetPassSec() > 0); };

	kmStra GetStr()
	{
		if(key.IsVkey())
		{
			return kmStra("%s %s %s %s", key .GetStr  ().P(), mac .GetStr().P(), 
				                         date.GetStrPt().P(), pkey.GetStr().P());
		}
		return kmStra("%s %s %s %s", key .GetStr  ().P(), mac .GetStr().P(), 
			                         date.GetStrPt().P(), addr.GetStr().P()); 
	};

	kmNetKeyElm& operator=(const kmNetKeyElm& elm)
	{
		memcpy((void*)this, (void*)&elm, sizeof(kmNetKeyElm));
		return *this;
	};
};
typedef kmMat1<kmNetKeyElm>  kmNetKeyElms;
typedef kmMat1<kmNetKeyElms> kmNetKeyTbl;

// net key signaling function class nks
class kmNetNks
{
	kmNetKeyTbl _ptbl; // (idx0)(idx1) pkey table 
	kmNetKeyTbl _vtbl; // (idx0)(idx1) vkey table

public:
	// create table
	void Create(int idx0_n = 32)
	{
		// create pkey and vkey tables
		_ptbl.Create(idx0_n); for(int i = 0; i < idx0_n; ++i) _ptbl(i).Create(32);
		_vtbl.Create(idx0_n); for(int i = 0; i < idx0_n; ++i) _vtbl(i).Create(32);
	};

	// register new one as pkey
	kmNetKey RegisterPkey(kmMacAddr mac, kmAddr4 addr)
	{
		// find empty key... idx0, idx1
		ushort idx0 = kmfrand(0, (int)_ptbl.N1() - 1), idx1 = 0;

		kmNetKeyElms& elms  = _ptbl(idx0);
		const int     elm_n = (int)elms.N1();

		for(; idx1 < elm_n; ++idx1) if(elms(idx1).IsInvalid()) break;

		// set key element
		kmNetKey key(kmNetKeyType::pkey, idx0, idx1, kmfrand(0u, 65535u));
		kmDate   date(time(NULL));

		// add key element to table
		if(idx1 < elm_n) elms(idx1) =  kmNetKeyElm(key, mac, date, addr);
		else             elms.PushBack(kmNetKeyElm(key, mac, date, addr));

		return key;
	};

	// register new one as vkey
	kmNetKey RegisterVkey(kmMacAddr mac, kmNetKey pkey, uint vld_sec)
	{
		// find empty key... idx0, idx1
		ushort idx0 = kmfrand(0, (int)_vtbl.N1() - 1), idx1 = 0;

		kmNetKeyElms& elms  = _vtbl(idx0);
		const int     elm_n = (int)elms.N1();

		for(; idx1 < elm_n; ++idx1) if(elms(idx1).IsInvalid()) break;

		// set key element
		kmNetKey key(kmNetKeyType::vkey, idx0, idx1, 0);
		kmDate   date(time(NULL) + vld_sec);

		// add key element to table
		if(idx1 < elm_n) elms(idx1) =  kmNetKeyElm(key, mac, date, pkey);
		else             elms.PushBack(kmNetKeyElm(key, mac, date, pkey));

		return key;
	};

	// find net key element with key
	kmNetKeyElm& Find(kmNetKey key, kmMacAddr mac = kmMacAddr(), bool mac_check = false)
	{	
		const auto idx0 = key.GetIdx0();
		const auto idx1 = key.GetIdx1();
		
		static kmNetKeyElm keyelm_invalid{};

		kmNetKeyTbl& tbl = (key.IsVkey()) ? _vtbl : _ptbl;

		// check index range 
		if(idx0 >=  tbl.N1() || idx1 >= tbl(idx0).N1())
		{
			return keyelm_invalid;
		}
		kmNetKeyElm& elm = tbl(idx0)(idx1);

		// check expiration time... vkey only
		if(elm.IsExpired()) return keyelm_invalid;

		// check pswd
		if(elm.key.GetPswd() != key.GetPswd())
		{
			print("** password is wrong (%d != %d)\n", elm.key.GetPswd(), key.GetPswd()); 
			
			return keyelm_invalid;
		}
		// check mac.. only for pkey
		if(mac_check && key.IsPkey() && elm.mac != mac)
		{
			print("** mac is wrong\n"); return keyelm_invalid;
		}
		return elm;
	};

	// find address and mac with key
	kmT2<kmAddr4, kmMacAddr> GetAddr(kmNetKey key, kmMacAddr mac)
	{	
		// get key element with vkey or pkey
		kmNetKeyElm& elm = (key.IsVkey()) ? Find(Find(key).pkey) : Find(key, mac, true);

		return kmT2(elm.addr, elm.mac);
	};

	// update address
	//   return : -1 (invalid key), 0 (not updated), 1 (updated since addr was changed)
	int Update(kmNetKey key, kmMacAddr mac, kmAddr4 addr)
	{
		// find key element
		kmNetKeyElm& ke = Find(key, mac, true);

		if(ke.IsInvalid()) return -1;

		// check if address is changed
		if(addr == ke.addr) return 0;

		// update address
		ke.addr = addr;
		ke.date.SetCur();

		return 1;
	};

	// save key table
	void Save(const wchar* path)
	{
		kmFile file(path, KF_NEW);

		file.WriteMat(&_ptbl);
		file.WriteMat(&_vtbl);
	};

	// load key table
	int Load(const wchar* path) try
	{
		kmFile file(path);

		file.ReadMat(&_ptbl);
		file.ReadMat(&_vtbl);

		return 1;
	}
	catch(kmException) { return 0; };

	// print every key
	void Print()
	{
		// print pkey
		print("* pkey [%d] \n", _ptbl.N1());

		for(int64 i = 0; i < _ptbl.N1(); ++i)
		{
			const kmNetKeyElms& elms = _ptbl(i);

			for(int64 j = 0; j < elms.N1(); ++j) if(elms(j).IsValid())
			{
				print("   (%d, %d) : %s\n", i, j, elms(j).GetStr().P());
			}
		}

		// print vkey
		print("* vkey [%d] \n", _vtbl.N1());

		for(int64 i = 0; i < _vtbl.N1(); ++i)
		{
			const kmNetKeyElms& elms = _vtbl(i);

			for(int64 j = 0; j < elms.N1(); ++j) if(elms(j).IsValid())
			{
				print("   (%d, %d) : %s\n", i, j, elms(j).GetStr().P());
			}
		}
	};
};

// basic protocol for key connection
class kmNetPtcNkey: public kmNetPtc
{
public:
	kmAddr4      _nks_addr;         // nks server's addr

	int          _rcv_key_flg  = 0; // 0 : not received, 1: received key
	int          _rcv_addr_flg = 0; // 0 : not received, 1: received addr

	kmNetKeyType _rcv_keytype;
	kmMacAddr    _rcv_mac;
	kmNetKey     _rcv_key;
	kmNetKey     _rcv_pkey;
	kmAddr4      _rcv_addr;	
	uint         _rcv_vld_sec; // valid time in sec
	uint         _rcv_vld_cnt; // valid count
	int          _rcv_sig_flg; // sig flag 0 : addr not changed, 1: change
	ushort       _rcv_src_id;

	kmAddr4      _snd_addr;
	kmMacAddr    _snd_mac;

	// receiving procedure
	virtual void RcvProc(char cmd_id, const kmAddr4& addr)
	{	
		switch(cmd_id)
		{
		case 0: RcvReqKey (addr); break;
		case 1: RcvKey    (addr); break;
		case 2: RcvReqAddr(addr); break;
		case 3: RcvAddr   (addr); break;
		case 4: RcvSig    (addr); break;
		case 5: RcvRepSig (addr); break;
		case 6: RcvReqAccs(addr); break;
		}
	};
	///////////////////////////////////////
	// interface functions

	// request key to nks with waiting (svr -> nks)
	kmNetKey ReqKey(kmNetKeyType keytype, uint vld_sec = 0, uint vld_cnt = 0, 
		            kmNetKey pkey = kmNetKey(), float tout_sec = 1.f)
	{
		// send request to nks
		SndReqKey(keytype, vld_sec, vld_cnt, pkey);

		// wait for timeout
		kmTimer time(1); _rcv_key_flg = 0;

		while(time.sec() < tout_sec && _rcv_key_flg == 0) Sleep(1);

		if(_rcv_key_flg == 0)
		{
			print("** kmNetPtcNkey::ReqKey - timeout\n"); 
			
			return kmNetKey();
		}		
		return _rcv_key;
	};

	// request vkey to svr with waiting (clt -> svr)
	kmNetKey ReqVkey(ushort src_id, ushort des_id, const kmAddr4& addr, 
		             uint vld_sec = 600, uint vld_cnt = 1, float tout_sec = 2.f)
	{
		// send request to svr
		SndReqKey(src_id, des_id, addr, kmNetKeyType::vkey, vld_sec, vld_cnt);

		// wait for timeout
		kmTimer time(1); _rcv_key_flg = 0;

		while(time.sec() < tout_sec && _rcv_key_flg == 0) Sleep(1);

		if(_rcv_key_flg == 0)
		{
			print("** kmNetPtcNkey::ReqVkey - timeout\n"); 

			return kmNetKey();
		}		
		return _rcv_key;
	};

	// request addr to nks server by key (clt -> nks)
	kmAddr4 ReqAddr(kmNetKey key, kmMacAddr mac, float tout_sec = 1.f)
	{
		// send request to nks server
		SndReqAddr(key, mac);

		// wait for timeout
		kmTimer time(1); _rcv_addr_flg = 0;

		while(time.sec() < tout_sec && _rcv_addr_flg == 0) Sleep(1);

		if(_rcv_addr_flg == 0)
		{
			print("** kmNetPtcNkey::ReqAddr - timeout\n"); 
			
			return kmAddr4(kmAddr4State::invalid);
		}
		return _rcv_addr;
	};

	// send key 
	void SendKey(ushort src_id, ushort des_id, const kmAddr4& addr, const kmNetKey& key)
	{
		SndKey(src_id, des_id, addr, key);
	};

	// send key from nks
	void SendKeyFromNks(const kmAddr4& addr, const kmNetKey& key)
	{
		SndKey(0xffff, 0xffff, addr, key);
	};

	// send signal to confirm the connection
	//  return : -1 (timeout) or echo time (msec, if received ack)
	float SendSig(kmNetKey key, kmMacAddr mac, float tout_msec = 300.f)
	{
		// reset flag
		_rcv_sig_flg = -1;

		// send signal
		SndSig(key, mac);

		// wait for timeout
		kmTimer time(1);

		while(time.msec() < tout_msec && _rcv_sig_flg == -1) { Sleep(1); };

		return (_rcv_sig_flg >= 0)? (float)time.msec():-1.f;
	};

	// reply signal (nks -> svr)
	//   flg 0 : addr was not changed, 1: changed
	inline void ReplySig(kmAddr4 addr, int flg) { SndRepSig(addr, flg); };

protected:
	////////////////////////////////////////
	// cmd 0 : request key

	// request key (svr -> nks)
	//   vld_sec, vld_cnt, pkey : only for requesting vkey
	void SndReqKey(kmNetKeyType keytype, uint vld_sec, uint vld_cnt, kmNetKey pkey)
	{	
		kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 0, (uchar)keytype, 0};

		*_net << hd << _net->_mac << vld_sec << vld_cnt << pkey;

		_net->Sendto(_nks_addr);
	};
	// requeset key (clt -> svr)
	void SndReqKey(ushort src_id, ushort des_id, const kmAddr4& addr,
		           kmNetKeyType keytype, uint vld_sec, uint vld_cnt)
	{
		kmNetHd hd = {src_id, des_id, _ptc_id, 0, (uchar)keytype, 0};

		*_net << hd << _net->_mac << vld_sec << vld_cnt << kmNetKey();

		_net->Sendto(addr);
	};
	void RcvReqKey(const kmAddr4& addr) // nks or svr
	{
		kmNetHd hd{};

		*_net >> hd >> _rcv_mac >> _rcv_vld_sec >> _rcv_vld_cnt;

		_rcv_src_id  = hd.des_id;		
		_rcv_keytype = (kmNetKeyType)hd.opt;
		_rcv_addr    = addr;

		if(_rcv_keytype == kmNetKeyType::vkey) *_net >> _rcv_pkey;
		else _rcv_pkey = kmNetKey();

		// call cb to register mac and send key 
		//   : _rcv_mac, _rcv_vld_sec, _rcv_vlc_cnt, _rcv_pkey
		(*_ptc_cb)(_net, 0, 0);
	};

	////////////////////////////////////////
	// cmd 1 : send key
	
	// send key 
	//   if sending key from nks, src_id and des_id will be 0xffff
	void SndKey(ushort src_id, ushort des_id, const kmAddr4& addr, kmNetKey key)
	{
		kmNetHd hd = { src_id, des_id, _ptc_id, 1, 0, 0};

		*_net << hd << key;

		_net->Sendto(addr);
	};
	void RcvKey(const kmAddr4& addr)
	{	
		kmNetHd hd{};

		*_net >> hd >> _rcv_key;

		_rcv_key_flg = 1;
	};

	////////////////////////////////////////
	// cmd 2 : request addr by key (clt -> nks)
	void SndReqAddr(kmNetKey key, kmMacAddr mac)
	{
		kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 2, 0, 0};

		*_net << hd << key << mac;

		_net->Sendto(_nks_addr);
	};
	void RcvReqAddr(const kmAddr4& addr) // nks server
	{
		kmNetHd hd{};

		*_net >> hd >> _rcv_key >> _rcv_mac;

		// call cb to get addr from key and send addr (_rcv_key, _rcv_mac -> _snd_addr, _snd_mac)
		(*_ptc_cb)(_net, 2, 0);

		// request svr to access clt
		if(_snd_addr.IsValid())
		{
			// check if lan or wan
			if(_snd_addr.ip == addr.ip) _snd_addr.SetFlgLan();
			else                        _snd_addr.SetFlgWan();

			// request _snd_addr to access addr
			SndReqAccs(_snd_addr, addr);
		}

		// send address(_snd_addr) to clt(addr)
		SndAddr(addr);
	};

	////////////////////////////////////////
	// cmd 3 : send addr (nks -> clt)
	void SndAddr(const kmAddr4& addr) // nks server
	{	
		kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 3, 0, 0};

		*_net << hd << _snd_addr << _snd_mac;

		_net->Sendto(addr);
	};
	void RcvAddr(const kmAddr4& addr)
	{	
		kmNetHd hd{};

		*_net >> hd >> _rcv_addr >> _rcv_mac;

		_rcv_addr_flg = 1;
	};

	////////////////////////////////////////
	// cmd 4 : send signal (svr -> nks)
	void SndSig(kmNetKey key, kmMacAddr mac)
	{
		kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 4, 0, 0};

		*_net << hd << key << mac;

		_net->Sendto(_nks_addr);
	};
	void RcvSig(const kmAddr4& addr) // nks server
	{
		kmNetHd hd{};

		*_net >> hd >> _rcv_key >> _rcv_mac;

		_rcv_addr = addr;

		(*_ptc_cb)(_net, 4, 0);
	};

	////////////////////////////////////////
	// cmd 5 : reply signal (nks --> svr)
	void SndRepSig(const kmAddr4& addr , int flg) // nks server
	{
		kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 5, 0, 0};

		*_net << hd << flg;

		_net->Sendto(addr);
	};
	void RcvRepSig(const kmAddr4& addr)
	{
		kmNetHd hd{}; 

		*_net >> hd >> _rcv_sig_flg;

		(*_ptc_cb)(_net, 5, 0);
	};

	////////////////////////////////////////
	// cmd 6 : request to access the target address (nks --> svr)
	void SndReqAccs(const kmAddr4& addr , const kmAddr4& trg_addr) // nks server
	{
		kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 6, 0, 0};

		*_net << hd << trg_addr;

		_net->Sendto(addr);
	};
	void RcvReqAccs(const kmAddr4& addr)
	{
		kmNetHd hd{}; 

		*_net >> hd >> _rcv_addr;

		(*_ptc_cb)(_net, 6, 0);
	};
};

// basic protocol for udp broadcasting
class kmNetPtcBrdc : public kmNetPtc
{
public:
	kmAddr4s*    _addrs;    // output address's pointer
	kmMacAddrs*  _macs;     // output mac address's pointer
	volatile int _wait = 0; // waiting status

	// receiving procedure
	virtual void RcvProc(char cmd_id, const kmAddr4& addr)
	{	
		switch(cmd_id)
		{
		case 0: RcvReqAck(addr); break;
		case 1: RcvAck   (addr); break;
		}
	};
	///////////////////////////////////////
	// interface functions

	// get every address within local network
	//   addrs        : output address
	//   macs         : output mac addresses
	//   timeout_msec : timeout in msec
	//   port         : port for broadcasting
	//
	//   return : number of addrs
	int GetAddrs(kmAddr4s& addrs, kmMacAddrs& macs, ushort port = KMNETDPORT, float tout_msec = 100.f)
	{
		// init output parameters
		_addrs = &addrs; _addrs->Recreate(0, 16);
		_macs  = &macs;  _macs ->Recreate(0, 16);
		_wait  = 1;

		// send UDP broadcasting
		SndReqAck(port);

		// wait for timeout
		kmTimer time(1);

		while(time.msec() < tout_msec) { Sleep(1); }
		_wait = 0;

		return (int)_addrs->N();
	};

	////////////////////////////////////////
	// cmd 0 : request ack
	void SndReqAck(ushort port)
	{
		// set buffer
		kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 0, 0, 0};

		*_net << hd << _net->_mac;

		// send as broadcast mode
		_net->SendtoBroadcast(port);
	};
	void RcvReqAck(const kmAddr4& addr)
	{
		// get from buffer
		kmNetHd hd{}; kmMacAddr mac;

		*_net >> hd >> mac;

		if(mac == _net->_mac) return; // to prevent self-ack

		print("**** rcvreqack\n");

		// send ack
		SndAck(addr);
	};

	///////////////////////////////////////////////
	// cmd 1 : ack
	void SndAck(const kmAddr4& addr)
	{
		// set buffer
		kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 1, 0, 0};

		print("**** sndack start\n");

		*_net << hd << _net->_mac;

		print("**** sndack buffer completed\n");

		// send buffer
		_net->Sendto(addr);

		print("**** sndack end\n");
	};
	void RcvAck(const kmAddr4& addr)
	{
		// get from buffer
		kmNetHd hd{}; kmMacAddr mac;

		*_net >> hd >> mac;

		// set output
		if(_wait == 1)
		{
			_addrs->PushBack(addr);
			_macs ->PushBack(mac);
		}
	};
};

// basic protocol for connection
class kmNetPtcCnnt : public kmNetPtc
{
public:		
	ushort    _rcv_src_id;
	ushort    _rcv_des_id;
	kmMacAddr _rcv_des_mac;
	kmAddr4   _rcv_des_addr;
	char      _rcv_des_name[64];
	int       _rcv_preack_flg = 0;
	kmAddr4   _rcv_preack_addr;
	int       _rcv_sig_flg    = 0;
	ushort    _rcv_sig_src_id;	
	kmAddr4   _rcv_sig_addr;
	int       _rcv_sig_res = 0;
	uint      _rcv_ack_id = 0;
	kmNetKey  _rcv_key{};

	ushort    _snd_src_id;
	ushort    _snd_des_id;
	kmMacAddr _snd_des_mac;
	char      _snd_des_name[64];
	uint      _snd_ack_id = 0;

	// receiving procedure
	virtual void RcvProc(char cmd_id, const kmAddr4& addr)
	{	
		switch(cmd_id)
		{
		case 0: RcvReqCnnt(addr); break;
		case 1: RcvAccept (addr); break;
		case 2: RcvPreCnnt(addr); break;
		case 3: RcvPreAck (addr); break;
		case 4: RcvSig    (addr); break;
		case 5: RcvRepSig (addr); break;
		}
	};
	///////////////////////////////////////
	// interface functions

	// connect to the target device	
	//   src_id  : source id
	//   addr    : destination address
	//	
	// return : kmT(des_id, des_mac)
	//   des_id  : destination id (0 ~ 0xffff-2 : accepted, 0xffff-1: not accepted, 0xffff : timeout)
	//   des_mac : mac of destination (output)
	//
	kmT2<ushort, kmMacAddr> Connect(ushort src_id, kmAddr4 addr, const kmStra& name, kmNetKey key = kmNetKey(), float tout_msec = 300.f)
	{
		// send reqconnect
		SndReqCnnt(src_id, addr, name, key);

		// wait for timeout
		kmTimer time(1);

		while(time.msec() < tout_msec && _snd_des_id == 0xffff) { Sleep(1); };

		if     (_snd_des_id == 0xffff  ) print("** kmNetPtcCnnt::Connect : timeout\n");
		else if(_snd_des_id == 0xffff-1) print("** kmNetPtcCnnt::Connect : reject from %s\n", addr.GetStr().P());

		return kmT2(_snd_des_id, _snd_des_mac);
	};

	// send reqack as pre-connection
	//  return : 0 (timeout), 1 <= (number of attempts to get ack)
	int SendPreCnnt(const kmAddr4& addr, float tout_msec = 200.f, int try_cnt = 3)
	{
		// reset flag
		_rcv_preack_flg  = 0;
		_rcv_preack_addr = 0;

		// send pre-cnnt
		for(int i = 0; i < try_cnt; ++i)
		{
			SndPreCnnt(addr);

			// wait for timeout
			kmTimer time(1);

			while(time.msec() < tout_msec && _rcv_preack_flg == 0) { Sleep(1); };

			if(_rcv_preack_flg > 0) return i + 1;
		}
		return 0;
	};

	// send signal to confirm the connection
	//  return : -1 (timeout) or echo time (msec, if received ack)
	float SendSig(ushort src_id, ushort des_id, const kmAddr4& addr, float tout_msec = 300.f, 
		          int dummy_byte = 0, int rep_n = 1, bool echo_dummy = false)
	{
		// rest flag
		_rcv_sig_flg = 0;

		kmTimer time(1);

		// send signal
		for(int n = rep_n - 1; n--;) SndSig(src_id, des_id, addr, dummy_byte, false);

		SndSig(src_id, des_id, addr, dummy_byte, true, echo_dummy);

		// wait for timeout
		while(time.msec() < tout_msec && _rcv_sig_flg == 0) { Sleep(0); };

		return (_rcv_sig_flg > 0)? (float)time.msec():-1.f;
	};

	// send accept ... only after receiving reqcnnt
	void SendAccept()
	{
		SndAccept(_rcv_src_id, _rcv_des_id, _rcv_des_addr, _net->_name);
	};

protected:
	///////////////////////////////////////
	// cmd 0 : request connect
	void SndReqCnnt(ushort src_id, kmAddr4 addr, const kmStra& name, kmNetKey key)
	{
		// set ack id
		_snd_ack_id = kmfrand(0u, end32u);
		
		// set snd_buf
		kmNetHd hd = { _snd_src_id = src_id, _snd_des_id = 0xffff, _ptc_id, 0, 0, 0};

		(*_net << hd << _net->_mac << _snd_ack_id).PutData(name.P(), (ushort)MIN(64,name.N()));

		*_net << key; // only for vkey

		// send snd_buf
		_net->Sendto(addr);
	};
	void RcvReqCnnt(const kmAddr4& addr)
	{
		// get from rcv_buf
		kmNetHd hd; ushort name_n;

		(*_net >> hd >> _rcv_des_mac >> _rcv_ack_id).GetData(_rcv_des_name, name_n);

		*_net >> _rcv_key; // only for vkey

		_rcv_des_id   = hd.src_id;
		_rcv_des_addr = addr;

		// call cb function.... get _src_id (will be 0xffff-1, if not accepted)
		(*_ptc_cb)(_net, 0, 0); // cbRcvPtcCnnt(cmd_id = 0)
	};

	////////////////////////////////////////////
	// cmd 1 : accept
	//
	//  if src_id == 0xffff-1, it means rejecting the connection.
	void SndAccept(ushort src_id, ushort des_id, const kmAddr4& addr, const kmStra& name)
	{
		// set snd_buf
		kmNetHd hd = { src_id, des_id, _ptc_id, 1, 0, 0};

		(*_net << hd << _net->_mac << _rcv_ack_id).PutData(name.P(), (ushort)MIN(64,name.N()));

		// send snd_buf
		_net->Sendto(addr);
	};
	void RcvAccept(const kmAddr4& addr)
	{
		// get from rcv_buf
		kmNetHd hd; ushort name_n; uint ack_id;

		(*_net >> hd >> _snd_des_mac >> ack_id).GetData(_snd_des_name, name_n);

		// check ack_id
		if(ack_id != _snd_ack_id)
		{
			print("*  RcvAccept : ack_id (%x) != _snd_ack_id (%x)\n", ack_id, _snd_ack_id);
			return;
		}
		_snd_des_id = hd.src_id; // if this is 0xffff - 1, it mesns reject

		// call cb function...cbRcvPtcCnnt(cmd_id = 1)
		(*_ptc_cb)(_net, 1, 0);
	};

	////////////////////////////////////////////
	// cmd 2 : precnnt
	void SndPreCnnt(const kmAddr4& addr)
	{
		kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 2, 0, 0}; *_net << hd;

		_net->Sendto(addr);
	}
	void RcvPreCnnt(const kmAddr4& addr)
	{
		//kmNetHd hd;  *_net >> hd;

		SndPreAck(addr);
		
		print("** kmNetPtcCnnt:RcvPreCnnt() from %s\n", addr.GetStr().P());
	};

	////////////////////////////////////////////
	// cmd 3 : preack
	void SndPreAck(const kmAddr4& addr)
	{	
		kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 3, 0, 0}; *_net << hd;

		_net->Sendto(addr);
	}
	void RcvPreAck(const kmAddr4& addr)
	{
		//kmNetHd hd;  *_net >> hd;

		_rcv_preack_flg  = 1;
		_rcv_preack_addr = addr;

		print("** kmNetPtcCnnt:RcvPreAck() from %s\n", addr.GetStr().P());
	};

	////////////////////////////////////////////
	// cmd 4 : send signal to confirm the connection

	// send sig
	//   dummy_byte is size excluding hd (8 bytes)
	void SndSig(ushort src_id, ushort des_id, const kmAddr4& addr, int dummy_byte = 0, 
		        bool echo_on = true, bool echo_dummy = false)
	{	
		kmNetHd hd = { src_id, des_id, _ptc_id, 4, 0, 0}; if(echo_on) hd.SetReqAck();

		*_net << hd << (echo_dummy ? dummy_byte : int(0));
		
		if(dummy_byte > 4) _net->_snd_buf.IncN1(dummy_byte - 4);

		_net->Sendto(addr);
	};
	void RcvSig(const kmAddr4& addr)
	{
		int dummy_byte;

		kmNetHd hd;	*_net >> hd >> dummy_byte; 

		if(dummy_byte > end16u - 28) dummy_byte = 0;

		// call cb function.. cmd_id = 4
		_rcv_sig_src_id = hd.des_id;
		_rcv_sig_addr   = addr;

		(*_ptc_cb)(_net, 4, 0); // update _rcv_sig_res... cbRcvPtcCnnt(cmd_id = 4)
		
		if(hd.IsReqAck() && _rcv_sig_res > 0) SndRepSig(hd.des_id, hd.src_id, addr, dummy_byte);
	};

	////////////////////////////////////////////
	// cmd 5 : reply signal
	void SndRepSig(ushort src_id, ushort des_id, const kmAddr4& addr, int dummy_byte = 0)
	{
		kmNetHd hd = { src_id, des_id, _ptc_id, 5, 0, 0};

		*_net << hd; if(dummy_byte > 0) _net->_snd_buf.IncN1(dummy_byte);

		_net->Sendto(addr);
	};
	void RcvRepSig(const kmAddr4& addr)
	{
		//kmNetHd hd; *_net >> hd;

		_rcv_sig_flg = 1;
	};
};

// basic protocol for send small data
class kmNetPtcData: public kmNetPtc
{
public:	
	// receiver's members
	ushort _src_id;
	ushort _des_id;
	uchar  _data_id;
	char*  _data;
	ushort _byte;
	ushort _rcv_ack_id[64]{}; // _rcv_ack_id[_src_id%64] = ack_id
	ushort _rcv_last_ack_id{};
	kmLock _lck;

	// sender's members
	int    _ack    = 0;
	ushort _ack_id = 0;

	// receiving procedure
	virtual void RcvProc(char cmd_id, const kmAddr4& addr)
	{	
		switch(cmd_id)
		{
		case 0: RcvData(addr); break;
		case 1: RcvAck (addr); break;
		}
	};
	///////////////////////////////////////
	// interface functions

	// send data to addr
	// 
	//   return value : -1 (fail), ack_id (ushort)
	int Send(ushort src_id, ushort des_id, const kmAddr4& addr, uchar data_id, char* data, int byte, 
		     float tout_msec, int retry_n, ushort ack_id)
	{
		kmLockGuard grd = _lck.Lock();

		// send data
		kmNetHdFlg flg = 0; flg.reqack = (tout_msec == 0) ? 0:1;

		_ack = 0;

		SndData(src_id, des_id, addr, data_id, data, byte, flg, ack_id);

		int ret_ack_id = (int)_ack_id;

		if(tout_msec == 0) return ret_ack_id;

		// wait for timeout
		kmTimer timer(1);

		for(; retry_n--;)
		{
			while(timer.msec() < tout_msec && _ack < 1) { Sleep(0); };

			if(_ack < 1) { SndDataAgain(addr); timer.Start(); } else break;
		}
		return ret_ack_id;
	};

	// get rcv_last_ack_id
	ushort GetLastAckId() { return _rcv_last_ack_id; };

protected:
	///////////////////////////////////////
	// cmd 0 : data
	void SndData(ushort src_id, ushort des_id, const kmAddr4& addr, uchar data_id, char* data, int byte, kmNetHdFlg flg, ushort ack_id = 0)
	{
		// set ack id
		_ack_id = (ack_id > 0) ? ack_id : (ushort)kmfrand(1, 0xffff);
	
		// set snd_buf
		kmNetHd hd = { src_id, des_id, _ptc_id, 0, data_id, flg};

		(*_net << hd << _ack_id).PutData(data, byte);

		// send snd_buf
		_net->Sendto(addr);
	};
	void SndDataAgain(const kmAddr4& addr) { _net->Sendto(addr); };
	void RcvData(const kmAddr4& addr)
	{
		// get from rcv_buf
		kmNetHd hd; ushort ack_id;

		*_net >> hd >> ack_id >> _byte; _data = _net->_rcv_buf.End1();

		// check if duplicate receiving
		const int idx = _src_id % numof(_rcv_ack_id);

		if(ack_id == _rcv_ack_id[idx]) return;

		// set varialbes
		_des_id          = hd.src_id;
		_src_id          = hd.des_id;
		_data_id         = hd.opt;
		_rcv_ack_id[idx] = _rcv_last_ack_id = ack_id;

		// send ack
		// * Note that it's good to be in front of the callback.
		// * since the callback may take a long time.
		if(hd.IsReqAck()) SndAck(ack_id, addr);

		// call cb function  (net, cmd_id, addr)
		(*_ptc_cb)(_net, 0, 0);
	};

	///////////////////////////////////////////////
	// cmd 1 : ack
	void SndAck(ushort ack_id, const kmAddr4& addr)
	{
		// set snd_buf
		kmNetHd hd = {_src_id, _des_id, _ptc_id, 1, 0, 0};

		*_net << hd << ack_id;

		// send buffer
		_net->Sendto(addr);
	};
	void RcvAck(const kmAddr4& addr)
	{
		// get from rcv_buf
		kmNetHd hd;
		ushort  ack_id;

		*_net >> hd >> ack_id;

		// check id
		if(_ack_id == ack_id) _ack = 1; else _ack = -1;
	};
};

// basic protocol for send large data
class kmNetPtcLrgd: public kmNetPtc
{
public:
	// receiver's members
	kmMat1blk _rcv_blk;        // receiving block buffer
	kmMat1bit _rcv_bsf;        // receiving block state flag (0 : not yet, 1: received)
	uint      _rcv_state = 0;  // 0 : not begin, 1: waiting blocks, 2: done
	int64     _rcv_byte  = 0;
	uint      _rcv_ieb   = 0;  // iblk of 1st empty block
	kmThread  _rcv_thrd;       // thread for time-out monitoring
	int       _rcv_flg   = 0;  // rcv flag for time-out monitoring
	int       _rcv_prm0  = 0;  // additional parameter 0
	int       _rcv_prm1  = 0;  // additional parameter 1

	// sender's members
	kmMat1blk _snd_blk;       // sending block buffer
	kmMat1bit _snd_bsf;       // sending block state flag (1: request, 0: not)
	uint      _snd_state = 0; // 0 : not begin, 1: waiting pre-ack, 2: sending blocks, 3: waiting ack, 4: done
	int64     _snd_byte  = 0;
	uint      _snd_ieb   = 0; // iblk of 1st empty block

	// receiving procedure
	virtual void RcvProc(char cmd_id, const kmAddr4& addr)
	{	
		switch(cmd_id)
		{
		case 1: RcvPreBlk(addr); break;
		case 2: RcvPreAck(addr); break;
		case 3: RcvBlk   (addr); break;
		case 4: RcvAck   (addr); break;
		}
	};
	
	///////////////////////////////////////
	// interface functions

	// send data to addr
	// 
	//   return : 0 (sending failed), 1 (sending is successful)
	int Send(ushort src_id, ushort des_id, const kmAddr4& addr, void* data, int64 byte, int prm0 = 0, int prm1 = 0)
	{
		// check state
		if(_snd_state != 0)
		{
			print("[kmNetPtcLrgd::Send in 2529] _snd_state is not zero\n"); return 0;
		}
		_snd_state = 1; // waiting pre-ack

		// init parameters
		const uint blk_byte  = UDP_MAX_DATA_BYTE/2 - sizeof(kmNetHd) - 8; // max limit
		//const uint blk_byte  = UDP_MSS_BYTE - sizeof(kmNetHd) - 8; // efficient size ??
		const uint blk_n     = uint((byte - 1)/blk_byte + 1);

		// init variables
		uint snd_blk_n = 0;

		// set buffer
		_snd_blk.Set((char*)data, byte, blk_byte, blk_n);
		
		_snd_bsf.RecreateIf(blk_n);
		_snd_bsf.SetZero();
		_snd_ieb = 0;

		// send pre block... cmd 0, state 1
		SndPreBlk(src_id, des_id, addr, blk_n, blk_byte, byte, prm0, prm1);

		// waiting pre ack... cmd 1, state 2
		const float preack_tout_msec = 1000.f; // time out for preack

		kmTimer timer(1);
		while(_snd_state == 1 && timer.msec() < preack_tout_msec) Sleep(0);

		if(_snd_state == 1) // time out and sending failed
		{
			print("* sending failed by timeout for preack\n"); _snd_state = 0; return 0;
		}
		const float echo_usec = (float)timer.usec();

		print("\n*** blk_n : %d, blk_byte : %d, echo time : %.1f usec\n", blk_n, blk_byte, echo_usec);

		// init congestion control parameters
		const uint  max_cw_n       = 256;
		const float max_snd_t_usec = blk_byte*(8.f/1000.f); // 1000MBPS

		uint  cw_n       = 2;              // size of congestion window		
		float snd_t_usec = max_snd_t_usec; // 100 usec
		float out_t_usec = echo_usec*2.f;  // out time
		uint  skip_cnt   = 0;              // skip counter

		print("*** sending block (snd_t_usec : %.2f usec, out_t : %.2f usec)\n", snd_t_usec, out_t_usec);

		// main loop for sending blocks
		for(uint iblk = 0; iblk < blk_n; iblk = _snd_ieb)
		{	
			// loop for congestion window
			for(uint icw = 0; icw < cw_n; ++icw, ++iblk)
			{
				// find iblk to send
				for(; iblk < blk_n; ++iblk) if(_snd_bsf(iblk)==0) break;
		
				if(iblk >= blk_n) break; // if there is no block to send
		
				// set request for ack... reqack
				bool reqack = true;
		
				if(icw != cw_n - 1) // if not the last of cw, reqack will be zero.
				{
					for(uint i = iblk + 1; i < blk_n; ++i) if(_snd_bsf(i) == 0) { reqack = false; break; }
				}

				// send blocks... cmd 2
				if(reqack) _snd_state = 3;

				//if(rand() > RAND_MAX/10) SndBlk(src_id, des_id, iblk, addr, reqack); // only for test
				SndBlk(src_id, des_id, iblk, addr, reqack);
		
				++snd_blk_n; if(reqack == 1) break;
		
				// control sending time
				timer.Wait(snd_t_usec);
			}
			// wait for ack... cmd 3
			for(timer.Start(); timer.usec() < out_t_usec && _snd_state == 3; Sleep(0)) {}

			// congestion control
			if(_snd_state == 3) // no ack
			{
				if(++skip_cnt > 8) // sending failed
				{
					_snd_state = 0; print("* sending failed : over no ack\n"); return 0;
				};
				// update cw_n and out_t_sec
				out_t_usec = MIN(out_t_usec*1.5f, 2e5f);
				cw_n       = MAX(cw_n/2, 2);

				print("** no ack  (cw_n : %d, out_t : %.2fusec)\n", cw_n, out_t_usec);
			}
			else if(_snd_state == 2) // got ack
			{
				// calc number of lost blocks
				uint lst_n = 0; for(uint i = _snd_ieb; i <= iblk; ++i) if(_snd_bsf(i) == 0) ++lst_n;

				float lst_rat = float(lst_n)/cw_n;

				// update snd_t_usec, out_t_usec and cw_n
				if(lst_rat > 0.01f) snd_t_usec *= (1.f + lst_rat);
				else
				{
					cw_n = MIN(cw_n*4, max_cw_n);
					snd_t_usec *= 0.8f;
				}
				if(skip_cnt == 0) out_t_usec *= 0.8f;

				// reset skip_cnt
				skip_cnt = 0;

				print("** get ack (lst_n : %d, cw_n : %d, out_t : %.1fusec, snd_t: %.2fusec)\n", lst_n, cw_n, out_t_usec, snd_t_usec);
			}
		}
		// end of sending		
		print("* end of sending large data\n");
		print("* sending count (%d / %d), loss (%.2f%%)\n", snd_blk_n, blk_n, 100.f*(snd_blk_n - blk_n)/blk_n);
		_snd_state = 0;
		return 1;
	};

	///////////////////////////////////////
	// cmd 1 : pre block
	void SndPreBlk(ushort src_id, ushort des_id, const kmAddr4& addr, uint blk_n, 
		           uint blk_byte, int64 byte, int prm0 = 0, int prm1 = 0)
	{
		// set buffer
		const char cmd_id = 1; kmNetHd hd = {src_id, des_id, _ptc_id, cmd_id, 0, 0};
		
		*_net << hd << blk_n << blk_byte << byte << prm0 << prm1;

		// send buffer
		_net->Sendto(addr);
	};
	void RcvPreBlk(const kmAddr4& addr)
	{	
		// get from buffer
		kmNetHd hd{}; uint blk_n, blk_byte; int64 byte;
		
		*_net >> hd >> blk_n >> blk_byte >> byte >> _rcv_prm0 >> _rcv_prm1;

		// check state
		if(_rcv_state != 0)
		{	
			SndPreAck(hd.des_id, hd.src_id, addr, 0); // to reject
			print("[kmNetPtcLrgd::RcvPreBlk in 2634] _rcv_state is not zero\n"); return;
		}
		_rcv_state = 1; // waiting blk
		
		// create rcv_buf and set rcv_bsf
		_rcv_blk.Recreate(byte, blk_byte, blk_n);

		_rcv_bsf.RecreateIf(blk_n);
		_rcv_bsf.SetZero();

		_rcv_byte = 0;
		_rcv_ieb  = 0;

		// create time out monitoring thread
		_rcv_thrd.Begin([](kmNetPtcLrgd* ptc)
		{
			const float tout_msec = 1e3f;
			kmTimer timer(1);
		
			while(timer.msec() < tout_msec)
			{
				if(ptc->_rcv_state != 1) return;
		
				if(ptc->_rcv_flg == 1) { ptc->_rcv_flg = 0; timer.Start(); }
				Sleep(2);
			}
			print("** receiving failed by time out\n");
			ptc->_rcv_state = 0;			
		}, this);

		// send ack
		SndPreAck(hd.des_id, hd.src_id, addr);

		// call cb
		(*_ptc_cb)(_net, 1, 0);
	};

	///////////////////////////////////////////////
	// cmd 2 : pre ack
	void SndPreAck(ushort src_id, ushort des_id, const kmAddr4& addr, uint accept = 1)
	{
		// set buffer
		const char cmd_id = 2; kmNetHd hd = {src_id, des_id, _ptc_id, cmd_id, 0, 0};
		
		*_net << hd << accept;

		// send buffer
		_net->Sendto(addr);
	};
	void RcvPreAck(const kmAddr4& addr)
	{
		// get from buffer
		kmNetHd hd{}; uint accept;

		*_net >> hd >> accept;

		// set state 
		if(_snd_state == 1)
		{
			if(accept == 1) _snd_state = 2;
			else 
			{
				_snd_state =0;
				print("[kmNetPtcLrgd::RcvPreAck in 2697] rejected from %d\n", hd.src_id);
			}
		}
	};

	///////////////////////////////////////////////
	// cmd 3 : blk
	void SndBlk(ushort src_id, ushort des_id, uint iblk, const kmAddr4& addr, bool reqack = false)
	{
		// set snd_buf
		const char cmd_id = 3; kmNetHd hd = { src_id, des_id, _ptc_id, cmd_id, 0, 0};

		if(reqack) hd.SetReqAck(); 

		(*_net << hd << iblk).PutData(_snd_blk.GetBlkPtr(iblk), (ushort)_snd_blk.GetBlkByte(iblk));

		// send buffer
		_net->Sendto(addr);

		// call cb
		(*_ptc_cb)(_net, -3, (void*)&iblk);
	};
	void RcvBlk(const kmAddr4& addr)
	{
		// get from buffer
		kmNetHd hd{}; uint iblk; ushort byte; char* data;

		*_net >> hd >> iblk >> byte; data = _net->_rcv_buf.End1();

		// copy data to _rcv_buf
		if(_rcv_bsf(iblk) == 0)
		{
			memcpy(_rcv_blk.GetBlkPtr(iblk), data, byte); 

			_rcv_bsf(iblk) = 1; _rcv_byte += byte;
		}
		_rcv_flg = 1;

		// update 1st empty block... _rcv_ieb
		if (_rcv_ieb == iblk)		
		for(_rcv_ieb =  iblk + 1; _rcv_ieb < _rcv_blk._blk_n; ++_rcv_ieb)
		{
			if(_rcv_bsf(_rcv_ieb) == 0) break;
		}

		// check if reqack
		if(hd.IsReqAck()) SndAck(hd.des_id, hd.src_id, iblk, addr);

		// call callback when receiving is done.
		int arg = (int)iblk;

		if(_rcv_ieb == _rcv_blk._blk_n) // receiving is done
		{
			_rcv_state = 2; (*_ptc_cb)(_net, 3, (void*)&(arg = -1)); _rcv_state = 0; 
		}
		else (*_ptc_cb)(_net, 3, (void*)&arg); // not done
	};

	///////////////////////////////////////////////
	// cmd 4 : ack
	void SndAck(ushort src_id, ushort des_id, uint iblk, const kmAddr4& addr)
	{	
		// init parameters
		const uint end = end32;
		
		// set buffer
		const char cmd_id = 4; kmNetHd hd = { src_id, des_id, _ptc_id, cmd_id, 0, 0};

		*_net << hd << iblk;
		
		for(uint i = _rcv_ieb; i < iblk; ++i)
		{
			if(_rcv_bsf(i) == 0) *_net << i; // add lost block index to snd_buf
		}
		*_net << end;

		// send buffer
		_net->Sendto(addr);
	};
	void RcvAck(const kmAddr4& addr)
	{
		// get from buffer
		kmNetHd hd{}; uint iblk, iblk_lst;

		*_net >> hd >> iblk >> iblk_lst;

		// update snd_bsf... _snd_bsf
		for(uint i = _snd_ieb; i <= iblk; ++i)
		{
			if(i < iblk_lst) _snd_bsf(i) = 1;
			else             *_net >> iblk_lst;
		}

		// update snd_ieb... _snd_ieb
		const uint blk_n = _snd_blk._blk_n;

		for(uint i = _snd_ieb; i < blk_n; ++i)
		{
			if(_snd_bsf(i) == 0) { _snd_ieb = i; _snd_state = 2; return; }  // not yet done
		}
		_snd_ieb = _snd_blk._blk_n; if(_snd_state != 0) _snd_state = 4; // sending done
	};
};

// control class of kmNetPtcFile
class kmNetPtcFileCtrl
{
public:
	int64  total_byte = 0; // total byte
	int64  byte       = 0; // sending or receiving byte	
	float  t_usec     = 0; // transfer time
	int    reject     = 0; // 0 : accept, 1: reject
	float  loss       = 0; // lost packet percentage
	kmStra file_name{};    // file name only... utf8
	kmStra file_path{};    // file name including full path.. .utf8

	// init
	void Init(const kmStra& file_path, const kmStra& file_name, int64 total_byte)
	{
		byte = 0; t_usec = 0; reject = 0; loss = 0;

		this->total_byte = total_byte;
		this->file_path  = file_path;
		this->file_name  = file_name;
	};
	
	// set path in cbRcvNetPtcFile() or vcbRcvPtcFile() at cmd_id == 1
	// * Note that you should call this function after file_name is fixed
	void SetPath(const kmStra& path)
	{
		file_path.SetStr("%s/%s.ing", path.P(), file_name.P());
	};

	// reject receiving in cbRcvNetPtcFile() or vcbRcvPtcFile() at cmd_id == 1
	void Reject() { reject = 1; };
};

// kmNetPtcFile's result enum class
//  inqueue = 2, success = 1, 
//  nonexistence = 0, sndstatewrong = -1, preacktimeout = -2, preackwrong = -3, 
//  preackrejected = -4, skipmax = -5, idwrong = -6, optwrong = -7
enum class kmNetPtcFileRes : int
{
	inqueue        =  2,
	success        =  1,
	nonexistence   =  0,	 
	sndstatewrong  = -1,
	preacktimeout  = -2, 
	preackwrong    = -3, 
	preackrejected = -4,
	skipmax        = -5, 
	idwrong        = -6, 
	optwrong       = -7,
	notsupported   = -8
};

// basic protocol to send a file
class kmNetPtcFile: public kmNetPtc
{
public:
	// receiver's members	
	kmFileBlk _rcv_blk;        // receiving file block buffer
	kmMat1bit _rcv_bsf;        // receiving block state flag (0 : not yet, 1: received)
	int       _rcv_state  = 0; // 0 : not begin, 1: waiting blocks, 2: done
	int64     _rcv_byte   = 0;
	uint      _rcv_ieb    = 0; // iblk of 1st empty block
	ushort    _rcv_src_id = 0;
	uint      _rcv_snd_id = 0;
	int       _rcv_prm[3] ={}; // additional parameter 

	// sender's members	
	kmFileBlk _snd_blk;          // sending file block buffer
	kmMat1bit _snd_bsf;          // sending block state flag (1: request, 0: not)
	int       _snd_state   = 0;  // 0 : not begin, 1: waiting pre-ack, 2: sending blocks, 3: waiting ack, 4: done, -1: preack wrong
	uint      _snd_id      = 0;  // to avoid mismatching between preblk and preack
	int64     _snd_byte    = 0;	
	uint      _snd_ieb     = 0;  // iblk of 1st empty block	
	int       _snd_ack_on  = 0;
	int       _snd_iack    = -1; // iblk of received ack
	ushort    _snd_src_id  = 0;
	int       _snd_prm[3]  = {}; // additional parameter
	kmNetCnd  _snd_cnd;          // tuning parameters
	kmLock    _snd_lck;
	uint      _snd_blk_byte = UDP_BLK_BYTE; // block size (byte)

	// control memebers
	kmNetPtcFileCtrl _rcv_ctrl;
	kmNetPtcFileCtrl _snd_ctrl;

	// receiving procedure
	virtual void RcvProc(char cmd_id, const kmAddr4& addr)
	{	
		switch(cmd_id)
		{
		case 1: RcvPreBlk  (addr); break;
		case 2: RcvPreAck  (addr); break;
		case 3: RcvBlk     (addr); break;
		case 4: RcvAck     (addr); break;
		case 5: RcvDone    (addr); break;
		case 6: RcvEmptyQue(addr); break;
		}
	};

	///////////////////////////////////////
	// interface functions

	// set block byte	
	void SetBlkByte(uint blk_byte = UDP_BLK_BYTE)
	{
		_snd_blk_byte = blk_byte;
	};

	// get block byte
	uint GetBlkByte() const { return _snd_blk_byte; };

	// set tuning parameters
	void SetSndCnd(const kmNetCnd& cnd) 
	{
		_snd_cnd.tkb_ms = cnd.tkb_ms;
		_snd_cnd.toh_ms = cnd.toh_ms;
		_snd_cnd.tlt_ms = cnd.tlt_ms;
	};

	// put tuning parameters to
	void PutSndCnd(kmNetCnd& cnd)
	{
		cnd.tkb_ms = _snd_cnd.tkb_ms;
		cnd.toh_ms = _snd_cnd.toh_ms;
		cnd.tlt_ms = _snd_cnd.tlt_ms;
	};

	// send file
	//   path : path only
	//   name : sub-path + file name
	// 
	//    ex) d:/folder/sub_folder/file1.exe 
	//           path = d:/folder, name = sub_folder/file1.exe
	// 
	//   return : <= 0 (sending failed), 1 (sending is successful)
	//             0 (nonexistance), -1 (snd_state is not 0), 
	//            -2 (preack timeout), -3 (preack wrong), -4 (preack reject), -5 (skip max)
	kmNetPtcFileRes Send(ushort src_id, ushort des_id, kmAddr4 addr, 
		const kmStra& path, const kmStra& name, int* prm = nullptr)
	{
		// check state
		if(_snd_state != 0)
		{
			print("[kmNetPtcFile::File in 1764 _snd_state is not zero\n"); 
			return kmNetPtcFileRes::sndstatewrong;
		}
		_snd_state = 1; // waiting pre-ack

		// init parameters
		const uint blk_byte = _snd_blk_byte;

		const kmStrw full(L"%s/%s", path.cuw().P(), name.cuw().P());

		if(!kmFile::Exist(full.P()))
		{
			_snd_state = 0; return kmNetPtcFileRes::nonexistence;
		}

		// open file ans set file block
		_snd_blk.OpenToRead(full.P(), blk_byte);

		const int64 byte  = _snd_blk.GetByte();
		const int   blk_n = _snd_blk.GetBlkN();

		// set bsf
		_snd_bsf.RecreateIf(blk_n);
		_snd_bsf.SetZero();
		_snd_ieb = 0;

		// set parameters
		if(prm == nullptr)  memset(_snd_prm,   0, sizeof(_snd_prm));
		else				memcpy(_snd_prm, prm, sizeof(_snd_prm));

		// send pre block... cmd 0, state 1
		SndPreBlk(_snd_src_id = src_id, des_id, addr, blk_n, blk_byte, byte, name);

		_snd_ctrl.Init(path, name, byte);

		(*_ptc_cb)(_net, -1, (void*)&_snd_ctrl);

		print("\n*** 1. send file : %d kbyte (%d byte x %d), id: %x, %s\n", 
			byte>>10, blk_byte, blk_n, _snd_id, _snd_cnd.GetStr().P());

		// waiting for preack... cmd 1, state 2
		float preack_tout_sec = 0.5f; // time out for preack

		kmTimer bps_timer, timer(1);

		for(int snd_cnt = 4; snd_cnt--; preack_tout_sec += preack_tout_sec)
		{
			timer.Start();

			while(_snd_state == 1 && timer.sec() < preack_tout_sec) Sleep(0);

			if(_snd_state == 2) // received preack
			{
				const float ec_msec = (float)timer.msec();

				//if (_snd_cnd.tlt_ms > ec_msec + 0.1f) _snd_cnd.tlt_ms = ec_msec + 0.1f;

				print("*** 2. rcv preack : %.2f msec\n", ec_msec);
				break;
			}
			else // time out and sending failed
			{
				if(_snd_state == 1) // preack timeout
				{
					if(snd_cnt > 0)
					{
						SndPreBlkAgain(addr); // send preblck once more
						print("*** sndpreblk again\n");
					}
					else
					{
						print("* sending failed by timeout for preack : %.1f sec\n", timer.sec()); 
						_snd_state = 0; _snd_blk.Close(); return kmNetPtcFileRes::preacktimeout;
					}
				}
				else if(_snd_state == -1) // preack is wrong
				{
					print("* preack is wrong and will send sndblk(reject)\n"); 
					SndBlk(src_id, des_id, 0, addr, 0, true); 
					_snd_state = 0; _snd_blk.Close(); return kmNetPtcFileRes::preackwrong;
				}
				else // reject
				{
					print("* sending has been rejected\n");
					_snd_state = 0; _snd_blk.Close(); return kmNetPtcFileRes::preackrejected;
				}
			}
		}

		// init congestion control parameters	
		const uint thrs_skip_cnt  = 16;

		int   cw_n       = MIN(MAX(32, 1024*512/blk_byte), 512); // blk number for reqack
		float snd_t_usec = _snd_cnd.GetSndTime(blk_byte)*1e3f;
		float out_t_msec = _snd_cnd.GetTimeOut(blk_byte, cw_n);

		// init variables
		uint skip_cnt = 0, nowait_cnt = 0, tout_cnt = 0;
		uint snd_cnt  = 0;                // number of sending blk without waiting
		int  iack_max = -1, snd_n = 0;
		int  ibb      = -1;               // biggest index fo blocks sent
		int  x{}, y{}, oldx{}, oldy{};    // cursor position

		_snd_ack_on = 0;

		//print("*** 3. send blocks : cw %d, snd %.1f usec, out %.1f msec\n", cw_n, snd_t_usec, out_t_msec);
		print("*** 3. send blocks : "); kmfgetcursor(x,y); print("\n");
		bps_timer.Start();

		// main loop for sending blocks		
		for(int iblk = 0, iack = -1, tbd_n = 0; _snd_state < 4;)
		{
			timer.Start();

			// find iblk to send... iblk
			for(; iblk < blk_n; ++iblk)
			{
				if(iblk < iack && _snd_bsf(iblk) == 0) break;
				if(ibb  < iblk) break;
			}
			// check if there is no more block to send... noblk
			bool noblk = (iblk >= blk_n);

			if(noblk)
			{
				for(;iblk--;) if(_snd_bsf(iblk) == 0) break;
				if(iblk < 0) iblk = 0;
			}
			
			// check if iblk is the last... islast
			bool islast = true;

			for(int i = iblk + 1; i < blk_n; ++i)
			{
				if(_snd_bsf(i) == 0) { islast = false; break;}
			}

			// set reqack and waitack ... reqack, waitack
			bool waitack = islast || noblk || (tbd_n > cw_n*4);
			bool reqack  = ((snd_n +1)%cw_n == 0) || waitack;

			if(waitack) { _snd_state = 3; }

			// send block
			SndBlk(src_id, des_id, iblk, addr, reqack); ++snd_n; ++snd_cnt; if(!noblk) ++tbd_n;

			// update ibb (biggest block of blocks to be sent)
			ibb = MAX(ibb, iblk); iblk++;

			// display sending status on console
			if( (iblk & 0b111111) == 0 || noblk || islast)
			{
				kmfgetcursor(oldx, oldy); kmfsetcursor(x, y);
				
				print("%5d:%d%d%d [%5d/%5d]", iblk-1, reqack, waitack, noblk, ibb + 1, blk_n);
				print(", cw %d, snd %.1fusec, out %.1fmsec",  cw_n, snd_t_usec, out_t_msec);

				kmfsetcursor(oldx, oldy);
			}

			// wait for sending time
			if(waitack == false)
			{
				if(timer.usec() > snd_t_usec) ++nowait_cnt;

				if(_snd_ack_on > 0) // check nonstop ack
				{
					// update parameters for ack
					iack     = _snd_iack;
					iblk     = _snd_ieb;
					iack_max = MAX(iack_max, iack);
					tbd_n    = ibb - iack_max;
					skip_cnt = 0;

					// get lost block number... lst_n, lst_rat
					int lst_n = 0;

					for(int i = _snd_ieb; i < iack; ++i) if(_snd_bsf(i) == 0) ++lst_n;

					float lst_rat = MIN(lst_n/(float)cw_n, 0.1f);

					//print("** nack(%d), lst_n / cw_n (%d / %d), nowait %d\n", iack1, lst_n, cw_n, nowait_cnt);

					// congestion control
					if(lst_n > 0) _snd_cnd.tkb_ms *= (1.f + lst_rat);
					else
					{
						if(nowait_cnt < 2) _snd_cnd.tkb_ms *= 0.98f;

						nowait_cnt = 0;
					}
					snd_t_usec = _snd_cnd.GetSndTime(blk_byte)*1e3f; _snd_ack_on = 0;
				}
				while(timer.usec() < snd_t_usec) Sleep(0);
				continue;
			}
			
			// wait for ack
			//if(skip_cnt < 3) out_t_msec = MIN(MAX( 10.f, _snd_cnd.GetTimeOut(blk_byte, snd_cnt)), 200.f);
			//else             out_t_msec = MIN(MAX(200.f, _snd_cnd.GetTimeOut(blk_byte, snd_cnt)), 500.f);
			out_t_msec = _snd_cnd.GetTimeOut(blk_byte, snd_cnt);

			//print("**** waiting for ack (iblk : %d, out : %.1f msec)", iblk - 1, out_t_msec);

			snd_cnt = 0;

			for(timer.Start(); timer.msec() < out_t_msec; Sleep(0))
			{
				if(_snd_state == 4)
				{
					print("...state is 4 : %.1f msec\n", timer.msec());
					break;
				}
				else if(_snd_state == 2)
				{
					// update parameters for ack
					iack     = _snd_iack;
					iack_max = MAX(iack_max, iack);
					tbd_n    = ibb - iack_max;

					if(_snd_iack == iblk - 1)
					{
						//print("...received ack : %.1f msec\n", timer.msec());
						break; // it is the ack to be waited
					}
					else // it's not the ack
					{
						_snd_ack_on = 0;
						_snd_state  = 3;
					}
				}
			}
			if(_snd_state == 4) break;

			// process after receiving ack
			if(_snd_state == 3) // timeout
			{
				++tout_cnt;

				//print("...time out (%d-%d) : %.1f msec\n", skip_cnt, tout_cnt, timer.msec());

				if(++skip_cnt > thrs_skip_cnt) // sending failed
				{
					print("******* sending failed\n");
					_snd_state = 0;  _snd_blk.Close();
					(*_ptc_cb)(_net, -4, (void*)&_snd_ctrl);
					return kmNetPtcFileRes::skipmax;
				}
				// congestion control
				if(skip_cnt > 1) _snd_cnd.tlt_ms *= 1.1f;

				_snd_cnd.tkb_ms *= 1.1f;

				snd_t_usec = _snd_cnd.GetSndTime(blk_byte)*1e3f;

				_snd_state = 2;
			}
			else if(_snd_state == 2) // receiving ack
			{
				// congestion control
				int ieb = _snd_ieb, lst_n = 0;
				
				for(int i = ieb; i < iack; ++i) if(_snd_bsf(i) == 0) ++lst_n;

				float lst_rat = MIN(lst_n/(float)cw_n, 0.5f);

				//print("****** wack(%d), lst_n / cw_n (%d / %d), nowait (%d)\n", iack, lst_n, cw_n, nowait_cnt);

				if(lst_n > 0) _snd_cnd.tkb_ms *= (1.f + lst_rat);
				else
				{
					if(nowait_cnt < 2) _snd_cnd.tkb_ms *= 0.98f;

					nowait_cnt = 0;
				}
				snd_t_usec = _snd_cnd.GetSndTime(blk_byte)*1e3f; _snd_ack_on = 0;

				// update variables
				iblk        = _snd_ieb;
				tbd_n       = 0;
				skip_cnt    = 0;
			}
		}
		const float snd_mbps = (float)(byte/bps_timer.usec());

		// call cb to display comment
		_snd_ctrl.t_usec = (float)bps_timer.usec();
		_snd_ctrl.byte   = _snd_ctrl.total_byte;
		_snd_ctrl.loss   = (snd_n == 0)? 0:100.f*(snd_n - blk_n)/snd_n;

		(*_ptc_cb)(_net, -3, (void*)&_snd_ctrl);

		print("*** 4. done : %.1f MB/s, loss %.2f %% (%d blocks), tout (%d)\n", 
			   _snd_ctrl.byte/_snd_ctrl.t_usec, _snd_ctrl.loss, snd_n - blk_n, tout_cnt);

		// update tlt_ms
		if(tout_cnt == 0) _snd_cnd.tlt_ms *= 0.98f;

		// send done
		SndDone(src_id, des_id, addr);

		// end of sending
		_snd_state = 0;  _snd_blk.Close();
		return kmNetPtcFileRes::success;
	};

	// send data to addr
	//   path : full path
	//
	kmNetPtcFileRes Send(ushort src_id, ushort des_id, kmAddr4 addr, kmStra& path, int* prm = nullptr)
	{
		kmStra name = path.SplitRvrs('/');

		if(name.N1() == 0) name = path.SplitRvrs('\\');

		return Send(src_id, des_id, addr, path, name, prm);
	};

	// stop receving procedure
	void StopRcv()
	{
		// reset receiving parameters
		_net->SetTomOff(); _rcv_blk.Close(); _rcv_state = 0;

		// delete ing file		
		kmStra cur_name = _rcv_ctrl.file_path;		

		if(kmFile::Exist(cur_name.cuw().P())) kmFile::Remove(cur_name.cuw().P());

		// callback
		(*_ptc_cb)(_net, 4, 0);
	};

	///////////////////////////////////////
	// cmd 1 : pre block
	void SndPreBlk(ushort src_id, ushort des_id, const kmAddr4& addr, uint blk_n, 
		uint blk_byte, int64 byte, const kmStra& file_str) // sender
	{
		// set id
		_snd_id = kmfrand(0u, end32u>>1);
		
		// set buffer
		const char cmd_id = 1; kmNetHd hd = {src_id, des_id, _ptc_id, cmd_id, 0, 0}; 

		*_net << hd << blk_n << blk_byte << byte << _snd_id;

		(*_net << _snd_prm[0] << _snd_prm[1] << _snd_prm[2]).PutData(file_str);

		// send buffer
		_net->Sendto(addr);
	};
	void SndPreBlkAgain(const kmAddr4& addr) { _net->Sendto(addr); }; // sender
	void RcvPreBlk(const kmAddr4& addr) // receiver
	{
		// get from buffer
		kmNetHd hd{}; uint blk_n, blk_byte, snd_id; int64 byte; kmStra file_str;

		*_net >> hd >> blk_n >> blk_byte >> byte >> snd_id;		

		// check state
		if(_rcv_state == 1)
		{
			if(snd_id == _rcv_snd_id) // received preackagain
			{
				SndPreAck(hd.des_id, hd.src_id, addr, snd_id);
				return;
			}
			else
			{
				if(_rcv_src_id == hd.des_id) // if it has the same src_id
				{
					StopRcv(); // stop and restart receiving
				}
				else // if it has different src_id
				{
					SndPreAck(hd.des_id, hd.src_id, addr, 0, 0); // to reject
					print("[kmNetPtcFile::RcvPreBlk] _rcv_state(%d) is not zero\n", _rcv_state);
					return;
				}
			}
		}
		else if(snd_id == _rcv_snd_id) // state 0 or 2
		{
			print("[RcvPreBlk] id (%x) is the same with old id in rcv state 0 or 2\n", snd_id);
			return;
		}

		// get the rest from buffer
		(*_net >> _rcv_prm[0] >> _rcv_prm[1] >> _rcv_prm[2]).GetData(file_str);

		// set state
		_rcv_state  = 1;          // waiting blk
		_rcv_src_id = hd.des_id;
		_rcv_snd_id = snd_id;

		// get path to save the file
		_rcv_ctrl.file_name  = file_str;
		_rcv_ctrl.total_byte = byte;
		_rcv_ctrl.reject     = 0;

		(*_ptc_cb)(_net, 1, (void*)&_rcv_ctrl);

		if(_rcv_ctrl.reject == 1)
		{	
			SndPreAck(hd.des_id, hd.src_id, addr, snd_id, 0);
			_rcv_state = 0; return;
		}

		// check if file_path is availables
		kmStra path = _rcv_ctrl.file_path;
		
		if(path.ReplaceRvrs('/', '\0') == 0) path.ReplaceRvrs('\\', '\0');

		kmFile::MakeDirs(path.cuw());

		// init rcv parameters
		_rcv_blk.OpenToWrite(_rcv_ctrl.file_path.cuw().P(), byte, blk_byte, blk_n);
		_rcv_bsf.RecreateIf(blk_n);
		_rcv_bsf.SetZero();
		_rcv_byte   = 0;
		_rcv_ieb    = 0;

		// send ack
		SndPreAck(hd.des_id, hd.src_id, addr, snd_id);

		_net->SetTom(3.f);

		print("\n*** 1. rcv preblk : %d kbyte (%d kbyte x %d), id: %x\n", 
			  byte>>10, blk_byte>>10, blk_n, _rcv_snd_id);
	};

	///////////////////////////////////////////////
	// cmd 2 : pre ack
	void SndPreAck(ushort src_id, ushort des_id, const kmAddr4& addr, uint snd_id, uint accept = 1) // receiver
	{
		// set buffer
		const char cmd_id = 2; kmNetHd hd = {src_id, des_id, _ptc_id, cmd_id, 0, 0};

		*_net << hd << snd_id << accept;

		// send buffer
		_net->Sendto(addr);
	};
	void RcvPreAck(const kmAddr4& addr) // sender
	{
		// get from buffer
		kmNetHd hd{}; uint accept, snd_id;

		*_net >> hd >> snd_id >> accept;

		// check state
		if(_snd_state != 1) return;

		// check snd_id and accept
		if(accept == 1)
		{	
			if(snd_id == _snd_id) _snd_state = 2;
			else
			{
				_snd_state = -1;
				print("** snd_id is wrong (%d != %d)\n", _snd_id, snd_id);
			};
		}
		else _snd_state = 0; // reject
	};

	///////////////////////////////////////////////
	// cmd 3 : blk
	void SndBlk(ushort src_id, ushort des_id, uint iblk, const kmAddr4& addr, bool reqack = false, bool reject = false) // sender
	{
		// set snd_buf
		const char cmd_id = 3; kmNetHd hd = { src_id, des_id, _ptc_id, cmd_id, 0, 0};

		if(reqack) hd.SetReqAck(); 
		if(reject) hd.SetReject();

		const ushort blk_byte = _snd_blk.GetBlkByte(iblk);

		*_net << hd << _snd_id << iblk << blk_byte;

		_snd_blk.ReadBlk(_net->_snd_buf.End1(), iblk); _net->_snd_buf.IncN1(blk_byte);

		// send buffer
		_net->Sendto(addr);
	};
	void RcvBlk(const kmAddr4& addr) // receiver
	{	
		// get from buffer
		kmNetHd hd{}; uint iblk; ushort byte; uint snd_id; char* data; 

		*_net >> hd >> snd_id >> iblk >> byte; data = _net->_rcv_buf.End1();

		// check rcv state
		if(_rcv_state == 2)
		{
			if(hd.IsReqAck())
			{
				SndAckLast(hd.des_id, hd.src_id, iblk, addr);
				print("* [RcvBlk] rcv_state : 2, sndacklast (iblk: %d, id: %x)\n", iblk, snd_id);
			}
			return;
		}
		else if(_rcv_state == 0) // skip
		{
			print("* [RcvBlk] rcv_state : 0 (iblk: %d, id: %x)\n", iblk, snd_id);
			return;
		}

		// check snd_id
		if(snd_id != _rcv_snd_id) // skip
		{
			print("* [RcvBlk] snd_id(%d) != _rcv_snd_id(%d)\n", snd_id, _rcv_snd_id);
			return;
		}

		// check reject
		if(hd.IsReject())
		{
			print("* [RcvBlk] rejection has been received (id :%x)\n", snd_id);
			StopRcv();
			return;
		}
		
		// copy data to _rcv_buf
		if(_rcv_bsf(iblk) == 0)
		{
			_rcv_blk.WriteBlk(iblk, data);

			_rcv_bsf(iblk) = 1; _rcv_byte += byte;
		}

		// update 1st empty block... _rcv_ieb
		if (_rcv_ieb == iblk)		
		for(_rcv_ieb =  iblk + 1; _rcv_ieb < _rcv_blk._blk_n; ++_rcv_ieb)
		{
			if(_rcv_bsf(_rcv_ieb) == 0) break;
		}

		// check tom
		_net->SetTomOn();

		// display receiving status on console
		static int x{}, y{}; int oldx, oldy;

		if(_rcv_byte == byte) { print("*** 2. rcv blk : "); kmfgetcursor(x, y); print("\n"); }

		if( (iblk & 0b11111111) == 0 || _rcv_ieb == _rcv_blk._blk_n)
		{
			kmfgetcursor(oldx, oldy); kmfsetcursor(x, y);

			print("%5d [%5d/%5d]", iblk, _rcv_ieb, _rcv_blk.GetBlkN());

			kmfsetcursor(oldx, oldy);
		}

		// call callback
		if(_rcv_ieb == _rcv_blk._blk_n) // receiving is done
		{
			// timeout monitoring off
			_net->SetTomOff();
			
			// set state as done
			_rcv_state = 2;			

			// send last ack
			SndAckLast(hd.des_id, hd.src_id, iblk, addr);

			print("*** 3. done\n");

			// close rcv block
			_rcv_blk.Close();

			// rename
			kmStra  cur_name = _rcv_ctrl.file_path;
			kmStra& new_name = _rcv_ctrl.file_path; new_name.Cutback(4);

			if(kmFile::Exist(new_name.cuw().P())) kmFile::Remove(new_name.cuw().P());

			kmFile::Rename(cur_name.cuw().P(), new_name.cuw().P());

			// call cb
			(*_ptc_cb)(_net, 3, nullptr);
		}
		else if(hd.IsReqAck()) // requested ack
		{
			// send ack
			SndAck(hd.des_id, hd.src_id, iblk, addr);

			// call callback
			_rcv_ctrl.byte = (int64)_rcv_ieb*_rcv_blk.GetBlkByte();

			(*_ptc_cb)(_net, 2, (void*)&_rcv_ctrl);
		}
	};

	///////////////////////////////////////////////
	// cmd 4 : ack
	void SndAck(ushort src_id, ushort des_id, uint iblk, const kmAddr4& addr) // receiver
	{	
		// init parameters
		const uint end = end32;

		// set buffer
		const char cmd_id = 4; kmNetHd hd = { src_id, des_id, _ptc_id, cmd_id, 0, 0};

		*_net << hd << _rcv_snd_id << iblk;
		
		for(uint i = _rcv_ieb; i < iblk; ++i)
		{
			if(_rcv_bsf(i) == 0) *_net << i; // add lost block index to snd_buf
		}
		*_net << end;

		// send buffer
		_net->Sendto(addr);
	};
	void SndAckLast(ushort src_id, ushort des_id, uint iblk, const kmAddr4& addr) // receiver
	{
		// init parameters
		const uint done = end32u; // for last ack

		// set buffer
		const char cmd_id = 4; kmNetHd hd = { src_id, des_id, _ptc_id, cmd_id, 0, 0};

		*_net << hd << _rcv_snd_id << iblk << done;

		// send buffer
		_net->Sendto(addr);
	};
	void SndAckRjt(ushort src_id, ushort des_id, uint iblk, uint snd_id, const kmAddr4& addr) // receiver
	{
		// set buffer
		const char cmd_id = 4; kmNetHd hd = { src_id, des_id, _ptc_id, cmd_id, 0, 0}; hd.SetReject();

		*_net << hd << _rcv_snd_id << iblk << end32;

		// send buffer
		_net->Sendto(addr);
	};
	void RcvAck(const kmAddr4& addr) // sender
	{
		// get from buffer
		kmNetHd hd{}; uint snd_id, iblk, iblk_lst;

		*_net >> hd >> snd_id >> iblk >> iblk_lst;

		//print("**************************  rcvack %d\n", iblk);

		// check id
		if(snd_id != _snd_id)
		{
			print("[RcvAck] snd_id(%x) is not _snd_id(%x)\n", snd_id, _snd_id); return;
		}

		// check done
		kmLockGuard grd = _snd_lck.Lock(); ///////////////////// lock

		if(iblk_lst == end32u) // received last ack
		{
			if(_snd_state == 3 || _snd_state == 2) _snd_state = 4;
			return;
		}

		// update snd_bsf... _snd_bsf
		for(uint i = _snd_ieb; i <= iblk; ++i)
		{
			if(i < iblk_lst) _snd_bsf(i) = 1;
			else             *_net >> iblk_lst;
		}		

		// update _snd_ieb, _snd_ack_on
		_snd_iack = iblk;

		const uint blk_n = _snd_blk._blk_n;

		for(uint i = _snd_ieb; i < blk_n; ++i)
		{
			if(_snd_bsf(i) == 0) // not yet done
			{
				_snd_ieb = i; _snd_ack_on = 1; if(_snd_state == 3)_snd_state = 2; 
				return; 
			}  
		}
		_snd_ieb = _snd_blk._blk_n;

		if(_snd_state == 3 || _snd_state == 2) _snd_state = 4; // done
	};

	////////////////////////////////////////////////////////////////////
	// cmd 5 : done
	void SndDone(ushort src_id, ushort des_id, const kmAddr4& addr) // sender
	{
		// set buffer
		const char cmd_id = 5; kmNetHd hd = { src_id, des_id, _ptc_id, cmd_id, 0, 0};

		*_net << hd << _snd_id;

		// send buffer
		_net->Sendto(addr);
	};
	void RcvDone(const kmAddr4& addr) // receiver
	{
		// get from buffer
		kmNetHd hd{}; uint snd_id;

		*_net >> hd >> snd_id;

		if(_rcv_state == 2 && snd_id == _rcv_snd_id)
		{
			print("*** 4. rcv done (id: %x)\n\n", snd_id);
			_rcv_state = 0;
		}
	};

	////////////////////////////////////////////////////////////////////
	// cmd 6 : empty queue
	//  * Note that this will be called in kmNet::_snd_thrd
	void SndEmptyQue(ushort src_id, ushort des_id, const kmAddr4& addr)
	{
		// set buffer
		const char cmd_id = 6; kmNetHd hd = { src_id, des_id, _ptc_id, cmd_id, 0, 0};

		*_net << hd;

		// send buffer
		_net->Sendto(addr);
	};
	void RcvEmptyQue(const kmAddr4& addr)
	{
		// call cb
		(*_ptc_cb)(_net, 6, nullptr);
	};
};

/////////////////////////////////////////////////////////////////////////////////////////
// kmNet class 

// struct for kmNet's snd queue... every kmStra is utf8
class kmNetSndFile
{
public: 
	ushort src_id{}; kmStra path{}; kmStra name{}; int prm[3]{}; 

	kmNetSndFile() {};
	kmNetSndFile(ushort src_id, const kmStra& path, const kmStra& name, int prm_[3]): src_id(src_id), path(path), name(name)
	{
		if(prm_ == nullptr) memset(prm,    0, sizeof(prm));
		else                memcpy(prm, prm_, sizeof(prm));
	};
};

// kmNetId's state enum class
enum class kmNetIdState : ushort { none = 0, connecting = 1, valid = 2, invalid = 3 };

// network device id
class kmNetId
{
public:
	kmMacAddr    mac;
	kmAddr4      addr;
	ushort       des_id   = 0xffff;	
	kmNetIdState state{};         // none, connecting, valid, invalid
	kmDate       time{};          // last connected time
	char         name[64] = {};   // ...utf8
	kmNetKey     vkey;
	kmNetCnd     cnd;

	void Print() const
	{
		print("  mac    : %s\n", mac .GetStr().P());
		print("  addr   : %s\n", addr.GetStr().P());
		print("  des Id : %d\n", des_id);
		print("  state  : %s\n", GetStateStr());
		print("  date   : %s\n", time.GetStrPt().P());
		print("  name   : %s\n", name);
		print("  vkey   : %s\n", vkey.GetStr().P());
		print("  cond   : %s\n", cnd .GetStr().P());
	};

	const char* GetStateStr() const
	{
		switch(state)
		{
		case kmNetIdState::none       : return "none";
		case kmNetIdState::connecting : return "connecting";
		case kmNetIdState::valid      : return "valid";
		case kmNetIdState::invalid    : return "invalid";
		}
		return "unknow";
	};

	const wchar* GetStateStrw() const
	{
		switch(state)
		{
		case kmNetIdState::none       : return L"none";
		case kmNetIdState::connecting : return L"connecting";
		case kmNetIdState::valid      : return L"valid";
		case kmNetIdState::invalid    : return L"invalid";
		}
		return L"unknow";
	};
};
typedef kmMat1<kmNetId> kmNetIds;

// typedef for net callback
using kmNetCb = int(*)(void* parent, uchar ptc_id, char cmd_id, void* arg);

// network base class (using UDP)
class kmNet : public kmNetBase
{
protected:		
	void*     _parent = nullptr;       // parent's pointer for callback
	kmNetCb   _netcb  = nullptr;       // callback function for parent
	kmNetIds  _ids;                    // client ids	

	// rcv thread members
	kmThread  _rcv_thrd;               // thread for receiving
	kmThread  _tom_thrd;               // thread for rcv time-out monitoring
	kmLock    _rcv_lck;                // mutex for RcvProc in multi-thread... double 

	// snd thread members
	kmThread             _snd_thrd;    // thread for sending with snd queue
	kmQue1<kmNetSndFile> _snd_que;     // sending queue for sndfile
	kmLock               _snd_que_lck; // mutex for _snd_que
	kmNetSndFile         _snd_last{};  // info of last sent file
	kmNetPtcFileRes      _snd_res {};  // result of last sent file

	// kal (keep-alive) thread members
	kmThread    _kal_thrd;             // thread for kal (keep-alive)

	// nks members
	kmNetKey    _pkey;                 // own permanent key	
	kmNetKeyRnw _rnw;                  // pkey renewal

	// protocol members
	kmMat1<kmNetPtc*> _ptcs;           // protocol array

	kmNetPtcBrdc      _ptc_brdc; 
	kmNetPtcCnnt      _ptc_cnnt;
	kmNetPtcData      _ptc_data;
	kmNetPtcLrgd      _ptc_lrgd;
	kmNetPtcFile      _ptc_file;
	kmNetPtcNkey      _ptc_nkey;

public:
	// constructor
	kmNet() {};

	// destructor
	virtual ~kmNet() { Close(); };

	// init
	//   return : 1 (success)
	//           -1 (binding failure)
	int Init(void* parent = nullptr, kmNetCb netcb = nullptr, ushort port = KMNETDPORT)
	{
		// set parent and callback
		_parent = parent;
		_netcb  = netcb;

		// get address
		_mac  = kmSock::GetMacAddr();
		_addr = kmSock::GetLocalAddr(port);

		// create buffer and header pointer
		// * Note that 64 KB is max size of UDP packet
		_rcv_buf.Recreate(64*1024);
		_snd_buf.Recreate(64*1024);

		// bind
		for(int i = 8; i--;)
		if(Bind() == -1)
		{
			if(i == 0) return -1;
			
			_addr.SetPort(_addr.GetPort() + 1);
			print("[kmNet::Init] binding failed. port is chaged [%d]", port = _addr.GetPort());
		}
		else break;

		// send dummy for broadcasting
		GetSock().SetSckOptBroadcast(true);
		kmMat1i8 buf(16); buf.SetVal(-1); GetSock().SendtoBroadcast(buf, port);
		GetSock().SetSckOptBroadcast(false);

		// init ids
		_ids.Recreate(0,16);

		// init and add basic protocols
		_ptcs.Recreate(0,16);

		_ptcs.PushBack(_ptc_brdc.Init(0, this));
		_ptcs.PushBack(_ptc_cnnt.Init(1, this, cbRcvPtcCnntStt));
		_ptcs.PushBack(_ptc_data.Init(2, this, cbRcvPtcDataStt));
		_ptcs.PushBack(_ptc_lrgd.Init(3, this, cbRcvPtcLrgdStt));
		_ptcs.PushBack(_ptc_file.Init(4, this, cbRcvPtcFileStt));
		_ptcs.PushBack(_ptc_nkey.Init(5, this, cbRcvPtcNkeyStt));

		// init timeout for ptc_file
		_tom.Set(1.);

		// create snd queue
		_snd_que.Recreate(16);

		// create threads
		CreateRcvThrd();
		CreateSndThrd();
		CreateTomThrd();

		// set addional socket option
		GetSock().SetSndBufByte(65536);    // any value greater than 65526 is ineffective
		GetSock().SetRcvBufByte(65536*8);  // bigger is better

		return 1;
	};

	// change port and re-bind
	//   return : 1 (success)
	//           -1 (binding failure)
	int ChangePort()
	{
		Close();

		_addr.SetPort(_addr.GetPort() + 1);

		// bind
		for(int i = 8; i--;)
		if(Bind() == -1)
		{
			if(i == 0) return -1;
			
			_addr.SetPort(_addr.GetPort() + 1);
			print("[kmNet::ChagePort] binding failed. port is chaged [%d]", _addr.GetPort());
		}
		else break;

		return 1;
	};

	///////////////////////////////////////////////
	// inner functions
protected:
	// create rcv time-out monitoring thread
	void CreateTomThrd()
	{
		_tom_thrd.Begin([](kmNet* net)
		{
			print("* tom thread starts\n");
			kmTimer ext_timer; // timer for extra work			

			while(1)
			{
				if(net->IsTomOn()) // check time out for rcvblk of ptc_file
				{
					if(net->IsTomOut())
					{
						print("** receiving failed with timeout (%.2fsec)\n", net->_tom.GetToutSec());						
						net->_ptc_file.StopRcv(); // including tom off
					}
					// timer control
					if(ext_timer.IsStarted()) ext_timer.Stop();
				}
				else
				{
					// do extra work
					if(ext_timer.sec() > 10.f) net->DoExtraWork();

					// timer control
					if(ext_timer.IsNotStarted()) ext_timer.Start();
				}
				Sleep(100); // because frequent monitoring isn't required
			}
			print("* end of rto thread\n");
		}, this);
		_tom_thrd.WaitStart();
	};

	// virtual function for extra work
	virtual void DoExtraWork() { Sleep(100); };

	// create receiving thread
	void CreateRcvThrd()
	{
		_rcv_thrd.Begin([](kmNet* net)
		{
			print("* rcv thread starts\n");
			int ret;

			while(1) { if((ret = net->RcvProc()) < 1) break; }

			// check error
			if(ret < 0) print("** %s\n", kmSock::GetErrStr().P());

			print("* end of rcv thread\n");
		}, this);
		_rcv_thrd.WaitStart();
	};	

	// receiving procedure... core
	//   [return] 0   : end of receiving process
	//            0 < : the number of bytes received or sent.
	//            0 > : error of the socket
	int RcvProc() 
	{
		// receive data
		kmLockGuard grd = _rcv_lck.Lock(); /////////////// lock-unlock

		kmAddr4 addr; int ret = Recvfrom(addr);

		if(ret <= 0) return ret;

		// get header and check source id, protocol id
		const ushort src_id = _rcv_buf.GetHd().des_id;
		const uchar  ptc_id = _rcv_buf.GetHd().ptc_id;
		const  char  cmd_id = _rcv_buf.GetHd().cmd_id;

		if(src_id < (ushort)_ids.N1())
		{
			if(_ids(src_id).addr == addr) _ids(src_id).time.SetCur();
		}		
		if(_ptcs.N1() <= ptc_id) return ret; // check if ptc_id comes within range

		// call receiving procedure of the protocol
		_ptcs(ptc_id)->RcvProc(cmd_id, addr);
		
		return ret;
	};

	// create sending thread
	void CreateSndThrd()
	{
		_snd_thrd.Begin([](kmNet* net)
		{
			print("* snd thread starts\n");

			while(1)
			{	
				if(net->_snd_que.N() > 0)
				for(int64 i = net->_snd_que.N(); i--;)
				{
					net->_snd_res = kmNetPtcFileRes::inqueue;

					net->_snd_que_lck.Lock();   /////////////  lock

					kmNetSndFile snd = *net->_snd_que.Dequeue();

					net->_snd_que_lck.Unlock(); //////////// unlock

					if(net->GetIds()(snd.src_id).cnd.isdone == 0) net->TestNetCnd(snd.src_id);

					kmNetPtcFileRes ret{}; int max_cnt = 32;

					for(int cnt = 0; cnt < max_cnt; ++cnt)
					{
						ret = net->SendFile(snd.src_id, snd.path, snd.name, snd.prm);

						if(ret == kmNetPtcFileRes::success ||
						   ret == kmNetPtcFileRes::skipmax ||
						   ret == kmNetPtcFileRes::nonexistence) break;

						switch(ret)
						{
						case kmNetPtcFileRes::sndstatewrong  : Sleep( 10); max_cnt = 64; break;
						case kmNetPtcFileRes::preacktimeout  : Sleep(  1); max_cnt =  1; break;
						case kmNetPtcFileRes::preackwrong    : Sleep(100); max_cnt =  3; break;
						case kmNetPtcFileRes::preackrejected : Sleep(500); max_cnt = 32; break;
						}
					}
					// set result
					net->_snd_res  = (kmNetPtcFileRes)ret;
					net->_snd_last = snd;

					// send emptyque if there is no more file to send
					if(net->_snd_que.N() == 0) net->NotifyEmptyQue(snd.src_id);
				}
				Sleep(10);
			}
			print("* end of snd thread\n");
		}, this);
		_snd_thrd.WaitStart();
	};

	// create kal (keep-alive) thread
	void CreateKalThrd()
	{
		_kal_thrd.Begin([](kmNet* net)
		{
			print("* kal thread starts\n");

			// init parameters
			const int exp_sec = 28;

			// init variables
			kmNetIds& ids = net->GetIds();
			kmDate    nks_time(time(NULL));

			// thread loop
			while(1)
			{
				// keep-alive for every valid id
				const int ids_n = (int) net->GetIdsN();

				for(int i = 0; i < ids_n; ++i)
				{
					// check expiration time
					if(ids(i).state == kmNetIdState::valid)
					if(ids(i).time.GetPassSec() > exp_sec)
					{
						const float ec_msec = net->SendSig(i, 500.f);

						//if(ec_msec > 0) print("* [kal thrd] sig to id(%d) : %.1f msec\n", i, ec_msec);
						//else            print("* [kal thrd] disonnected to src_id(%d)\n", i);
					}
					Sleep(100);
				}
				// keep-alive for nks
				if(net->GetPkey().IsValid())
				{
					if(nks_time.GetPassSec() > exp_sec)
					{
						const float ec_msec = net->SendSigToNks();

						nks_time.SetCur();

						//if(ec_msec > 0) print("* [kal thrd] sig to nks : %.1f msec\n", ec_msec);
						//else            print("* [kal thrd] sig to nks : time-out\n");
					}
				}
				Sleep(1000);
			}
		}, this);
	};
	
	///////////////////////////////////////////////
	// callback functions for ptc
		
	// callback function for ptc_cnnt
	static int cbRcvPtcCnntStt(kmNetBase* net, char cmd_id, void* arg)
	{
		return ((kmNet*)net)->cbRcvPtcCnnt(cmd_id, arg);
	};
	int cbRcvPtcCnnt(char cmd_id, void* arg)
	{
		if(cmd_id == 0) // RcvReqCnnt... to get _ptc_cnnt._rcv_src_id
		{
			// init parameters		
			const kmMacAddr mac  = _ptc_cnnt._rcv_des_mac;
			const kmAddr4   addr = _ptc_cnnt._rcv_des_addr;

			// check if already connected
			for(int i = 0; i < _ids.N(); ++i)
			{
				if(_ids(i).mac == mac || _ids(i).addr == addr)
				{
					_ids(i).mac    = mac;
					_ids(i).addr   = addr;
					_ids(i).des_id = _ptc_cnnt._rcv_des_id;
					_ids(i).state  = kmNetIdState::valid;

					int src_id = _ptc_cnnt._rcv_src_id = i; // set src id

					memcpy(_ids(i).name, _ptc_cnnt._rcv_des_name, 64);

					print("** mac(%s) already connected with id(%d)\n", mac.GetStr().P(), src_id);

					// send accept
					_ptc_cnnt.SendAccept();

					// call cb
					vcbRcvPtcCnnt(src_id, cmd_id);

					return (*_netcb)(_parent, _ptc_cnnt._ptc_id, cmd_id, 0);
				}
			}
			// set ids and get new id
			kmNetId id = { mac, _ptc_cnnt._rcv_des_addr, _ptc_cnnt._rcv_des_id, kmNetIdState::valid};
			
			id.time.SetCur();

			int src_id = _ptc_cnnt._rcv_src_id = (ushort)_ids.PushBack(id); // set src_id
			
			memcpy(_ids(src_id).name, _ptc_cnnt._rcv_des_name, 64);

			const kmNetKey& key = _ptc_cnnt._rcv_key;

			if(key.IsVkey()) id.vkey = key;

			// send accept
			_ptc_cnnt.SendAccept();

			// call cb
			vcbRcvPtcCnnt(src_id, cmd_id);

			return (*_netcb)(_parent, _ptc_cnnt._ptc_id, cmd_id, 0);
		}
		else if(cmd_id == 1) // RcvAccept
		{
			if(_ptc_cnnt._snd_des_id < 0xffff - 1) // accpeted
			{
				// update ids
				int src_id = _ptc_cnnt._snd_src_id;

				_ids(src_id).mac    = _ptc_cnnt._snd_des_mac;
				_ids(src_id).des_id = _ptc_cnnt._snd_des_id;
				_ids(src_id).state  = kmNetIdState::valid;

				memcpy(_ids(src_id).name, _ptc_cnnt._snd_des_name, 64);

				// call cb
				vcbRcvPtcCnnt(src_id, cmd_id);

				return (*_netcb)(_parent, _ptc_cnnt._ptc_id, cmd_id, 0);
			}
			else {}// rejected
		}		
		else if(cmd_id == 4) // RcvSig
		{
			int src_id = _ptc_cnnt._rcv_sig_src_id;
			
			if(src_id < _ids.N1())
			{
				_ids(src_id).time.SetCur();
				_ids(src_id).state = kmNetIdState::valid;
				_ids(src_id).addr  = _ptc_cnnt._rcv_sig_addr;
				_ptc_cnnt._rcv_sig_res = 1;
			}
			else _ptc_cnnt._rcv_sig_res = -1;
		}
		return -1;
	};

	// callback function for ptc_data
	static int cbRcvPtcDataStt(kmNetBase* net, char cmd_id, void* arg)
	{
		return ((kmNet*)net)->cbRcvPtcData(cmd_id, arg);
	};
	int cbRcvPtcData(char cmd_id, void* arg)
	{
		if(cmd_id == 0) // RcvData
		{
			kmNetBuf buf(_ptc_data._data, _ptc_data._byte);

			vcbRcvPtcData(_ptc_data._src_id, _ptc_data._data_id, buf);

			return (*_netcb)(_parent, _ptc_data._ptc_id, cmd_id, arg);
		}
		else if(cmd_id == 1) {} // RcvAck
		return -1;
	};

	// callback function for ptc_lrgd
	static int cbRcvPtcLrgdStt(kmNetBase* net, char cmd_id, void* arg)
	{
		return ((kmNet*)net)->cbRcvPtcLrgd(cmd_id, arg);
	};
	int cbRcvPtcLrgd(char cmd_id, void* arg)
	{
		// call netcb
		return (*_netcb)(_parent, _ptc_lrgd._ptc_id, cmd_id, arg);
	};

	// callback function for ptc_file
	//  cmd_id  1: rcv preblk, 2: receiving, 3: rcv done,  4: rcv failure, 5: emtpy equeue
	//         -1: snd preblk,              -3: snd done, -4: snd failure
	static int cbRcvPtcFileStt(kmNetBase* net, char cmd_id, void* arg)
	{
		return ((kmNet*)net)->cbRcvPtcFile(cmd_id, arg);
	};
	int cbRcvPtcFile(char cmd_id, void* arg)
	{
		// get src_id
		ushort src_id = (cmd_id < 0) ? _ptc_file._snd_src_id : _ptc_file._rcv_src_id;

		// call netcb
		if(cmd_id < 0) vcbRcvPtcFile(src_id, cmd_id, _ptc_file._snd_prm);
		else           vcbRcvPtcFile(src_id, cmd_id, _ptc_file._rcv_prm);

		return (*_netcb)(_parent, _ptc_file._ptc_id, cmd_id, arg);
	};

	// callback function for ptc_nkey
	//  cmd_id 0 : reqkey, 1: sndkey, 2: reqaddr, 3: sndaddr
	static int cbRcvPtcNkeyStt(kmNetBase* net, char cmd_id, void* arg)
	{
		return ((kmNet*)net)->cbRcvPtcNkey(cmd_id, arg);
	};
	int cbRcvPtcNkey(char cmd_id, void* arg)
	{
		// call netcb
		vcbRcvPtcNkey(cmd_id);

		return (*_netcb)(_parent, _ptc_nkey._ptc_id, cmd_id, arg);
	};

	///////////////////////////////////////////////
	// virtual functions for rcv callback

	// virtual callback for ptc_cnnt
	virtual void vcbRcvPtcCnnt(ushort src_id, char cmd_id) {};

	// virtual callback for ptc_data
	virtual void vcbRcvPtcData(ushort src_id, uchar data_id, kmNetBuf& buf) {};

	// virtual callback for ptc_file
	//  cmd_id  1: rcv preblk, 2: receiving, 3: rcv done,  4: rcv failure, 5 : empty queue
	//         -1: snd preblk,              -3: snd done, -4: snd failure
	virtual void vcbRcvPtcFile(ushort src_id, char cmd_id, int* prm) {}; 

	// virtual callback for ptc_nkey
	//  cmd_id 0 : rcv reqkey, 1 : rcv key, 2 : rcv reqaddr, 3 : rcv addr
	virtual void vcbRcvPtcNkey(char cmd_id) {};

	///////////////////////////////////////////////
	// interface functions
public:
	// get client ids
	kmNetIds& GetIds() { return _ids; };

	// get client id
	kmNetId& GetId(int idx) { return _ids(idx); };

	// get number of client ids
	int GetIdsN() { return (int)_ids.N(); };

	// find src_id from mac
	//  return : src_id (if finding failed, it will be -1)
	int FindId(const kmMacAddr mac)
	{
		for(int i = 0; i < _ids.N(); ++i) if(_ids(i).mac == mac) return i;

		return -1;
	};

	// set name
	void SetName(const kmStra& name) { _name = name; };

	// get name
	kmStra& GetName() { return _name; };

	// get pkey
	kmNetKey& GetPkey() { return _pkey; };

	// get connectiong src_id.. cmd_id 0 : receiver, cmd_id 1 : sender	
	ushort GetCnntingId(char cmd_id = 0)
	{
		return (cmd_id == 0) ? _ptc_cnnt._rcv_src_id : _ptc_cnnt._snd_src_id;
	};

	// set address for nks server
	void SetNksAddr(kmAddr4 nks_addr)
	{
		_ptc_nkey._nks_addr = nks_addr;
	};
	// get address for nks server
	kmAddr4 GetNksAddr() { return _ptc_nkey._nks_addr; };

	///////////////////////////////////////////////
	// interface functions for communication

	// get addrs with ptc_brdc (UDP broadcasting)
	int GetAddrsInLan(kmAddr4s& addrs, kmMacAddrs& macs, ushort port, float tout_msec = 300.f)
	{
		return _ptc_brdc.GetAddrs(addrs, macs, port, tout_msec);
	};

	// connect to new device with ptc_cnnt... core
	//  return : src_id (if connecting failed, it will be -1)
	int Connect(kmAddr4 addr, const kmStra& name, kmNetKey key, float tout_msec = 300.f)
	{
		// check if there is already id
		for(int i = (int)_ids.N1(); i--;)
		{
			if(_ids(i).addr == addr)
			{
				print("* [kmNet::Connect] already connected to %s\n", addr.GetStr().P());
				return i;
			}
		}
		// get new id
		kmNetId id = { 0, addr, 0, kmNetIdState::connecting};

		const ushort src_id = (ushort)_ids.PushBack(id);

		// connect to addr
		ushort des_id; kmMacAddr des_mac; 
		
		kmT2(des_id, des_mac) = _ptc_cnnt.Connect(src_id, addr, name, key, tout_msec);

		// post processing
		if(des_id < 0xffff - 1) // connecting is successful
		{
			// this processing has been done in cbRcvPtcCnnt(cmd_id = 1)
		}
		else // connecting failed
		{
			// cancel connecting 
			_ids.PopBack()->state = kmNetIdState::none; return -1;
		}
		return (int)src_id;
	};
	// connect to new device with ptc_cnnt
	//  return : src_id (if connecting failed, it will be -1)
	int Connect(kmAddr4 addr, const kmStra& name, float tout_msec = 300.f)
	{
		return Connect(addr, name, kmNetKey(), tout_msec);
	};

	// connect to new device  with vkey
	//  port can be needed if it's lan environment
	//  return : src_id (if connecting failed, it will be -1)
	int Connect(kmNetKey vkey, const kmStra& name, ushort port, float tout_msec = 300.f)
	{
		// check vkey
		if(!vkey.IsVkey()) { print("* [%s] is not vkey\n", vkey.GetStr().P()); return -1; }

		// request addr with key
		kmAddr4 addr; kmMacAddr mac; 
		
		kmT2(addr, mac) = RequestAddrToNks(vkey);

		if(addr.IsInvalid()) { print("* failed to get addr with vkey\n"); return -1; }

		if(addr.IsFlgLan()) // lan
		{
			// get addrs in lan
			kmAddr4s addrs; kmMacAddrs macs;

			int n = GetAddrsInLan(addrs, macs, port);

			// connect
			for(int i = 0; i < n; ++i) if(macs(i) == mac) 
			{
				print("** connect to %s in lan\n", addrs(i).GetStr().P());

				return Connect(addrs(i), name, vkey, tout_msec);
			}
		}
		else // wan
		{
			// send preconnect
			int try_n; kmAddr4 rcv_addr;

			kmT2(try_n, rcv_addr) = SendPreCnnt(addr, 300.f, 6);

			if(try_n == 0)
			{
				print("* sendprecnnt : no answer\n"); return -1;
			}
			print("** sendprecnnt : get answer after %d times from %s\n", try_n, rcv_addr.GetStr().P());

			// connect
			return Connect(addr, name, vkey, tout_msec);
		}
		return -1;
	};

	// connect to new device with pkey and destination's mac
	//  return : src_id (if connecting failed, it will be -1)
	int Connect(kmNetKey pkey, kmMacAddr mac, const kmStra& name, float tout_msec = 300.f)
	{
		// check pkey
		if(!pkey.IsPkey()) { print("* [%s] is not vkey\n", pkey.GetStr().P()); return -1; }

		// request addr with key
		kmAddr4 addr = RequestAddrToNks(pkey, mac);

		if(addr.IsInvalid()) { print("* failed to get addr with pkey\n"); return -1; }

		// send preconnect
		int try_n; kmAddr4 rcv_addr;

		kmT2(try_n, rcv_addr) = SendPreCnnt(addr, 300.f, 6);

		if(try_n == 0)
		{
			print("* sendprecnnt : no answer\n"); return -1;
		}
		print("** sendprecnnt : get answer after %d times from %s\n", try_n, rcv_addr.GetStr().P());

		// connect
		return Connect(addr, name, pkey, tout_msec);
	};

	// reconnect as new addr... core
	//  return : src_id (if connecting failed, it will be -1)
	int Reconnect(ushort src_id, kmAddr4 addr, kmNetKey key, float tout_msec = 300.f)
	{
		// check src_id
		if(src_id >= _ids.N1()) return -1;

		kmNetId& netid = _ids(src_id);

		netid.addr  = addr;
		netid.state = kmNetIdState::connecting;
		
		// connect to addr
		ushort des_id; kmMacAddr des_mac; 

		kmT2(des_id, des_mac) = _ptc_cnnt.Connect(src_id, addr, _name, key, tout_msec);

		// post processing
		if(des_id < 0xffff - 1) // connecting is successful
		{
			// this processing has been done in cbRcvPtcCnnt(cmd_id = 1)
		}
		else // connecting failed
		{
			netid.state = kmNetIdState::invalid; return -1;
		}
		return src_id;
	};
	// reconnect as new addr
	//  return : src_id (if connecting failed, it will be -1)
	int Reconnect(ushort src_id, kmAddr4 addr, float tout_msec = 300.f)
	{
		return Reconnect(src_id, addr, kmNetKey(), tout_msec);
	};

	/// send reqack as pre-connection
	//  return kmT2<try_n, rcv_addr>
	//    try_n    : 0 (timeout), 1 <= (number of attempts to get ack)
	//    rcv_addr : received address (it can be different with addr if it's symmetric nat)
	kmT2<int,kmAddr4> SendPreCnnt(const kmAddr4 addr, float tout_msec = 200.f, int try_cnt = 3)
	{
		int try_n = _ptc_cnnt.SendPreCnnt(addr, tout_msec, try_cnt);

		return  kmT2(try_n, _ptc_cnnt._rcv_preack_addr);
	};

	// send signal to confirm the connection
	//  return : -1 (timeout) or echo time (msec, if received ack)
	float SendSig(ushort src_id, float tout_msec = 200.f, int try_cnt = 3)
	{
		// get id
		kmNetId& id = _ids(src_id);

		// send signal
		for(;try_cnt--; )
		{
			const float ec_msec = _ptc_cnnt.SendSig(src_id, id.des_id, id.addr, tout_msec);

			if     (ec_msec > 0 )  return ec_msec;
			else if(try_cnt  == 0) id.state = kmNetIdState::invalid;
		}
		return -1.f;
	};

	// test network 
	//  return : average time to send byte x snd_n (msec)
	float TestNet(ushort src_id, int byte, int snd_n, int rep_n)
	{
		// get id
		kmNetId& id = _ids(src_id);

		// send signal
		float sum_msec = 0; int cnt = 0;

		for(int i = 0; i < rep_n; ++i)
		{
			float ec_msec = _ptc_cnnt.SendSig(src_id, id.des_id, id.addr, 500, byte, snd_n);

			if(ec_msec > 0) { sum_msec += ec_msec; ++cnt; }
		}
		if(cnt != rep_n) { print("* Test Net : test failed (%d/%d)\n", cnt, rep_n); return -1; }

		const float sum_mb = byte*snd_n*cnt/(1024.f*1024.f);
		const float mps    = sum_mb/sum_msec*1e3f;

		print("* Test Net (%5d byte x %5d x %2d:%2d))", byte, snd_n, cnt, rep_n);
		print(" : %6.1f MB, %6.0f msec, %5.1f MB/s\n", sum_mb, sum_msec, mps);

		return sum_msec/cnt;
	};

	// test network condition
	void TestNetCnd(ushort src_id)
	{
		print("\n** test network condtion : src_id(%d)\n", src_id);

		//const int ts_kb[] = {1,  1,  1,   1, 10, 10, 10,  10};
		//const int ts_n [] = {1, 10, 40, 120,  1, 10, 40, 120};

		const int ts_kb[] = {1,  1, 10, 10};
		const int ts_n [] = {1, 32,  1, 32};

		kmNetCnd& cnd = _ids(src_id).cnd; cnd.ClearTest();

		for(int i = 0; i < numof(ts_kb); ++i)
		{
			const float tec_msec = TestNet(src_id, ts_kb[i]<<10, ts_n[i], 3);

			if(tec_msec < 0)
			{
				print("* TestNetCnd was failed. it was set as the worst condition\n"); 
				cnd.tkb_ms = 1.f;
				cnd.toh_ms = 0.1f;
				cnd.tlt_ms = 500.f;
				cnd.isdone = 1;
				return; 
			}			
			cnd.AddTest(tec_msec, ts_kb[i], ts_n[i]);
		}
		cnd.CalcTest();
		cnd.Print();

		//int snd_max_byte = GetSock().GetSndMaxByte(); print("**** snd max byte : %d\n", snd_max_byte);
	};

	// test network UDP packet size
	void TestNetPckSize(ushort src_id, int start_byte, int end_byte, int del_byte)
	{
		print("\n** test network packet size : src_id(%d)\n\n", src_id);

		// get id
		kmNetId& id = _ids(src_id);

		for(int byte = start_byte; byte <= end_byte; byte += del_byte)
		{
			float ec_msec = _ptc_cnnt.SendSig(src_id, id.des_id, id.addr, 300, byte - 8, 1, true); // 8 byte is header size

			print("* Test UDP packet size : %d byte --> echo %.1f msec\n", byte, ec_msec);
		}
	};

	// send data through ptc_data
	//   tout_msec : 0 (not wait for ack), > 0 (wait for ack)
	// 	
	//   return value : -1 (fail), ack_id (ushort)
	int Send(ushort src_id, uchar data_id, char* data, ushort byte, 
		     float tout_msec = 200.f, int retry_n = 3, ushort ack_id = 0)
	{
		// get id
		kmNetId& id = _ids(src_id);

		// send data
		return _ptc_data.Send(src_id, id.des_id, id.addr, data_id, data, byte, tout_msec, retry_n, ack_id);
	};

	// get data from ptc_data
	//   return : kmT4(data_id, data, byte, src_id)  
	kmT4<uchar, char*, ushort, ushort> GetData()
	{
		return kmT4(_ptc_data._data_id, _ptc_data._data, _ptc_data._byte, _ptc_data._src_id);
	};

	// get data_id only from ptc_data
	uchar GetDataId() {	return _ptc_data._data_id; };

	// send large data through ptc_lrgd
	// 
	//   return : 0 (sending failed), 1 (sending is successful)
	int SendLrgd(ushort src_id, char* data, int64 byte)
	{
		// get id
		kmNetId& id = _ids(src_id);

		// send large data
		return _ptc_lrgd.Send(src_id, id.des_id, id.addr, data, byte);
	};

	// get data from ptc_lrgd
	//   return : kmT2(data, byte)
	kmT2<char*, int64> GetLrgd()
	{
		char* data = _ptc_lrgd._rcv_blk.P();
		int64 byte = _ptc_lrgd._rcv_blk.Byte();
		
		return kmT2(data, byte);
	};

	// get rcv info from ptc_lrgd
	//   return : kmT2(received byte, total byte)
	kmT2<int64, int64> GetLrgdInfoRcv()
	{
		int64 byte = _ptc_lrgd._rcv_blk.Byte();

		return kmT2(_ptc_lrgd._rcv_byte, byte);
	};

	// get snd info from ptc_lrgd
	//   return : kmT2(sent byte, total byte)
	kmT2<int64, int64> GetLrgdInfoSnd()
	{
		int64 byte = _ptc_lrgd._snd_blk.Byte();

		return kmT2(_ptc_lrgd._snd_byte, byte);
	};

	// set file block size (byte)
	//     UDP_MASS_BYTE = 1500 - 28 = 1472
	//
	//     blk_byte = UDP_MSS_BYTE    - sizeof(kmNetHd) - 10; // min size
	//     blk_byte = UDP_MSS_BYTE*4  - sizeof(kmNetHd) - 10; // optimal size
	//     blk_byte = UDP_MSS_BYTE*32 - sizeof(kmNetHd) - 10; // max size
	//     blk_byte = UDP_BLK_BYTE;                           // safe size that doesn't split
	void SetFileBlkByte(uint blk_byte = UDP_BLK_BYTE)
	{
		_ptc_file.SetBlkByte(blk_byte);
	};

	// get file block size
	uint GetFileBlkByte() const { return _ptc_file.GetBlkByte(); };

	// send file through ptc_file
	//   prm : additional parameters (optional)
	//   return : <= 0 (sending failed), 1 (sending is successful)
	//             0 (nonexistence), -1 (snd_state is not 0)
	//            -2 (preack timeout), -3 (preack wrong), -4 (preack reject), -5 (skip max)
	kmNetPtcFileRes SendFile(ushort src_id, kmStra path, int* prm = nullptr)
	{
		kmStra name = path.SplitRvrs('/');

		if(name.N1() == 0) name = path.SplitRvrs('\\');

		return SendFile(src_id, path, name, prm);
	};

	// send file through ptc_file... core
	//   prm : additional parameters (optional)	
	//   return : <= 0 (sending failed), 1 (sending is successful)
	//             0 (nonexistence), -1 (snd_state is not 0)
	//            -2 (preack timeout), -3 (preack wrong), -4 (preack reject), -5 (skip max)
	kmNetPtcFileRes SendFile(ushort src_id, const kmStra& path, const kmStra& name, int* prm = nullptr)
	{	
		// get id
		kmNetId& id = _ids(src_id);

		// set sender's network condition
		_ptc_file.SetSndCnd(id.cnd);

		// send file
		auto res = _ptc_file.Send(src_id, id.des_id, id.addr, path, name, prm);

		// put sender's network condition
		_ptc_file.PutSndCnd(id.cnd);

		return res;
	};

	// send file through ptc_file with seperated thread	
	//   prm : additional parameters (optional)
	void EnqSendFile(ushort src_id, kmStra path, int* prm = nullptr)
	{
		kmStra name = path.SplitRvrs('/');

		if(name.N1() == 0) name = path.SplitRvrs(L'\\');

		EnqSendFile(src_id, path, name, prm);
	};

	// send file through ptc_file with seperated thread	
	//   prm : additional parameters (optional)
	void EnqSendFile(ushort src_id, const kmStra& path, const kmStra& name, int* prm = nullptr)
	{
		_snd_que_lck.Lock();   ////////////////// lock
		_snd_que    .Enqueue(kmNetSndFile(src_id, path, name, prm));
		_snd_que_lck.Unlock(); ////////////////// unlock
	};

	// get num of remained element in snd queue
	int GetSndQueN() { return (int)_snd_que.N1(); };

	// send empty queue
	void NotifyEmptyQue(ushort src_id)
	{
		// get id
		kmNetId& id = _ids(src_id);

		// send noti for empty queue
		_ptc_file.SndEmptyQue(src_id, id.des_id, id.addr);
	};

	// request pkey (svr -> nks)
	//   return : 0 (failed), 1 (successful)
	int RequestPkeyToNks()
	{
		// get key from keysvr
		_pkey = _ptc_nkey.ReqKey(kmNetKeyType::pkey);

		if(_pkey.IsValid() == false)
		{
			print("* failed to get pkey\n"); return 0;
		}
		_pkey.Print();
		return 1;
	};

	// request vkey (svr -> nks)
	//  vld_sec : valid time in sec
	//  vld_cnt : valid count (number of times address can be requested)
	kmNetKey RequestVkeyToNks(kmNetKey pkey, uint vld_sec = 600, uint vld_cnt = 1)
	{
		// check pkey
		if(!pkey.IsValid()) { print("* pkey is not valid\n"); return kmNetKey(); }

		// get key from nks
		return _ptc_nkey.ReqKey(kmNetKeyType::vkey, vld_sec, vld_cnt, pkey);
	};

	// request vkey (clt -> svr)
	//  vld_sec : valid time in sec
	//  vld_cnt : valid count (number of times address can be requested)
	kmNetKey RequestVkey(ushort src_id, uint vld_sec = 600, uint vld_cnt = 1)
	{
		// get id
		kmNetId& id = _ids(src_id);

		// get key from svr
		return _ptc_nkey.ReqVkey(src_id, id.des_id, id.addr, vld_sec, vld_cnt);
	};

	// request address with pkey to nks (clt -> nks)	
	//   return : addr of pkey (if successful) or invalid addr (if it fails)
	kmAddr4 RequestAddrToNks(kmNetKey pkey, kmMacAddr mac)
	{
		return _ptc_nkey.ReqAddr(pkey, mac);
	};

	// request address with vkey to nks (clt -> nks)
	//   
	//   return : addr of vkey (if successful) or invalid addr (if it fails)
	//            mac  of vkey (if it is lan)
	kmT2<kmAddr4, kmMacAddr> RequestAddrToNks(kmNetKey vkey)
	{
		kmAddr4 addr = _ptc_nkey.ReqAddr(vkey, kmMacAddr());

		return kmT2(addr, _ptc_nkey._rcv_mac);
	};

	// send key
	void SendKey(ushort src_id, kmNetKey key)
	{
		// get id
		kmNetId& id = _ids(src_id);

		_ptc_nkey.SendKey(src_id, id.des_id, id.addr, key);
	};

	// send key from nks... nsk only
	void SendKeyFromNks(kmAddr4 addr, kmNetKey key)
	{
		_ptc_nkey.SendKeyFromNks(addr, key);
	};

	// send sig to nks (svr -> nks)
	//  return : -1 (timeout) or echo time (msec, if received ack)
	float SendSigToNks()
	{
		return _ptc_nkey.SendSig(_pkey, _mac);
	};

	// reply nks signal (nks -> svr)
	//   flg 0 : addr was not changed, 1: changed
	void ReplySigFromNks(kmAddr4 addr, int flg)
	{
		_ptc_nkey.ReplySig(addr, flg); 
	};

	// print ids
	void PrintIds()
	{
		const int n = GetIdsN();

		print("* kmNet::_ids.N1() : %d\n", n);

		for(int i = 0; i < n; ++i)
		{
			print("* ids(%d)\n", i); _ids(i).Print();
		}
	};
};

#endif /* __km7Net_H_INCLUDED_2021_05_31__ */
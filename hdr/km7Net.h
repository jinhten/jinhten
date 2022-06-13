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
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <chrono>
#include <thread>
#include <wchar.h>
#include <errno.h>

///////////////////////////////////////////////////////////////
// enum and definition for net

// * Note that TCP's MTU is from ethernet v2's limit, 1500byte.
#define TCP_MTU_BYTE       1500
#define TCP_MSS_BYTE       1460  // 1500 - 20 (ip header) - 20 (tcp header)

// * Note that UDP's MTU and MSS are only for efficient transmit.
#define UDP_MTU_BYTE       1500
#define UDP_MSS_BYTE       1472  // 1500 - 20 (ip header) - 8 (udp header) 
//#define UDP_MSS_BYTE       34000 - 28 // 1500 - 20 (ip header) - 8 (udp header)

#define UDP_MAX_PKT_BYTE   65535
#define UDP_MAX_DATA_BYTE  65507 // 65525 - 20 (ip header) - 8 (udp header)

//#define DEFAULT_PORT 60215
#define DEFAULT_PORT 60165
#define RTC_PORT 60165

#define INVALID_SOCKET -1

#define ID_NAME 64

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
inline uint64 ntoh(uint64 a) { return (((uint64)ntohl(a)) << 32) + ntohl(a >> 32); };

inline char   ntoh(char   a) { return        a ; };
inline short  ntoh(short  a) { return ntohs (a); };
inline int    ntoh(int    a) { return ntohl (a); };
inline int64  ntoh(int64  a) { return (((int64)ntohl(a)) << 32) + ntohl(a >> 32); };

//inline float  ntoh(float  a) { return ntohf(*((uint*  )&a)); };
//inline double ntoh(double a) { return ntohd(*((uint64*)&a)); };

inline uchar  hton(uchar  a) { return        a ; };
inline wchar  hton(wchar  a) { return htons (a); };
inline ushort hton(ushort a) { return htons (a); };
inline uint   hton(uint   a) { return htonl (a); };
inline uint64 hton(uint64 a) { return (((uint64)htonl(a)) << 32) + htonl(a >> 32); };

inline char   hton(char   a) { return        a ; };
inline short  hton(short  a) { return htons (a); };
inline int    hton(int    a) { return htonl (a); };
inline int64  hton(int64 a) { return (((int64)htonl(a)) << 32) + htonl(a >> 32); };

//inline float  hton(float  a) { uint   b = htonf(a); return *((float* )&b); };
//inline double hton(double a) { uint64 b = htond(a); return *((double*)&b); };

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

    // set from char[8] or uchar[8]
    void Set(uchar* addr) { for(int i = 0; i < 8; ++i) c[i] = *(addr + i); };

    // assignment operator
             kmMacAddr& operator=(const          kmMacAddr& a)          { i64 = a.i64; return *this; };    
    volatile kmMacAddr& operator=(const volatile kmMacAddr& a) volatile { i64 = a.i64; return *this; };

    // operator
    bool operator==(const kmMacAddr& b) const { return i64 == b.i64; };
    bool operator!=(const kmMacAddr& b) const { return i64 != b.i64; };

    // get string
    kmStra GetStr() const
    {
        return kmStra("%02X-%02X-%02X-%02X-%02X-%02X-%02X-%02X", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
    };
    kmStrw GetStrw() const
    {
        return kmStrw(L"%02X-%02X-%02X-%02X-%02X-%02X-%02X-%02X", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
    };
};
typedef kmMat1<kmMacAddr> kmMacAddrs;

// ipv4 address's state... valid, invalid, pending
enum class kmAddr4State : ushort { valid = 0, invalid = 1, pending = 2 };

// ipv4 address... compatible with sockaddr_in (4 + 2 + 2 byte)
class kmAddr4
{
public:
    union { uint ip; uchar c[4];}; // compatible with sockaddr_in.sin_addr.s_addr
    ushort port;                   // compatible with sockaddr_in.sin_port
    kmAddr4State state{};          // optional * Note that it's 8 byte even without this

    // constructor
    kmAddr4()                        { ip = 0;                     port = 0;};
    kmAddr4(int     a, ushort p = 0) { ip = a;                     port = htons(p);};
    kmAddr4(uint    a, ushort p = 0) { ip = a;                     port = htons(p);};
    kmAddr4(ulong   a, ushort p = 0) { ip = a;                     port = htons(p);};
    kmAddr4(in_addr a, ushort p = 0) { ip = a.s_addr;              port = htons(p);};    
    kmAddr4(LPCSTR  s, ushort p = 0) { inet_pton(AF_INET, s, &ip); port = htons(p);};
    //kmAddr4(LPCWSTR s, ushort p = 0) { InetPtonW(AF_INET, s, &ip); port = htons(p);};
    kmAddr4(sockaddr_in saddr)
    {
        ip = saddr.sin_addr.s_addr; port = saddr.sin_port;
    };
    kmAddr4(uchar a0, uchar a1, uchar a2, uchar a3, ushort p = DEFAULT_PORT)
    {
        c[0] = a0; c[1] = a1; c[2] = a2; c[3] = a3; port = htons(p);
    };    

    // operator
    uchar operator()(int i) { return c[i]; };

    bool operator==(const kmAddr4& b) { return (ip == b.ip && port == b.port); };

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
};
typedef kmMat1<kmAddr4> kmAddr4s;

///////////////////////////////////////////////////////////////
// socket class with winsock2 and tcp/ip
//
// * Note that you should not define a destructor which will close the socket.
//
class kmSock
{
public:
    int    _sck   = INVALID_SOCKET;
    int    _state = 0; // 0 : free, 1 : socket, 2 : bind, 3: listen, 4: connected

    void Init() { _sck = INVALID_SOCKET; _state = 0; };

    /////////////////////////////////
    // static functions

    static bool GetIntfAddr(kmAddr4& ipAddr)
    {
        char buf[8192] = {0,};
        struct ifconf ifc = {0,};
        struct ifreq *ifr = NULL;
        int sck = 0;
        char ip[INET6_ADDRSTRLEN] = {0,};
        struct sockaddr_in *addr;

        /* Get a socket handle. */
        sck = socket(PF_INET, SOCK_DGRAM, 0);
        if(sck < 0)
        {
            perror("socket");
            return false;
        }

        /* Query available interfaces. */
        ifc.ifc_len = sizeof(buf);
        ifc.ifc_buf = buf;
        if(ioctl(sck, SIOCGIFCONF, &ifc) < 0)
        {
            perror("ioctl(SIOCGIFCONF)");
            return false;
        }

        /* Iterate through the list of interfaces. */
        ifr = ifc.ifc_req;
        addr = (struct sockaddr_in*)&(ifr[1].ifr_addr);

        /* Get the IP address*/
        if(ioctl(sck, SIOCGIFADDR, &ifr[1]) < 0)
        {
            perror("ioctl(OSIOCGIFADDR)");
        }

        if (inet_ntop(AF_INET, &(((struct sockaddr_in *)addr)->sin_addr), ip, sizeof ip) == NULL) //vracia adresu interf
        {
            perror("inet_ntop");
            return false;
        }

        // Get the MAC address
        if(ioctl(sck, SIOCGIFHWADDR, &ifr[1]) < 0) {
            perror("ioctl(SIOCGIFHWADDR)");
            return false;
        }

        // display result
        kmAddr4 tempIpAddr(ip, DEFAULT_PORT);
        ipAddr = tempIpAddr;

        close(sck);
        return true;
    };

/* TODO
    // get own local ip address
    static kmAddr4 GetLocalAddr(int idx = 0, ushort port = 0, int display_on = 0)
    {
        // get addr table
        MIB_IPADDRTABLE tbl[16]; ulong size = sizeof(tbl);
        
        if(GetIpAddrTable(tbl, &size, 0) != 0) return 0;

        // get address
        const int n = tbl->dwNumEntries;

        if(display_on > 0)
        {
            PRINTFA("* GetLocalAddr()\n");
            for(int i = 0; i < n; ++i)
            {
                PRINTFA("  [%d] ip :%s\n", i, kmAddr4(tbl->table[i].dwAddr).GetIpStr().P());
            }
        }
        return (idx >= n)? 0 : kmAddr4(tbl->table[idx].dwAddr, port);
    };

    // get udp broadcasting address
    static kmAddr4 GetBrdcAddr(ushort port = 0)
    {
        kmAddr4 addr = kmSock::GetLocalAddr(0, port); addr.c[3] = 255;

        return addr;
    };
*/

    // get WSA error
    static kmStra GetErrStr(int err_code = 0)
    {
        const char* str = nullptr;
        const int   err = err_code;//(err_code == 0) ? WSAGetLastError():err_code;

        switch(err)
        {
        case EISCONN            : str = "the socket is not connected"; break;
        case EPIPE              : str = "the connection has been broken"; break;
/*
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
*/
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
        ASSERTFA(_state == 0, "kmSock::GetSocket in 172");

        if(type == kmSockType::tcp) _sck = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        else                        _sck = socket(AF_INET, SOCK_DGRAM , IPPROTO_UDP);

        if(_sck != INVALID_SOCKET) _state = 1;

        ASSERTFA(_sck != INVALID_SOCKET, "kmSock::GetSocket in 178");
    };

    // set socket option
    void SetSckOptBroadcast(int on = true)
    {
        int ret = setsockopt(_sck, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on));

        ASSERTFA(ret == 0, "kmSock::SetSckOpt in 320");
    };

    // get socket type
    kmSockType GetSckType() const
    {
        int type; socklen_t len = sizeof(int);

        getsockopt(_sck, SOL_SOCKET, SO_TYPE, (char*)&type, &len);

        return (type == SOCK_STREAM) ? kmSockType::tcp : kmSockType::udp;
    };
    
    // bind
    //  [return] 0                : if no error occurs
    //           SOCKET_ERROR(-1) : otherwise
    int Bind(kmAddr4 addr, kmSockType type = kmSockType::tcp)
    {
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
        kmSock client; sockaddr_in saddr; socklen_t len = sizeof(saddr);
    
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
    int Recvfrom(kmStra& str, kmAddr4& addr)
    {
        sockaddr_in saddr; socklen_t len = sizeof(saddr);

        int ret = ::recvfrom(_sck, str.P(), (int) str.Size(), 0, (sockaddr*)&saddr, &len);

        addr = kmAddr4(saddr);

        return ret;
    };

    // recive from (for udp)
    template<typename T> int Recvfrom(kmMat1<T>& data, kmAddr4& addr)
    {
        sockaddr_in saddr; socklen_t len = sizeof(saddr);

        int ret = ::recvfrom(_sck, data.P(), (int) data.Byte(), 0, (sockaddr*)&saddr, &len);

        data.SetN1(MAX(0, ret));

        addr = kmAddr4(saddr);

        return ret;
    }

    // send to (for udp)
    int Sendto(const kmStra& str, kmAddr4 addr)
    {
        sockaddr_in saddr = addr.GetSckAddr();

        return ::sendto(_sck, str.P(), (int)str.GetLen(), 0, (sockaddr*)&saddr, sizeof(saddr));
    };

    // send to (for udp)
    template<typename T> int Sendto(const kmMat1<T>& data, kmAddr4 addr)
    {
        sockaddr_in saddr = addr.GetSckAddr();

        return ::sendto(_sck, data.P(), (int)data.N1()*sizeof(T), 0, (sockaddr*)&saddr, sizeof(saddr));
    }

    // send to (for udp broadcast) using global broadcast address (255.255.255.255)
    template<typename T> int SendtoBroadcast(const kmMat1<T>& data, ushort port = DEFAULT_PORT)
    {
        sockaddr_in saddr;
        saddr.sin_family      = AF_INET;
        saddr.sin_addr.s_addr = 0xffffffff; 
        saddr.sin_port        = htons(port);

        return ::sendto(_sck, data.P(), (int)data.N1()*sizeof(T), 0, (sockaddr*)&saddr, sizeof(saddr));
    };

/*
    // send to (for udp broadcast) using local broadcast address (ex. 10.114.75.255)
    template<typename T> int SendtoBroadcastLocal(const kmMat1<T>& data, ushort port = DEFAULT_PORT)
    {
        kmAddr4      addr = kmSock::GetBrdcAddr(DEFAULT_PORT);
        sockaddr_in saddr = addr.GetSckAddr();

        print("* broadcasting to %s\n", addr.GetStr().P());

        return ::sendto(_sck, data.P(), (int)data.N1()*sizeof(T), 0, (sockaddr*)&saddr, sizeof(saddr));
    };
*/

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

        //const int ret = ::closesocket(_sck); Init();
        const int ret = ::close(_sck); Init();

        if(ret != 0) PRINTFA("** [close] error occurs : %s\n", GetErrStr().P());
        
        return ret;
    };

    // get state
    //   0 : free, 1 : socket, 2 : bind, 3: listen, 4: connected
    int GetState() const { return _state; };

    // get source address
    kmAddr4 GetSrcAddr() const 
    {
        sockaddr_in saddr; socklen_t len = sizeof(saddr);

        getsockname(_sck, (sockaddr*)&saddr, &len);

        return kmAddr4(saddr);
    };

    // get destination address
    kmAddr4 GetDstAddr() const
    {
        sockaddr_in saddr; socklen_t len = sizeof(saddr);

        getpeername(_sck, (sockaddr*)&saddr, &len);

        return kmAddr4(saddr);
    };

    // print info of socket
    void PrintInfo(LPCSTR str = nullptr) const
    {
        if(str != nullptr) PRINTFA("%s\n", str);
        else               PRINTFA("[socket info]\n");
        PRINTFA("  handle      : %p\n", &_sck);
        PRINTFA("  state       : %d\n"  , _state); if(_state == 0) return;
        PRINTFA("  type        : %s\n"  , (GetSckType() == kmSockType::tcp) ? "TCP":"UDP");
        PRINTFA("  src address : %s\n"  , GetSrcAddr().GetStr().P());
        PRINTFA("  dst address : %s\n"  , GetDstAddr().GetStr().P());
    };
};

//////////////////////////////////////////////////////////////////////
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
/*
    int Bind(ushort port = DEFAULT_PORT)
    {
        return _sck.Bind(kmSock::GetLocalAddr(0, port), kmSockType::udp);
    };
*/

    // connect    
    int Connect(kmAddr4 addr = INADDR_ANY)
    {
        if(addr.port == 0) addr.port = htons(DEFAULT_PORT);

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
    
        if(n > 0) PRINTFA("-> receive from (%s) : %s\n", addr.GetStr().P(), str.P());

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
            PRINTFA("* receiving\n");
            int ret;

            while(1) { if((ret = net->RcvProc()) < 1) break; }

            // check error
            if(ret < 0) PRINTFA("** %s\n", kmSock::GetErrStr().P());

            // close socket
            net->Close();

            PRINTFA("* end of receiving\n");
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
        if(str != nullptr) PRINTFA("%s", str);
        PRINTFA("=====================================UDP\n");

        _sck.PrintInfo();

        PRINTFA("=========================================\n");
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
public:
    double  _tout_sec = 2;
    kmTimer _timer;

    inline void Set(double tout_sec) { _tout_sec = tout_sec; };

    inline void On()  { _timer.Start(); };
    inline void Off() { _timer.Stop (); };

    inline bool IsOut() { return _timer.IsStarted() && (_timer.sec() > _tout_sec); };
    inline bool IsOff() { return _timer.IsNotStarted(); };
    inline bool IsOn () { return _timer.IsStarted(); };
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
    kmStrw    _name{};     // net user's name
    kmLock    _lck;        // mutex for _snd_buf
    kmNetTom  _tom;        // rcv timeout monitoring

    int Bind()  { return _sck.Bind(_addr, kmSockType::udp); };
    int Close() { return _sck.Close();  };

    int Recvfrom(      kmAddr4& addr) { return _sck.Recvfrom(_rcv_buf, addr); };
    int Sendto  (const kmAddr4& addr)
    {
        const int ret = _sck.Sendto(_snd_buf, addr); 
        UnlockSnd();  return ret; ///////////////////////////// unlock
    };
    int SendtoBroadcast()
    {
        _sck.SetSckOptBroadcast(1);
        
        const int ret = _sck.SendtoBroadcast(_snd_buf);
        //const int ret = _sck.SendtoBroadcastLocal(_snd_buf);

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

    //void convWC4to2(wchar* wc, char* c, const ushort& n) { for (ushort i = 0; i < n; ++i) { c[i*2+1] = (char)wc[i]; } }
    void convWC4to2(wchar* wc, ushort* c, const ushort& n) { for (ushort i = 0; i < n; ++i) { c[i] = (ushort)wc[i]; } }
    void convWC2to4(ushort* c, wchar* wc, const ushort& n) { for (ushort i = 0; i < n; ++i) { wc[i] = (wchar)c[i]; } }
};

// net key type enum class... pkey, vkey, tkey
enum class kmNetKeyType : uchar { invalid = 0, pkey = 1, vkey = 2, tkey = 3 };

// net key class... 8byte
class kmNetKey
{
protected: // * Note that key is saved with big-endian for network transfer
           // * so, you should not directly access this members
    kmNetKeyType _type{};  // key type
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

    kmNetKeyType GetType() const { return _type;       };
    uchar        GetSvri() const { return _svri;       };
    ushort       GetIdx0() const { return ntoh(_idx0); };
    uint         GetIdx1() const { return ntoh(_idx1); };
    ushort       GetPswd() const { return ntoh(_pswd); };

    bool IsValid() const { return _type != kmNetKeyType::invalid; };

    const char* GetTypeStr()
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
    kmStra GetStr()
    {
        return kmStra("[%s] %d %d %d %d", 
                      GetTypeStr(), GetIdx0(), GetIdx1(), GetPswd(), GetSvri());
    };
    void Print() { print("* key : %s\n", GetStr().P()); };
};

// net key element class for nks server... 32 byte
class kmNetKeyElm
{
public:
    kmNetKey      key {};
    kmMacAddr     mac {};
    kmDate        date{};
    kmAddr4       addr{};

    // constructor
    kmNetKeyElm() {};
    kmNetKeyElm(kmAddr4State state) { addr.state = state; };
    kmNetKeyElm(kmNetKey key, kmMacAddr mac, kmDate date, kmAddr4 addr) :
        key(key), mac(mac), date(date), addr(addr) {};

    // member functions
    bool IsValid  () { return addr.state == kmAddr4State::valid;   };
    bool IsPending() { return addr.state == kmAddr4State::pending; };
    bool IsInvalid() { return addr.state == kmAddr4State::invalid; };

    kmStra GetStr()
    {
        return kmStra("%s %s %s %s", key .GetStr().P(), mac .GetStr().P(), 
                                     date.GetStrPt().P(), addr.GetStr().P()); 
    };
};
typedef kmMat1<kmNetKeyElm> kmNetKeyElms;

// net key signaling function class nks
class kmNetNks
{
    kmMat1<kmNetKeyElms> _tbl; // (idx0)(idx1)    

public:
    // create table
    void Create(int idx0_n = 32)
    {
        _tbl.Create(idx0_n);

        for(int i = 0; i < idx0_n; ++i) _tbl(i).Create(0, 32);
    };

    // register new one as pkey
    kmNetKey Register(kmMacAddr mac, kmAddr4 addr)
    {
        // find empty key... idx0, idx1
        ushort idx0 = kmrand(0, (int)_tbl.N1() - 1), idx1 = 0;

        kmNetKeyElms& elms  = _tbl(idx0);
        const int     elm_n = (int)elms.N1();

        for(; idx1 < elm_n; ++idx1) if(elms(idx1).IsInvalid()) break;

        // set key element
        kmNetKey key(kmNetKeyType::pkey, idx0, idx1, kmrand(0u, 65535u));
        kmDate   date(time(NULL));

        // add key element to table
        if(idx1 < elm_n) elms(idx1) =  kmNetKeyElm(key, mac, date, addr);
        else             elms.PushBack(kmNetKeyElm(key, mac, date, addr));

        return key;
    };

    // find net key element with key
    kmNetKeyElm Find(kmNetKey key, kmMacAddr mac)
    {    
        const auto idx0 = key.GetIdx0();
        const auto idx1 = key.GetIdx1();

        // check index range
        if(idx0 >=  _tbl.N1() || idx1 >= _tbl(idx0).N1())
        {
            return kmAddr4State::invalid;
        }
        // check pswd
        if(_tbl(idx0)(idx1).key.GetPswd() != key.GetPswd())
        {
            print("** password is wrong\n"); return kmAddr4State::invalid;
        }
        // check mac
        if(_tbl(idx0)(idx1).mac != mac)
        {
            print("** mac is wrong\n"); return kmAddr4State::invalid;
        }
        return _tbl(idx0)(idx1);
    };

    // save key table
    void Save(const wchar* path)
    {
        kmFile file(path, KF_NEW);

        file.WriteMat(&_tbl);
    };

    // load key table
    int Load(const wchar* path) try
    {
        kmFile file(path);

        file.ReadMat(&_tbl);

        return 1;
    }
    catch(kmException) { return 0; };

    // print every key
    void Print()
    {
        print("* idx0_n : %lld \n", _tbl.N1());
        for(int64 i = 0; i < _tbl.N1(); ++i)
        {
            const kmNetKeyElms& elms = _tbl(i);

            for(int64 j = 0; j < elms.N1(); ++j)
            {
                print("   (%lld, %lld) : %s\n", i, j, elms(j).GetStr().P());
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
    kmAddr4      _rcv_addr;

    kmNetKey     _snd_key;
    kmAddr4      _snd_addr;

    // receiving procedure
    virtual void RcvProc(char cmd_id, const kmAddr4& addr)
    {    
        switch(cmd_id)
        {
        case 0: RcvReqKey (addr); break;
        case 1: RcvKey    (addr); break;
        case 2: RcvReqAddr(addr); break;
        case 3: RcvAddr   (addr); break;
        }
    };
    ///////////////////////////////////////
    // interface functions

    // request pkey to key server
    kmNetKey ReqKey(kmNetKeyType keytype, float tout_sec = 1.f)
    {
        // send request to key server
        SndReqKey(keytype);

        // wait for timeout
        kmTimer time(1); _rcv_key_flg = 0;

        while(time.GetTime_sec() < tout_sec && _rcv_key_flg == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));

        if(_rcv_key_flg == 0)
        {
            print("** kmNetPtcNkey::ReqKey - timeout\n"); 
            
            return kmNetKey();
        }        
        return _rcv_key;
    };

    // request addr to key server by key
    kmAddr4 ReqAddr(kmNetKey key, kmMacAddr mac, float tout_sec = 1.f)
    {
        // send request to nks server
        SndReqAddr(key, mac);

        // wait for timeout
        kmTimer time(1); _rcv_addr_flg = 0;

        while(time.GetTime_sec() < tout_sec && _rcv_addr_flg == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));;

        if(_rcv_addr_flg == 0)
        {
            print("** kmNetPtcNkey::ReqAddr - timeout\n"); 
            
            return kmAddr4();
        }
        return _rcv_addr;
    };

    ////////////////////////////////////////
    // cmd 0 : request key
    void SndReqKey(kmNetKeyType keytype)
    {
        kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 0, (uchar)keytype, 0};

        *_net << hd << _net->_mac;

        _net->Sendto(_nks_addr);
    };
    void RcvReqKey(const kmAddr4& addr) // nks server
    {
        kmNetHd hd{};

        *_net >> hd >> _rcv_mac;

        _rcv_keytype = (kmNetKeyType)hd.opt;
        _rcv_addr    = addr;

        // call cb to register mac and send key (_rcv_addr, _rcv_mac -> _snd_key)
        (*_ptc_cb)(_net, 0, 0);

        SndKey(addr);
    };

    ////////////////////////////////////////
    // cmd 1 : send key
    void SndKey(const kmAddr4& addr) // nks server
    {    
        kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 1, 0, 0};

        *_net << hd << _snd_key;
    
        _net->Sendto(addr);
    };
    void RcvKey(const kmAddr4& addr)
    {    
        kmNetHd hd{};

        *_net >> hd >> _rcv_key;

        _rcv_key_flg = 1;
    };

    ////////////////////////////////////////
    // cmd 2 : request addr by key to nks server
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

        // call cb to get addr from key and send addr (_rcv_key, _rcv_mac -> _snd_addr)
        (*_ptc_cb)(_net, 2, 0);

        SndAddr(addr);
    };

    ////////////////////////////////////////
    // cmd 3 : send addr
    void SndAddr(const kmAddr4& addr) // nks server
    {    
        kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 3, 0, 0};

        *_net << hd << _snd_addr;

        _net->Sendto(addr);
    };
    void RcvAddr(const kmAddr4& addr)
    {    
        kmNetHd hd{};

        *_net >> hd >> _rcv_addr;

        _rcv_addr_flg = 1;
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
    //
    //   return : number of addrs
    int GetAddrs(kmAddr4s& addrs, kmMacAddrs& macs, float tout_msec = 100.f)
    {
        // init output parameters
        _addrs = &addrs; _addrs->Recreate(0, 16);
        _macs  = &macs;  _macs ->Recreate(0, 16);
        _wait  = 1;

        // send UDP broadcasting
        SndReqAck();

        // wait for timeout
        kmTimer time(1);

        while(time.GetTime_msec() < tout_msec) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        _wait = 0;

        return (int)_addrs->N();
    };

    ////////////////////////////////////////
    // cmd 0 : request ack
    void SndReqAck()
    {
        // set buffer
        kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 0, 0, 0};

        *_net << hd << _net->_mac;

        // send as broadcast mode
        _net->SendtoBroadcast();
    };
    void RcvReqAck(const kmAddr4& addr)
    {
        // get from buffer
        kmNetHd hd{}; kmMacAddr mac;

        *_net >> hd >> mac;

        if(mac == _net->_mac) return; // to prevent self-ack.

        // send ack
        SndAck(addr);
    };

    ///////////////////////////////////////////////
    // cmd 1 : ack
    void SndAck(const kmAddr4& addr)
    {
        // set buffer
        kmNetHd hd = { 0xffff, 0xffff, _ptc_id, 1, 0, 0};

        *_net << hd << _net->_mac;

        // send buffer
        _net->Sendto(addr);
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
    wchar     _rcv_des_name[64];

    ushort    _snd_src_id;
    ushort    _snd_des_id;
    kmMacAddr _snd_des_mac;
    wchar     _snd_des_name[64];

    // receiving procedure
    virtual void RcvProc(char cmd_id, const kmAddr4& addr)
    {
        switch(cmd_id)
        {
        case 0: RcvReqConnect(addr); break;
        case 1: RcvAccept    (addr); break;
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
    kmT2<ushort, kmMacAddr> Connect(ushort src_id, const kmAddr4& addr, const kmStrw& name, float tout_msec = 100.f)
    {
        // send reqconnect
        SndReqConnect(src_id, addr, name);

        // wait for timeout_usec
        kmTimer time(1);

        while(time.GetTime_msec() < tout_msec && _snd_des_id == 0xffff) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); };

        if     (_snd_des_id == 0xffff  ) print("** kmNetPtcCnnt::Connect : timeout\n");
        else if(_snd_des_id == 0xffff-1) print("** kmNetPtcCnnt::Connect : reject from %s\n", addr.GetStr().P());

        return kmT2(_snd_des_id, _snd_des_mac);
    };    
    
    ///////////////////////////////////////
    // cmd 0 : request connect
    void SndReqConnect(ushort src_id, const kmAddr4& addr, const kmStrw& name)
    {
        // set snd_buf
        kmNetHd hd = { _snd_src_id = src_id, _snd_des_id = 0xffff, _ptc_id, 0, 0, 0};

        ushort cname[64] = {0,};
        convWC4to2(name.P(), cname, name.N());
        (*_net << hd << _net->_mac).PutData(cname, (ushort)MIN(64,name.N()));

        // send snd_buf
        _net->Sendto(addr);
    };
    void RcvReqConnect(const kmAddr4& addr)
    {
        // get from rcv_buf
        kmNetHd hd; ushort name_n;

        ushort name[64] = {0,};
        (*_net >> hd >> _rcv_des_mac).GetData(name, name_n);
        convWC2to4(name, _rcv_des_name, name_n);

        _rcv_des_id   = hd.src_id;
        _rcv_des_addr = addr;

        // call cb function.... get _src_id (will be 0xffff-1, if not accepted)
        (*_ptc_cb)(_net, 0, 0);

        // send accept
        SndAccept(_rcv_src_id, _rcv_des_id, addr, _net->_name);
    };

    ////////////////////////////////////////////
    // cmd 1 : accept
    //
    //  if src_id == oxffff-1, it means recjecting the connection.
    void SndAccept(ushort src_id, ushort des_id, const kmAddr4& addr, const kmStrw& name)
    {
        // set snd_buf
        kmNetHd hd = { src_id, des_id, _ptc_id, 1, 0, 0};

        (*_net << hd << _net->_mac).PutData(name.P(), (ushort)MIN(64,name.N()));

        // send snd_buf
        _net->Sendto(addr);

        // call cb function.. cmd_id = -1
        (*_ptc_cb)(_net, -1, 0);
    };
    void RcvAccept(const kmAddr4& addr)
    {
        // get from rcv_buf
        kmNetHd hd; ushort name_n;

        ushort name[64] = {0,};
        (*_net >> hd >> _snd_des_mac).GetData(name, name_n);
        convWC2to4(name, _snd_des_name, name_n);

        _snd_des_id = hd.src_id;

        // call cb function.. cmd_id = 1
        (*_ptc_cb)(_net, 1, 0);
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
    //   return : 0 (not ack), 1 (received ack)
    int Send(ushort src_id, ushort des_id, const kmAddr4& addr, uchar data_id, char* data, int byte, float tout_msec = 0.f)
    {    
        // send data
        kmNetHdFlg flg = 0; flg.reqack = (tout_msec == 0) ? 0:1;

        _ack = 0;

        SndData(src_id, des_id, addr, data_id, data, byte, flg);

        //const kmNetBuf& buf = _net->_snd_buf;

        if(tout_msec == 0) return 0;

        // wait for timeout
        kmTimer time(1);

        while(time.GetTime_msec() < tout_msec && _ack < 1) { std::this_thread::sleep_for(std::chrono::milliseconds(0)); };

        return _ack;
    };

    ///////////////////////////////////////
    // cmd 0 : data
    void SndData(ushort src_id, ushort des_id, const kmAddr4& addr, uchar data_id, char* data, int byte, kmNetHdFlg flg)
    {
        // set ack id
        struct timespec tp;
        //clock_gettime(clock_monotonic, &tp);
        clock_gettime(CLOCK_MONOTONIC, &tp);
        _ack_id = (tp.tv_sec*1000ull) + (tp.tv_nsec/1000ull/1000ull);
        //_ack_id = (ushort)GetTickCount64();

        // set snd_buf
        kmNetHd hd = { src_id, des_id, _ptc_id, 0, data_id, flg};

        (*_net << hd << _ack_id).PutData(data, byte);

        // send snd_buf
        _net->Sendto(addr);
    };
    void RcvData(const kmAddr4& addr)
    {
        // get from rcv_buf
        kmNetHd hd; ushort  ack_id;

        *_net >> hd >> ack_id >> _byte; _data = _net->_rcv_buf.End1();

        _des_id  = hd.src_id;
        _src_id  = hd.des_id;
        _data_id = hd.opt;

        // send ack
        // * Note that it's good to be in front of the callback.
        // * since the callback may take a long time.
        if(hd.IsReqAck()) SndAck(ack_id, addr);

        { // KKT
        kmStra ip(addr.GetStr());
        print("\n* received data\n");
        print("** dst id : %02X, src id : %02X, data id : %02X\n", _des_id, _src_id, _data_id);
        print("** ip addr : %s\n\n", ip.P());
        }

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
        const float preack_tout_msec = 500.f; // time out for preack

        kmTimer timer(1);
        while(_snd_state == 1 && timer.msec() < preack_tout_msec) std::this_thread::sleep_for(std::chrono::milliseconds(0));

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
            for(timer.Start(); timer.GetTime_usec() < out_t_usec && _snd_state == 3; std::this_thread::sleep_for(std::chrono::milliseconds(0))) {}

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

        { // KKT
        kmStra ip(addr.GetStr());
        print("\n* received large file\n");
        print("** block num : %u, block byte : %u\n", blk_n, blk_byte);
        print("** ip addr : %s\n\n", ip.P());
        }

        // create time out monitoring thread
        _rcv_thrd.Begin([](kmNetPtcLrgd* ptc)
        {
            const float tout_msec = 1e3f;
            kmTimer timer(1);
        
            while(timer.msec() < tout_msec)
            {
                if(ptc->_rcv_state != 1) return;
        
                if(ptc->_rcv_flg == 1) { ptc->_rcv_flg = 0; timer.Start(); }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
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
    kmStrw file_name{};    // file name only
    kmStrw file_path{};    // file name including full path

    // init
    void Init(const kmStrw& file_path, const kmStrw& file_name, int64 total_byte)
    {
        byte = 0; t_usec = 0; reject = 0; loss = 0;

        this->total_byte = total_byte;
        this->file_path  = file_path;
        this->file_name  = file_name;
    };

    // set path in cbRcvNetPtcFile() or vcbRcvPtcFile() at cmd_id == 1
    void SetPath(const kmStrw& path)
    {
        file_path.SetStr(L"%S/%S.ing", path.P(), file_name.P());
    };

    // reject receiving in cbRcvNetPtcFile() or vcbRcvPtcFile() at cmd_id == 1
    void Reject() { reject = 1; };
};

// basic protocol to send a file
class kmNetPtcFile: public kmNetPtc
{
public:
    // receiver's members    
    kmFileBlk _rcv_blk;        // receiving file block buffer
    kmMat1bit _rcv_bsf;        // receiving block state flag (0 : not yet, 1: received)
    uint      _rcv_state  = 0; // 0 : not begin, 1: waiting blocks, 2: done
    int64     _rcv_byte   = 0;
    uint      _rcv_ieb    = 0; // iblk of 1st empty block
    ushort    _rcv_src_id = 0;
    int       _rcv_prm0   = 0; // additional parameter 0
    int       _rcv_prm1   = 0; // additional parameter 1

    // sender's members    
    kmFileBlk _snd_blk;        // sending file block buffer
    kmMat1bit _snd_bsf;        // sending block state flag (1: request, 0: not)
    uint      _snd_state  = 0; // 0 : not begin, 1: waiting pre-ack, 2: sending blocks, 3: waiting ack, 4: done
    int64     _snd_byte   = 0;
    uint      _snd_ieb    = 0; // iblk of 1st empty block
    ushort    _snd_src_id = 0;
    int       _snd_prm0   = 0; // additional parameter 0
    int       _snd_prm1   = 0; // additional parameter 1

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
        case 5: RcvEmptyQue(addr); break;
        }
    };

    ///////////////////////////////////////
    // interface functions

    // send file
    //   path : path only
    //   name : sub-path + file name
    // 
    //    ex) d:/folder/sub_folder/file1.exe 
    //           path = d:/folder, name = sub_folder/file1.exe
    // 
    //   return : 0 (sending failed), 1 (sending is successful)
    int Send(ushort src_id, ushort des_id, kmAddr4 addr, const kmStrw& path, const kmStrw& name, int prm0 = 0, int prm1 = 0)
    {
        // check state
        if(_snd_state != 0)
        {
            print("[kmNetPtcFile::File in 1764 _snd_state is not zero\n"); return 0;
        }
        _snd_state = 1; // waiting pre-ack

        // init parameters
        const uint blk_byte = UDP_MAX_DATA_BYTE - sizeof(kmNetHd) - 8; // max limit
        const kmStrw full(L"%S/%S", path.P(), name.P());

        // open file ans set file block
        _snd_blk.OpenToRead(full.P(), blk_byte);

        const int64 byte  = _snd_blk.GetByte();
        const uint  blk_n = _snd_blk.GetBlkN();
        //const float mbyte = (float)byte/(1024.f*1024.f);

        // init variables
        uint snd_blk_n = 0;

        // set bsf
        _snd_bsf.RecreateIf(blk_n);
        _snd_bsf.SetZero();
        _snd_ieb = 0;

        // send pre block... cmd 0, state 1
        SndPreBlk(_snd_src_id = src_id, des_id, addr, blk_n, blk_byte, byte, name, 
                  _snd_prm0 = prm0, _snd_prm1 = prm1);
        
        _snd_ctrl.Init(path, name, byte);
            
        (*_ptc_cb)(_net, -1, (void*)&_snd_ctrl);

        // waiting pre ack... cmd 1, state 2
        const float preack_tout_sec = 10.f; // time out for preack

        kmTimer bps_timer, timer(1);
        while(_snd_state == 1 && timer.sec() < preack_tout_sec) std::this_thread::sleep_for(std::chrono::milliseconds(0));

        if(_snd_state != 2) // time out and sending failed
        {
            if(_snd_state == 1) print("* sending failed by timeout for preack\n"); 
            else                print("* sending has been rejected\n");
            _snd_state = 0; _snd_blk.Close();
            return 0;
        }        
        const float echo_usec = (float)timer.usec();

        print("\n*** blk_n : %d, blk_byte : %d, echo time : %.1f usec\n", blk_n, blk_byte, echo_usec);
    
        // init congestion control parameters
        const uint  cw_max_n       = 128;
        const uint  skip_max_cnt   = 16;
        const float snd_t_max_usec = blk_byte*(8.f/1000.f)*8.f;    // 1000MBPS / 8
        const float snd_t_min_usec = blk_byte*(8.f/1000.f)*0.05f; // 1000MBPS * 20

        uint    cw_n       = 2;                   // size of congestion window
        float   snd_t_usec = snd_t_min_usec*20.f; // 100 usec
        float   out_t_msec = 100.f;               // out time
        uint    skip_cnt   = 0;                   // skip counter

        print("*** sending block (snd_t_usec : %.2f usec, out_t : %.2f msec)\n", snd_t_usec, out_t_msec);
        bps_timer.Start();

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
            for(timer.Start(); timer.msec() < out_t_msec && _snd_state == 3; std::this_thread::sleep_for(std::chrono::milliseconds(0))) {}

            // congestion control
            if(_snd_state == 3) // no ack
            {
                if(++skip_cnt > skip_max_cnt) // sending failed
                {
                    _snd_state = 0;  _snd_blk.Close();
                    (*_ptc_cb)(_net, -4, (void*)&_snd_ctrl);
                    return 0;
                };
                // update cw_n and out_t_sec
                out_t_msec = MIN(out_t_msec*1.5f, 100.f);
                cw_n       = MAX(cw_n/2, 1);
                snd_t_usec = MIN(snd_t_max_usec, 1.2f*snd_t_usec);

                print("** no ack : cw_n (%d) out_t (%.1fmsec) snd_t (%.1fusec)\n", cw_n, out_t_msec, snd_t_usec);
            }
            else if(_snd_state == 2) // got ack
            {
                // calc number of lost blocks
                uint lst_n = 0; for(uint i = _snd_ieb; i <= iblk; ++i) if(_snd_bsf(i) == 0) ++lst_n;

                float lst_rat = float(lst_n)/cw_n;

                // update snd_t_usec, out_t_usec and cw_n
                if(lst_rat > 0.05f) snd_t_usec = MIN(snd_t_max_usec, (1.f + lst_rat)*snd_t_usec);
                else
                {
                    cw_n       = MIN(      cw_max_n,    2*cw_n);
                    snd_t_usec = MAX(snd_t_min_usec, 0.9f*snd_t_usec);
                }
                if(skip_cnt == 0) out_t_msec = MAX(out_t_msec*0.9f, 5.f);

                // reset skip_cnt
                skip_cnt = 0;

                // call cb
                _snd_ctrl.t_usec = (float)bps_timer.usec();
                _snd_ctrl.byte   = (int64)_snd_ieb*blk_byte;
                _snd_ctrl.loss   = (snd_blk_n == 0)? 0:100.f*(snd_blk_n - blk_n)/snd_blk_n;

                (*_ptc_cb)(_net, -2, (void*)&_snd_ctrl);

                //print("** get ack (lst_n : %d, cw_n : %d, out_t : %.1fmsec, snd_t: %.2fusec)\n", 
                //       lst_n, cw_n, out_t_msec, snd_t_usec);
            }
        }
        //const float snd_mbps = (float)(byte/bps_timer.usec());

        // call cb to display comment
        _snd_ctrl.t_usec = (float)bps_timer.usec();
        _snd_ctrl.byte   = _snd_ctrl.total_byte;
        _snd_ctrl.loss   = (snd_blk_n == 0)? 0:100.f*(snd_blk_n - blk_n)/snd_blk_n;

        (*_ptc_cb)(_net, -3, (void*)&_snd_ctrl);

        // end of sending
        _snd_state = 0;  _snd_blk.Close();
        return 1;
    };

    // send data to addr
    //   path : full path
    // 
    //   return : 0 (sending failed), 1 (sending is successful)
    int Send(ushort src_id, ushort des_id, kmAddr4 addr, kmStrw& path, int prm0 = 0, int prm1 = 0)
    {
        kmStrw name = path.SplitRvrs(L'/');

        return Send(src_id, des_id, addr, path, name, prm0, prm1);
    };

    // stop receving procedure
    void StopRcv()
    {    
        _rcv_state = 0; _rcv_blk.Close();
        (*_ptc_cb)(_net, 4, 0);
    };

    ///////////////////////////////////////
    // cmd 1 : pre block
    void SndPreBlk(ushort src_id, ushort des_id, const kmAddr4& addr, uint blk_n, 
                   uint blk_byte, int64 byte, const kmStrw& file_str, int prm0 = 0, int prm1 = 0)
    {    
        // set buffer
        const char cmd_id = 1; kmNetHd hd = {src_id, des_id, _ptc_id, cmd_id, 0, 0}; 

        ushort cFile[200] = {0,};
        ushort len = wcslen(file_str.P());
        convWC4to2(file_str.P(), cFile, len);
        (*_net << hd << blk_n << blk_byte << byte << prm0 << prm1).PutData(cFile, file_str.N());

        // send buffer
        _net->Sendto(addr);
    };
    void RcvPreBlk(const kmAddr4& addr)
    {
        // get from buffer
        kmNetHd hd{}; uint blk_n, blk_byte; int64 byte; kmStrw file_str;

        ushort file_n = 0;
        ushort file_name[256] = {0,};
        wchar  wfile_name[128] = {0,};
        (*_net >> hd >> blk_n >> blk_byte >> byte >> _rcv_prm0 >> _rcv_prm1).GetData(file_name, file_n);
        convWC2to4(file_name, wfile_name, file_n);
        file_str.SetStr(wfile_name);

        { // KKT
        kmStra ip(addr.GetStr());
        print("\n* received file\n");
        print("** block num : %u, block byte : %u\n", blk_n, blk_byte);
        print("** ip addr : %s\n", ip.P());
        wcout<<L"** file name : ";
        char cname[100] = {0,};
        wcstombs(cname, file_str.P(), wcslen(file_str.P())*2);
        cout<<cname<<endl<<endl;
        }

        // check state
        if(_rcv_state != 0)
        {    
            SndPreAck(hd.des_id, hd.src_id, addr, 0); // to reject
            print("[kmNetPtcFile::RcvPreBlk in 1932] _rcv_state is not zero\n"); return;
        }
        _rcv_state  = 1;          // waiting blk
        _rcv_src_id = hd.des_id;

        // get path to save the file
        _rcv_ctrl.file_name  = file_str;
        _rcv_ctrl.total_byte = byte;
        _rcv_ctrl.reject     = 0;

        (*_ptc_cb)(_net, 1, (void*)&_rcv_ctrl);

        if(_rcv_ctrl.reject == 1)
        {    
            SndPreAck(hd.des_id, hd.src_id, addr, 0);
            _rcv_state = 0; return;
        }

        // check if file_path is availables
        kmStrw path = _rcv_ctrl.file_path; path.ReplaceRvrs(L'/', L'\0');

        kmFile::MakeDirs(path);

        // init rcv parameters
        _rcv_blk.OpenToWrite(_rcv_ctrl.file_path.P(), byte, blk_byte, blk_n);
        _rcv_bsf.RecreateIf(blk_n);
        _rcv_bsf.SetZero();
        _rcv_byte   = 0;
        _rcv_ieb    = 0;

        // send ack
        SndPreAck(hd.des_id, hd.src_id, addr);

        _net->SetTom(3.f);
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
        if(_snd_state == 1) _snd_state = (accept == 1) ? 2 : 0;        
    };

    ///////////////////////////////////////////////
    // cmd 3 : blk
    void SndBlk(ushort src_id, ushort des_id, uint iblk, const kmAddr4& addr, bool reqack = false)
    {
        // set snd_buf
        const char cmd_id = 3; kmNetHd hd = { src_id, des_id, _ptc_id, cmd_id, 0, 0};

        if(reqack) hd.SetReqAck(); 

        const ushort blk_byte = _snd_blk.GetBlkByte(iblk);

        *_net << hd << iblk << blk_byte;

        _snd_blk.ReadBlk(_net->_snd_buf.End1(), iblk); _net->_snd_buf.IncN1(blk_byte);

        // send buffer
        _net->Sendto(addr);
    };
    void RcvBlk(const kmAddr4& addr)
    {
        // check rcv state
        if(_rcv_state != 1) return;

        // get from buffer
        kmNetHd hd{}; uint iblk; ushort byte; char* data;

        *_net >> hd >> iblk >> byte; data = _net->_rcv_buf.End1();

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

        // check if reqack
        if(hd.IsReqAck()) SndAck(hd.des_id, hd.src_id, iblk, addr);

        // call callback
        if(_rcv_ieb == _rcv_blk._blk_n) // receiving is done
        {
            // timeout monitoring off
            _net->SetTomOff();

            // close rcv block
            _rcv_state = 2;
            _rcv_blk.Close(); 

            // rename
            kmStrw  cur_name = _rcv_ctrl.file_path;
            kmStrw& new_name = _rcv_ctrl.file_path; new_name.Cutback(4);

            if(kmFile::Exist(new_name.P())) kmFile::Remove(new_name.P());

            kmFile::Rename(cur_name.P(), new_name.P());

            // call cb
            (*_ptc_cb)(_net, 3, nullptr);

            _rcv_state = 0;
        }
        else if(hd.IsReqAck()) // receiving
        {
            _rcv_ctrl.byte = (int64)_rcv_ieb*_rcv_blk.GetBlkByte();

            (*_ptc_cb)(_net, 2, (void*)&_rcv_ctrl);
        }
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

    ////////////////////////////////////////////////////////////////////
    // cmd 5 : empty queue
    //  * Note that this will be called in kmNet::_snd_thrd
    void SndEmptyQue(ushort src_id, ushort des_id, const kmAddr4& addr)
    {
        // set buffer
        const char cmd_id = 5; kmNetHd hd = { src_id, des_id, _ptc_id, cmd_id, 0, 0};

        *_net << hd;

        // send buffer
        _net->Sendto(addr);
    };
    void RcvEmptyQue(const kmAddr4& addr)
    {
        // call cb
        (*_ptc_cb)(_net, 5, nullptr);
    };
};

/////////////////////////////////////////////////////////////////////////////////////////
// kmNet class 

// struct for kmNet's snd queue
struct kmNetSndFile { ushort src_id{}; kmStrw path{}; kmStrw name{}; int prm0{}; int prm1{};};

// network device id
class kmNetId
{
public:
    kmMacAddr mac;    
    kmAddr4   addr;
    ushort    des_id   = 0xffff;
    short     state    = 0;     // 0 : not connect, 1: connecting, 2: valid, 3: invalid
    wchar     name[ID_NAME] = {0,};

    void Print() const
    {
        print ( "  mac    : %s\n", mac .GetStr().P());
        print ( "  addr   : %s\n", addr.GetStr().P());
        print ( "  des Id : %d\n", des_id);
        print ( "  state  : %d\n", state);
        wcout<<L"  name   : "<<name<<endl;
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

    // snd thread members
    kmThread             _snd_thrd;    // thread for sending with snd queue
    kmQue1<kmNetSndFile> _snd_que;     // sending queue for sndfile
    kmLock               _snd_que_lck; // mutex for _snd_que

    // nks members
    kmNetKey    _pkey;                 // own pkey

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
    void Init(void* parent = nullptr, kmNetCb netcb = nullptr)
    {
        // set parent and callback
        _parent = parent;
        _netcb  = netcb;
        _name = getHostName();

        // get address
        if (kmSock::GetIntfAddr(_addr) == false)
            cout<<"Get Local Address Error!!"<<endl;

        // create buffer and header pointer
        // * Note that 64 KB is max size of UDP packet
        _rcv_buf.Recreate(64*1024);
        _snd_buf.Recreate(64*1024);

        // bind
        Bind();

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
            kmTimer timer(1);

            while(1)
            {
                // check time out for rcvblk of ptc_file
                if(net->IsTomOut())
                {
                    print("** receiving failed with timeout (%.2fsec)\n", net->_tom._tout_sec);                    
                    net->SetTomOff();
                    net->_ptc_file.StopRcv();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));

                // timer control
                if     (timer.IsNotStarted() && net->IsTomOff()) timer.Start();
                else if(timer.IsStarted()    && net->IsTomOn())  timer.Stop ();

                // do extra work when nothing to do
                if(net->IsTomOff() && timer.sec() > 3.f)
                {
                    net->DoExtraWork();
                }
                else                
                {    
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            print("* end of rto thread\n");
        }, this);
        _tom_thrd.WaitStart();
    };

    // virtual function for extra work
    virtual void DoExtraWork() { std::this_thread::sleep_for(std::chrono::milliseconds(100)); };

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
        kmAddr4 addr; int ret = Recvfrom(addr);

        if(ret <= 0) return ret;

        // get header and protocol id
        const uchar ptc_id = _rcv_buf.GetHd().ptc_id;
        const  char cmd_id = _rcv_buf.GetHd().cmd_id;
        
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
                    net->_snd_que_lck.Lock();   /////////////  lock

                    kmNetSndFile* snd = net->_snd_que.Dequeue();

                    net->_snd_que_lck.Unlock(); //////////// unlock

                    net->SendFile(snd->src_id, snd->path, snd->name, snd->prm0, snd->prm1);
                    
                    // send emptyque if there is no more file to send
                    if(net->_snd_que.N() == 0) net->NotifyEmptyQue(snd->src_id);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            print("* end of snd thread\n");
        }, this);
        _snd_thrd.WaitStart();
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
        if(cmd_id == 0) // RcvReqConnect... to get _ptc_cnnt._src_id
        {
            // init parameters        
            const kmMacAddr mac = _ptc_cnnt._rcv_des_mac;

            // check if already connected
            for(int i = 0; i < _ids.N(); ++i)
            {
                if(_ids(i).mac == mac)
                {
                    _ids(i).addr   = _ptc_cnnt._rcv_des_addr;
                    _ids(i).des_id = _ptc_cnnt._rcv_des_id;

                    _ptc_cnnt._rcv_src_id = i;

                    memcpy(_ids(i).name, _ptc_cnnt._rcv_des_name, 64);

                    print("** mac(%s) already connected with id(%d)\n", mac.GetStr().P(), i);
                    return -1;
                }
            }
            // set ids and get new id
            kmNetId id = { mac, _ptc_cnnt._rcv_des_addr, _ptc_cnnt._rcv_des_id, 2};

            int src_id = _ptc_cnnt._rcv_src_id = (ushort)_ids.PushBack(id);
            
            memcpy(_ids(src_id).name, _ptc_cnnt._rcv_des_name, 64);

            return (*_netcb)(_parent, _ptc_cnnt._ptc_id, cmd_id, 0);
        }
        else if(cmd_id == 1) // RcvAccept
        {
            // update ids
            int src_id = _ptc_cnnt._snd_src_id;

            _ids(src_id).mac    = _ptc_cnnt._snd_des_mac;
            _ids(src_id).des_id = _ptc_cnnt._snd_des_id;

            memcpy(_ids(src_id).name, _ptc_cnnt._snd_des_name, 64);

            // call cb
            vcbRcvPtcCnnt(src_id, cmd_id);

            return (*_netcb)(_parent, _ptc_cnnt._ptc_id, cmd_id, 0);
        }
        else if(cmd_id == -1) // SndAccept
        {
            int src_id = _ptc_cnnt._rcv_src_id;

            // call cb
            vcbRcvPtcCnnt(src_id, cmd_id);
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
        // call netcb
        if(cmd_id < 0) vcbRcvPtcFile(_ptc_file._snd_src_id, cmd_id, _ptc_file._snd_prm0, _ptc_file._snd_prm1);
        else           vcbRcvPtcFile(_ptc_file._rcv_src_id, cmd_id, _ptc_file._rcv_prm0, _ptc_file._rcv_prm1);

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
    virtual void vcbRcvPtcFile(ushort src_id, char cmd_id, int prm0, int prm1) {}; 

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
    void SetName(const kmStrw& name) { _name = name; };

    // get name
    kmStrw& GetName() { return _name; };

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

    ///////////////////////////////////////////////
    // interface functions for communication

    // get addrs with ptc_brdc (UDP broadcasting)
    int GetAddrsInLan(kmAddr4s& addrs, kmMacAddrs& macs, float tout_msec = 100.f)
    {
        return _ptc_brdc.GetAddrs(addrs, macs, tout_msec);
    };

    // connect to new device with ptc_cnnt
    //  return : src_id (if connecting failed, it will be -1)
    int Connect(const kmAddr4 addr, const kmStrw& name, float tout_msec = 100.f)
    {
        // get new id
        kmNetId id = { 0, addr, 0, 1};

        const ushort src_id = (ushort)_ids.PushBack(id);

        // connect to addr
        ushort des_id; kmMacAddr des_mac; 

        kmT2(des_id, des_mac) = _ptc_cnnt.Connect(src_id, addr, name, tout_msec);

        // post processing
        if(des_id < 0xffff - 1) // connecting is successful
        {
            // update id
            kmNetId& netid = _ids(src_id);

            netid.mac    = des_mac;
            netid.des_id = des_id;
            netid.state  = 2;

            wcsncpy(netid.name, _ptc_cnnt._snd_des_name, ID_NAME);
        }
        else // connecting failed
        {
            // cancel connecting 
            _ids.PopBack()->state = 0; return -1;
        }
        return (int)src_id;
    };

    // send data through ptc_data
    //   tout_msec : 0 (not wait for ack), > 0 (wait for ack)
    // 
    //   return : 0 (not received ack), 1 (received ack)
    int Send(ushort src_id, uchar data_id, char* data, ushort byte, float tout_msec = 0.f)
    {
        // get id
        kmNetId& id = _ids(src_id);

        // send data
        return _ptc_data.Send(src_id, id.des_id, id.addr, data_id, data, byte, tout_msec);
    };

    // get data from ptc_data
    //   return : kmT4(data_id, data, byte, src_id)
    kmT4<uchar, char*, ushort, ushort> GetData()
    {
        return kmT4(_ptc_data._data_id, _ptc_data._data, _ptc_data._byte, _ptc_data._src_id);
    };

    // get data_id only from ptc_data
    uchar GetDataId() {    return _ptc_data._data_id; };

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
    //   return : kmT2(sended byte, total byte)
    kmT2<int64, int64> GetLrgdInfoSnd()
    {
        int64 byte = _ptc_lrgd._snd_blk.Byte();

        return kmT2(_ptc_lrgd._snd_byte, byte);
    };

    // send file through ptc_file
    //   prm0, prm1 : additional parameters (optional)
    //   return : 0 (sending failed), 1 (sending is successful)
    int SendFile(ushort src_id, kmStrw path, int prm0 = 0, int prm1 = 0)
    {
        kmStrw name = path.SplitRvrs(L'/');

        return SendFile(src_id, path, name, prm0, prm1);
    };

    // send file through ptc_file
    //   prm0, prm1 : additional parameters (optional)
    //   return : 0 (sending failed), 1 (sending is successful)
    int SendFile(ushort src_id, const kmStrw& path, const kmStrw& name, int prm0 = 0, int prm1 = 0)
    {
        // get id
        kmNetId& id = _ids(src_id);

        // send file
        return _ptc_file.Send(src_id, id.des_id, id.addr, path, name, prm0, prm1);
    };

    // send file through ptc_file with seperated thread    
    //   prm0, prm1 : additional parameters (optional)
    void EnqueueSndFile(ushort src_id, kmStrw path, int prm0 = 0, int prm1 = 0)
    {
        kmStrw name = path.SplitRvrs(L'/');

        EnqueueSndFile(src_id, path, name, prm0, prm1);
    };

    // send file through ptc_file with seperated thread
    //   prm0, prm1 : additional parameters (optional)
    void EnqueueSndFile(ushort src_id, const kmStrw& path, const kmStrw& name, int prm0 = 0, int prm1 = 0)
    {
        _snd_que_lck.Lock();   ////////////////// lock
        _snd_que    .Enqueue(kmNetSndFile({src_id, path, name, prm0, prm1}));
        _snd_que_lck.Unlock(); ////////////////// unlock
    };

    // send empty queue
    void NotifyEmptyQue(ushort src_id)
    {
        // get id
        kmNetId& id = _ids(src_id);

        // send noti for empty queue
        _ptc_file.SndEmptyQue(src_id, id.des_id, id.addr);
    };

    kmStrw getHostName()
    {
        char host[100] = {0,};
        gethostname(host, sizeof(host));
        wchar whost[100] = {0,};
        mbstowcs(whost, host, strlen(host));
        kmStrw ret;
        ret.SetStr(whost);
        return ret;
    }

    // request pkey
    //   return : 0 (failed), 1 (successful)
    int RequestPkey()
    {
        // get key from keysvr
        _pkey = _ptc_nkey.ReqKey(kmNetKeyType::pkey);

        if(_pkey.IsValid() == false)
        {
            return 0;
        }
        _pkey.Print();
        return 1;
    };

    // request address to nks
    //   return : 0 (failed), 1 (successful)
    int RequestAddr(kmNetKey key, kmMacAddr mac)
    {
        _ptc_nkey.ReqAddr(key, mac);

        return 1;
    };
};

#endif /* __km7Net_H_INCLUDED_2021_05_31__ */

#ifndef __zbNet_H_INCLUDED_2022_04_07__
#define __zbNet_H_INCLUDED_2022_04_07__

/* Note ----------------------
* zbNet has been created by Choi, Kiwan
* This is version 1
* zbNet is based on kmNet (ver. 7)
*/

//#define KMNETDPORT 60165  // default port... only for jkt test
#define KMNETDPORT 60166  // default port

#define zbTestVkey kmNetKey(kmNetKeyType::vkey, 999, 999)

// base header
#include "km7Net.h"
#include "km7Jsn.h"

/////////////////////////////////////////////////////////////////////////////////////////
// zbNet class

//////////////////////////////////////
// zibsvr file

// zibsvr list type enum class... user list, strg list, file list
enum class zbListType : short { user, strg, file };

// zibsvr file type enum class... image, movie, data, folder, dummy
enum class zbFileType : short { image, movie, data, folder, dummy };

// zibsvr file state enum class... bkupno, bkup, bkuponly, deletednc, deleted, bkupban, bkupbannc, none
//               clt / svr
//   bkupban   :   o  /  x   : none
//   bkupno    :   o  /  x   : none
//   bkup      :   o  /  o   : bkupno/bkup
//   bkupbannc :   o  /  o   : bkupno/bkup
//   bkuponly  :   x  /  o   : bkupno/bkup
//   deletednc :   x  /  o   : bkupno/bkup
//   deleted   :   x  /  x   : none
//   
enum class zbFileState : short { bkupno, bkup, bkuponly, deletednc, deleted, bkupban, bkupbannc, none };

// zibsvr file flag
class zbFileFlg
{
public:
    uchar file    : 1 = 0; // 0: no file        1: there is the file
    uchar thumb   : 1 = 0; // 0: no thumbnail   1: there is the thumbnail 
    uchar cache   : 1 = 0; // 0: no cache       1: there is the cache
    uchar encrypt : 1 = 0; // 0: not encrypted  1: encrypted
    uchar isin    : 1 = 0; // teamperary flag
};

// zibsvr file class... type, state, flg, encrypt, name, date, gps, crypt
class zbFile
{
public:
    zbFileType  type   {};  // image, movie, data, folder
    zbFileState state  {};  // bkupno, bkup, bkuponly, deletednc, deleted, bkupban, bkupbannc, none
    zbFileFlg   flg    {};  // thumb, encrypted    
    ushort      encrypt{};  // encryption key... reserved
    kmStra      name   {};  // relative path from strg.srcpath + name ... utf8
    kmDate      date   {};  // time created or written
    kmGps       gps    {};  // gps info
};
typedef kmMat1<zbFile> zbFiles;

//////////////////////////////////////
// zibsvr storage 

// zibsvr storage type enum class
//  : imgb (image bkup), sync, pssv (passive)
//  : imgl (image link for svr), imgs (image shrd for clt)
enum class zbStrgType { imgb, sync, pssv, imgl, imgs };

// zibsvr storage class
class zbStrg
{
public:
    zbStrgType type{};     // imgb (bkup), sync, pssv, imgl (link), imgs (shrd)
    kmStra     name{};     // name only... utf8
    kmStra     path{};     // full path... utf8
    kmStra     srcpath{};  // path of source files... utf8
    zbFiles    files{};    // file list
    int        chknfid{};  // next fid of checked file only for imgb to make thumb
    short      lnk_uid{};  // only for imgl
    short      lnk_sid{};  // only for imgl

    // get string for rol
    kmStra GetTypeStr()
    {
        switch(type)
        {
        case zbStrgType::imgb : return "imgb";
        case zbStrgType::sync : return "sync";
        case zbStrgType::pssv : return "pssv";
        case zbStrgType::imgl : return "imgl";
        case zbStrgType::imgs : return "imgs";
        }
        return "unknown";
    }
};
typedef kmMat1<zbStrg> zbStrgs;

//////////////////////////////////////
// zibsvr shared item

// zibsvr shared item type enum class... file, strg, user
//  file : shared album (itm(0,0,0), itm(0,0,1), ... )
//  strg : shared storage (member cannot add but can delete as authority) (itm(0,0,-1), itm(0,1,-1), ...)
//  user : shared users (patner) (itm is not used)
enum class zbShrdType { file, strg, user };

// zibsvr shared state enum class... valid, pending, deleted
enum class zbShrdState { valid, pending, deleted };

// zibsvr shared authority enum class... none, readonly, readwrite, admin
//   readonly  : can read every files
//   readwrite : readonly  + can write own files
//   admin     : readwrite + can delete every files and change shared info
enum class zbShrdAuth { none, readonly, readwrite, admin };

// zibsvr shared member info
class zbShrdMember
{
public:
    int uid{}; zbShrdAuth auth{};

    json ToJson();
};
typedef kmMat1<zbShrdMember> zbShrdMembers;

// index for shared item
class zbShrdItm
{
public : 
    int uid{}, sid{}, fid{}; // -1 means every 

    json ToJson();
};
typedef kmMat1<zbShrdItm> zbShrdItms;

// zibsvr shared info (user, storage, file) 
class zbShrd
{
public:
    zbShrdType    type{};      // user, strg, file
    zbShrdState   state{};     // valid, pending, deleted
    kmStra        name{};      // name or description ... utf8
    int           owner_uid{}; // owner's uid
    kmDate        time{};      // created time
    zbShrdMembers mmbs;        // shared members (without owner)
    zbShrdItms    itms;        // shared itmes (uid, sid, fid)

    // constructor
    zbShrd() { };
    zbShrd(zbShrdType type, kmStra name, int owner_uid)
        : type(type), name(name), owner_uid(owner_uid) {
    };

    void Init()
    {
        mmbs.Recreate(0, 2);
        itms.Recreate(0, 8);
    }

    // add item (user or storage or file)
    zbShrd& AddItem(int uid, int sid = -1, int fid = -1)
    {
        if(itms.Size() == 0) itms.Recreate(0,16);

        itms.PushBack(zbShrdItm(uid, sid, fid)); return *this; 
    };

    // add member
    zbShrd& AddMember(int uid, zbShrdAuth auth = zbShrdAuth::readonly)
    {
        if(mmbs.Size() == 0) mmbs.Recreate(0,8);

        mmbs.PushBack(zbShrdMember(uid, auth)); return *this; 
    };

    // get string for type
    kmStra GetTypeStr() const
    {       
        switch(type)
        {
        case zbShrdType::user : return "user";
        case zbShrdType::strg : return "strg";
        case zbShrdType::file : return "file";
        }
        return "unknown";
    };

    // get string for type
    kmStra GetStateStr() const
    {       
        switch(state)
        {
        case zbShrdState::valid   : return "valid";
        case zbShrdState::pending : return "pending";
        case zbShrdState::deleted : return "deleted";
        }
        return "unknown";
    };

    // get json string
    json ToJson(int uid);

    // set from json string
    void FromJson(const kmStra& jsna);
};
typedef kmMat1<zbShrd> zbShrds;

//////////////////////////////////////
// zibsvr user

// zibsvr role enum class... owner, member, ownsvr, svr, family, guest
enum class zbRole { owner, member, ownsvr, svr, family, guest };

// zibsvr user class
class zbUser
{
public:    
    kmMacAddr  mac{};
    kmStra     name{};      // user name ... utf8
    zbRole     role{};      // owner, member, ownsvr, svr, family, guest
    kmNetKey   key{};       // pkey
    kmNatType  nattype{};   // nat type
    kmStra     path{};      // path of user (full path)... utf8
    kmDate     time{};      // last connected time
    kmAddr4    addr{};      // last connected ip
    ushort     src_id = -1; // src_id for _ids
    zbStrgs    strgs{};     // storage list
    
    // cosntructor
    zbUser() {};    
    zbUser(kmMacAddr mac, const kmStra& name, zbRole role, kmAddr4 addr, ushort src_id, kmNetKey key = {})
        : mac(mac), name(name), role(role), addr(addr), src_id(src_id), key(key) {};

    // reset connected time
    void ResetTime() { time.SetCur(); };

    // display user info
    void PrintInfo()
    {
        print("name : %s\nmac : %s\nip : %s\n\n", 
            name.P(), mac.GetStr().P(), addr.GetStr().P());
    };

    // get string for role
    kmStra GetRoleStr()
    {
        switch(role)
        {
        case zbRole::owner  : return "owner";
        case zbRole::member : return "member";
        case zbRole::ownsvr : return "ownsvr";
        case zbRole::svr    : return "svr";
        case zbRole::family : return "family";
        case zbRole::guest  : return "guest";
        }
        return "unknown";
    };

    // set role
    void SetRole(const kmStra& str)
    {
        if     (str == "owner" ) role = zbRole::owner;
        else if(str == "member") role = zbRole::member;
        else if(str == "ownsvr") role = zbRole::ownsvr;
        else if(str == "svr")    role = zbRole::svr;
        else if(str == "family") role = zbRole::family;
        else if(str == "guest")  role = zbRole::guest;
    }

    // is server
    bool IsSvr() { return role == zbRole::ownsvr || role == zbRole::svr; };

    // get json string... return value is UTF-8
    json ToJson(int uid);

    // set from json string... jsna is UTF-8
    //   return : self uid
    int FromJson(const kmStra& jsna);
};
typedef kmMat1<zbUser> zbUsers;

///////////////////////////////////////////////////////////
// zbNet main class

// zibsvr notify enum class
enum class zbNoti : int { none };

// zibsvr mode enum class
enum class zbMode : int { svr, clt, nks };

// zibsvr info
class zbNetInfo
{
public:
    kmMacAddr mac{};
    kmStra    name{};    // ... utf8
    zbMode    mode{};
    int       user_n{};
    ushort    src_id{};
    kmNetKey  vkey{};
};
typedef kmMat1<zbNetInfo> zbNetInfos;
typedef kmQue1<zbNetInfo> zbNetInfoQue;

// receiving file info
class zbNetRcvFileInfo
{
public:
    int _uid, _sid, _fid, _opt;
    int _state = 0;     // 0: none, 1: receving, 2: done

    bool IsReceiving(int uid, int sid, int fid, int opt)
    {
        if(_state == 1 && uid == _uid && sid == _sid && fid == _fid && opt == _opt) return true;
        return false;
    };
    bool IsDone(int uid, int sid, int fid, int opt)
    {
        if(_state == 2 && uid == _uid && sid == _sid && fid == _fid && opt == _opt) return true;
        return false;
    };
    void SetReceiving(int uid, int sid, int fid, int opt)
    {
        _uid = uid; _sid = sid; _fid = fid; _opt = opt; _state = 1;
    };
    void SetDone(int uid, int sid, int fid, int opt)
    {
        _uid = uid; _sid = sid; _fid = fid; _opt = opt; _state = 2;
    };
    void Clear() { _state = 0; }
};

// vkey element for table
class zbVkeyElm
{
public:
    kmNetKey key{};
    kmDate   date{};      // expiration time
    uint     cnt   =  0;  // valid count
    int      hid   = -1;  // shared index

    kmStra GetStr() const
    {
        return kmStra("%s : hid %d, cnt %d, exp %s", 
                      key.GetStr().P(), hid, cnt, date.GetStrPt().P());
    };
    void Print() const { print("* key elm : %s\n", GetStr().P()); };
};
typedef kmMat1<zbVkeyElm> zbVkeyElms;

// class for json buffer
class zbNetJsn
{
public:
    ushort ack_id = 0;
    ushort state  = 0; // 0 : empty, 1: waiting, 2: received wack, 3: recieve nack
    kmStra jsna{};     // json string ... utf8
};

class zbNetJsnBuf
{
protected:
    kmMat1<zbNetJsn> _buf{};
    kmLock           _lck{};

public:
    // create buffer
    void Create(int64 n)
    {
        kmLockGuard grd = _lck.Lock(); ///////////// lock & unlock

        _buf.Recreate(n);  
    };

    // regist ack_id waiting for (wack)
    //   return : registed index
    int Regist(ushort ack_id)
    {
        kmLockGuard grd = _lck.Lock(); ///////////// lock & unlock

        for(int i = 0; i < _buf.N1(); ++i)
        {
            if(_buf(i).state == 0)
            {
                _buf(i).ack_id = ack_id;  
                _buf(i).state  = 1;    return i;
            }
        }
        // if there is no empty buf
        const int64 idx = _buf.PushBack();

        _buf(idx).ack_id = ack_id;
        _buf(idx).state  = 1;      return (int)idx;
    };

    // find ack_id and put jsna    
    //   return : jsna's referencefound index
    int FindPut(ushort ack_id, const kmStra& jsna)
    {
        kmLockGuard grd = _lck.Lock(); ///////////// lock & unlock

        // if it is ack waiting for (wack)
        for(int i = 0; i < _buf.N1(); ++i)
        if(_buf(i).state == 1 && _buf(i).ack_id == ack_id)
        {
            _buf(i).jsna  = jsna;
            _buf(i).state = 2;     return i;
        }
        // if it is ack not waiting for (nack)
        for(int i = 0; i < _buf.N1(); ++i)
        if(_buf(i).state == 0)
        {
            _buf(i).ack_id = ack_id;
            _buf(i).jsna   = jsna;
            _buf(i).state  = 3;    return i;
        }
        // if there is no empty buf
        const int64 idx = _buf.PushBack();

        _buf(idx).ack_id = ack_id;
        _buf(idx).jsna   = jsna;
        _buf(idx).state  = 3;    return (int)idx;
    };

    // find, get and del jsna
    // return : jsna (if there is no ack_id, jsna.N1() will be zero)
    kmStra FindGetDel(ushort ack_id)
    {
        kmLockGuard grd = _lck.Lock(); ///////////// lock & unlock

        for(int i = 0; i < _buf.N1(); ++i)
        if((_buf(i).state == 2 || _buf(i).state == 3) && _buf(i).ack_id == ack_id)
        {   
            _buf(i).state = 0;  return _buf(i).jsna;
        }        
        return kmStra();
    };

    // find and delete
    void FindDel(ushort ack_id)
    {
        kmLockGuard grd = _lck.Lock(); ///////////// lock & unlock

        for(int i = 0; i < _buf.N1(); ++i)
        if(_buf(i).ack_id == ack_id)
        {
            _buf(i).ack_id = 0;
            _buf(i).state  = 0; return;
        }
    };

    // find and delete if nack
    void FindDelNack(ushort ack_id)
    {
        kmLockGuard grd = _lck.Lock(); ///////////// lock & unlock

        for(int i = 0; i < _buf.N1(); ++i)
        if(_buf(i).ack_id == ack_id)
        {
            if(_buf(i).state == 3)
            {
                _buf(i).ack_id = 0;
                _buf(i).state  = 0; 
            }
            return;
        }
    };

    // get jsna
    kmStra& GetJsna(int idx)  { return _buf(idx).jsna;  };

    // get buffer number of state... // 0 : empty, 1: waiting, 2: received
    int GetBufN(int state = 0)
    {
        int n = 0; for(int i = 0; i < _buf.N1(); ++i) if(_buf(i).state == state) ++n;

        return n;
    };

    // get lock
    kmLock& GetLock() { return _lck; };

    // print buffer info
    void Print()
    {
        kmLockGuard grd = _lck.Lock(); ///////////// lock & unlock

        print("*** zbNetJsnBuf : total(%d), empty(%d), waiting(%d), rcv wack(%d), rcv nack(%d)\n",
            _buf.N1(), GetBufN(0), GetBufN(1), GetBufN(2), GetBufN(3));

        for(int i = 0; i < _buf.N1(); ++i)
        if(_buf(i).state == 1) print("***  waiting(%d) : (%d) %s\n", i, _buf(i).ack_id, _buf(i).jsna.P());

        for(int i = 0; i < _buf.N1(); ++i)
        if(_buf(i).state == 2) print("***  rcv wack(%d) : (%d) %s\n", i, _buf(i).ack_id, _buf(i).jsna.P());

        for(int i = 0; i < _buf.N1(); ++i)
        if(_buf(i).state == 3) print("***  rcv nack(%d) : (%d) %s\n", i, _buf(i).ack_id, _buf(i).jsna.P());
    };
};

struct AlarmData
{
    unsigned char targetId = 0;
    kmStra data;
};

//#include <shared_mutex>
//typedef std::shared_mutex Lock;

#define MAX_ALARM_BUFF 20

class zbNetJsnAlarmBuf
{
protected:
    AlarmData _buf[MAX_ALARM_BUFF];
    int _bufIdxS = 0;
    int _bufIdxE = 0;

    //Lock _alarmLock;

    kmThread  _thrd;
    kmLock    _lck{};
    void*     _pNet = nullptr;

public:
    // create buffer
    void Init(void* net);
    void Push(AlarmData data);
    bool Pop(AlarmData& data);
    bool Empty();
    void sendNoti();
    //void printBuf();
};

// zibsvr main class
class zbNet : public kmNet
{
public:
    zbMode           _mode{};           // svr (server), clt (client), nks (net key signaling server)
    ushort           _port{KMNETDPORT}; // port
    kmStra           _path{};           // zibsvr root's path... utf8
    zbUsers          _users{};          // users info
    zbShrds          _shrds{};          // shared info between users... actually svr only
    kmStra           _srcpath{};        // image source path ... client only, utf8
    kmStra           _dwnpath{};        // download path     ... client only, utf8
    kmStra           _tmppath{};        // temprary path     ... client only, utf8
    zbNetInfoQue     _rcvinfo_que;      // queue of received info
    kmLock           _rcvinfo_que_lck;  // mutex of _rcvinfo_que
    kmNetNks         _nks;              // net key signaling function only for zbMode::nks
    zbNetRcvFileInfo _rcvfile{};        // info of receiving file
    kmWorks          _wrks;             // worker to send data    
    ushort           _jsn_snd_ack_id;   // json ack id to send rep 
    zbNetJsnBuf      _jsnbuf;           // json buffer
    zbNetJsnAlarmBuf _jsnAlarm;         // svr -> clt noti msg for push alarm
    zbVkeyElms       _vkeys{};          // vkey element array only for svr

    // parameters
    float _cnnt_tout_msec       = 400.f;   // timeout for cnnt
    float _precnnt_tout_msec    = 300.f;   // timeout for precnnt
    int   _precnnt_try_n        = 6;       // try number for precnnt
    float _accs_tout_msec       = 300.f;   // timeout for precnnt of reqaccs
    int   _accs_try_n           = 16;      // try number for precnnt of reqaccs

    // init
    void Init(void* parent, kmNetCb netcb)
    {
        // init kmnet
        kmNet::Init(parent, netcb, _port);

        // load setting... _pkey
        LoadSetting(); _pkey.Print();
        
        // load users and strgs
        if(_mode == zbMode::svr || _mode == zbMode::clt)
        {
            LoadUsers();

            for(int uid = 0; uid < _users.N1(); ++uid)
            {
                LoadStrgs(uid);

                for(int sid = 0; sid < _users(uid).strgs.N1(); ++sid)
                {
                    LoadFileList(uid,sid);  PrintFileList(uid,sid);
                }
            }
        }

        // create network key table only for nks
        if(_mode == zbMode::nks)
        {
            if(LoadNks() == 0) _nks.Create();
        }

        // create vkey table only for svr
        if(_mode == zbMode::svr) _vkeys.Recreate(0, 32);

        // request pkey or send sig to nks svr
        if(_mode == zbMode::svr)
        {
            // load shrds
            if (_mode == zbMode::svr) LoadShrdList();

            // request pkey or send sig to nks svr
            if(_pkey.IsValid())
            {
                const float ec_msec = SendSigToNks();

                if(ec_msec > 0) print("* sig to nks : %.1f msec\n", ec_msec);
                else            print("* sig to nks : time-out\n");
            }
            else // pkey is invalid
            {
                if(RequestPkeyToNks() == 1) SaveSetting(); 
            }
        }

        // init for rcvinfo queue
        _rcvinfo_que.Recreate(8);

        // init for jsn
        _jsnbuf.Create(128);

        // init for alarm
        _jsnAlarm.Init(this);

        // create work thread 
        _wrks.Create([](kmWork& wrk, zbNet* net)
        {    
            ushort src_id; zbRole role; kmAddr4 addr; char* jsnap; int n, msec; ushort ack_id; 

            uint vld_sec, vld_cnt; kmNetKey vkey;

            switch(wrk.Id())
            {
            case 0: wrk.Get(src_id);                   net->SendInfo     (src_id);                   break;
            case 1: wrk.Get(src_id, role, vkey);       net->RequestRegist(src_id, role, vkey);       break;
            case 2: wrk.Get(src_id, role);             net->AcceptRegist (src_id, role);             break;
            case 3: wrk.Get(addr);                     net->Preconnect   (addr);                     break;
            case 4: wrk.Get(src_id, jsnap, n, ack_id); net->ParseJson    (src_id, jsnap, n, ack_id); break;
            case 5: wrk.Get(src_id);                   net->TestNetCnd   (src_id);                   break;
            case 6: wrk.Get(msec);                     Sleep             (msec);                     break;
            case 7: wrk.Get(src_id, vld_sec, vld_cnt); net->ProcVkey     (src_id, vld_sec, vld_cnt); break;
            }    
        }, this);

        // create kal (keep-alive) thread only for svr
        if(_mode == zbMode::svr) CreateKalThrd();
    };

    // remove every user, strg, file, shrd and init 
    void Reset()
    {
        print("** zbNet will be reset!!\n");

        // remove file
        kmFile::RemoveAll(_path.cuw().P());

        // reset lists
        _users.Recreate(0,4);
        _shrds.Recreate(0,4);

        // load setting
        _pkey = kmNetKey();

        LoadSetting();

        // recreate vkey
        if(_mode == zbMode::svr) _vkeys.Recreate(0,32);

        // request pkey or send sig to nks svr
        if(_mode == zbMode::svr)
        {   
            // request pkey or send sig to nks svr
            if(_pkey.IsValid())
            {
                const float ec_msec = SendSigToNks();

                if(ec_msec > 0) print("* sig to nks : %.1f msec\n", ec_msec);
                else            print("* sig to nks : time-out\n");
            }
            else // pkey is invalid
            {
                if(RequestPkeyToNks() == 1) SaveSetting(); 
            }
        }
    };

    // get functions
    zbUser&       GetUser (int uid)          { return _users(uid); };
    zbStrg&       GetStrg (int uid, int sid) { return _users(uid).strgs(sid); };
    zbNetInfoQue& GetRcvInfoQue()            { return _rcvinfo_que; };
    zbNetInfo     GetRcvInfo()
    {
        kmLockGuard grd = _rcvinfo_que_lck.Lock(); ///////////// lock & unlock

        zbNetInfo* info = _rcvinfo_que.Dequeue();

        return (info == nullptr) ? zbNetInfo() : *info;
    };
    int GetRcvInfoN()
    {
        kmLockGuard grd = _rcvinfo_que_lck.Enter(); ///////////// enter & leave
        
        return (int)_rcvinfo_que.N1(); 
    };

    // get src_id from uid
    //  return : 0xffff (not available), 0 <= : src_id
    ushort GetSrcId(int uid)
    {
        if(uid < 0 || uid >= _users.N1()) return 0xffff;

        const ushort src_id = _users(uid).src_id;

        if(src_id >= _ids.N1()) return 0xffff;

        return src_id;
    };

    // get files to cope with imgl type
    zbFiles& GetFiles(int uid, int sid)
    {
        zbStrg& strg = _users(uid).strgs(sid);

        if(strg.type == zbStrgType::imgl)
        {
            return _users(strg.lnk_uid).strgs(strg.lnk_sid).files;
        }
        return strg.files;
    };

    // get path... utf8
    //  opt : 0 (          origin file, saved to _srcpath or strg path)
    //        1 (           cache file, saved to _tmppath)
    //        2 (       thumbnail file, saved to strg0/.thumb)
    //        3 (          shared file, saved to _dwnpath)
    //        4 (    cache shared file, saved to _tmppath/.shrd/)
    //        5 (thumbnail shared file, saved to _path/shrd/.thumb/)
    //        6 (    own profile image, save to _path/)
    //        7 ( user's profile image, save to _path/user0/)
    //        8 (  mmb's profile image, save to _path/user0/)
    //        9 ( shrd's profile iamge, save to _path/shrd/, svr)
    //       10 ( shrd's profile iamge, save to _path/user0/shrd/, clt)
    kmStra GetPath(int uid, int sid, int opt = 0)
    {    
        switch(opt)
        {
        case  0: return _users(uid).strgs(sid).srcpath;
        case  1: return _tmppath;
        case  2: return _users(uid).strgs(sid).path + "/.thumb";
        case  3: return _dwnpath;
        case  4: return _tmppath + "/.shrd";
        case  5: return _path + "/shrd/.thumb";
        case  6: return _path;
        case  7: 
        case  8: return _users(uid).path;
        case  9: return _path + "/shrd";
        case 10: return _users(uid).path + "/shrd";
        }
        return kmStra();
    };

    // get full path... opt : 0 (file), 1 (cache), 2 (thumbnail).. utf8
    kmStra GetFullPath(int uid, int sid, int fid, int opt = 0)
    {
        return GetPath(uid,sid,opt) + "/" + GetFiles(uid, sid)(fid).name;
    };

    // set functions
    void SetPort(ushort port)
    {
        if(_port == port) return;

        _addr.SetPort(_port = port); 
        
        if(GetSock()._state > 1)
        {
            Close(); Init(_parent, _netcb);
        }
    };

    // enqueue work
    template<typename... Ts>
    void EnqueueWork(int id, Ts... args) { _wrks.Enqueue(id, args...); }
    
    ///////////////////////////////////////////////
    // virtual functions for rcv callback
protected:
    // virtual callback for ptc_cnnt 
    //   cmd_id will be 0 (rcvreqcnnt), 1 (rcvaccept)
    virtual void vcbRcvPtcCnnt(ushort src_id, char cmd_id)
    {
        // get id
        kmNetId& id = _ids(src_id); id.cnd.isdone = 0;

        // find user already registered
        int uid = FindUser(id.mac);

        if(uid < 0) // no registered user
        {
            if(cmd_id == 0 && _mode == zbMode::svr) EnqSendInfo(src_id);
        }
        else 
        {
            _users(uid).src_id = src_id;
            _users(uid).addr   = id.addr;
            _users(uid).ResetTime();

            SaveUsers();
        }
    };

    // virtual callback for ptc_data
    virtual void vcbRcvPtcData(ushort src_id, uchar data_id, kmNetBuf& buf)
    {
        switch(data_id)
        {
        case  0: RcvInfo      (src_id, buf); break;
        case  1: RcvReqRegist (src_id, buf); break;
        case  2: RcvAcpRegist (src_id, buf); break;
        case  3: RcvNotify    (src_id, buf); break;
        case  4: RcvReqFile   (src_id, buf); break;
        case  5: RcvDelFile   (src_id, buf); break;
        case  6: RcvAckDelFile(src_id, buf); break;
        case  7: RcvReqThumb  (src_id, buf); break;
        case  8: RcvReqList   (src_id, buf); break;
        case  9: RcvTxt       (src_id, buf); break;
        case 10: RcvJson      (src_id, buf); break;
        case 11: RcvReqInfo   (src_id, buf); break;
        }
    };

    // virtual callback for ptc_file
    //  cmd_id  1: rcv preblk, 2: receiving, 3: rcv done,  4: rcv failure
    //         -1: snd preblk,              -3: snd done, -4: snd failure
    virtual void vcbRcvPtcFile(ushort src_id, char cmd_id, int* prm)
    {
        if(cmd_id == 2) return;

        kmNetPtcFileCtrl& ctrl = _ptc_file._rcv_ctrl;
        const int         uid  = FindUser(src_id);
                
        int sid = prm[0], fid = prm[1], opt = LOWORD(prm[2]), uuid = HIWORD(prm[2]);

        if(cmd_id == 1) // rcv preblk
        {
            if(opt == 3) // shared file
            {
                // remove sub-path from file_name
                int idx = ctrl.file_name.FindRvrs('/');
                
                ctrl.file_name = ctrl.file_name.Get(kmI(idx+1, end32));
            }
            else if(opt == 4 || opt == 5) // shared file cache or thumbnail
            {
                // get extension
                kmStra& name = ctrl.file_name;

                int sla_idx = name.FindRvrs('/');
                int dot_idx = name.FindRvrs('.');

                kmStra ext = (sla_idx < dot_idx)? name.Get(kmI(dot_idx, end32)) : kmStra("");
                
                // rename for unique naming
                name.SetStr("sh%d-%d-%d-%d%s",uid,uuid,sid,fid,ext.P());
            }
            else if(opt == 8) // member's profile image
            {
                ctrl.file_name.SetStr("user%d-%d.jpg",uid,uuid);
            }
            
            if(CheckId(uid, sid) || opt > 2) // if it's shared file. don't care about uid, sid
            {
                ctrl.SetPath(GetPath(uid, sid, opt));

                _rcvfile.SetReceiving(uid, sid, fid, opt);
            }
            else  ctrl.Reject(); 
        }
        else if(cmd_id == 3) // rcv done
        {
            // get date from the file
            kmFileInfo fileinfo(ctrl.file_path.cuw().P());

            zbFile file = {{}, {}, {}, 0, ctrl.file_name, fileinfo.date};

            _rcvfile.SetDone(uid, sid, fid, opt);

            if(opt == 0) // file
            {
                if(!CheckId(uid, sid)) return;

                AddFile(uid, sid, fid, file); SaveFileList(uid, sid);
            }
            else if(opt == 1 || opt == 2) // 1 : cache, 2 : thumb
            {
                if(!CheckId(uid, sid, fid)) return;

                if(opt == 1) GetFiles(uid, sid)(fid).flg.cache = 1;
                else         GetFiles(uid, sid)(fid).flg.thumb = 1;

                SaveFileList(uid, sid);
            }
        }
        else if(cmd_id == 4) // rcv failure
        {
            _rcvfile.Clear();
        }
        else if(cmd_id == 5) // empty queue --> it means that there is no more file to send
        {
        }
        else if(cmd_id == -1) // snd preblk
        {
        }
        else if(cmd_id == -3) // snd done
        {
            if(opt == 0 && _users(uid).IsSvr() && sid >= 0)
            {
                _users(uid).strgs(sid).files(fid).state = zbFileState::bkup;
                SaveFileList(uid, sid);
            }
        }
    };

    // virtual callback for ptc_nkey
    //  cmd_id 0 : rcv reqkey, 1 : rcv key, 2 : rcv reqaddr, 3 : rcv addr
    virtual void vcbRcvPtcNkey(char cmd_id)
    {
        switch(cmd_id)
        {
        case 0 : RcvNkeyReqKey (); break;
        case 1 : RcvNkeyKey    (); break;
        case 2 : RcvNkeyReqAddr(); break;
        case 3 : RcvNkeyAddr   (); break;
        case 4 : RcvNkeySig    (); break;
        case 5 : RcvNkeyRepSig (); break;
        case 6 : RcvNkeyReqAccs(); break;
        }
    };
    void RcvNkeyReqKey()
    {
        // get parameters of ptc_nkey... _rcv_src_id
        kmAddr4      addr    = _ptc_nkey._rcv_addr;
        kmMacAddr    mac     = _ptc_nkey._rcv_mac;
        kmNetKey     pkey    = _ptc_nkey._rcv_pkey;
        kmNetKeyType type    = _ptc_nkey._rcv_keytype;
        uint         vld_sec = _ptc_nkey._rcv_vld_sec;
        uint         vld_cnt = _ptc_nkey._rcv_vld_cnt;
        ushort       src_id  = _ptc_nkey._rcv_src_id;

        print("**** rcv request key from %s\n", addr.GetStr().P());

        if(_mode == zbMode::nks)
        {
            // register key
            kmNetKey key;

            if     (type == kmNetKeyType::pkey) key = _nks.RegisterPkey(mac, addr);
            else if(type == kmNetKeyType::vkey) key = _nks.RegisterVkey(mac, pkey, vld_sec);

            // send key (nks -> svr)
            _ptc_nkey.SendKeyFromNks(addr, key);
        }
        else if(_mode == zbMode::svr)
        {
            // request, register and send vkey
            EnqProcVkey(src_id, vld_sec, vld_cnt);
        }
    };
    void RcvNkeyKey()
    {
    };
    void RcvNkeyReqAddr() // _rcv_key, _rcv_mac -> _snd_addr, _snd_mac
    {
        print("**** rcv request addr by %s\n", _ptc_nkey._rcv_key.GetStr().P());

        kmT2(_ptc_nkey._snd_addr, _ptc_nkey._snd_mac) = _nks.GetAddr(_ptc_nkey._rcv_key, _ptc_nkey._rcv_mac);
    };
    void RcvNkeyAddr()
    {
    };
    void RcvNkeySig() // only for nks
    {
        print("*** rcvnkeysig\n");

        if(_mode != zbMode::nks) return;        

        // update key
        int flg = _nks.Update(_ptc_nkey._rcv_key, _ptc_nkey._rcv_mac, _ptc_nkey._rcv_addr);

        if     (flg == 0) { print("*** key was not changed\n"); }
        else if(flg == 1) { print("*** key was changed\n"); SaveNks(); }

        if(flg >= 0) ReplySigFromNks(_ptc_nkey._rcv_addr, flg);
    };
    void RcvNkeyRepSig()
    {
        int flg = _ptc_nkey._rcv_sig_flg;

        if     (flg == 0) { print("*** key was not changed\n"); }
        else if(flg == 1) { print("*** key was changed\n");     }
    };
    void RcvNkeyReqAccs()
    {
        static kmAddr4 trg_addr; trg_addr = _ptc_nkey._rcv_addr;

        print("*** rcvnkeyreqaccs : %s\n", trg_addr.GetStr().P());

        // preconnect in work thread
        EnqPreconnect(trg_addr);
    };

    // virtual function for extra work to make thumbnails
    virtual void DoExtraWork()
    {
        // make thumbnails
        if(_mode == zbMode::svr)
        for(int uid = (int)_users.N1();            uid--;)
        for(int sid = (int)_users(uid).strgs.N1(); sid--;)
        {
            zbStrg& strg = _users(uid).strgs(sid);

            if(strg.type == zbStrgType::imgb)
            {            
                if(strg.chknfid < strg.files.N1())
                {
                    MakeThumb(uid, sid, strg.chknfid++);

                    if(strg.chknfid == strg.files.N1())
                    {
                        SaveStrgs(uid); 
                        SaveFileList(uid, sid);
                    }
                }                
            }
        }
    };

    /////////////////////////////////////////////////////////////
    // _ptc_data (ptc_id = 2) protocol functions

    // request info    ... data_id = 11
    void ReqInfo(ushort src_id)
    {
        print("************* reqinfo\n");

        uchar data_id = 11; kmNetBuf buf(256);

        buf << _mac << _name << (int)_mode << (int)_users.N1();

        Send(src_id, data_id, buf.P(), (ushort)buf.N1()); // no waiting
    };
    void RcvReqInfo(ushort src_id, kmNetBuf& buf)
    {
        print("************* rcvreqinfo\n");

        zbNetInfo info{}; info.src_id = src_id;

        buf >> info.mac >> info.name >> *(int*)&info.mode >> info.user_n;

        // send info
        if(_mode == zbMode::svr) EnqSendInfo(src_id);
    };

    // send my info to opposite    ...data_id = 0
    void SndInfo(ushort src_id)
    {
        print("************* sndinfo\n");

        uchar data_id = 0; kmNetBuf buf(256);

        buf << _mac << _name << (int)_mode << (int)_users.N1();

        Send(src_id, data_id, buf.P(), (ushort)buf.N1()); // no waiting
    };
    void RcvInfo(ushort src_id, kmNetBuf& buf)
    {
        print("************* rcvinfo\n");

        zbNetInfo info{}; info.src_id = src_id;

        buf >> info.mac >> info.name >> *(int*)&info.mode >> info.user_n;

        _rcvinfo_que_lck.Lock();   ///////////////////////// lock

        _rcvinfo_que.Enqueue(info);

        _rcvinfo_que_lck.Unlock(); ///////////////////////// unlock
    };

    // request registration (clt to svr)...data_id = 1
    void ReqRegist(ushort src_id, zbRole role, kmNetKey vkey = kmNetKey())
    {
        print("**** 2. ReqRegist : src_id(%d)\n", src_id);

        uchar data_id = 1; kmNetBuf buf(256);

        buf << _mac << _name << role << vkey;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1()); // no waiting
    };
    void RcvReqRegist(ushort src_id, kmNetBuf& buf)
    {
        print("**** 3. RcvReqRegist : src_id(%d)\n", src_id);

        // check mode
        if(_mode != zbMode::svr) return;

        // get from buffer
        kmMacAddr mac; kmStra name{}; zbRole role; kmNetKey vkey;

        buf >> mac >> name >> role >> vkey;

        // check if already registered
        if(FindUser(mac) >= 0) return;

        // check role
        if(role == zbRole::owner  && _users.N1() >  0) { print("*** there is already owner\n"); return; }
        if(role == zbRole::member && _users.N1() == 0) { print("*** there is no owner\n");      return; }

        // check vkey... under construction
        if(role != zbRole::owner)
        {
            int ivkey = -1;

            for(int n = (int)_vkeys.N1(); n--; ) if(_vkeys(n).key == vkey) { ivkey = n; break; }

            if(ivkey < 0) print("** vkey(%s) cannot be found\n", vkey.GetStr().P());
            else          print("** vkey(%s) is correct\n"     , vkey.GetStr().P());
        }

        // add users
        zbUser user(mac, name, role, _ids(src_id).addr, src_id);

        AddUser(user);
        SaveUsers();

        print("**** add user : src_id(%d)\n", src_id);

        // accept the opposite        
        EnqAcceptRegist(src_id, role);

        NotiUser((int)_users.N1() - 1);
    };

    // accept registration (svr to clt)...data_id = 2
    void AcpRegist(ushort src_id, zbRole role)
    {
        print("**** 4. AcpRegist : src_id(%d), role(%d)\n", src_id, (int)role);

        uchar data_id = 2; kmNetBuf buf(256);

        buf << _mac << _name << role << _pkey;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1()); // no waiting
    };
    void RcvAcpRegist(ushort src_id, kmNetBuf& buf)
    {
        print("**** 5. RcvAcpRegist : src_id(%d)\n", src_id);

        // check mode
        if(_mode != zbMode::clt) return;

        // get from buffer
        kmMacAddr mac; kmStra name; zbRole myrole; kmNetKey pkey;

        buf >> mac >> name >> myrole >> pkey;

        // check
        if(FindUser(mac) >= 0) return; // already registered

        // set role
        zbRole role = zbRole::family;

        switch(myrole)
        {
        case zbRole::owner  : role = zbRole::ownsvr; break;
        case zbRole::member : role = zbRole::svr;    break;
        case zbRole::family : role = zbRole::family; break;
        case zbRole::guest  : role = zbRole::guest;  break;
        }

        // add users        
        zbUser user(mac, name, role, _ids(src_id).addr, src_id, pkey);

        print("** user was added\n"); user.PrintInfo();

        AddUser(user);
        SaveUsers();

        NotiUser((int)_users.N1() - 1);
    };

    // notify... data_id = 3
    void Notify(ushort src_id, zbNoti noti)
    {
        uchar data_id = 3; kmNetBuf buf(8);

        buf << noti;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };
    void RcvNotify(ushort src_id, kmNetBuf& buf)
    {
        zbNoti noti; int uid = FindUser(src_id);

        buf >> noti;

        switch(noti)
        {
        case zbNoti::none : print("* receive zbNot::none\n"); break;
        }
    };

    // request file... data_id = 4
    //   opt : 0 (file), 1(cache)    
    void ReqFile(int uid, int sid, int fid_s, int fid_e = -1, uchar cache = 0)
    {
        const ushort src_id = _users(uid).src_id;
        
        uchar data_id = 4;  kmNetBuf buf(32);

        buf << sid << fid_s << fid_e << cache;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };    
    void RcvReqFile(ushort src_id, kmNetBuf& buf)
    {
        int sid, fid_s, fid_e; uchar cache;

        buf >> sid >> fid_s >> fid_e >> cache;

        // get uid
        int uid = FindUser(src_id); 

        // check id
        if(!CheckId(uid, sid, fid_s)) return;

        // set fid_e
        const int file_n = (int)GetFiles(uid,sid).N1();

        fid_e = MIN(MAX(fid_s, fid_e), file_n - 1);

        print("*** rcvreqfile : uid(%d) sid(%d) fid(%d,%d)\n", uid, sid, fid_s, fid_e);

        // send files
        for(int fid = fid_s; fid <= fid_e; ++fid)
        {
            if(cache == 0) SendFile     (uid, sid, fid);
            else           SendFileCache(uid, sid, fid);
        }
    };    

    // delete file on svr... date_id = 5
    void DelFile(int uid, int sid, int fid)
    {
        // check id
        if(!CheckId(uid, sid, fid)) return;

        const ushort src_id = _users(uid).src_id; if(src_id >= _ids.N1()) return;

        // send 
        uchar data_id = 5; kmNetBuf buf(8);

        buf << sid << fid;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };
    void RcvDelFile(ushort src_id, kmNetBuf& buf)
    {    
        int uid = FindUser(src_id), sid, fid;

        buf >> sid >> fid;

        // check id
        if(!CheckId(uid, sid, fid)) return;

        // delete file
        zbFile& file  = _users(uid).strgs(sid).files(fid);

        if(file.state == zbFileState::bkupno || file.state == zbFileState::bkup)
        {    
            kmFile::Remove(GetFullPath(uid, sid, fid).cuw());
        }
        file.state = zbFileState::none;

        // save file list
        SaveFileList(uid, sid);

        // send ack
        AckDelFile(uid, sid, fid);
    };

    // ack for delfile... date_id = 6
    void AckDelFile(int uid, int sid, int fid)
    {
        // check
        if(!CheckId(uid, sid, fid)) return;

        const ushort src_id = _users(uid).src_id; if(src_id >= _ids.N1()) return;

        // send
        uchar data_id = 6; kmNetBuf buf(8);

        buf << sid << fid;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };
    void RcvAckDelFile(ushort src_id, kmNetBuf& buf)
    {
        int uid = FindUser(src_id), sid, fid;

        buf >> sid >> fid;

        // check id
        if(!CheckId(uid, sid, fid)) return;

        // modifie file state
        zbFileState& state = _users(uid).strgs(sid).files(fid).state;

        if     (state == zbFileState::deletednc) { state = zbFileState::deleted; SaveFileList(uid, sid); }
        else if(state == zbFileState::bkupbannc) { state = zbFileState::bkupban; SaveFileList(uid, sid); }
    };

    // request thumbnail image... data_id = 7
    void ReqThumb(int uid, int sid, int fid_s, int fid_e, int w_pix = 0, int h_pix = 0)
    {
        // check
        if(!CheckId(uid, sid, fid_s)) return;

        const ushort src_id = _users(uid).src_id; if(src_id >= _ids.N1()) return;

        // send
        uchar data_id = 7; kmNetBuf buf(24);

        buf << sid << fid_s << fid_e << w_pix << h_pix;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 1200.f);
    };
    void RcvReqThumb(ushort src_id, kmNetBuf& buf)
    {
        int uid = FindUser(src_id), sid, fid_s, fid_e, w_pix, h_pix;

        buf >> sid >> fid_s >> fid_e >> w_pix >> h_pix;

        // check id
        if(!CheckId(uid, sid, fid_s)) return;

        // set fid_e
        const int file_n = (int)GetFiles(uid,sid).N1();

        fid_e = MIN(MAX(fid_s, fid_e), file_n - 1);

        print("*** rcvreqthumb : uid(%d) sid(%d) fid(%d,%d)\n", uid, sid, fid_s, fid_e);

        for(int fid = fid_s; fid <= fid_e; ++fid)
        {
            SendThumb(uid, sid, fid);
        }
    };

    // request list (user, strg, file)... data_id = 8
    //    if sid   < 0 --> request strg list
    void ReqList(zbListType type, int uid, int sid = 0)
    {
        // check
        if(uid < 0 || uid >= _users.N1()) return;

        const ushort src_id = _users(uid).src_id; if(src_id >= _ids.N1()) return;

        // send
        uchar data_id = 8;  kmNetBuf buf(32);

        buf << type << sid;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };    
    void RcvReqList(ushort src_id, kmNetBuf& buf)
    {
        // check
        int uid = FindUser(src_id); if(uid < 0) return;

        // rcv buf
        zbListType type; int sid;

        buf >> type >> sid;

        switch(type)
        {
        case zbListType::user : break;
        case zbListType::strg : break;
        case zbListType::file : break;
        }
    };

    // send text (utf-8)... data_id = 9
    void SndTxt(int uid, const kmStra& txt)
    {
        // check
        if(uid < 0 || uid >= _users.N1()) return;

        const ushort src_id = _users(uid).src_id; if(src_id >= _ids.N1()) return;

        if(txt.Byte() > 1024*10)
        {
            print("* SndTxt in 833 : text is too long\n"); return;
        }
        // send
        uchar data_id = 9;

        Send(src_id, data_id, txt.P(), (ushort)txt.Byte(), 100);
    };
    void RcvTxt(ushort src_id, kmNetBuf& buf) {};

    // send json (utf-8)... data_id = 10
    //   return value : -1 (fail), ack_id (ushort)
    int SndJson(int uid, const kmStra& json, ushort ack_id = 0)
    {
        // check
        if(uid < 0 || uid >= _users.N1()) return -1;

        const ushort src_id = _users(uid).src_id; if(src_id >= _ids.N1()) return -1;

        if(json.Byte() > 1024*10)
        {
            print("* SndJson in 855 : text is too long\n"); return -1;
        }
        // send
        uchar data_id = 10;

        return (int) Send(src_id, data_id, json.P(), (ushort)json.Byte(), 200.f, 3, ack_id);
    };
    void RcvJson(ushort src_id, kmNetBuf& buf)
    {
        // get jsna from _jsnbuf
        ushort ack_id = _ptc_data.GetLastAckId();

        int idx = _jsnbuf.FindPut(ack_id, kmStra((char*)buf.P()));

        // enqueue json
        kmLockGuard grd = _jsnbuf.GetLock().Lock();

        kmStra& jsna = _jsnbuf.GetJsna(idx);

        EnqParseJson(src_id, jsna.P(), (int)jsna.N1(), ack_id);
    };

    // util function
    string GetFirstParam(string& url);
    bool   CheckDigit(const string& str);
    int    GetAlreayExistUserOwnerAlbum(int uid);
    string getStrFileType(const zbFileType& type);
    void   SetFileInfoToJson(int uuid, int uid, int sid, int fid, json& j, bool sharedName = true);
    bool   GetRangeIndex(string& range, int& startIdx, int& endIdx);

    // parse json
    //   ret : 0 (failure), 1 (success)
    void    ParseJson(ushort src_id, char* jsnap, int n, ushort ack_id = 0);
    void EnqParseJson(ushort src_id, char* jsnap, int n, ushort ack_id = 0)
    {
        EnqueueWork(4, src_id, jsnap, n, ack_id); 
    };

    // parse json
    void ParseJsonGet   (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonPut   (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonChange(int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonDel   (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonRep   (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonCreate(int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonUpload(int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonReset (int uid, kmStra& jsna, string& url, json& body);

    // parse json user list
    void ParseJsonGetUserList(int uid, kmStra& jsna, string& url, json& body);

    // parse json shared album
    // create
    void ParseJsonCreateShrd(int uid, kmStra& jsna, string& url, json& body);

    // get
    void ParseJsonGetShrd             (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonGetShrdAlbum        (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonGetShrdAlbumList    (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonGetShrdAlbumFile    (int uid,               string& url, json& body);
    void ParseJsonGetShrdAlbumDownload(int uid,               string& url, json& body);
    void ParseJsonGetShrdAlbumThumb   (int uid,               string& url, json& body);
    void ParseJsonGetShrdAlbumFileList(int uid,               string& url, zbShrd& shrd, json& body);
    void ParseJsonGetShrdMember       (int uid, kmStra& jsna, string& url, json& body);

    // put/add
    void ParseJsonPutShrd      (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonPutShrdMember(int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonPutShrdAlbum (int uid, kmStra& jsna, string& url, json& body);

    // delete
    void ParseJsonDelShrd          (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonDelShrdAlbum     (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonDelShrdAlbumFile(int uid, kmStra& jsna, int album_id, json& body);
    void ParseJsonDelShrdMember    (int uid, kmStra& jsna, string& url, json& body);
    // shared album end

    // change
    void ParseJsonChangeName(int uid, kmStra& jsna, string& url, json& body);

    // parse json partner
    // get
    void ParseJsonGetPartner     (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonGetPartnerAlbum(int uid, kmStra& jsna, string& url, json& body);

    // add
    void ParseJsonPutPartner(int uid, kmStra& jsna, string& url, json& body);

    // delete
    void ParseJsonDelPartner(int uid, kmStra& jsna, string& url, json& body);
    // partner end

    // upload start
    void ParseJsonUploadUser(int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonUploadShrd(int uid, kmStra& jsna, string& url, json& body);
    // upload end

    // profile
    void ParseJsonGetProfile     (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonGetProfileUser (int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonGetProfileAlbum(int uid, kmStra& jsna, string& url, json& body);

    // other command
    void ParseJsonGetMemoryInfo(int uid, kmStra& jsna, string& url, json& body);
    void ParseJsonGetMyInfo    (int uid, kmStra& jsna, string& url, json& body);
    // other command end

    // server -> client noti message start
    void SendNotiPartner(int src_uid, int trg_uid, string action);
    void SendNotiMember (int src_uid, int trg_uid, int album_id, string action);
    void SendNotiAlbum  (int src_uid, int trg_uid, int album_id, string action);
    void SendNotiFile   (int src_uid, int trg_uid, int album_id, string action);


    void NotiUser(int uid);
    void SendNotiUser(int sourceUid, int targetUid, string action);
    // noti message end

    // parse json user list  
    //void ParseJsonGetFile(int uid, kmStra& jsna, string& url, json& body);

    /////////////////////////////////////////////////////////////
    // network interface functions
public:
    /*
    // connect... core
    //  return : src_id (if connecting failed, it will be -1)
    int Connect(const kmAddr4 addr, const kmNetKey vkey = kmNetKey())
    {
        int src_id = kmNet::Connect(addr, _name, vkey, _cnnt_tout_msec);
    
        if(src_id >= 0) EnqSendInfo((ushort) src_id);
    
        return src_id;
    };

    // connect with vkey... core
    int Connect(const kmNetKey vkey)
    {
        int src_id = kmNet::Connect(vkey, _name, _port, _cnnt_tout_msec);

        if(src_id >= 0) EnqSendInfo((ushort) src_id);

        return src_id;
    };    

    // connect to new device in lan
    int ConnectNew()
    {
        // get addrs in lan
        kmAddr4s addrs; kmMacAddrs macs;

        int n = GetAddrsInLan(addrs, macs, _port);

        // connect to new devices
        int cnnt_n = 0;

        for(int i = 0; i < n; ++i)
        {
            if(FindId(macs(i)) > -1) continue; // check if already connected

            if(Connect(addrs(i)) < 0)
            {
                print("* connecting failed (to %s)\n", addrs(i).GetStr().P());
            }
            else ++cnnt_n;
        }
        return cnnt_n;
    };
    //*/

    // connect with addr or vkey ... core
    //  return : src_id (if connecting failed, it will be -1)
    int Connect(kmAddr4  addr) { return kmNet::Connect(addr, _name,        _cnnt_tout_msec); };
    int Connect(kmNetKey vkey) { return kmNet::Connect(vkey, _name, _port, _cnnt_tout_msec); };

    // connect with uid and addr... core
    //  return : src_id (if connecting failed, it will be -1)
    int Connect(int uid, kmAddr4 addr)
    {   
        const ushort src_id = GetSrcId(uid);

        if(src_id == 0xffff) return kmNet::Connect  (  addr, _name, _cnnt_tout_msec);
        else                 return kmNet::Reconnect(src_id,  addr, _cnnt_tout_msec);
    };

    // find new zib server
    //  return : the number of new zib servers which have not been registered yet
    int FindNewSvr(zbNetInfos& infos)
    {
        // init parameters
        int new_n = 0;
        
        // get addrs in lan
        kmAddr4s addrs; kmMacAddrs macs;

        int n = GetAddrsInLan(addrs, macs, _port);

        if(n == 0) return 0;

        // ask if new server
        for(int i = 0; i < n; ++i)
        {
            if(FindUser(macs(i)) > -1) continue; // check if already registered

            int src_id = FindId(macs(i));

            if(src_id > -1) // check if alread connected
            {
                RequestInfo(src_id); ++new_n;
            }
            else
            {
                src_id = Connect(addrs(i));

                if(src_id < 0) print("* cnnt failed (to %s)\n", addrs(i).GetStr().P());
                else           ++new_n;
            }
        }        
        if(new_n == 0) return 0;

        infos.Recreate(0,new_n);

        // wait for rcvinfo, timeout is 1sec
        kmTimer time(1);

        while(GetRcvInfoN() < new_n && time.sec() < 0.3) { Sleep(50); };
        
        // set info
        for(int n = GetRcvInfoN(); n--;)
        {
            zbNetInfo info = GetRcvInfo();

            if(info.mac.i64 != 0) infos.PushBack(info);
        }
        return (int)infos.N1();
    };

    // connect to new server as owner
    void ConnectNewSvrAsOwner(zbNetInfo info)
    {
        EnqRequestRegist(info.src_id, zbRole::owner, kmNetKey());
    };

    // connect to new server as member
    void ConnectNewSvrAsMember(kmNetKey vkey)
    {
        int src_id = Connect(vkey);

        if(src_id > -1) EnqRequestRegist(src_id, zbRole::member, vkey);
    };

    // connect to new server as member only for test
    void __ConnectNewSvrAsMember(kmAddr4 addr)
    {
        int src_id = Connect(addr);

        if(src_id > -1) EnqRequestRegist(src_id, zbRole::member, zbTestVkey);
    };

    // connect with uid
    //  return : src_id (if connecting failed, it will be -1)
    //  return : -1 (failed), 0 (last), 1 (inLan), 2 (inWan)
    int Connect(int uid)
    {
        int ret;

        print("** 1\n"); ret = ConnectLastAddr(uid); if(ret >= 0) return 0;
        print("** 2\n"); ret = ConnectInLan   (uid); if(ret >= 0) return 1;
        print("** 3\n"); ret = ConnectInWan   (uid); if(ret >= 0) return 2;

        print("** connecting fails\n");

        return -1;
    };

    // connect with last connected address
    //  return : src_id (if connecting failed, it will be -1)
    int ConnectLastAddr(int uid)
    {
        if(uid >= _users.N1()) return -1;

        print("** connect to the last address (%s)\n", _users(uid).addr.GetStr().P());

        return Connect(uid, _users(uid).addr);
    };

    // connect in lan
    //  return : src_id (if connecting failed, it will be -1)
    int ConnectInLan(int uid)
    {
        if(uid >= _users.N1()) return -1;

        // get addrs in lan
        kmAddr4s addrs; kmMacAddrs macs;

        int n = GetAddrsInLan(addrs, macs, _port);

        print("** get addrs (%d) in lan\n", n);

        // find uid with mac and connect
        for(int i = 0; i < n; ++i) if(macs(i) == _users(uid).mac)
        {
            print("** connect to %s in lan\n", addrs(i).GetStr().P());

            return Connect(uid, addrs(i));
        }
        return -1;
    };

    // connect in wan
    //  return : src_id (if connecting failed, it will be -1)
    int ConnectInWan(int uid)
    {
        if(uid >= _users.N1()) return -1;

        // check key
        if(!_users(uid).key.IsValid()) return -1;

        // request addr with key
        kmAddr4 addr = RequestAddrToNks(_users(uid).key, _users(uid).mac);

        if(!addr.IsValid()) return -1;

        print("** connect to %s in wan\n", addr.GetStr().P());

        // send precnnt
        int try_n; kmAddr4 rcv_addr;

        kmT2(try_n, rcv_addr) = SendPreCnnt(addr, _precnnt_tout_msec, _precnnt_try_n);

        if(try_n == 0)
        {
            print("** sendprecnnt : no answer\n"); return -1;
        }
        print("** sendprecnnt : get answer after %d times from %s\n", try_n, rcv_addr.GetStr().P());

        // connect
        return Connect(uid, rcv_addr);
    };

    // connect in wan with other port (new port = old port + 1)
    int ConnectInWanWithOtherPort(int uid)
    {
        ChangePort();

        print("** new port is %d\n", _addr.GetPort());

        return ConnectInWan(uid);
    };

    // preconnect... only for rcvreqaccs
    int Preconnect(kmAddr4 addr)
    {
        int try_n; kmAddr4 rcv_addr;

        kmT2(try_n, rcv_addr) = SendPreCnnt(addr, _accs_tout_msec, _accs_try_n);

        if(try_n == 0) print("** preconnect : no answer\n");
        else           print("** preconnect : get answer after %d times from %s\n", try_n, rcv_addr.GetStr().P());

        return try_n;
    };
    void EnqPreconnect(kmAddr4 addr) { EnqueueWork(3, addr); };

    // check if connected
    bool IsConnected(int uid)
    {
        // check src_id
        const ushort src_id = GetSrcId(uid);

        if(src_id == 0xffff) return false;

        // check state
        if(_ids(src_id).state == kmNetIdState::invalid) return false;

        // check time
        if(_ids(src_id).time.GetPassSec() < 3) return true;

        // send sig
        const float ec_msec = kmNet::SendSig(src_id, 300.f, 3);

        return (ec_msec >=0) ? true : false;
    };

    // send data through ptc_data including connection checking
    // 
    //   tout_msec : -1 (not wait for ack), > 0 (wait for ack)
    int Send(ushort src_id, uchar data_id, char* data, ushort byte, 
             float tout_msec = 0.f, int retry_n = 1, ushort ack_id = 0)
    {
        return kmNet::Send(src_id, data_id, data, byte, tout_msec, retry_n, ack_id);
    };

    // sleep in worker
    void EnqSleep(int msec) { EnqueueWork(6, msec); };

    // test network condition
    void EnqTestNetCnd(ushort src_id) { EnqueueWork(5, src_id); };

    // request info
    void RequestInfo(ushort src_id) { ReqInfo(src_id); };

    // send info
    void SendInfo   (ushort src_id) { SndInfo(src_id); };
    void EnqSendInfo(ushort src_id) { EnqueueWork(0, src_id); };

    // request registration to svr
    void RequestRegist(ushort src_id, zbRole role, kmNetKey vkey)
    {
        if(_mode == zbMode::clt) ReqRegist(src_id, role, vkey);
    };
    void EnqRequestRegist(ushort src_id, zbRole role, kmNetKey vkey)
    {
        EnqueueWork(1, src_id, role, vkey);
    };

    // accept registration
    void AcceptRegist(ushort src_id, zbRole role)
    {
        if(_mode == zbMode::svr) AcpRegist(src_id, role);
    };
    void EnqAcceptRegist(ushort src_id, zbRole role) { EnqueueWork(2, src_id, role); };

    // request vkey (svr -> nks), register vkey (svr) and send vkey (svr -> clt)
    void ProcVkey(ushort src_id, uint vld_sec, uint vld_cnt)
    {
        // request vkey (svr -> nks)
        kmNetKey vkey = kmNet::RequestVkeyToNks(_pkey, vld_sec, vld_cnt);

        print("*** [ProcVeky] %s\n", vkey.GetStr().P());

        if(vkey.IsInvalid())
        {
            print("*** [ProcVkey] failed to get vkey from nks\n"); return;
        }
        // register vkey table (svr)
        zbVkeyElm elm = {vkey, kmDate(time(NULL) + vld_sec), vld_cnt, -1};

        _vkeys.PushBack(elm);

        // send vkey (svr -> clt)
        kmNet::SendKey(src_id, vkey);
    };
    void EnqProcVkey(ushort src_id, uint vld_sec, uint vld_cnt)
    {
        EnqueueWork(7, src_id, vld_sec, vld_cnt); 
    };
        
    // send txt with utf8
    void SendTxt(int uid, const kmStra& txt) { SndTxt(uid, txt); };

    // send json with utf8
    void SendJson(int uid, const kmStra& json) { SndJson(uid, json); };

    // send json with utf8 and waiting for rep
    string SendJsonSync(int uid, const kmStra& json)
    {
        int ack_id = SndJson(uid, json);

        // regist to buffer
        _jsnbuf.Regist(ack_id);

        // wait for rep
        const float tout_msec = 1000;

        for (kmTimer time(1); time.msec() < tout_msec; Sleep(1))
        {
            kmStra jsna = _jsnbuf.FindGetDel(ack_id);

            if (jsna.N1() > 0)
            {
                print("*** received rep (%d)\n", ack_id);
                return string(jsna.P());
            }
        }

        // timeout
        _jsnbuf.FindDel(ack_id);

        print("*** didn't receive rep (%d)\n", ack_id);
        return "{\"result_code\":500}";
    };

    // send json for reply with utf8
    void SendJsonRep(int uid, const kmStra& json)
    {
        SndJson(uid, json, _jsn_snd_ack_id);
    };

    /////////////////////////////////////////////////////////////
    // setting functions

    // save setting.
    void SaveSetting()
    {
        kmFile file(kmStrw(L"%s/.zbnetsetting", _path.cuw().P()).P(), KF_NEW);

        file.Write(&_pkey);
    };

    // load setting
    int LoadSetting() try
    {
        kmFile file(kmStrw(L"%s/.zbnetsetting", _path.cuw().P()).P());

        file.Read(&_pkey);

        return 1;
    }
    catch(kmException) { return 0; };

    /////////////////////////////////////////////////////////////
    // user control functions

    // find with mac in _users
    //   return   <  0 : there is no user which has the mac.
    //            >= 0 : uid
    int FindUser(kmMacAddr mac)
    {
        for(int i = (int)_users.N1(); i--;) if(_users(i).mac == mac) return i;
        return -1;
    };

    // find with src_id in _users
    //   return   <  0 : there is no user which has the src_id
    //            >= 0 : uid
    int FindUser(ushort src_id)
    {
        for(int i = (int)_users.N1(); i--; ) if(_users(i).src_id == src_id) return i;
        return -1;
    };

    // find with user's role
    //   return   <  0 : there is no owner
    //            >= 0 : uid
    int FindUser(zbRole role)
    {
        for(int i = (int)_users.N1(); i--; ) if(_users(i).role == role) return i;
        return -1;
    };

    // add user
    void AddUser(zbUser& user)
    {
        if(_users.Size() == 0) _users.Recreate(0,8);

        // set path
        user.path.SetStr("%s/user%d", _path.P(), (int)_users.N1());

        // add user
        int uid = (int)_users.PushBack(user);

        // make user folder
        kmFile::MakeDir(user.path.cuw().P());

        // add storage
        AddStrg(uid, zbStrgType::imgb, user.IsSvr() ? _srcpath : kmStra());

        // add link storage ... only test for imgl and imgs
        //if(user.role == zbRole::member) // svr
        //{
        //    AddStrgLnk(uid, 0, 0);
        //}
        //else if(user.role == zbRole::svr) // clt
        //{
        //    AddStrg(uid, zbStrgType::imgs, _dwnpath);
        //}

        // save strg.list
        SaveStrgs(uid);
    };

    // save users
    void SaveUsers()
    {
        if(_users.N1() == 0) return;
        
        kmFile file(kmStra("%s/.userlist", _path.P()).cuw().P(), KF_NEW);

        file.WriteMat(&_users.Pack());

        for(int i = 0; i < _users.N1(); ++i)
        {    
            file.WriteMat(&_users(i).name);
            file.WriteMat(&_users(i).path);
        }
    };

    // load users.. .return value is num of user
    int LoadUsers() try
    {
        kmFile file(kmStra("%s/.userlist", _path.P()).cuw().P());

        file.ReadMat(&_users);

        // restore and read sub-mats
        for(int i = 0; i < _users.N1(); ++i)
        {
            file.ReadMat(&_users(i).name.Restore());
            file.ReadMat(&_users(i).path.Restore());

            _users(i).strgs.Restore();            
            _users(i).src_id = -1;
        }
        return (int)_users.N1();
    }
    catch(kmException) { return 0; };

    /////////////////////////////////////////////////////////////
    // storage control functions

    // add storage
    int AddStrg(int uid, zbStrgType type, const kmStra& srcpath)
    {
        // init variables
        zbStrgs&  strgs  = _users(uid).strgs;
        const int strg_n = (int)strgs.N1();

        if(strg_n == 0) strgs.Recreate(0,4);

        // set strg
        zbStrg strg = {type, kmStra("strg%d",strg_n)};

        strg.path.SetStr("%s/%s", _users(uid).path.P(), strg.name.P());

        if(srcpath.N1() > 1) strg.srcpath = srcpath;
        else                 strg.srcpath = strg.path;

        // make strg folder
        kmFile::MakeDir(strg.path.cuw().P());

        // add strg
        return (int)strgs.PushBack(strg);
    };

    // add link storage
    int AddStrgLnk(int uid, int lnk_uid, int lnk_sid)
    {
        // init variables
        zbStrgs&  strgs  = _users(uid).strgs;
        const int strg_n = (int)strgs.N1();

        if(strg_n == 0) strgs.Recreate(0,4);

        // set strg
        zbStrg strg = {zbStrgType::imgl, kmStra("strg%d",strg_n)};

        strg.path    = _users(lnk_uid).strgs(lnk_sid).path;
        strg.srcpath = _users(lnk_uid).strgs(lnk_sid).srcpath;

        strg.lnk_uid = lnk_uid;
        strg.lnk_sid = lnk_sid;

        // add strg
        return (int)strgs.PushBack(strg);
    }

    // save strgs list
    void SaveStrgs(int uid)
    {
        // init variables
        zbStrgs& strgs = _users(uid).strgs;

        if(strgs.N1() == 0) return;

        // save file
        kmFile file(kmStra("%s/.strglist", _users(uid).path.P()).cuw().P(), KF_NEW);

        print("** start to save strglists\n");

        file.WriteMat(&strgs.Pack());

        for(int i = 0; i < strgs.N1(); ++i)
        {
            file.WriteMat(&strgs(i).name);
            file.WriteMat(&strgs(i).path);
            file.WriteMat(&strgs(i).srcpath);
        }        
        print("** finish to save strglists\n");
    };

    // load strgs list... return value is num of strg
    int LoadStrgs(int uid) try
    {
        // init variables
        zbStrgs& strgs = _users(uid).strgs;

        // load file
        kmFile file(kmStra("%s/.strglist", _users(uid).path.P()).cuw().P());

        file.ReadMat(&strgs);

        // restore and read sub-mats
        for(int i = 0; i < strgs.N1(); ++i)
        {
            file.ReadMat(&strgs(i).name   .Restore());
            file.ReadMat(&strgs(i).path   .Restore());
            file.ReadMat(&strgs(i).srcpath.Restore());

            strgs(i).files.Restore();

            for(int k = 0; k < strgs(i).files.N1(); ++k)
            {
                strgs(i).files(k).name.Restore();
            }
        }
        return (int)strgs.N1();
    }
    catch(kmException e) { kmPrintException(e); return 0; };

    // get number of storage for every users
    int GetAllStrgN()
    {
        int strg_n = 0;

        for(int uid = 0; uid < _users.N1(); ++uid)
        {
            strg_n += (int)_users(uid).strgs.N1();
        }
        return strg_n;
    };

    /////////////////////////////////////////////////////////////
    // shrd list control functions

    // add shrd... return hid
    int AddShrd(zbShrd& shrd)
    {
        if(_shrds.Size() == 0) _shrds.Recreate(0,8);

        // set created time
        shrd.time.SetCur();

        // find deleted shrd
        int hid = 0; 
        
        for(; hid < _shrds.N1(); ++hid) if(_shrds(hid).state == zbShrdState::deleted) break;
        
        // add shrd
        if(hid == _shrds.N1()) _shrds.PushBack(shrd);
        else                   _shrds(hid) = shrd;

        // save shrd list
        SaveShrdList();

        return hid;
    };

    // delete shrd
    void DeleteShrd(int hid)
    {
        if(hid >= _shrds.N1()) return;

        // set state
        _shrds(hid).state = zbShrdState::deleted;
        _shrds(hid).mmbs.Init();
        _shrds(hid).itms.Init();

        // delete file
        kmStra path("%s/shrd/.shrd%d", _path.P(), hid);

        kmFile::Remove(path.cuw().P());
    };

    // save shrd list including shrd
    void SaveShrdList()
    {
        if(_shrds.N1() == 0) return;

        // save shrdlist
        kmFile file(kmStra("%s/.shrdlist", _path.P()).cuw().P(), KF_NEW);

        file.WriteMat(&_shrds.Pack());

        for(int i = 0; i < _shrds.N1(); ++i)
        {
            file.WriteMat(&_shrds(i).name);
        }
        file.Close();

        // save shrd
        for(int i = 0; i < _shrds.N1(); ++i) SaveShrd(i);
    };

    // load shrd list... return value is number of shrd
    int LoadShrdList() try
    {
        // read shrdlist
        kmFile file(kmStra("%s/.shrdlist", _path.P()).cuw().P());

        file.ReadMat(&_shrds);

        // resotre and read sub-mats
        for(int i = 0; i < _shrds.N1(); ++i)
        {
            file.ReadMat(&_shrds(i).name.Restore());

            _shrds(i).mmbs.Restore();
            _shrds(i).itms.Restore();            
        }
        file.Close();

        // load shrd        
        for(int i = 0; i < _shrds.N1(); ++i) LoadShrd(i);

        // print shrd
        PrintShrdList();

        return (int)_shrds.N1();
    }
    catch(kmException) { return 0; };

    // save shrd
    void SaveShrd(int hid)
    {
        if(hid >= _shrds.N1()) return;

        if(_shrds(hid).state == zbShrdState::deleted) return;

        // make directory for shrd
        kmStrw path(L"%s/shrd", _path.cuw().P());

        kmFile::MakeDir(path.P());

        // save shrd
        const zbShrd& shrd = _shrds(hid);

        kmFile file(kmStrw(L"%s/.shrd%d", path.P(), hid).P(), KF_NEW);

        file.Write(&shrd);

        file.WriteMat(&shrd.name);
        file.WriteMat(&shrd.mmbs);
        file.WriteMat(&shrd.itms);
    };

    // load shrd
    void LoadShrd(int hid) try
    {
        if(hid >= _shrds.N1()) return;

        if(_shrds(hid).state == zbShrdState::deleted) return;

        // load shrd
        zbShrd& shrd = _shrds(hid);

        kmFile file(kmStra("%s/shrd/.shrd%d", _path.P(), hid).cuw().P()); 

        file.Read(&shrd);

        file.ReadMat(&shrd.name.Restore());
        file.ReadMat(&shrd.mmbs.Restore());
        file.ReadMat(&shrd.itms.Restore());
    }
    catch(kmException) {};

    // print shrd list
    void PrintShrdList()
    {
        for(int i = 0; i < _shrds.N1(); ++i) PrintShrdList(i);
    };

    // print shrd list
    void PrintShrdList(int hid)
    {
        const zbShrd& shrd = _shrds(hid);

        print("\n* shrd list (hid: %d)\n", hid);
        print("   type  : %s\n", shrd.GetTypeStr ().P());
        print("   state : %s\n", shrd.GetStateStr().P());
        print("   name  : %s\n", shrd.name         .P());
        print("   owner : %d\n", shrd.owner_uid);
        print("   time  : %s\n", shrd.time.GetStrPt().P());
        print("   member: ");

        for(int i = 0; i < shrd.mmbs.N1(); ++i)
        {
            print("%d ", shrd.mmbs(i).uid);
        }
        print("\n   item  : uid, sid, fid\n");

        for(int i = 0; i < shrd.itms.N1(); ++i)
        {
            const zbShrdItm& itm = shrd.itms(i);

            print("            %d    %d    %d\n", itm.uid, itm.sid, itm.fid);
        }
    };

    /////////////////////////////////////////////////////////////
    // file list control functions
    
    // update files of every user's every storage and save file list (no sending files to svr)
    void UpdateFile()
    {
        const int usr_n = (int)_users.N1(); if(usr_n == 0) return;

        kmStra buf(1024);

        for(int uid = 0; uid < usr_n; ++uid)
        {
            if(!_users(uid).IsSvr()) continue;

            const int stg_n = (int)_users(uid).strgs.N1();

            for(int sid = 0; sid < stg_n; ++sid)
            {
                // reset isin flag
                zbFiles& files  = GetFiles(uid, sid);
                int      file_n = (int) files.N1();

                for(int i = 0; i < file_n; ++i) files(i).flg.isin = 0;
                
                // add file
                int add_n = UpdateFile(uid, sid, buf);

                // change deleted file's state
                int del_n = 0;

                for(int i = 0; i < file_n; ++i)
                if(files(i).flg.isin == 0 && files(i).state == zbFileState::bkup)
                {
                    files(i).state = zbFileState::bkuponly; del_n++;
                }

                // save file list
                print("** update file : added (%d), deleted (%d)\n", add_n, del_n);

                if(add_n > 0 || del_n > 0) SaveFileList(uid, sid);
            }
        }
    };

    // update file
    // * Notet that subpath is for recursive function. 
    // * subpath must be an empty string allocated with sufficient size
    int UpdateFile(int uid, int sid, kmStra& subpath)
    {
        // check if user is svr
        if(!_users(uid).IsSvr()) return 0;

        // get files        
        zbStrg&  strg  = _users(uid).strgs(sid);
        zbFiles& files = strg.files;

        // get file list
        kmFileList flst(kmStra("%s/%s*.*", strg.srcpath.P(), subpath.P()).cuw().P());

        //flst.Print();

        // add file and check if deleted
        int add_n = 0, files_n = (int)files.N1(), flst_n = (int)flst.N1();

        kmMat1u8 flg(files_n); flg.SetZero();

        for(int i = 0; i < flst_n; ++i)
        {
            const kmFileInfo& flsti = flst(i);
            const kmStra name = flsti.name.cu(); // utf8
            
            if(flsti.IsRealDir())
            {
                kmStra path("%s%s/", subpath.P(), name.P());

                (subpath += name) += "/";
                
                add_n += UpdateFile(uid, sid, subpath);

                subpath.Cutback(name.GetLen()); // including L'/'
            }
            else if(flsti.IsNormal())
            {
                // check if flst(i) is already in flies
                int isin = 0; subpath += name;

                for(int k = 0; k < files_n; ++k)
                {
                    if(subpath == files(k).name)
                    {
                        files(k).flg.isin = isin = 1; break; 
                    }
                }
                if(isin == 0)
                {
                    // add file to files
                    zbFile file = {zbFileType::image};

                    file.name  = subpath;
                    file.date  = flsti.date;
                    file.state = zbFileState::bkupno;

                    int fid = AddFile(uid, sid, file); ++add_n;
                }
                subpath.Cutback(name.GetLen()-1);
            }
        }
        return add_n;
    };

    // send bkupno files
    void SendBkupno(bool sync = false)
    {
        const int usr_n = (int)_users.N1(); if(usr_n == 0) return;

        for(int uid = 0; uid < usr_n; ++uid)
        {    
            const int stg_n = (int)_users(uid).strgs.N1();

            for(int sid = 0; sid < stg_n; ++sid)
            {
                // send files
                int snd_n = SendBkupno(uid, sid);

                // save file list
                if(snd_n > 0) SaveFileList(uid, sid);
            }
        }
        // wait for sending to finish
        if(sync) while(GetSndQueN() > 0) { Sleep(10); }
    };
    int SendBkupno(int uid, int sid)
    {
        int snd_n = 0;

        zbFiles& files = _users(uid).strgs(sid).files; int file_n = (int)files.N1();

        for(int fid = 0; fid < file_n; ++fid)
        {
            if(files(fid).state == zbFileState::bkupno)
            {
                SendFile(uid,sid,fid); ++snd_n;
            }
        }
        return snd_n;
    };
    
    // add new file to file list ... core
    //  return : fid
    int AddFile(int uid, int sid, zbFile& file)
    {
        print("** add file (%d,%d) : %s\n", uid, sid, file.name.P());

        // init
        zbFiles& files = _users(uid).strgs(sid).files;

        // add file
        if(files.Size() == 0) files.Recreate(0,8);

        // read file info
        if(file.type == zbFileType::image || file.type == zbFileType::movie)
        {
            const kmStra& path  = _users(uid).strgs(sid).srcpath;

            kmMdf mdf(kmStra("%s/%s", path.P(), file.name.P()).cuw().P());

            if(mdf._date != kmDate()) file.date = mdf._date;
            if(mdf._gps  != kmGps())  file.gps  = mdf._gps;

            // set file type
            if     (mdf._type == kmMdfType::jpg) file.type = zbFileType::image;
            else if(mdf._type == kmMdfType::png) file.type = zbFileType::image;
            else if(mdf._type == kmMdfType::mp4) file.type = zbFileType::movie;
            else                                 file.type = zbFileType::data;
        }
        return (int)files.PushBack(file);
    };

    // add file to file list
    void AddFile(int uid, int sid, int fid, zbFile& file)
    {
        // init
        zbFiles& files  = _users(uid).strgs(sid).files;
        int      file_n = (int)files.N1();

        if(fid < file_n) // old one
        {
            print("** add file (%d,%d,%d) : %s\n", uid, sid, fid, file.name.P());

            if(files(fid).state == zbFileState::bkuponly) // downloaded from svr
            {
                // check file
                if(file.name != files(fid).name)
                {
                    print("* [zbNet::AddFile in 782] file.name(%s) is not same with (%s)\n",
                           file.name.P(), files(fid).name.P());
                    return;
                }
                files(fid).state = zbFileState::bkup;
            }
            else if(files(fid).state == zbFileState::none)
            {
                const kmStra& path = _users(uid).strgs(sid).srcpath;

                kmMdf mdf(kmStra("%s/%s", path.P(), file.name.P()).cuw().P());

                if(mdf._type != kmMdfType::unknown)
                {
                    file.date = mdf._date;
                    file.gps  = mdf._gps;
                }
                files(fid) = file;
            }
        }
        else // new one
        {
            for(;file_n < fid; ++file_n)
            {
                zbFile dummy = {zbFileType::dummy, zbFileState::none, {}, 0, "---"};
                AddFile(uid, sid, dummy);
            }
            AddFile(uid, sid, file);
        }
    };

    // delete file on clt    
    void DeleteFileClt(int uid, int sid, int fid)
    {
        // init parameters
        kmStra& path  = _users(uid).strgs(sid).srcpath;
        zbFile& file  = _users(uid).strgs(sid).files(fid);        

        // delete file
        if(file.state == zbFileState::bkup) // bkup --> bkuponly
        {
            kmFile::Remove(kmStra("%s/%s",path.P(), file.name.P()).cuw().P());

            file.state = zbFileState::bkuponly;
        }        
        else if(file.state == zbFileState::bkupban) // bkupban --> deleted
        {
            kmFile::Remove(kmStra("%s/%s",path.P(), file.name.P()).cuw().P());

            file.state = zbFileState::deleted;
        }
    };

    // delete file on both svr and clt
    void DeleteFileBoth(int uid, int sid, int fid, bool sync = false)
    {
        // init parameters
        kmStra& path  = _users(uid).strgs(sid).srcpath;
        zbFile& file  = _users(uid).strgs(sid).files(fid);

        // delete file on clt
        if(file.state == zbFileState::bkupno    || file.state == zbFileState::bkup || 
           file.state == zbFileState::bkupbannc || file.state == zbFileState::bkupban)
        {
            kmFile::Remove(kmStra("%s/%s",path.P(), file.name.P()).cuw().P());
        }        

        // delete file on svr
        if(file.state == zbFileState::bkup     || file.state == zbFileState::deletednc ||
           file.state == zbFileState::bkuponly || file.state == zbFileState::bkupbannc)
        {
            DelFile(uid, sid, fid);
        }
        file.state = zbFileState::deletednc;

        // wait for receiving ack from svr
        if(sync) while(file.state != zbFileState::deleted) { Sleep(10); }
    };

    // ban backup
    void BanBkup(int uid, int sid, int fid, bool sync = false)
    {
        // init parameters
        zbFileState& state  = _users(uid).strgs(sid).files(fid).state;

        if(state == zbFileState::bkupno)
        {
            state = zbFileState::bkupban;
        }
        else if(state == zbFileState::bkup)
        {
            state = zbFileState::bkupbannc;

            DelFile(uid, sid, fid);

            // wait for receiving ack from svr
            if(sync) while(state != zbFileState::bkupban) { Sleep(10); }
        }
    };

    // lift on the ban for backup
    void LiftBanBkup(int uid, int sid, int fid)
    {
        // init parameters
        zbFileState& state  = _users(uid).strgs(sid).files(fid).state;

        if(state == zbFileState::bkupban)
        {
            state = zbFileState::bkupno;
        }
        else if(state == zbFileState::bkupbannc)
        {
            state = zbFileState::bkup;
        }
    };

    // request file
    //   ret : < 0 (error), 0 (time out or no waiting), 1(received the file)
    int RequestFile(int uid, int sid, int fid, float tout_sec = 0, uchar cache = 0)
    {
        // check id
        if(!CheckId(uid, sid, fid)) return -1;

        // check connection
        if(_users(uid).src_id >= _ids.N1()) return -2;
        
        // request file
        ReqFile(uid, sid, fid, fid, cache);

        // wait for receiving file to finish
        if(tout_sec > 0)
        {
            const kmStrw path = GetFullPath(uid, sid, fid, cache).cuw(); kmTimer time(1);

            // wait for receiving
            while(time.sec() < tout_sec)
            {
                if(kmFile::Exist(path.P())) return 1;
                if(_rcvfile.IsReceiving(uid, sid, fid, cache)) break;
                Sleep(1);
            }
            // wait for done
            while(_rcvfile.IsReceiving(uid, sid, fid, cache))
            {
                if(kmFile::Exist(path.P())) return 1;
                Sleep(1);
            }
            if(kmFile::Exist(path.P())) return 1;
        }
        return 0;
    };

    // request cache file
    //   ret : < 0 (error), 0 (time out or no waiting), 1(received the file)
    int RequestCache(int uid, int sid, int fid, float tout_sec = 0)
    {
        return RequestFile(uid, sid, fid, tout_sec, 1);
    };

    // request thumbnail
    //   ret : < 0 (error), 0 (time out or no waiting), 1(received the file)
    int RequestThumb(int uid, int sid, int fid, float tout_sec = 0)
    {
        // check id
        if(!CheckId(uid, sid, fid)) return -1;

        // check connection
        if(_users(uid).src_id >= _ids.N1()) return -2;

        // check if it is an (image or movie)
        const zbFileType type = GetFiles(uid,sid)(fid).type;

        if(type != zbFileType::image && type != zbFileType::movie) return -3;

        // request file
        ReqThumb(uid, sid, fid, fid);

        // wait for timeout
        if(tout_sec > 0)
        {
            const kmStrw path = GetFullPath(uid, sid, fid, 2).cuw(); kmTimer time(1);

            // wait for receiving
            while(time.sec() < tout_sec)
            {
                if(kmFile::Exist(path.P())) return 1;
                if(_rcvfile.IsReceiving(uid, sid, fid, 2)) break;
                Sleep(1);
            }
            // wait for done
            while(_rcvfile.IsReceiving(uid, sid, fid, 2))
            {
                if(kmFile::Exist(path.P())) return 1;
                Sleep(1);
            }
            if(kmFile::Exist(path.P())) return 1;
        }
        return 0;
    };

    // send file(uid,sid,fid) to user(dst_uid)... core
    // * Note that if you set sync as true, 
    // * you should not call this function in other thread before it returns
    // 
    //  opt : 0 (          origin file, saved to _srcpath or strg path), 
    //        1 (           cache file, saved to _tmppath),    
    //        2 (       thumbnail file, saved to strg0/.thumb)
    //        3 (          shared file, saved to _dwnpath without sub-path)
    //        4 (    cache shared file, saved t0 _tmppath/.shrd/ with new naming)   
    //        5 (thumbnail shared file, saved to shrd/.thumb with new naming)
    kmNetPtcFileRes SendFileTo(int dst_uid, int uid, int sid, int fid, int opt = 0, bool sync = false)
    {
        // check uid, sid, fid anc opt
        if(!CheckId(uid, sid, fid)) return kmNetPtcFileRes::idwrong;        
        if(opt < 0 || 5 < opt)      return kmNetPtcFileRes::optwrong;

        // check dst_uid and get src_id
        ushort src_id = GetSrcId(dst_uid);

        if(src_id == 0xffff) return kmNetPtcFileRes::idwrong;

        // init strg and file        
        const zbStrg& strg   = _users(uid).strgs(sid);
        const zbFile& file   = GetFiles(uid,sid)(fid);

        int prm[] = {sid, fid, MAKELONG(opt, uid)};

        if(file.state == zbFileState::none) return kmNetPtcFileRes::nonexistence;

        // check if there is the src file
        const kmStra path = strg.srcpath + "/" + file.name;

        if(!kmFile::Exist(path.cuw().P())) return kmNetPtcFileRes::nonexistence;

        // make thumbnail image if opt is thumb and there is no thumb yet
        if((opt == 2 || opt == 5) && file.flg.thumb == 0)
        {
            if(file.type == zbFileType::image) MakeThumb(uid, sid, fid);
            else
            {
                print("** %s does not support thumbnail\n", path.P()); 
                return kmNetPtcFileRes::notsupported;
            }
        }

        // enqueue the file to send
        if(opt == 2 || opt == 5)
        {
            kmStra name = file.name;

            if(file.type == zbFileType::movie) name.CutbackRvrs('.') += ".jpg";
            
            kmNet::EnqSendFile(src_id, strg.path + "/.thumb", name, prm);
        }
        else  kmNet::EnqSendFile(src_id, strg.srcpath, file.name, prm);

        // wait for sending to finish
        if(sync)
        {
            while(GetSndQueN() > 0) Sleep(10);
            while(_snd_res == kmNetPtcFileRes::inqueue) Sleep(10);
        }
        return (sync) ? _snd_res : kmNetPtcFileRes::inqueue;
    };

    // send file(uid,sid,fid) to user(uid)... opt = 0
    kmNetPtcFileRes SendFile(int uid, int sid, int fid, bool sync = false)
    {
        return SendFileTo(uid, uid, sid, fid, 0, sync);
    };

    // send file cache(uid,sid,fid) to user(uid).. opt = 1
    kmNetPtcFileRes SendFileCache(int uid, int sid, int fid, bool sync = false)
    {
        return SendFileTo(uid, uid, sid, fid, 1, sync);
    };

    // send thumbnail(uid,sid,fid) to user(uid).. opt = 2
    kmNetPtcFileRes SendThumb(int uid, int sid, int fid, bool sync = false)
    {
        return SendFileTo(uid, uid, sid, fid, 2, sync);
    };

    // send shared file(uid,sid,fid) to user(mmb_uid)... opt = 3
    kmNetPtcFileRes SendFileShrd(int mmb_uid, int uid, int sid, int fid, bool sync = false)
    {
        return SendFileTo(mmb_uid, uid, sid, fid, 3, sync);
    };

    // send shared file cache(uid,sid,fid) to user(mmb_uid).. opt = 4
    kmNetPtcFileRes SendFileShrdCache(int mmb_uid, int uid, int sid, int fid, bool sync = false)
    {
        return SendFileTo(mmb_uid, uid, sid, fid, 4, sync);
    };

    // send thumb shared file(uid,sid,fid) to user(mmb_uid).. opt = 5
    kmNetPtcFileRes SendThumbShrd(int mmb_uid, int uid, int sid, int fid, bool sync = false)
    {
        return SendFileTo(mmb_uid, uid, sid, fid, 5, sync);
    };

    // send user's profile image(uid) to user(dst_uid)... core
    // 
    //  opt :  6 (own svr's prof, clt -> svr, own's    prof)
    //         7 (own       prof, clt -> svr, user's   prof)
    //         8 (member's  prof, svr -> clt, member's prof), uid
    kmNetPtcFileRes SendUserProfTo(int dst_uid, int uid, int opt, bool sync = false)
    {
        // check opt and uid
        if(opt < 6 || 8 < opt) return kmNetPtcFileRes::optwrong;
        if(opt == 8 && (uid < 0 || _users.N1() <= uid)) return kmNetPtcFileRes::idwrong;

        // check dst_uid and get src_id
        ushort src_id = GetSrcId(dst_uid);

        if(src_id == 0xffff) return kmNetPtcFileRes::idwrong;

        // init prm
        int prm[] = {0, 0, MAKELONG(opt, uid)};

        // init path for profile image... path
        kmStra path = _path;

        if     (opt == 6) path += kmStra("/user%d/", dst_uid);
        else if(opt == 8) path += kmStra("/user%d/", uid);
        else if(opt == 7) path += "/";

        if(!kmFile::Exist((path + "profile.jpg").cuw().P())) return kmNetPtcFileRes::nonexistence;

        // enqueue the file to send
        kmNet::EnqSendFile(src_id, path, "profile.jpg", prm);

        // wait for sending to finish
        if(sync)
        {
            while(GetSndQueN() > 0) Sleep(10);
            while(_snd_res == kmNetPtcFileRes::inqueue) Sleep(10);
        }
        return (sync) ? _snd_res : kmNetPtcFileRes::inqueue;
    };
    // send own svr's profile image to svr(dst_uid)
    kmNetPtcFileRes SendOwnSvrProf(int dst_uid, bool sync = false)
    {
        return SendUserProfTo(dst_uid, dst_uid, 6, sync);
    };
    // send own profile image to svr(dst_uid)
    kmNetPtcFileRes SendOwnProf(int dst_uid, bool sync = false)
    {
        return SendUserProfTo(dst_uid, dst_uid, 7, sync);
    };
    // send mmb(uid)'s profile image to clt(dst_uid)
    kmNetPtcFileRes SendMmbProf(int dst_uid, int uid, bool sync = false)
    {
        return SendUserProfTo(dst_uid, uid, 8, sync);
    };    

    // send shared profile image(hid) to user(dst_uid)... core
    //
    // opt :   9 (shrd's profile, clt -> svr), hid
    //        10 (shrd's profile, svr -> clt), hid
    kmNetPtcFileRes SendShrdProfTo(int dst_uid, int hid, int opt, bool sync = false)
    {
        // check opt 
        if(opt < 9 || 10 < opt) return kmNetPtcFileRes::optwrong;        

        // check dst_uid and get src_id
        ushort src_id = GetSrcId(dst_uid);

        if(src_id == 0xffff) return kmNetPtcFileRes::idwrong;

        // init prm
        int prm[] = {0, 0, MAKELONG(opt, hid)};

        // init path and file name for profile image... path, name
        kmStra path = _path, name;

        if     (opt ==  9) path += kmStra("/user%d/shrd/", dst_uid);
        else if(opt == 10) path += kmStra("/shrd/");

        name.SetStr("shrd%d.jpg",hid);

        if(!kmFile::Exist((path + name).cuw().P())) return kmNetPtcFileRes::nonexistence;

        // enqueue the file to send
        kmNet::EnqSendFile(src_id, path, name, prm);

        // wait for sending to finish
        if(sync)
        {
            while(GetSndQueN() > 0) Sleep(10);
            while(_snd_res == kmNetPtcFileRes::inqueue) Sleep(10);
        }
        return (sync) ? _snd_res : kmNetPtcFileRes::inqueue;
    };
    // send shrd profile image(hid) to svr(dst_uid)
    kmNetPtcFileRes SendShrdProfToSvr(int dst_uid, int hid, bool sync = false)
    {
        return SendShrdProfTo(dst_uid, hid, 9, sync);
    };
    // send shrd profile image(hid) to clt(dst_uid)
    kmNetPtcFileRes SendShrdProfToClt(int dst_uid, int hid, bool sync = false)
    {
        return SendShrdProfTo(dst_uid, hid, 10, sync);
    };

    // make thumbnail image
    void MakeThumb(int uid, int sid, int fid, int w_pix = 256, int h_pix = 256)
    {
        // check uid, sid, fid
        if(!CheckId(uid, sid, fid)) return;

        // check strg
        zbStrg& strg = _users(uid).strgs(sid);

        if(strg.type != zbStrgType::imgb || fid >= strg.files.N1()) return;

        // check file
        zbFile& file = strg.files(fid);

        if(file.flg.thumb == 1 || file.type != zbFileType::image) return;
        
        if(file.state == zbFileState::none) return;

        // set path
        kmStra srcpath("%s/%s",        strg.srcpath.P(), file.name.P());
        kmStra despath("%s/.thumb/%s", strg.srcpath.P(), file.name.P());

        // check if despath is availables
        kmStra despath0 = despath; despath0.ReplaceRvrs('/', '\0');

        kmFile::MakeDirs(despath0.cuw());

        // make thumbnail imags        
        kmMdf::MakeThumbImg(srcpath.cuw(), despath.cuw());

        // set flag
        file.flg.thumb = 1;

        print("* thumbnail : %s\n", despath.P());
    };

    // save file list
    void SaveFileList(int uid, int sid = -1)
    {
        // for every storage
        if(sid == -1)
        {    
            for(int i = (int)_users(uid).strgs.N1(); i--;) SaveFileList(uid, i);
            return;
        }
        // init varialbes
        zbStrg&  strg   = _users(uid).strgs(sid);
        zbFiles& files  = strg.files;
        int      file_n = (int)files.N1();

        if(file_n == 0) return;

        kmStrw name(L"%s/.filelist", strg.path.cuw().P());

        // switch hidden to normal
        if(kmFile::Exist(name.P())) kmFile::SetNormal(name.P());

        // save file list
        kmFile file(name.P(), KF_NEW);

        file.Write(&file_n);        

        for(int i = 0; i < file_n; ++i)
        {
            file.Write   (&files(i));
            file.WriteMat(&files(i).name);
        }
        // switch normal to hidden
        kmFile::SetHidden(name.P());
    };

    // load file list... return value is num of files
    int LoadFileList(int uid, int sid) try
    {
        // init variables
        zbStrg&  strg  = _users(uid).strgs(sid); if(strg.type == zbStrgType::imgl) return 0;
        zbFiles& files = strg.files;
        kmStra   path("%s/.filelist", strg.path.P());

        // load file
        kmFile file(path.cuw());
        int    file_n = 0;

        file.Read(&file_n);  files.Recreate(file_n);

        // restore and read sub-mats
        for(int i = 0; i < file_n; ++i)
        {
            file.Read   (&files(i));
            file.ReadMat(&files(i).name.Restore());
        }
        return (int)files.N1();
    }
    catch(kmException e) { kmPrintException(e); return 0; };

    // check file list... return vaule is num of changed files
    int CheckFileList(int uid, int sid)
    { 
        if(_users.N1() == 0) return 0;

        const kmStra&  path  = _users(uid).strgs(sid).srcpath;
        const zbFiles& files = _users(uid).strgs(sid).files;

        int chng_n = 0;

        for(int fid = (int)files.N1(); fid--;)
        {
            zbFile& file = files(fid);

            if(file.state == zbFileState::bkupno || file.state == zbFileState::bkup)
            {
                if(kmFile::Exist(kmStra("%s/%s", path.P(), file.name.P()).cuw())) continue;

                ++chng_n;

                if(file.state == zbFileState::bkup) file.state = zbFileState::bkuponly;
                else                                file.state = zbFileState::deleted;
            }
        }
        return chng_n;
    };

    // print file list for every storage
    void PrintFileList()
    {
        for(int uid = 0, u_n = (int)_users.N1();            uid < u_n; ++uid)
        for(int sid = 0, s_n = (int)_users(uid).strgs.N1(); sid < s_n; ++sid)
        {
            PrintFileList(uid, sid);
        }
    };

    // print file list
    void PrintFileList(int uid, int sid)
    {
        zbFiles& files = _users(uid).strgs(sid).files;

        printw(L"\n* file list : uid %d, sid %d, num %d\n\n", uid, sid, files.N1());

        for(int i = 0; i < files.N1(); ++i)
        {
            zbFile& file = files(i);

            printw(L"[%d] %s  %s ", i, file.date.GetStrwPt().P(), file.gps.GetStrw().P());

            switch(file.type)
            {
            case zbFileType::data   : print("data   "); break;
            case zbFileType::dummy  : print("dummy  "); break;
            case zbFileType::image  : print("image  "); break;
            case zbFileType::movie  : print("movie  "); break;
            case zbFileType::folder : print("folder "); break;
            }
            switch(file.state)
            {
            case zbFileState::bkupno   : print("bkupno    "); break;
            case zbFileState::bkup     : print("bkup      "); break;
            case zbFileState::bkuponly : print("bkuponly  "); break;
            case zbFileState::deletednc: print("deletednc "); break;
            case zbFileState::deleted  : print("deleted   "); break;
            case zbFileState::none     : print("none      "); break;
            case zbFileState::bkupbannc: print("bkupbannc "); break;
            case zbFileState::bkupban  : print("bkupban   "); break;
            }
            print(file.flg.thumb ? "o":"x");
            print(file.flg.cache ? "o":"x");
            print(" %s\n", file.name.P());
        }
        printw(L"\n");
    };

    // check id
    bool CheckId(int uid, int sid, int fid)
    {
        if(uid < 0 || uid >= _users.N1()            ||
           sid < 0 || sid >= _users(uid).strgs.N1() ||
           fid < 0 || fid >= GetFiles(uid, sid).N1())
        {
            print("* [CheckId](uid %d, sid %d, fid %d)\n", uid, sid, fid); return false;
        }
        return true;
    };
    bool CheckId(int uid, int sid)
    {
        if(uid < 0 || uid >= _users.N1() ||
           sid < 0 || sid >= _users(uid).strgs.N1() )
        {
            print("* [CheckId](uid %d, sid %d)\n", uid, sid); return false;
        }
        return true;
    };

    // check if the file exists or not
    //  opt : 0 (file), 1 (cache), 2 (thumbnail)
    bool Exist(int uid, int sid, int fid, int opt = 0)
    {
        return kmFile::Exist(GetFullPath(uid, sid, fid, opt).cuw().P());
    };

    /////////////////////////////////////////////////////////////
    // nks functions

    // save nks table
    void SaveNks()
    {
        if(_mode != zbMode::nks) return;

        kmStra path("%s/.nkstable", _path.P());

        _nks.Save(path.cuw().P());
    };

    // load nks table
    int LoadNks() try
    {
        if(_mode != zbMode::nks) return 0;

        kmStra path("%s/.nkstable", _path.P());

        return _nks.Load(path.cuw().P());
    }
    catch(kmException) { return 0; };

    ///////////////////////////////////////////////
    // only for debugging

    kmNetPtcFile& GetPtcFile() { return _ptc_file; };
};

#endif /* __zbNet_H_INCLUDED_2022_04_07__ */

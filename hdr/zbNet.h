#ifndef __zbNet_H_INCLUDED_2022_04_07__
#define __zbNet_H_INCLUDED_2022_04_07__

/* Note ----------------------
* zbNet has been created by Choi, Kiwan
* This is version 1
* zbNet is based on kmNet (ver. 7)
*/

// base header
#include "km7Net.h"
#include <string>

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
    kmStrw      name   {};  // relative path from strg.srcpath + name
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
    kmStrw     name{};     // name only
    kmStrw     path{};     // full path
    kmStrw     srcpath{};  // path of source files
    zbFiles    files{};    // file list
    int        chknfid{};  // next fid of checked file only for imgb to make thumb
    short      lnk_uid{};  // only for imgl
    short      lnk_sid{};  // only for imgl

    // get string for rol
    kmStrw GetTypeStrw()
    {
        switch(type)
        {
        case zbStrgType::imgb : return L"imgb";
        case zbStrgType::sync : return L"sync";
        case zbStrgType::pssv : return L"pssv";
        case zbStrgType::imgl : return L"imgl";
        case zbStrgType::imgs : return L"imgs";
        default: break;
        }
        return L"unknown";
    }
};
typedef kmMat1<zbStrg> zbStrgs;

//////////////////////////////////////
// zibsvr shared item

// zibsvr shared item type enum class... user, strg, file
enum class zbShrdType { user, strg, file };

// zibsvr shared item (user, storage, file)
class zbShrd
{
public:
    zbShrdType type{}; // user, strg, file
    kmStrw     name{};
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
    kmStrw     name{};
    zbRole     role{};      // owner, member, ownsvr, svr, family, guest
    kmNetKey   key{};       // pkey
    kmNatType  nattype{};   // nat type
    kmStrw     path{};      // path of user (full path)
    kmDate     time{};      // last connected time
    kmAddr4    addr{};      // last connected ip
    ushort     src_id = -1; // src_id for _ids
    zbStrgs    strgs{};     // storage list
    zbShrds    shrds{};     // shared  list

    // cosntructor
    zbUser() {};
    zbUser(kmMacAddr mac, const kmStrw& name, zbRole role, kmAddr4 addr, ushort src_id, kmNetKey key = {})
        : mac(mac), name(name), role(role), addr(addr), src_id(src_id), key(key) {};

    // reset connected time
    void ResetTime() { time.SetCur(); };

    // display user info
    string PrintInfo()
    {
        char cname[300] = {0,};
        wcstombs(cname, name.P(), name.GetLen());
        string name(cname);

        ostringstream oss;
        oss<<"  name : "<<name<<"\n  mac  : "<<mac.GetStr().P()<<"\n  ip   : "<<addr.GetStr().P()<<endl;
        return oss.str();
    };

    // get string for rol
    kmStrw GetRoleStrw()
    {
        switch(role)
        {
        case zbRole::owner  : return L"owner";
        case zbRole::member : return L"member";
        case zbRole::ownsvr : return L"ownsvr";
        case zbRole::svr    : return L"svr";
        case zbRole::family : return L"family";
        case zbRole::guest  : return L"guest";
        default: break;
        }
        return L"unknown";
    };

    // is server
    bool IsSvr() { return role == zbRole::ownsvr || role == zbRole::svr; };
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
    kmStrw    name{};
    zbMode    mode{};
    int       user_n{};
    ushort    src_id{};
};

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

// zibsvr main class
class zbNet : public kmNet
{
public:
    struct PathSet
    {
        string path = "";
        string srcpath = "";
        string dlpath = "";
        string cachepath = "";
    };

    zbMode           _mode = zbMode::clt; // svr (server), clt (client), nks ( net key signaling server)
    ushort           _port{DEFAULT_PORT}; // port                                                       
    kmStrw           _path{};           // zibsvr root's path
    zbUsers          _users;
    kmStrw           _srcpath{};        // image source path ... client only
    kmStrw           _dwnpath{};        // download path     ... client only
    kmStrw           _tmppath{};        // temprary path     ... client only
    zbNetInfo        _rcvinfo;          // last received info
    kmNetNks         _nks;              // net key signaling function only for zbMode::nks
    zbNetRcvFileInfo _rcvfile{};        // info of receiving file
    kmWorks _wrks;                      // worker to send data

    // init
    void Init(void* parent, kmNetCb netcb, const PathSet& pathSet, const string& deviceID, const string& deviceName)
    {
        setDeviceID(deviceID);
        setDeviceName(deviceName);

        // init kmnet
        kmNet::Init(parent, netcb, _port);

        setFilePath(pathSet);

/*
        sockaddr_in saddr;
        saddr.sin_family = AF_INET;
        saddr.sin_port = RTC_PORT;
        //saddr.sin_addr.s_addr = 0x3448720A; // 10.114.75.52
        saddr.sin_addr.s_addr = 0x0100007F; // 127.0.0.1
        kmAddr4 nks_addr(saddr);
        kmNet::SetNksAddr(nks_addr);

        // load setting... _pkey
        LoadSetting(); _pkey.Print();
*/

        // load users and strgs
        //if (_mode == zbMode::svr || _mode == zbMode::clt)
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

        // create work thread 
        _wrks.Create([](kmWork& wrk, zbNet* net)
        {   
            ushort src_id; zbRole role; kmAddr4 addr;

            switch(wrk.Id())
            {
            case 0: wrk.Get(src_id);       net->SendInfo     (src_id);       break;
            case 1: wrk.Get(src_id);       net->RequestRegist(src_id);       break;
            case 2: wrk.Get(src_id, role); net->AcceptRegist (src_id, role); break;
            case 3: wrk.Get(addr); net->Preconnect (addr); break;
            default: break;
            }   
        }, this);
    };

    // get functions
    zbUser&    GetUser (int uid)          { return _users(uid); };
    zbStrg&    GetStrg (int uid, int sid) { return _users(uid).strgs(sid); };
    zbNetInfo& GetLastRcvInfo()           { return _rcvinfo; };

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

    // get path... opt : 0 (file), 1 (cache), 2 (thumbnail)
    kmStrw GetPath(int uid, int sid, int opt = 0)
    {
        switch(opt)
        {
        case 0: return _users(uid).strgs(sid).srcpath;           // file
        case 1: return _tmppath;                                 // caching file
        case 2: return _users(uid).strgs(sid).path + L"/.thumb"; // thumbnail
        default: break;
        }
        return kmStrw();
    };

    // get full path... opt : 0 (file), 1 (cache), 2 (thumbnail)
    kmStrw GetFullPath(int uid, int sid, int fid, int opt = 0)
    {
        return GetPath(uid,sid,opt) + L"/" + GetFiles(uid, sid)(fid).name;
    };

/*
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
*/

    // enqueue work
    template<typename... Ts>
    void EnqueueWork(int id, Ts... args) { _wrks.Enqueue(id, args...); }

    ///////////////////////////////////////////////
    // virtual functions for rcv callback
protected:
    // virtual callback for ptc_cnnt
    //   cmd_id will be 1 (rcvaccept) or -1 (sndaccept)
    virtual void vcbRcvPtcCnnt(ushort src_id, char cmd_id)
    {
        // get id
        kmNetId& id = _ids(src_id);

        // find user already registered
        int uid = FindUser(id.mac);

        if(uid < 0) // no registered user
        {
            EnqSendInfo(src_id);
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
        //case  7: RcvReqThumb  (src_id, buf); break;
        case  8: RcvReqList   (src_id, buf); break;
        case  9: RcvTxt       (src_id, buf); break;
        case 10: RcvJson      (src_id, buf); break;
        default: break;
        }
    };

    // virtual callback for ptc_file
    //  cmd_id  1: rcv preblk, 2: receiving, 3: rcv done,  4: rcv failure
    //         -1: snd preblk,              -3: snd done, -4: snd failure
    virtual void vcbRcvPtcFile(ushort src_id, char cmd_id, int* prm)
    {
        kmNetPtcFileCtrl& ctrl = _ptc_file._rcv_ctrl;
        const int         uid  = FindUser(src_id);

        int sid = prm[0], fid = prm[1], opt = prm[2]; // opt : 0 (file), 1 (caching file), 2 (thumbnail)

        if(cmd_id == 1) // rcv preblk
        {
            if(CheckId(uid, sid))
            {
                ctrl.SetPath(GetPath(uid, sid, opt));

                _rcvfile.SetReceiving(uid, sid, fid, opt);
            }
            else  ctrl.Reject(); 
        }
        else if(cmd_id == 3) // rcv done
        {
            // get date from the file
            kmFileInfo fileinfo(ctrl.file_path);

            zbFile file = {{}, {}, {}, 0, ctrl.file_name, fileinfo.date};

            _rcvfile.SetDone(uid, sid, fid, opt);

            if(opt == 0) // file
            {
                if(!CheckId(uid, sid)) return;

                AddFile(uid, sid, fid, file); SaveFileList(uid, sid);
            }
            else if(opt == 1 || opt == 2) // cache
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
            if(_users(uid).IsSvr() && sid >= 0)
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
        default: break;
        }
    };
    void RcvNkeyReqKey() // _rcv_addr, _rcv_mac ->_snd_key
    {
        print("**** rcv request key from %s\n", _ptc_nkey._rcv_addr.GetStr().P());
        
        _ptc_nkey._snd_key = _nks.Register(_ptc_nkey._rcv_mac, _ptc_nkey._rcv_addr);
    };
    void RcvNkeyKey()
    {
    };
    void RcvNkeyReqAddr() // _rcv_key, _rcv_mac -> _snd_addr
    {
        print("**** rcv request addr by %s\n", _ptc_nkey._rcv_key.GetStr().P());

        _ptc_nkey._snd_addr = _nks.GetAddr(_ptc_nkey._rcv_key, _ptc_nkey._rcv_mac);
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

        if(flg >= 0) ReplyNksSig(_ptc_nkey._rcv_addr, flg);
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

    /////////////////////////////////////////////////////////////
    // _ptc_data (ptc_id = 2) protocol functions

    // send my info to opposite    ...data_id = 0
    void SndInfo(ushort src_id)
    {
        uchar data_id = 0; kmNetBuf buf(256);

        ushort tmpName[MAX_FILE_NAME] = {0,};
        convWC4to2(_name.P(), tmpName, _name.N());
        (buf << _mac).PutData(tmpName, _name.N());
        buf << _mode << (int)_users.N1();

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };
    void RcvInfo(ushort src_id, kmNetBuf& buf)
    {
        zbNetInfo& info = _rcvinfo; _rcvinfo.src_id = src_id;

        ushort name[ID_NAME] = {0,};
        ushort name_n;
        (buf >> info.mac).GetData(name, name_n);
        wchar wname[ID_NAME] = {0,};
        convWC2to4(name, wname, name_n);
        info.name.SetStr(wname);
        buf >> *(int*)&info.mode >> info.user_n;
    };

    // request registration (clt to svr)...data_id = 1
    void ReqRegist(ushort src_id)
    {
        uchar data_id = 1; kmNetBuf buf(256);

        ushort sname[ID_NAME] = {0,};
        ushort len = wcslen(_name.P());
        convWC4to2(_name.P(), sname, len);
        (buf << _mac).PutData(sname, MIN(64, _name.N()));

        Send(src_id, data_id, buf.P(), (ushort)buf.N1()); // no waiting
    };
    void RcvReqRegist(ushort src_id, kmNetBuf& buf)
    {
        print("**** 3. RcvRegRegist : src_id(%d)\n", src_id);

        kmMacAddr mac; kmStrw name{};

        buf >> mac >> name;

        // check
        if(FindUser(mac) >= 0)   return; // already registered
        if(_mode == zbMode::nks) return;

        // set role
        zbRole role = zbRole::family;

        if(_mode == zbMode::svr) role = (_users.N1() == 0) ? zbRole::owner : zbRole::member;

        // add users
        zbUser user(mac, name, role, _ids(src_id).addr, src_id);

        AddUser(user);
        SaveUsers();

        print("**** add user : src_id(%d)\n", src_id);

        // accept the opposite        
        //AcceptRegist(src_id, role);
        EnqAcceptRegist(src_id, role);
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

        kmMacAddr mac; kmStrw name; zbRole myrole; kmNetKey pkey;

        ushort sname[ID_NAME] = {0,};
        ushort name_n;

        (buf >> mac).GetData(sname, name_n);
        buf >> myrole >> pkey;
        //buf >> pkey;

        wchar wname[ID_NAME] = {0,};
        convWC2to4(sname, wname, name_n);
        name.SetStr(wname);

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
        default: break;
        }

        // add users
        zbUser user(mac, name, role, _ids(src_id).addr, src_id, pkey);

        print("** user was added\n"); cout<<user.PrintInfo();

        AddUser(user);
        SaveUsers();
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
        zbNoti noti; FindUser(src_id);

        buf >> noti;

        switch(noti)
        {
        case zbNoti::none : print("* receive zbNot::none\n"); break;
        default: break;
        }
    };

    // request file... data_id = 4
    //   opt : 0 (file), 1(caching)
    void ReqFile(int uid, int sid, int fid_s, int fid_e = -1, uchar caching = 0)
    {
        const ushort src_id = _users(uid).src_id; if(src_id >= _ids.N1()) return;

        uchar data_id = 4;  kmNetBuf buf(32);

        buf << sid << fid_s << fid_e << caching;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };
    void RcvReqFile(ushort src_id, kmNetBuf& buf)
    {
        int sid, fid_s, fid_e; uchar caching;

        buf >> sid >> fid_s >> fid_e >> caching;

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
            SendFile(uid, sid, fid, (int)caching);
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
            kmFile::Remove(GetFullPath(uid, sid, fid));
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
        default: break;
        }
    };

    // send text (utf-8)... data_id = 9
    void SndTxt(int uid, kmStra& txt)
    {
        // check
        if(uid < 0 || uid >= _users.N1()) return;

        const ushort src_id = _users(uid).src_id; if(src_id >= _ids.N1()) return;

        if(txt.Byte() > 2048) { return; }
        // send
        uchar data_id = 9;

        Send(src_id, data_id, txt.P(), (ushort)txt.Byte(), 100);
    };
    void RcvTxt(ushort src_id, kmNetBuf& buf) {};

    // send json (utf-8)... data_id = 10
    void SndJson(int uid, kmStra& json)
    {
        // check
        if(uid < 0 || uid >= _users.N1()) return;

        const ushort src_id = _users(uid).src_id; if(src_id >= _ids.N1()) return;

        if(json.Byte() > 2048)
        {
            print("* SndJson in 855 : text is too long\n");    return;
        }
        // send
        uchar data_id = 10;

        Send(src_id, data_id, json.P(), (ushort)json.Byte(), 100);
    };
    void RcvJson(ushort src_id, kmNetBuf& buf) {};

    /////////////////////////////////////////////////////////////
    // network interface functions
public:
    // connect
    //  return : src_id (if connecting failed, it will be -1)
    int Connect(const kmAddr4 addr, float tout_msec = 300.f)
    {
        return kmNet::Connect(addr, _name, tout_msec);
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

    // connect with uid
    //  return : src_id (if connecting failed, it will be -1)
    int Connect(int uid)
    {
        int ret;

        //print("** 1\n"); ret = ConnectLastAddr(uid); if(ret >= 0) return ret;
        print("** 2\n"); ret = ConnectInLan   (uid); if(ret >= 0) return ret;
        print("** 3\n"); ret = ConnectInWan   (uid); if(ret >= 0) return ret;

        print("** connecting fails\n");

        return -1;
    };

    // connect with last connected address
    //  return : src_id (if connecting failed, it will be -1)
    int ConnectLastAddr(int uid)
    {
        if(uid >= _users.N1()) return -1;

        print("** connect to the last address (%s)\n", _users(uid).addr.GetStr().P());

        return Connect(_users(uid).addr);
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

            return Connect(addrs(i));
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
        kmAddr4 addr = RequestAddr(_users(uid).key, _users(uid).mac);

        if(!addr.isValid()) return -1;

        print("** connect to %s in wan\n", addr.GetStr().P());

        // send precnnt
        Preconnect(addr);

        // connect
        return Connect(addr);
    };

    // preconnect
    int Preconnect(kmAddr4 addr)
    {
        int n_try = SendPreCnnt(addr, 200, 6);

        if(n_try == 0) print("** precnnt : no answer\n");
        else           print("** precnnt : get answer after %d times\n", n_try);

        return n_try;
    };
    void EnqPreconnect(kmAddr4 addr) { EnqueueWork(3, addr); };

    // check if connected
    bool IsConnected(int uid)
    {
        // check src_id
        const ushort src_id = GetSrcId(uid);

        if(src_id == 0xffff) return false;

        // send sig
        const float ec_msec = kmNet::SendSig(src_id, 300.f, 3);

        return (ec_msec >=0) ? true : false;
    };

    // send data through ptc_data including connection checking
    // 
    //   tout_msec : 0 (not wait for ack), > 0 (wait for ack)
    int Send(ushort src_id, uchar data_id, char* data, ushort byte, 
             float tout_msec = 0.f, int retry_n = 1)
    {
        return kmNet::Send(src_id, data_id, data, byte, tout_msec, retry_n);
    };

    // send info
    void SendInfo   (ushort src_id) { SndInfo(src_id); };
    void EnqSendInfo(ushort src_id) { EnqueueWork(0, src_id); };

    // request registration to svr
    void RequestRegist(ushort src_id)
    {
        ReqRegist(src_id);
    };
    void EnqRequestRegist(ushort src_id) { EnqueueWork(1, src_id); };

    // accept registration
    void AcceptRegist(ushort src_id, zbRole role)
    {
        if(_mode == zbMode::svr) AcpRegist(src_id, role);
    };
    void EnqAcceptRegist(ushort src_id, zbRole role) { EnqueueWork(2, src_id, role); };

    // send txt with UTF-16
    void SendTxt(int uid, kmStrw& txtw)
    {
        int len = txtw.GetLen()+1;
        ushort uTxt[len] = {0,};
        convWC4to2(txtw.P(), uTxt, len);
        //kmStra txt = txtw.EncWtoA(CP_UTF8); // convert utf-16 to utf-8
        char cTxt[len] = {0,};
        kmStra txt(cTxt);

        SndTxt(uid, txt);
    };
    // send txt with UTF-8
    void SendTxt(int uid, kmStra& txt) { SndTxt(uid, txt); };

    // send json with UTF-16
    void SendJson(int uid, kmStrw& jsonw)
    {    
        int len = jsonw.GetLen()+1;
        ushort uJson[len] = {0,};
        convWC4to2(jsonw.P(), uJson, len);
        //kmStra json = jsonw.EncWtoA(CP_UTF8); // convert utf-16 to utf-8
        char cJson[len] = {0,};
        kmStra json(cJson);

        SndJson(uid, json);
    };
    // send json with UTF-8
    void SendJson(int uid, kmStra& json) { SndJson(uid, json); };

    /////////////////////////////////////////////////////////////
    // setting functions

    // save setting.
    void SaveSetting()
    {
        kmFile file(kmStrw(L"%S/.zbnetsetting", _path.P()).P(), KF_NEW);

        file.Write(&_pkey);
    };

    // load setting
    int LoadSetting() try
    {
        kmFile file(kmStrw(L"%S/.zbnetsetting", _path.P()).P());

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
        user.path.SetStr(L"%S/user%d", _path.P(), (int)_users.N1());

        // add user
        int uid = (int)_users.PushBack(user);

        // make user folder
        kmFile::MakeDir(user.path.P());

        // add storage
        AddStrg(uid, zbStrgType::imgb, user.IsSvr() ? _srcpath : kmStrw());

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
        
        kmFile file(kmStrw(L"%S/.userlist", _path.P()).P(), KF_NEW);

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
        kmFile file(kmStrw(L"%S/.userlist", _path.P()).P());

        file.ReadMat(&_users);

        // restore and read sub-mats
        for(int i = 0; i < _users.N1(); ++i)
        {
            file.ReadMat(&_users(i).name.Restore());
            file.ReadMat(&_users(i).path.Restore());

            _users(i).strgs.Restore();
            _users(i).shrds.Restore();
            _users(i).src_id = -1;
        }
        return (int)_users.N1();
    }
    catch(kmException) { return 0; };

    /////////////////////////////////////////////////////////////
    // storage control functions

    // add storage
    int AddStrg(int uid, zbStrgType type, const kmStrw& srcpath)
    {
        // init variables
        zbStrgs&  strgs  = _users(uid).strgs;
        const int strg_n = (int)strgs.N1();

        if(strg_n == 0) strgs.Recreate(0,4);

        // set strg
        zbStrg strg = {type, kmStrw(L"strg%d",strg_n)};

        strg.path.SetStr(L"%S/%S", _users(uid).path.P(), strg.name.P());

        if(srcpath.N1() > 1) strg.srcpath = srcpath;
        else                 strg.srcpath = strg.path;

        // make strg folder
        kmFile::MakeDir(strg.path.P());

        // make thumb folder
        kmStrw thumb(L"%S/.thumb",strg.path.P());
        kmFile::MakeDir(thumb.P());

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
        zbStrg strg = {zbStrgType::imgl, kmStrw(L"strg%d",strg_n)};

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
        kmFile file(kmStrw(L"%S/.strglist", _users(uid).path.P()).P(), KF_NEW);

        file.WriteMat(&strgs.Pack());

        for(int i = 0; i < strgs.N1(); ++i)
        {
            file.WriteMat(&strgs(i).name);
            file.WriteMat(&strgs(i).path);
            file.WriteMat(&strgs(i).srcpath);
        }
    };

    // load strgs list... return value is num of strg
    int LoadStrgs(int uid) try
    {
        // init variables
        zbStrgs& strgs = _users(uid).strgs;

        // load file
        kmFile file(kmStrw(L"%S/.strglist", _users(uid).path.P()).P());

        file.ReadMat(&strgs);

        // restore and read sub-mats
        for(int i = 0; i < strgs.N1(); ++i)
        {
            file.ReadMat(&strgs(i).name.Restore());
            file.ReadMat(&strgs(i).path.Restore());
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
    // file list control functions

    int UpdateFileOfList(int uid, int sid)
    {
        zbFiles& files = _users(uid).strgs(sid).files;
        if (files.N1() < 1) return 0;

        SendBkupno(true);
        return 1;
    }

    int checkStateOfLastFile(int uid, int sid)
    {
        zbFiles& files = _users(uid).strgs(sid).files;

        if (files.N1() < 1) return 0;

        while (files(files.N1()-1).state == zbFileState::bkupno) {sleep(1);}

        return 1;
    }

    // update files of every user's every storage
    void UpdateFile(bool fileSendFlag = true)
    {
        const int usr_n = (int)_users.N1(); if(usr_n == 0) return;

        kmStrw buf(1024);

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
                int add_n = UpdateFile(uid, sid, buf, fileSendFlag);

                // change deleted file's state
                int del_n = 0;

                for(int i = 0; i < file_n; ++i)
                {
                    if(files(i).flg.isin == 0 && files(i).state == zbFileState::bkup)
                    {
                        files(i).state = zbFileState::bkuponly; del_n++;
                    }
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
    int UpdateFile(int uid, int sid, kmStrw& subpath, bool fileSendFlag)
    {
        // check if user is svr
        if(!_users(uid).IsSvr()) return 0;

        // get files        
        zbStrg&  strg  = _users(uid).strgs(sid);
        zbFiles& files = strg.files;

        // get file list
        kmFileList flst(kmStrw(L"%S/%S*", strg.srcpath.P(), subpath.P()).P());

        //flst.Print();

        // add file and check if deleted
        int add_n = 0, files_n = (int)files.N1(), flst_n = (int)flst.N1();

        kmMat1u8 flg(files_n); flg.SetZero();

        for(int i = 0; i < flst_n; ++i)
        {
            const kmFileInfo& flsti = flst(i);

            if(flsti.IsRealDir())
            {
                kmStrw path(L"%s%s/", subpath.P(), flsti.name.P());

                (subpath += flsti.name) += L"/";
                
                add_n += UpdateFile(uid, sid, subpath, fileSendFlag);

                subpath.Cutback(flsti.name.GetLen()); // including L'/'
            }
            else if(flsti.IsNormal())
            {
                // check if flst(i) is already in flies
                int isin = 0; subpath += flsti.name;

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

                    int fid = AddFile(uid, sid, file);

                    // send file to svr
                    if (fileSendFlag) SendFile(uid,sid,fid);

                    ++add_n;
                }
                subpath.Cutback(flsti.name.GetLen()-1);
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
        if(sync) while(GetSndQueN() > 0) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
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
        char cname[300] = {0,};
        wcstombs(cname, file.name.P(), wcslen(file.name.P())*2);

        // init
        zbFiles& files = _users(uid).strgs(sid).files;

        // add file
        if(files.Size() == 0) files.Recreate(0,8);

        // read file info
        if(file.type == zbFileType::image || file.type == zbFileType::movie)
        {
            const kmStrw& path  = _users(uid).strgs(sid).srcpath;

            kmMdf mdf(kmStrw(L"%S/%S", path.P(), file.name.P()).P());

            if(mdf._date != kmDate()) file.date = mdf._date;
            if(mdf._gps  != kmGps())  file.gps  = mdf._gps;

            // set file type
            if     (mdf._type == kmMdfType::jpg) file.type = zbFileType::image;
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
            char cname[300] = {0,};
            wcstombs(cname, file.name.P(), wcslen(file.name.P())*2);
            if(files(fid).state == zbFileState::bkuponly) // downloaded from svr
            {
                // check file
                if(file.name != files(fid).name) return;

                files(fid).state = zbFileState::bkup;
            }
            else if(files(fid).state == zbFileState::none)
            {
                const kmStrw& path = _users(uid).strgs(sid).srcpath;

                kmMdf mdf(kmStrw(L"%S/%S", path.P(), file.name.P()).P());

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
                zbFile dummy = {zbFileType::dummy, zbFileState::none, {}, 0, L"---"};
                AddFile(uid, sid, dummy);
            }
            AddFile(uid, sid, file);
        }
    };

    // delete file on clt    
    void DeleteFileClt(int uid, int sid, int fid)
    {
        // init parameters
        kmStrw& path  = _users(uid).strgs(sid).srcpath;
        zbFile& file  = _users(uid).strgs(sid).files(fid);        

        // delete file
        if(file.state == zbFileState::bkup) // bkup --> bkuponly
        {
            kmFile::Remove(kmStrw(L"%S/%S",path.P(), file.name.P()).P());

            file.state = zbFileState::bkuponly;
        }        
        else if(file.state == zbFileState::bkupban) // bkupban --> deleted
        {
            kmFile::Remove(kmStrw(L"%S/%S",path.P(), file.name.P()).P());

            file.state = zbFileState::deleted;
        }
        SaveFileList(uid, sid);
    };

    // delete file on both svr and clt
    void DeleteFileBoth(int uid, int sid, int fid, bool sync = false)
    {
        // init parameters
        kmStrw& path  = _users(uid).strgs(sid).srcpath;
        zbFile& file  = _users(uid).strgs(sid).files(fid);

        // delete file on clt
        if(file.state == zbFileState::bkupno    || file.state == zbFileState::bkup || 
           file.state == zbFileState::bkupbannc || file.state == zbFileState::bkupban)
        {
            kmFile::Remove(kmStrw(L"%S/%S",path.P(), file.name.P()).P());
        }        

        // delete file on svr
        if(file.state == zbFileState::bkup     || file.state == zbFileState::deletednc ||
           file.state == zbFileState::bkuponly || file.state == zbFileState::bkupbannc)
        {
            DelFile(uid, sid, fid);
        }
        file.state = zbFileState::deletednc;

        // wait for receiving ack from svr
        if(sync) while(file.state != zbFileState::deleted) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
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
            if(sync) while(state != zbFileState::bkupban) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
        }
        SaveFileList(uid, sid);
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
        SaveFileList(uid, sid);
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
            const kmStrw path = GetFullPath(uid, sid, fid, cache); kmTimer time(1);

            // wait for receiving
            while(time.sec() < tout_sec)
            {
                if(kmFile::Exist(path.P())) return 1;
                if(_rcvfile.IsReceiving(uid, sid, fid, cache)) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            // wait for done
            while(_rcvfile.IsReceiving(uid, sid, fid, cache))
            {
                if(kmFile::Exist(path.P())) return 1;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if(kmFile::Exist(path.P())) return 1;
        }
        return 0;
    };

    // request caching file
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

        // check if it is an image
        if(GetFiles(uid,sid)(fid).type != zbFileType::image) return -3;

        // request file
        ReqThumb(uid, sid, fid, fid);

        // wait for timeout
        if(tout_sec > 0)
        {
            const kmStrw path = GetFullPath(uid, sid, fid, 2); kmTimer time(1);

            // wait for receiving
            while(time.sec() < tout_sec)
            {
                if(kmFile::Exist(path.P())) return 1;
                if(_rcvfile.IsReceiving(uid, sid, fid, 2)) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            // wait for done
            while(_rcvfile.IsReceiving(uid, sid, fid, 2))
            {
                if(kmFile::Exist(path.P())) return 1;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if(kmFile::Exist(path.P())) return 1;
        }
        return 0;
    };

    // send file to user
    // * Note that if you set sync as true, 
    // * you should not call this function in other thread before it returns
    //
    //   opt : 0 (file), 1 (caching file)    
    kmNetPtcFileRes SendFile(int uid, int sid, int fid, int opt = 0, bool sync = false)
    {
        // check uid, sid, fid and opt
        if(!CheckId(uid, sid, fid)) return kmNetPtcFileRes::idwrong;
        if(opt != 0 && opt != 1)    return kmNetPtcFileRes::optwrong;

        // init variables
        ushort  src_id = _users(uid).src_id;
        zbStrg& strg   = _users(uid).strgs(sid);
        zbFile& file   = GetFiles(uid,sid)(fid);

        int prm[] = {sid, fid, opt};

        if(file.state == zbFileState::none) return kmNetPtcFileRes::nonexistence;

        // check if there is the src file
        const kmStrw path = strg.srcpath + L"/" + file.name;

        if(!kmFile::Exist(path.P())) return kmNetPtcFileRes::nonexistence;

        // enqueue the file to send
        kmNet::EnqSendFile(src_id, strg.srcpath, file.name, prm);

        // wait for sending to finish
        if(sync)
        {
            while(GetSndQueN() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
            while(_snd_res == kmNetPtcFileRes::inqueue) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return (sync) ? _snd_res : kmNetPtcFileRes::inqueue;
    };

    // send file to user
    // * Note that if you set sync as true, 
    // * you should not call this function in other thread before it returns
    //
    //   opt : 0 (file), 1 (caching file)    
    kmNetPtcFileRes SendFile(int uid, int sid, int fid, bool sync)
    {
        return SendFile(uid, sid, fid, 0, sync);
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

        kmStrw name(L"%S/.filelist", strg.path.P());

        // save file list
        kmFile file(name.P(), KF_NEW);

        file.Write(&file_n);        

        for(int i = 0; i < file_n; ++i)
        {
            file.Write   (&files(i));
            file.WriteMat(&files(i).name);
        }
    };

    // load file list... return value is num of files
    int LoadFileList(int uid, int sid) try
    {
        // init variables
        zbStrg&  strg  = _users(uid).strgs(sid); if(strg.type == zbStrgType::imgl) return 0;
        zbFiles& files = strg.files;
        kmStrw   path(L"%S/.filelist", strg.path.P());

        // load file
        kmFile file(path);
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

        const kmStrw&  path  = _users(uid).strgs(sid).srcpath;
        const zbFiles& files = _users(uid).strgs(sid).files;

        int chng_n = 0;

        for(int fid = (int)files.N1(); fid--;)
        {
            zbFile& file = files(fid);

            if(file.state == zbFileState::bkupno || file.state == zbFileState::bkup)
            {
                if(kmFile::Exist(kmStrw(L"%S/%S", path.P(), file.name.P()))) continue;

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

        wcout<<L"\n* file list : uid "<<uid<<L", sid "<<sid<<L", num "<<files.N1()<<L"\n"<<endl;

        for(int i = 0; i < files.N1(); ++i)
        {
            zbFile& file = files(i);

            wcout<<L"["<<i<<L"] "<<file.date.GetStrwPt().P()<<L"  "<<file.gps.GetStrw().P()<<L"  ";
            char cname[300] = {0,};
            wcstombs(cname, file.name.P(), wcslen(file.name.P())*2);
            cout<<cname<<"  ";

            switch(file.type)
            {
            case zbFileType::data   : wcout<<L" [data]   "; break;
            case zbFileType::dummy  : wcout<<L" [dummy]  "; break;
            case zbFileType::image  : wcout<<L" [image]  "; break;
            case zbFileType::movie  : wcout<<L" [movie]  "; break;
            case zbFileType::folder : wcout<<L" [folder] "; break;
            default: break;
            }
            switch(file.state)
            {
            case zbFileState::bkupno   : wcout<<L"[bkupno]"<<endl;   break;
            case zbFileState::bkup     : wcout<<L"[bkup]"<<endl;     break;
            case zbFileState::bkuponly : wcout<<L"[bkuponly]"<<endl; break;
            case zbFileState::deletednc: wcout<<L"[deletednc]"<<endl;break;
            case zbFileState::deleted  : wcout<<L"[deleted]"<<endl;  break;
            case zbFileState::none     : wcout<<L"[none]"<<endl;     break;
            case zbFileState::bkupbannc: wcout<<L"[bkupbannc]"<<endl;break;
            case zbFileState::bkupban  : wcout<<L"[bkupban]"<<endl;  break;
            default: break;
            }
            print(file.flg.thumb ? "o":"x");
            print(file.flg.cache ? "o":"x");
            wcout<<L" "<<file.name.P()<<endl;
        }
        cout<<endl;
    };

    // print file list
    string PrintFileListTest(int uid, int sid)
    {
        stringstream oss;
        zbFiles& files = _users(uid).strgs(sid).files;

        oss<<"\n* file list : uid "<<uid<<", sid "<<sid<<", num "<<files.N1()<<"\n\n";

        for(int i = 0; i < files.N1(); ++i)
        {
            zbFile& file = files(i);

            oss<<"["<<i<<"] "<<file.date.GetStrPt().P()<<"  ";
            char cgps[200] = {0,};
            wcstombs(cgps, file.gps.GetStrw().P(), wcslen(file.gps.GetStrw().P())*2);
            oss<<cgps<<"  ";
            char cname[200] = {0,};
            wcstombs(cname, file.name, wcslen(file.name)*2);
            oss<<cname<<"  ";

            switch(file.type)
            {
            case zbFileType::data   : oss<<" [data]   "; break;
            case zbFileType::dummy  : oss<<" [dummy]  "; break;
            case zbFileType::image  : oss<<" [image]  "; break;
            case zbFileType::movie  : oss<<" [movie]  "; break;
            case zbFileType::folder : oss<<" [folder] "; break;
            default: break;
            }
            switch(file.state)
            {
            case zbFileState::bkupno   : oss<<"[bkupno]\n";   break;
            case zbFileState::bkup     : oss<<"[bkup]\n";     break;
            case zbFileState::bkuponly : oss<<"[bkuponly]\n"; break;
            case zbFileState::deletednc: oss<<"[deletednc]\n";break;
            case zbFileState::deleted  : oss<<"[deleted]\n";  break;
            case zbFileState::none     : oss<<"[none]\n";     break;
            case zbFileState::bkupbannc: oss<<"[bkupbannc]\n";break;
            case zbFileState::bkupban  : oss<<"[bkupban]\n";  break;
            default: break;
            }
            print(file.flg.thumb ? "o":"x");
            print(file.flg.cache ? "o":"x");
            wcout<<L" "<<file.name.P()<<endl;
        }
        oss<<"\n";
        return oss.str();
    };

    void convWC4to2(wchar* wc, ushort* c, const ushort& n) { for (ushort i = 0; i < n; ++i) { c[i] = (ushort)wc[i]; } };
    void convWC2to4(ushort* c, wchar* wc, const ushort& n) { for (ushort i = 0; i < n; ++i) { wc[i] = (wchar)c[i]; } };
    void setFilePath(const PathSet& pathSet)
    {
        wchar temp[256] = {0,};
        mbstowcs(temp, pathSet.path.c_str(), pathSet.path.length()*2);
        _path.SetStr(temp);
        kmStrw path(L"%S",temp);
        kmFile::MakeDir(path.P());

        mbstowcs(temp, pathSet.srcpath.c_str(), pathSet.srcpath.length()*2);
        _srcpath.SetStr(temp);

        mbstowcs(temp, pathSet.dlpath.c_str(), pathSet.dlpath.length()*2);
        _dwnpath.SetStr(temp);
        kmStrw dwpath(L"%S",temp);
        kmFile::MakeDir(dwpath.P());

        mbstowcs(temp, pathSet.cachepath.c_str(), pathSet.cachepath.length()*2);
        _tmppath.SetStr(temp);
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
    //  opt : 0 (file), 1 (caching), 2 (thumbnail)
    bool Exist(int uid, int sid, int fid, int opt = 0)
    {
        return kmFile::Exist(GetFullPath(uid, sid, fid, opt).P());
    };

    /////////////////////////////////////////////////////////////
    // nks functions

    // save nks table
    void SaveNks()
    {
        if(_mode != zbMode::nks) return;

        kmStrw path(L"%S/.nkstable", _path.P());

        _nks.Save(path.P());
    };

    // load nks table
    int LoadNks() try
    {
        if(_mode != zbMode::nks) return 0;

        kmStrw path(L"%S/.nkstable", _path.P());

        return _nks.Load(path.P());
    }
    catch(kmException) { return 0; };

    ///////////////////////////////////////////////
    // only for debugging

    kmNetPtcFile& GetPtcFile() { return _ptc_file; };

    /////////////////////////////////////////////////////////////
    // mac address - device id
    void setDeviceID(const string& deviceID)
    {
        if (deviceID.length() > 16) return;

        uint64 longMac = strtoull(deviceID.c_str(), nullptr, 16); // string, idx, base

        //macAddr.Set((uchar*)ifr[1].ifr_hwaddr.sa_data);
        char tmpMac[8] = {(char)(longMac>>56&0xFF),
                          (char)(longMac>>48&0xFF),
                          (char)(longMac>>40&0xFF),
                          (char)(longMac>>32&0xFF),
                          (char)(longMac>>24&0xFF),
                          (char)(longMac>>16&0xFF),
                          (char)(longMac>>8&0xFF),
                          (char)(longMac&0xFF)};

        _mac.Set((uchar*)tmpMac);
    }
    void setDeviceName(const string& deviceName)
    {
        wchar wname[300] = {0,};
        mbstowcs(wname, deviceName.c_str(), deviceName.length());
        _name.SetStr(wname);
    }
};

#endif /* __zbNet_H_INCLUDED_2022_04_07__ */

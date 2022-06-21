#ifndef __zbNet_H_INCLUDED_2022_04_07__
#define __zbNet_H_INCLUDED_2022_04_07__

/* Note ----------------------
* zbNet has been created by Choi, Kiwan
* This is version 1
* zbNet is based on kmNet (ver. 7)
*/

// base header
#include "km7Net.h"
#include <sstream>
#include <string>

/////////////////////////////////////////////////////////////////////////////////////////
// zbNet class

//////////////////////////////////////
// zibsvr file

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
    uchar thumb   : 1 = 0; // 0: no thumbnail,  1: there is thumbnail 
    uchar encrypt : 1 = 0; // 0: not encrypted, 1: encrypted
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
    ushort     src_id = 0xFFFF; // src_id for _ids
    zbStrgs    strgs{};     // storage list
    zbShrds    shrds{};     // shared  list

    // cosntructor
    zbUser() {};
    zbUser(kmMacAddr mac, const kmStrw& name, zbRole role, kmAddr4 addr, ushort src_id, kmNetKey key = {})
        : mac(mac), name(name), role(role), addr(addr), src_id(src_id), key(key) {};

    // reset connected time
    void ResetTime() { time.SetCur(); };

    // display user info
    void PrintInfo()
    {
        wcout<<L"name : "<<name.P()<<L"\nmac : "<<mac.GetStrw().P()<<L"\nip : "<<addr.GetStrw().P()<<L"\n"<<endl;
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

    zbMode      _mode = zbMode::clt;  // svr (server), clt (client), nks ( net key signaling server)
    ushort    _port{DEFAULT_PORT}; // port
    kmStrw    _path{};           // zibsvr root's path
    zbUsers   _users;             
    kmStrw    _srcpath{};        // image source path ... client only
    kmStrw    _dwnpath{};        // download path ... client only
    kmStrw    _cachepath{};      // app cache path ... client only
    zbNetInfo _rcvinfo;          // last received info    
    kmNetNks  _nks;              // net key signaling function only for zbMode::nks

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
    };

    // get functions
    zbUser&    GetUser (int uid)          { return _users(uid); };
    zbStrg&    GetStrg (int uid, int sid) { return _users(uid).strgs(sid); };
    zbNetInfo& GetLastRcvInfo()           { return _rcvinfo; };

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

    ///////////////////////////////////////////////
    // virtual functions for rcv callback

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
            SendInfo(src_id);
        }
        else 
        {
            _users(uid).src_id = src_id;
            _users(uid).ResetTime();
        }
    };

    // virtual callback for ptc_data
    virtual void vcbRcvPtcData(ushort src_id, uchar data_id, kmNetBuf& buf)
    {
        switch(data_id)
        {
        case 0: RcvInfo        (src_id, buf); break;
        case 1: RcvReqRegist   (src_id, buf); break;
        case 2: RcvAcceptRegist(src_id, buf); break;
        case 3: RcvNotify      (src_id, buf); break;
        case 4: RcvReqList     (src_id, buf); break;
        case 5: RcvDelFile     (src_id, buf); break;
        case 6: RcvAckDelFile  (src_id, buf); break;
        case 7: RcvReqThumb    (src_id, buf); break;
        }
    };

    // virtual callback for ptc_file
    //  cmd_id  1: rcv preblk, 2: receiving, 3: rcv done,  4: rcv failure
    //         -1: snd preblk,              -3: snd done, -4: snd failure
    virtual void vcbRcvPtcFile(ushort src_id, char cmd_id, int sid, int fid)
    {
        kmNetPtcFileCtrl& ctrl = _ptc_file._rcv_ctrl;
        const int         uid  = FindUser(src_id);

        if(cmd_id == 1) // rcv preblk
        {
            if(uid < 0) ctrl.Reject(); // src_id is not one of registered users
            else
            {
                //if(sid >= 0) ctrl.SetPath(_users(uid).strgs(sid).srcpath);
                if(sid >= 0) ctrl.SetPath(_dwnpath);
                else         ctrl.SetPath(_users(uid).strgs(-sid-1).path + L"/.thumb"); // thumbnail
            }
        }
        else if(cmd_id == 3) // rcv done
        {
            // get date from the file
            kmFileInfo fileinfo(ctrl.file_path); ctrl.file_name.Printf();

            zbFile file = {{}, {}, {}, 0, ctrl.file_name, fileinfo.date};

            if(sid >= 0)
            {
                AddFile(uid, sid, fid, file);

                SaveFileList(uid, sid);
            }            
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
        }
    };
    void RcvNkeyReqKey() // _rcv_addr, _rcv_mac ->_snd_key
    {
        _ptc_nkey._snd_key = _nks.Register(_ptc_nkey._rcv_mac, _ptc_nkey._rcv_addr);
    };
    void RcvNkeyKey()
    {
    };
    void RcvNkeyReqAddr() // _rcv_key, _rcv_mac -> _snd_addr
    {
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
        else if(flg == 1) { print("*** key was changed\n");     }

        if(flg >= 0) ReplySig(_ptc_nkey._rcv_addr, flg);
    };
    void RcvNkeyRepSig()
    {
        int flg = _ptc_nkey._rcv_sig_flg;

        if     (flg == 0) { print("*** key was not changed\n"); }
        else if(flg == 1) { print("*** key was changed\n");     }
    };

    /////////////////////////////////////////////////////////////
    // network interface functions

    // connect
    //  return : src_id (if connecting failed, it will be -1)
    int Connect(const kmAddr4 addr, float tout_msec = 100.f)
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
                print("** connect failed (to %s)\n", addrs(i).GetStr().P());
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

        ret = ConnectLastAddr(uid); if(ret >= 0) return ret;
        ret = ConnectInLan   (uid); if(ret >= 0) return ret;
        ret = ConnectInWan   (uid); if(ret >= 0) return ret;

        print("** connecting fails\n");

        return -1;
    };

    // connect with last connected address
    //  return : src_id (if connecting failed, it will be -1)
    int ConnectLastAddr(int uid)
    {
        if(uid >= _users.N1()) return -1;

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

        // connect
        return Connect(addr);
    };

    // send my info to opposite    ...data_id = 0
    void SendInfo(ushort src_id)
    {
        uchar data_id = 0; kmNetBuf buf(256);

        ushort tmpName[20] = {0,};
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

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };
    void RcvReqRegist(ushort src_id, kmNetBuf& buf)
    {
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

        // accept the opposite
        AcceptRegist(src_id, role);
    };

    // accept registration (svr to clt)...data_id = 2
    void AcceptRegist(ushort src_id, zbRole role)
    {
        uchar data_id = 2; kmNetBuf buf(256);

        buf << _mac << _name << role << _pkey;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };
    void RcvAcceptRegist(ushort src_id, kmNetBuf& buf)
    {
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
        }

        // add users
        zbUser user(mac, name, role, _ids(src_id).addr, src_id, pkey);

        print("** user was added\n"); user.PrintInfo();

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
        case zbNoti::none        : print("* receive zbNot::none\n");        break;
        default: break;
        }
    };

    // request list (user, strg, file)... data_id = 4    
    //    if sid   < 0 --> request strg list
    //    if fid_s < 0 --> request file list
    //    else         --> file from fid_s to fid_e
    void ReqList(int uid, int sid = -1, int fid_s = -1, int fid_e = -1)
    {
        uchar data_id = 4;  kmNetBuf buf(32);

        buf << sid << fid_s << fid_e;

        const ushort src_id = _users(uid).src_id;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };
    void RcvReqList(ushort src_id, kmNetBuf& buf)
    {
        int sid, fid_s, fid_e;

        buf >> sid >> fid_s >> fid_e;
        
        if(sid < 0) // request strg list
        {
        }
        else if(fid_s < 0) // request file list
        {
        }
        else // requset files (from fid_s to fid_e)
        {
            int uid = FindUser(src_id);

            const int file_n = (int)GetFiles(uid,sid).N1();

            if(file_n == 0) return;

            fid_e = MIN(MAX(fid_s, fid_e), file_n - 1);

            // send files
            for(int fid = fid_s; fid <= fid_e; ++fid)
            {
                SendFile(uid, sid, fid);
            }
        }
    };

    // request file (user, strg, file)... data_id = 4
    inline void ReqFile(int uid, int sid, int fid_s, int fid_e = -1)
    {
        ReqList(uid, sid, fid_s, fid_e); 
    };

    // delete file on svr... date_id = 5
    void DelFile(int uid, int sid, int fid)
    {
        uchar data_id = 5; kmNetBuf buf(8);

        buf << sid << fid;

        const ushort src_id = _users(uid).src_id;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };
    void RcvDelFile(ushort src_id, kmNetBuf& buf)
    {
        int uid = FindUser(src_id), sid, fid;

        buf >> sid >> fid;

        // delete file
        kmStrw& path  = _users(uid).strgs(sid).path;
        zbFile& file  = _users(uid).strgs(sid).files(fid);

        if(file.state == zbFileState::bkupno || file.state == zbFileState::bkup)
        {    
            kmFile::Remove(kmStrw(L"%S/%S",path.P(), file.name.P()).P());
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
        uchar data_id = 6; kmNetBuf buf(8);

        buf << sid << fid;

        const ushort src_id = _users(uid).src_id;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 100.f);
    };
    void RcvAckDelFile(ushort src_id, kmNetBuf& buf)
    {
        int uid = FindUser(src_id), sid, fid;

        buf >> sid >> fid;

        // modifie file state
        zbFileState& state = _users(uid).strgs(sid).files(fid).state;

        if     (state == zbFileState::deletednc) { state = zbFileState::deleted; SaveFileList(uid, sid); }
        else if(state == zbFileState::bkupbannc) { state = zbFileState::bkupban; SaveFileList(uid, sid); }
    };

    // request thumbnail image... data_id = 7
    void ReqThumb(int uid, int sid, int fid_s, int fid_e, int w_pix = 0, int h_pix = 0)
    {
        uchar data_id = 7; kmNetBuf buf(24);

        buf << sid << fid_s << fid_e << w_pix << h_pix;

        const ushort src_id = _users(uid).src_id;

        Send(src_id, data_id, buf.P(), (ushort)buf.N1(), 1200.f);
    };
    void RcvReqThumb(ushort src_id, kmNetBuf& buf)
    {
/* for server
        int uid = FindUser(src_id), sid, fid_s, fid_e, w_pix, h_pix;

        buf >> sid >> fid_s >> fid_e >> w_pix >> h_pix;

        // send files
        zbFiles& files = GetFiles(uid, sid);

        fid_e = MIN(fid_e, (int)files.N1());

        for(int fid = fid_s; fid <= fid_e; ++fid)
        {
            SendThumb(uid, sid, fid, w_pix, h_pix);
        }
*/
    };

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

        // add link storage ... only for test
        if(user.role == zbRole::member) // svr
        {
            AddStrgLnk(uid, 0, 0);
        }
        else if(user.role == zbRole::svr) // clt
        {
            AddStrg(uid, zbStrgType::imgs, _dwnpath);
        }

        // save strg.list
        SaveStrgs(uid);
    };

    // save users
    void SaveUsers()
    {
        if(_users.N1() == 0) return;
        
        kmFile file(kmStrw(L"%S/.user.list", _path.P()).P(), KF_NEW);

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
        kmFile file(kmStrw(L"%S/.user.list", _path.P()).P());

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
        kmFile file(kmStrw(L"%S/.strg.list", _users(uid).path.P()).P(), KF_NEW);

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
        kmFile file(kmStrw(L"%S/.strg.list", _users(uid).path.P()).P());

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

        for(int i = 0; i < files.N1(); ++i)
        {
            if (files(i).state != zbFileState::bkupno) continue;

            SendFile(uid,sid,i);
        }

        return 1;
    }

    int checkStateOfLastFile(int uid, int sid)
    {
        zbFiles& files = _users(uid).strgs(sid).files;

        if (files.N1() < 1) return 0;

        while (files(files.N1()-1).state != zbFileState::bkup) {sleep(1);}

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
                // add file
                int add_n = UpdateFile(uid, sid, buf, fileSendFlag);

                // save file list
                if(add_n > 0) SaveFileList(uid, sid);
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

        // add file
        int add_n = 0;

        for(int i = 0; i < flst.N1(); ++i)
        {
            const kmFileInfo& flsti = flst(i);
            
            if(flsti.IsRealDir())
            {
                kmStrw path(L"%S%S/", subpath.P(), flsti.name.P());

                (subpath += flsti.name) += L"/";
                
                add_n += UpdateFile(uid, sid, subpath, fileSendFlag);

                subpath.Cutback(flsti.name.GetLen()); // including L'/'
            }
            else if(flsti.IsNormal())
            {
                // check if flst(i) is already in flies
                int isin = 0; subpath += flsti.name;

                for(int k = 0; k < files.N1(); ++k)
                {
                    if(subpath == files(k).name) { isin = 1; break; }
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
    void SendBkupno()
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
        char cname[100] = {0,};
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

            if(mdf._type == kmMdfType::unknown) file.type = zbFileType::data;
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
            char cname[100] = {0,};
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
    };

    // delete file on both svr and clt
    void DeleteFileBoth(int uid, int sid, int fid)
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
    };

    // ban backup
    void BanBkup(int uid, int sid, int fid)
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
        SaveFileList(uid, sid);
    };

    // send file to user
    void SendFile(int uid, int sid, int fid)
    {
        // init variables
        ushort  src_id = _users(uid).src_id;
        zbStrg& strg   = _users(uid).strgs(sid);
        zbFile& file   = GetFiles(uid,sid)(fid);

        // enqueue the file to send
        kmNet::EnqueueSndFile(src_id, strg.srcpath, file.name, sid, fid);
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

        kmStrw name(L"%S/.file.list", strg.path.P());

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
        kmStrw   path(L"%S/.file.list", strg.path.P());

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
            char cname[100] = {0,};
            wcstombs(cname, file.name.P(), wcslen(file.name.P())*2);
            cout<<cname<<"  ";

            switch(file.type)
            {
            case zbFileType::data   : wcout<<L" [data]   "; break;
            case zbFileType::dummy  : wcout<<L" [dummy]  "; break;
            case zbFileType::image  : wcout<<L" [image]  "; break;
            case zbFileType::movie  : wcout<<L" [movie]  "; break;
            case zbFileType::folder : wcout<<L" [folder] "; break;
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
            }
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
            }
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
        _cachepath.SetStr(temp);
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

    /////////////////////////////////////////////////////////////
    // nsk functions

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
        wchar wname[100] = {0,};
        mbstowcs(wname, deviceName.c_str(), deviceName.length());
        _name.SetStr(wname);
    }
};

#endif /* __zbNet_H_INCLUDED_2022_04_07__ */

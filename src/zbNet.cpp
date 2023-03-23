// base header
#include "../inc/zbNet.h"
#include "../inc/km7Jsn.h"

const auto JSON_INDENT = 2;

#define NAME "name"
#define TYPE "type"
#define ROLE "role"
#define KEY  "key"
#define MAC  "MAC"
#define TIME "time"
#define OWNER "owner"

#define UID  "uid"
#define SID  "sid"
#define FID  "fid"

#define CMD  "cmd"
#define URL  "url"
#define BODY "body"
#define RESP_MSG "resp_msg"
#define RES_CODE "result_code"

enum CODE {
    SUCCESS_E = 200,
    INVALID_E = 400,
    TIMEOUT_E = 500
};

#define ORIGIN   "origin"
#define USERINFO "userinfo"

////////////////////////
// command value
////////////////////////
#define CRTE "create"
#define GET  "get"
#define ADD  "add"
#define PUT  "put"
#define CHG  "change"
#define DEL  "delete"
#define UPLOAD "upload"
#define REP  "rep"
#define NOTI "noti"
#define RESET "reset"

////////////////////////
// item value
////////////////////////
#define USERLIST "userlist"
#define SHRD     "shrd"
#define PROFILE  "profile"
#define FILELIST "filelist"
#define FILES    "files"
#define THUMB    "thumb"
#define DOWNLOAD "download"

////////////////////////
// Shared Album
////////////////////////
#define FILE     "file"
#define USER     "user"
#define STRG     "strg"

#define ALBUM     "album"
#define ALBUMS    "albums"
#define ALBUM_NAME "album_name"
#define ALBUMLIST "albumlist"
#define ALBUM_ID  "album_id"
#define ALBUM_TYPE "album_type"
#define ALBUM_IMG "album_img"

#define PARTNER     "partner"
#define PARTNERS    "partners"
#define TO          "to"
#define FROM        "from"
#define PARTNER_ID  "partner_id"
#define PARTNER_NAME "partner_name"
#define PARTNERALBUM "partneralbum"
#define PARTNER_INFO "partner_info"
#define MEMBER "member"
#define MEMBERS "members"
#define MEMBER_UID  "uid"
#define MEMBER_AUTH "auth"

#define AMOUNT "amount"

/////////////////
// other
/////////////////
#define MEMORYINFO "memoryinfo"
#define TOTAL "total"
#define USED  "used"
#define FREE  "free"

#define MYINFO "myinfo"

#define MAX_ALBUMLIST 10
#define MAX_FILELIST  50


#define RETRY_COUNT 3

// + uid, sid, fid

// for debug
#define KKT(A) {cout<<"[kkt.jin] "<<__FILE__<<":"<<__func__<<":"<<__LINE__<<", "<<A<<endl;}

///////////////////////////////////////////////////////////////
// base util function
vector<string> split(const string& str, const char& Delimiter) {
    istringstream iss(str);
    string buffer = "";

    vector<string> result;

    while (getline(iss, buffer, Delimiter)) {
        result.push_back(buffer);
    }

    return result;
}

void setResultCode(json& j, const int& code, string str = "")
{
    j[RES_CODE] = code;
    if (code != 200)
    {
        j[BODY][RESP_MSG] = str;
    }
}

///////////////////////////////////////////////////////////////
// json functions

kmStra getstr(json& jsn, const char* str) try
{
    return kmStra(jsn.at(str).get<string>().c_str());
}
catch (json::out_of_range& e) { cout << e.what() << endl; return kmStra(); }
catch (json::type_error& e) { cout << e.what() << endl; return kmStra(); };

string getstring(json& jsn, const char* str) try
{
    return jsn.at(str).get<string>();
}
catch (json::out_of_range& e) { cout << e.what() << endl; return string(); }
catch (json::type_error& e) { cout << e.what() << endl; return string(); };

json getjson(json& jsn, const char* str) try
{
    return jsn.at(str);
}
catch (json::out_of_range& e) { cout << e.what() << endl; return nullptr; }
catch (json::type_error& e) { cout << e.what() << endl; return nullptr; };

template<typename T>
T get(json& jsn, const char* str, T default_value = 0) try
{
    return jsn.at(str).get<T>();
}
catch (json::out_of_range& e) { cout << e.what() << endl; return default_value; }
catch (json::type_error& e) { cout << e.what() << endl; return default_value; };

///////////////////////////////////////////////////////////////
// zbUser class members

// get json string... return value is UTF-8
json zbUser::ToJson(int uid)
{
    json jsn;

    jsn[NAME] = name.P();
    jsn[ROLE] = GetRoleStr().P();
    jsn[KEY]  = key.i64();
    jsn[MAC]  = mac.i64;
    jsn[UID]  = uid;

    return jsn;
};

// set from json string... jsna is UTF-8
//   return : self uid
int zbUser::FromJson(const kmStra& jsna)
{
    json jsn = json::parse(jsna.P());

    // set name
    name = kmStra(jsn.at(NAME).get<string>().c_str());

    // set role
    SetRole(kmStra(jsn.at(ROLE).get<string>().c_str()));

    // set mac and kye
    mac.i64 = jsn.at(MAC).get<int64>();
    *((int64*)&key) = jsn.at(KEY).get<int64>();

    // return self uid
    return jsn.at(UID).get<int>();
};

///////////////////////////////////////////////////////////////
// zbShrd class members

// get json string
json zbShrd::ToJson(int album_id)
{
    json jsn;

    jsn[ALBUM_ID] = album_id;
    string typeStr = "file";
    if (type == zbShrdType::strg) typeStr = "strg";
    else if (type == zbShrdType::user) typeStr = "user";
    jsn[ALBUM_TYPE] = typeStr;
    jsn[NAME]  = name.P();
    jsn[TIME]  = time.GetStrPt().P();
    jsn[OWNER] = owner_uid;

    for (int i = 0; i < mmbs.N1(); ++i)
    {
        jsn[MEMBERS].push_back(mmbs(i).uid);
    }

    return jsn;
};

// set from json string
void zbShrd::FromJson(const kmStra& jsna)
{
};

///////////////////////////////////////////////////////////////
// zbShrdItm class members

// get json string... return value is UTF-8
json zbShrdItm::ToJson()
{
    json jsn;

    jsn[UID] = uid;
    jsn[SID] = sid;
    jsn[FID] = fid;

    return jsn;
};

///////////////////////////////////////////////////////////////
// zbShrdMember class members
json zbShrdMember::ToJson()
{
    json jsn;

    jsn[MEMBER_UID] = uid;
    string authStr = "none";
    if (auth == zbShrdAuth::readonly) authStr = "readonly";
    else if (auth == zbShrdAuth::readwrite) authStr = "readwrite";
    else if (auth == zbShrdAuth::admin) authStr = "admin";
    jsn[MEMBER_AUTH] = authStr;

    return jsn;
}

///////////////////////////////////////////////////////////////
// zbNet class members

#include <chrono>
using namespace chrono;

//////////////////////////////
// parse json... level 1
void zbNet::ParseJson(ushort src_id, char* jsnap, int n, ushort ack_id)
{
    int uid = FindUser(src_id); if (uid < 0) return;

    // set json string
    kmStra jsna; jsna.Set(jsnap, n);

    // find first character
    int is = 0; for (; is < n; ++is) if (jsna(is) != ' ') break;

    if (is == n) return;

    // get cmd, url(path + item + prms), body    
    string cmd, url; json body;

    if (jsna(is) != '{') // cmd + url + json
    {
        // get command
        int ec_idx = 0; // end of command

        cmd = jsna.FindWord(ec_idx).P();

        // get url and body
        int sl_idx = jsna.Find('/');                // first slash
        int br_idx = jsna.Find('{');                // first brace
        int eu_idx = (br_idx < 0) ? n - 1 : br_idx - 1; // end of url

        kmStra url_str = (sl_idx < 0) ? kmStra() : jsna.Get(kmI(sl_idx, eu_idx));
        kmStra body_str = (br_idx < 0) ? kmStra() : jsna.Get(kmI(br_idx, n - 1));

        if (url_str.N1() > 0) url = url_str.P();
        if (body_str.N1() > 0) body = json::parse(body_str.P());
    }
    else // full json
    {
        print("***** full json\n");

        json jsn = json::parse(jsna.Get(kmI(is, end32)).P());

        cmd = getstring(jsn, CMD);
        url = getstring(jsn, URL);
        body = getjson(jsn, BODY);
    }
    //cmd.RemoveSpace();
    //url.RemoveSpace();

    cout << "** cmd  : " << cmd << endl;
    cout << "** url  : " << url << endl;

    // get path, item prms from url
    /*
    kmStra path, item, prms;

    if (url.N1() > 1)
    {
        int qm_idx = url.Find('?');     // question mark
        int ls_idx = url.FindRvrs('/'); // last slash

        path = (ls_idx < 1) ? kmStra() : url.Get(kmI(0, ls_idx - 1));
        item = (ls_idx < 0) ? kmStra() : url.Get(kmI(ls_idx + 1, qm_idx - 1));
        prms = (qm_idx < 0) ? kmStra() : url.Get(kmI(qm_idx + 1, end32));
    }

    print("** path : %s\n", path.P());
    print("** item : %s\n", item.P());
    print("** prms : %s\n", prms.P());
    print("** body : %s\n", body.P());
    */
    // set _jsn_snd_ack_id for SendJsonRep
    if (cmd != REP) _jsn_snd_ack_id = ack_id;

    // parse command
    if      (cmd == GET)    ParseJsonGet   (uid, jsna, url, body);
    else if (cmd == ADD)    ParseJsonPut   (uid, jsna, url, body);
    else if (cmd == PUT)    ParseJsonPut   (uid, jsna, url, body);
    else if (cmd == CRTE)   ParseJsonCreate(uid, jsna, url, body);
    else if (cmd == CHG)    ParseJsonChange(uid, jsna, url, body);
    else if (cmd == DEL)    ParseJsonDel   (uid, jsna, url, body);
    else if (cmd == REP)    ParseJsonRep   (uid, jsna, url, body);
    else if (cmd == UPLOAD) ParseJsonUpload(uid, jsna, url, body);
    else if (cmd == RESET)  ParseJsonReset (uid, jsna, url, body);
    else
    {
        // set json
        json jsn;

        jsn[CMD] = REP;
        if (!body.is_null()) jsn[ORIGIN] = body;
        setResultCode(jsn, 400, "invalid cmd!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }

    // * Note that if wack, it will be deleted in SendJsonSync()
    _jsnbuf.FindDelNack(ack_id);
};

//////////////////////////////
string zbNet::GetFirstParam(string& url)
{
    if (url.empty()) return "";

    size_t pos = url.find('/', 1);
    if (pos >= url.length()) pos = url.length();

    string item = url.substr(1, pos - 1);
    url = url.substr(pos, url.length() - pos);

    return item;
}

bool zbNet::CheckDigit(const string& str)
{
    if (str.length() < 1) return false;

    for (int i = 0; i < str.length(); ++i)
    {
        if (str[i] < '0' || str[i] > '9') return false;
    }
    return true;
}

int zbNet::GetAlreayExistUserOwnerAlbum(int uid)
{
    for (int i = 0; i < _shrds.N1(); ++i)
    {
        if (_shrds(i).owner_uid == uid &&
            _shrds(i).type == zbShrdType::user &&
            _shrds(i).state != zbShrdState::deleted) return i;
    }
    return -1;
}

string zbNet::getStrFileType(const zbFileType& type)
{
    string str = "image";

    switch (type)
    {
    case zbFileType::data:   str = "data"; break;
    case zbFileType::folder: str = "folder"; break;
    case zbFileType::image:  str = "image"; break;
    case zbFileType::movie:  str = "movie"; break;
    default: break;
    }

    return str;
}

void zbNet::SetFileInfoToJson(int uuid, int uid, int sid, int fid, json& j, bool sharedName)
{
    zbFile& file = _users(uid).strgs(sid).files(fid);
    if (file.name.N1() > 1)
    {
        if (sharedName)
        {
            string filename;
            filename.assign(file.name);
            size_t dotIdx = filename.rfind('.');
            j[NAME] = "sh"+to_string(uuid)+"-"+to_string(uid)+"-"+to_string(sid)+"-"+to_string(fid)+filename.substr(dotIdx);
        }
        else
        {
            j[NAME] = file.name;
        }
    }

    j[TYPE] = getStrFileType(file.type);
}

/**
@brief range string data에서 start index와 end index를 추출.
@detail range의 format은 2가지
           1. {start idx}-{end idx}   : ex) 0-22
           2. specific idx            : ex) 2

@param range[in] : range string data
@param startIdx[out] : range에서 뽑아낸 start index
                       format 2 인 경우, start index
@param endIdx[out] : range에서 뽑아낸 end index
                     format 2 인 경우, end index = start index

@return - true : detail의 1, 2번 format을 만족한 경우
        - false : 1. 입력 range string이 비어있는 경우
                  2. start index나 end index가 숫자가 아닌 경우
                  3. start index가 end index보다 큰 경우
*/
bool zbNet::GetRangeIndex(string& range, int& startIdx, int& endIdx)
{
    // user_id
    if (range.empty())
        return false;

    size_t pos = range.find('-');
    string startStr = range;
    string endStr = startStr;
    if (pos != std::string::npos)
    {
        startStr = range.substr(0, pos);
        endStr = range.substr(pos + 1);
    }

    if (!CheckDigit(startStr) || !CheckDigit(endStr))
        return false;

    startIdx = stoi(startStr);
    endIdx = stoi(endStr);

    if (startIdx > endIdx)
        return false;

    return true;
}

//////////////////////////////
// parse json... level 2

// pasre json for command get
void zbNet::ParseJsonGet(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGet\n");

    string item = GetFirstParam(url);
    if (item == USERLIST)  ParseJsonGetUserList(uid, jsna, url, body);
    else if (item == SHRD) ParseJsonGetShrd(uid, jsna, url, body);
    else if (item == PARTNERALBUM) ParseJsonGetPartnerAlbum(uid, jsna, url, body);
    else if (item == PARTNER) ParseJsonGetPartner(uid, jsna, url, body);
    else if (item == PROFILE) ParseJsonGetProfile(uid, jsna, url, body);
    else if (item == MEMORYINFO) ParseJsonGetMemoryInfo(uid, jsna, url, body);
    else if (item == MYINFO) ParseJsonGetMyInfo(uid, jsna, url, body);
    //else if (item == FILE) ParseJsonGetFile(uid, jsna, url, body);
    else
    {
        // set json
        json jsn;

        jsn[CMD] = REP;
        if (!body.is_null()) jsn[ORIGIN] = body;
        setResultCode(jsn, 400, "get invalid param!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }
};

// parse json for command put
void zbNet::ParseJsonPut(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonPut\n");

    string item = GetFirstParam(url);
    if (item == SHRD) ParseJsonPutShrd(uid, jsna, url, body);
    else if (item == PARTNER) ParseJsonPutPartner(uid, jsna, url, body);
    else
    {
        // set json
        json jsn;

        jsn[CMD] = REP;
        if (!body.is_null()) jsn[ORIGIN] = body;
        setResultCode(jsn, 400, "put/add invalid param!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }
};

// parse json for command del
void zbNet::ParseJsonCreate(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonCreate\n");

    string prm = GetFirstParam(url);
    if (prm == SHRD) ParseJsonCreateShrd(uid, jsna, url, body);
    else
    {
        // set json
        json jsn;

        jsn[CMD] = REP;
        if (!body.is_null()) jsn[ORIGIN] = body;
        jsn[BODY][RESP_MSG] = "create invalid param!!";
        setResultCode(jsn, 400, "create invalid param!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }
};

// parse json for command change
void zbNet::ParseJsonChange(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonCange\n");

    string item = GetFirstParam(url);
    if (item == NAME) ParseJsonChangeName(uid, jsna, url, body);
    else
    {
        // set json
        json jsn;

        jsn[CMD] = REP;
        if (!body.is_null()) jsn[ORIGIN] = body;
        setResultCode(jsn, 400, "delete invalid param!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }
};

void zbNet::ParseJsonChangeName(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonChangeName\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    int code = 200;
    string reason = "";

    string name = GetFirstParam(url);
    if (name.empty())
    {
        code = 400;
        reason = "invalid user name!!";
    }    
    _users(uid).name = kmStra(name.c_str());
    SaveUsers();

    setResultCode(jsn, code, reason);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};

// parse json for command del
void zbNet::ParseJsonDel(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonDel\n");

    string item = GetFirstParam(url);
    if (item == SHRD) ParseJsonDelShrd(uid, jsna, url, body);
    else if (item == PARTNER) ParseJsonDelPartner(uid, jsna, url, body);
    else
    {
        // set json
        json jsn;

        jsn[CMD] = REP;
        if (!body.is_null()) jsn[ORIGIN] = body;
        setResultCode(jsn, 400, "delete invalid param!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }
};

// parse json rep for command rep
void zbNet::ParseJsonRep(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonRep\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;
    setResultCode(jsn, 400, "rep invalid param!!");
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};

// parse json rep for command upload
void zbNet::ParseJsonUpload(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonUpload\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string prm = GetFirstParam(url);
    if (prm != PROFILE)
    {
        setResultCode(jsn, 400, "invalid url, for profile!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    prm = GetFirstParam(url);
    if (prm == USER) ParseJsonUploadUser(uid, jsna, url, body);
    else if (prm == SHRD) ParseJsonUploadShrd(uid, jsna, url, body);
    else
    {
        setResultCode(jsn, 400, "upload profile invalid param!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }
};

// parse json rep for command reset
void zbNet::ParseJsonReset(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonReset\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    setResultCode(jsn, 200);

    // send json
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));

    Reset();
};

// parse json rep for command Profile
void zbNet::ParseJsonGetProfile(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetProfile\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string prm = GetFirstParam(url);
    if (prm == USER) ParseJsonGetProfileUser(uid, jsna, url, body);
    else if (prm == ALBUM) ParseJsonGetProfileAlbum(uid, jsna, url, body);
    else
    {
        setResultCode(jsn, 400, "upload profile invalid param!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }
};

//////////////////////////////
// parse json... level 3

// parse json to get userlist
void zbNet::ParseJsonGetUserList(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetUserList\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    for (int i = 0; i < _users.N1(); ++i)
    {
        jsn[BODY][USERINFO].push_back(_users(i).ToJson(i));
    }

    setResultCode(jsn, 200);

    // send json
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
}

// parse json to get shared
void zbNet::ParseJsonGetShrd(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetShrd\n");

    string prm = GetFirstParam(url);
    if (prm == ALBUMLIST) ParseJsonGetShrdAlbumList(uid, jsna, url, body);
    else if (prm == ALBUM) ParseJsonGetShrdAlbum(uid, jsna, url, body);
    else if (prm == MEMBER) ParseJsonGetShrdMember(uid, jsna, url, body);
    else
    {
        // set json
        json jsn;

        jsn[CMD] = REP;
        if (!body.is_null()) jsn[ORIGIN] = body;
        setResultCode(jsn, 400, "need param for get command!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }

    return;
}

// parse json to get shared
void zbNet::ParseJsonDelShrd(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonDelShrd\n");

    string prm = GetFirstParam(url);
    if (prm == ALBUM) ParseJsonDelShrdAlbum(uid, jsna, url, body);
    else if (prm == MEMBER) ParseJsonDelShrdMember(uid, jsna, url, body);
    else
    {
        // set json
        json jsn;

        jsn[CMD] = REP;
        if (!body.is_null()) jsn[ORIGIN] = body;
        setResultCode(jsn, 400, "need param for delete command!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }

    return;
}

// parse json to get shared
void zbNet::ParseJsonCreateShrd(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonCreateShrd\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string prm = GetFirstParam(url);
    if (prm.empty())
    {
        setResultCode(jsn, 400, "invalud create param!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    vector<int> partnerVector;
    zbShrd shrd = { zbShrdType::file, "", uid };
    if (prm == USER)
    {
        shrd.type = zbShrdType::user;
        if (GetAlreayExistUserOwnerAlbum(uid) != -1)
        {
            setResultCode(jsn, 400, "user's shared album already exist!!");
            SendJsonRep(uid, kmStra(jsn.dump().c_str()));
            return;
        }

        if (!body.is_null())
        {
            //if (!body[NAME].is_null())
            {
                //string name = body[NAME];
                //if (!name.empty())
                {
                    shrd.name = _users(uid).name + kmStra("의 앨범");
                }
            }
            if (!body[PARTNERS].is_null())
            {
                for (auto v : body[PARTNERS])
                {
                    if (!v.is_null())
                    {
                        if (v < _users.N1() && v != uid)
                        {
                            shrd.AddMember(v);
                            partnerVector.push_back(v);
                        }
                    }
                }
            }
        }
    }
    else if (prm == FILE)
    {
        if (!body[NAME].is_null())
        {
            string name = body[NAME];
            if (!name.empty())
            {
                shrd.name = kmStra(name.c_str());
            }
        }
    }
    else if (prm == STRG)
    {
        shrd.type = zbShrdType::strg;

        if (!body.is_null() && !body[PARTNERS].is_null())
        {
            for (auto v : body[PARTNERS])
            {
                if (!v.is_null())
                {
                    if (v < _users.N1() && v != uid)
                    {
                        shrd.AddMember(v);
                        partnerVector.push_back(v);
                    }
                }
            }
        }
    }

    shrd.AddMember(uid);

    int shrd_id = AddShrd(shrd);
    SaveShrdList();

    jsn[BODY][ALBUM_ID] = shrd_id;
    setResultCode(jsn, 200);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));

    for (auto v : partnerVector)
    {
        SendNotiPartner(uid, v, ADD);
    }

    return;
}

void zbNet::ParseJsonGetShrdAlbum(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetShrdAlbum\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string albumStr = GetFirstParam(url);
    if (albumStr.empty() || !CheckDigit(albumStr))
    {
        setResultCode(jsn, 400, "get shared invalid album_id");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    int album_id = stoi(albumStr);
    if (_shrds.N1() <= album_id)
    {
        setResultCode(jsn, 400, "album does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    auto& shrd = _shrds(album_id);
    if (shrd.state == zbShrdState::deleted)
    {
        setResultCode(jsn, 400, "deleted album!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    string prm = GetFirstParam(url);
    if (prm == FILELIST) ParseJsonGetShrdAlbumFileList(uid, url, shrd, body);
    else if (prm == FILE) ParseJsonGetShrdAlbumFile(uid, url, body);
    else if (prm == DOWNLOAD) ParseJsonGetShrdAlbumDownload(uid, url, body);
    else if (prm == THUMB) ParseJsonGetShrdAlbumThumb(uid, url, body);
    else
    {
        jsn[BODY][ALBUM] = shrd.ToJson(album_id);
        setResultCode(jsn, 200);
        // send json
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }
};

void zbNet::ParseJsonGetShrdAlbumFileList(int uid, string& url, zbShrd& shrd, json& body)
{
    print("*** ParseJsonGetShrdAlbumFileList\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string param = GetFirstParam(url);
    if (param == AMOUNT)
    {
        int64 amount = 0;
        if (shrd.type == zbShrdType::file)
        {
            amount = shrd.itms.N1();
        }
        else if (shrd.type == zbShrdType::user)
        {
            for (int i = 0; i < _users(shrd.owner_uid).strgs.N1(); ++i)
            {
                amount += _users(shrd.owner_uid).strgs(i).files.N1();
            }
        }
        jsn[BODY][AMOUNT] = amount;
        setResultCode(jsn, 200);
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    int startIdx = 0;
    int endIdx = 0;
    if (!GetRangeIndex(param, startIdx, endIdx))
    {
        setResultCode(jsn, 400, "invalid file range. ex)/shrd/album/2/filelist/0-2!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    if (shrd.type == zbShrdType::file)
    {
        if (shrd.itms.N1() < 1)
        {
            setResultCode(jsn, 400, "empty files!!");
            SendJsonRep(uid, kmStra(jsn.dump().c_str()));
            return;
        }

        for (int i = startIdx; (i < shrd.itms.N1()) && (i <= endIdx); ++i)
        {
            auto j = shrd.itms(i).ToJson();
            string type;
            type.assign(_users(shrd.itms(i).uid).strgs(shrd.itms(i).sid).files(shrd.itms(i).fid).name.P());
            type = type.substr(type.rfind('.')+1);
            j[TYPE] = type;
            jsn[BODY][FILELIST].push_back(j);
        }
    }
    else if (shrd.type == zbShrdType::user)
    {
        for (int j = 0; j < _users(shrd.owner_uid).strgs.N1(); ++j)
        {
            zbFiles& files = _users(shrd.owner_uid).strgs(j).files;
            for (int i = startIdx; (i < files.N1()) && (i <= endIdx); ++i)
            {
                json file;
                file[UID] = shrd.owner_uid;
                file[SID] = 0;
                file[FID] = i;
                string type;
                type.assign(files(i).name.P());
                type = type.substr(type.rfind('.')+1);
                file[TYPE] = type;
                jsn[BODY][FILELIST].push_back(file);
            }
        }
    }

    setResultCode(jsn, 200);
    // send json
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};

void zbNet::ParseJsonGetShrdAlbumFile(int uid, string& url, json& body)
{
    print("*** ParseJsonGetShrdAlbumFile\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    // user_id
    string userStr = GetFirstParam(url);
    if (!CheckDigit(userStr))
    {
        setResultCode(jsn, 400, "invalid uid!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    if (!CheckDigit(userStr))
    {
        setResultCode(jsn, 400, "invalid uid!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int user_id = stoi(userStr);
    if (_users.N1() <= user_id)
    {
        setResultCode(jsn, 400, "user does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    // sid
    string sidStr = GetFirstParam(url);
    if (!CheckDigit(sidStr))
    {
        setResultCode(jsn, 400, "invalid sid!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    if (!CheckDigit(sidStr))
    {
        setResultCode(jsn, 400, "invalid sid");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int sid = stoi(sidStr);
    if (_users(user_id).strgs.N1() <= sid)
    {
        setResultCode(jsn, 400, "user's storage does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    // fid
    string fidStr = GetFirstParam(url);
    if (!CheckDigit(fidStr))
    {
        setResultCode(jsn, 400, "invalid fid!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    if (!CheckDigit(fidStr))
    {
        setResultCode(jsn, 400, "invalid fid!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int fid = stoi(fidStr);
    if (_users(user_id).strgs(sid).files.N1() <= fid)
    {
        setResultCode(jsn, 400, "file does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    kmNetPtcFileRes resp = SendFileShrdCache(uid, user_id, sid, fid, true);
    int code = 200;
    string str = "";

    switch (resp)
    {
    case kmNetPtcFileRes::idwrong: code = 400; str = "idwrong"; break;
    case kmNetPtcFileRes::inqueue: code = 400; str = "inqueue"; break;
    case kmNetPtcFileRes::nonexistence: code = 400; str = "nonexistence"; break;
    case kmNetPtcFileRes::optwrong: code = 400; str = "optwrong"; break;
    case kmNetPtcFileRes::preackrejected: code = 400; str = "preackrejected"; break;
    case kmNetPtcFileRes::preacktimeout: code = 400; str = "preacktimeout"; break;
    case kmNetPtcFileRes::preackwrong: code = 400; str = "preackwrong"; break;
    case kmNetPtcFileRes::skipmax: code = 400; str = "skipmax"; break;
    case kmNetPtcFileRes::sndstatewrong: code = 400; str = "sndstatewrong"; break;
    default: break;
    }

    SetFileInfoToJson(uid, user_id, sid, fid, jsn[BODY]);

    setResultCode(jsn, code, str);

    // send json
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};

void zbNet::ParseJsonGetShrdAlbumDownload(int uid, string& url, json& body)
{
    print("*** ParseJsonGetShrdAlbumDownload\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    // user_id
    string userStr = GetFirstParam(url);
    if (!CheckDigit(userStr))
    {
        setResultCode(jsn, 400, "invalid uid!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int user_id = stoi(userStr);
    if (_users.N1() <= user_id)
    {
        setResultCode(jsn, 400, "user does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    // sid
    string sidStr = GetFirstParam(url);
    if (!CheckDigit(sidStr))
    {
        setResultCode(jsn, 400, "invalid sid!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int sid = stoi(sidStr);
    if (_users(user_id).strgs.N1() <= sid)
    {
        setResultCode(jsn, 400, "user's storage does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    // fid
    string fidStr = GetFirstParam(url);
    if (!CheckDigit(fidStr))
    {
        setResultCode(jsn, 400, "invalid fid!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int fid = stoi(fidStr);
    if (_users(user_id).strgs(sid).files.N1() <= fid)
    {
        setResultCode(jsn, 400, "file does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    if (SendFileShrd(uid, user_id, sid, fid, true) == kmNetPtcFileRes::success)
    {
        SetFileInfoToJson(uid, user_id, sid, fid, jsn[BODY], false);
        setResultCode(jsn, 200);
    }
    else
    {
        setResultCode(jsn, 400, "get shared album download Fail!!");
    }

    // send json
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};

void zbNet::ParseJsonGetShrdAlbumThumb(int uid, string& url, json& body)
{
    print("*** ParseJsonGetShrdAlbumThumb\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    // user_id
    string userStr = GetFirstParam(url);
    if (!CheckDigit(userStr))
    {
        setResultCode(jsn, 400, "invalid uid!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int user_id = stoi(userStr);
    if (_users.N1() <= user_id)
    {
        setResultCode(jsn, 400, "user does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    // sid
    string sidStr = GetFirstParam(url);
    if (!CheckDigit(sidStr))
    {
        setResultCode(jsn, 400, "invalid sid!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int sid = stoi(sidStr);
    if (_users(user_id).strgs.N1() <= sid)
    {
        setResultCode(jsn, 400, "user's storage does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    // fid
    string fidStr = GetFirstParam(url);
    if (!CheckDigit(fidStr))
    {
        setResultCode(jsn, 400, "invalid fid!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int fid = stoi(fidStr);
    if (_users(user_id).strgs(sid).files.N1() <= fid)
    {
        setResultCode(jsn, 400, "file does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    if (SendThumbShrd(uid, user_id, sid, fid, true) == kmNetPtcFileRes::success)
    {
        SetFileInfoToJson(uid, user_id, sid, fid, jsn[BODY]);
        setResultCode(jsn, 200);
    }
    else
    {
        setResultCode(jsn, 400, "get shared album thumb Fail!!");
    }

    // send json
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};

void zbNet::ParseJsonGetShrdAlbumList(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetShrdAlbumList\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    if (_shrds.N1() < 1)
    {
        setResultCode(jsn, 400, "get ahred album not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    string param = GetFirstParam(url);
    if (param == AMOUNT)
    {
        int count = 0;
        for (int i = 0; i < _shrds.N1(); ++i)
        {
            if (_shrds(i).state != zbShrdState::deleted)
                count++;
        }
        jsn[BODY][AMOUNT] = count;
        setResultCode(jsn, 200);
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    int startIdx = 0;
    int endIdx = 0;
    if (!GetRangeIndex(param, startIdx, endIdx))
    {
        setResultCode(jsn, 400, "invalid album range. ex)/shrd/albumlist/0-2!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    for (int i = startIdx; (i < _shrds.N1()) && (i <= endIdx); ++i)
    {
        if (_shrds(i).state != zbShrdState::deleted)
        {
            json album = _shrds(i).ToJson(i);

            if (_shrds(i).itms.N1() > 0)
            {
                auto file = _shrds(i).itms(0);

                string type;
                type.assign(_users(file.uid).strgs(file.sid).files(file.fid).name.P());
                type = type.substr(type.rfind('.') + 1);

                album[ALBUM_IMG] = file.ToJson();
                album[ALBUM_IMG][TYPE] = type;
            }
            else
            {
                album[ALBUM_IMG] = "";
            }

            jsn[BODY][ALBUMS].push_back(album);
        }
    }

    if (jsn[BODY][ALBUMS].is_null())
    {
        setResultCode(jsn, 400, "album empty!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    // send json
    setResultCode(jsn, 200);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
}

void zbNet::ParseJsonGetPartnerAlbum(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetPartnerAlbum\n");
    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    int index = GetAlreayExistUserOwnerAlbum(uid);
    if (index == -1)
    {
        setResultCode(jsn, 400, "user's shared album not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    jsn[BODY][TO] = _shrds(index).ToJson(index);

    for (int i = 0; i < _shrds.N1(); ++i)
    {
        for (int m = 0; m < _shrds(i).mmbs.N1(); ++m)
        {
            if (_shrds(i).mmbs(m).uid == uid &&
                _shrds(i).state != zbShrdState::deleted &&
                _shrds(i).type == zbShrdType::user)
            {
                jsn[BODY][FROM].push_back(_shrds(i).ToJson(i));
            }
        }
    }

    setResultCode(jsn, 200);
    // send json
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};

void zbNet::ParseJsonGetPartner(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetShrdPartner\n");
    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string prm = GetFirstParam(url);
    if (prm.length() != 1 || !CheckDigit(prm))
    {
        setResultCode(jsn, 400, "partner id error!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    int partner_id = stoi(prm);
    bool find = false;
    zbShrd shrd;
    for (int index = 0; index < _shrds.N1(); ++index)
    {
        if (_shrds(index).owner_uid == partner_id)
        {
            auto& members = _shrds(index).mmbs;
            for (int i = 0; i < members.N1(); ++i)
            {
                if (members(i).uid == uid)
                {
                    shrd = _shrds(index);
                    find = true;
                }
            }
        }
    }

    if (!find)
    {
        setResultCode(jsn, 400, "shared album does not exist or does not shared to me!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    prm.clear();
    prm = GetFirstParam(url);
    if (prm == FILELIST)
    {
        string param = GetFirstParam(url);
        if (param == AMOUNT)
        {
            jsn[BODY][AMOUNT] = shrd.itms.N1();
            setResultCode(jsn, 200);
            SendJsonRep(uid, kmStra(jsn.dump().c_str()));
            return;
        }

        int startIdx = 0;
        int endIdx = 0;
        if (!GetRangeIndex(param, startIdx, endIdx))
        {
            setResultCode(jsn, 400, "invalid file range. ex)/partner/2/filelist/0-2!!");
            SendJsonRep(uid, kmStra(jsn.dump().c_str()));
            return;
        }

        json partnerInfo;
        partnerInfo[PARTNER_INFO] = _users(partner_id).ToJson(partner_id);
        for (int i = startIdx; (i < shrd.itms.N1()) && (i <= endIdx); ++i)
        {
            jsn[BODY][FILELIST].push_back(shrd.itms(i).ToJson());
        }
        setResultCode(jsn, 200);
    }
    else if (prm == PROFILE)
    {
        if (SendMmbProf(uid, partner_id, true) == kmNetPtcFileRes::success)
            setResultCode(jsn, 200);
        else
            setResultCode(jsn, 400, "partner profile sends fail!!");
    }
    else
    {
        setResultCode(jsn, 400, "invalid param!!");
    }

    // send json
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};

// parse json to get file
/*
void zbNet::ParseJsonGetFile(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetFile\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
        if (!body.is_null()) jsn[ORIGIN] = body;
    setResultCode(jsn, 400, "get file invalid param!!");
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};
*/

// parse json to put shrd
void zbNet::ParseJsonPutShrd(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonPutShrd\n");

    string prm = GetFirstParam(url);
    if (prm == ALBUM) ParseJsonPutShrdAlbum(uid, jsna, url, body);
    else if (prm == MEMBER) ParseJsonPutShrdMember(uid, jsna, url, body);
    else
    {
        // set json
        json jsn;

        jsn[CMD] = REP;
        if (!body.is_null()) jsn[ORIGIN] = body;
        setResultCode(jsn, 400, "add shared invalid param!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }
};

// parse json to put partner
void zbNet::ParseJsonPutPartner(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonPutPartner\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    int album_id = GetAlreayExistUserOwnerAlbum(uid);
    if (album_id == -1)
    {
        setResultCode(jsn, 400, "does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    string prm = GetFirstParam(url);
    if (prm.empty())
    {
        setResultCode(jsn, 400, "need partner_id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    if (!CheckDigit(prm))
    {
        setResultCode(jsn, 400, "partner_id error!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    int partner_id = stoi(prm);
    auto& shrd = _shrds(album_id);
    for (int i = 0; i < shrd.mmbs.N1(); ++i)
    {
        if (partner_id == shrd.mmbs(i).uid)
        {
            setResultCode(jsn, 400, "already exist partner!!");
            SendJsonRep(uid, kmStra(jsn.dump().c_str()));
            return;
        }
    }

    shrd.AddMember(partner_id);
    SaveShrd(partner_id);
    SaveShrdList();

    setResultCode(jsn, 200);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));

    SendNotiPartner(uid, partner_id, ADD);
};

void zbNet::ParseJsonPutShrdAlbum(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonPutShrdAlbum for file\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string albumStr = GetFirstParam(url);
    if (albumStr.empty())
    {
        setResultCode(jsn, 400, "need album_id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    if (!CheckDigit(albumStr))
    {
        setResultCode(jsn, 400, "invalid album_id");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int album_id = stoi(albumStr);

    if (_shrds.N1() <= album_id || _shrds(album_id).state == zbShrdState::deleted)
    {
        setResultCode(jsn, 400, "album does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    string file = GetFirstParam(url);
    if (file != FILE)
    {
        setResultCode(jsn, 400, "need url action(only file)!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    if (body[FILES].is_null())
    {
        setResultCode(jsn, 400, "file array empty!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    bool addFlag = false;
    auto& shrd = _shrds(album_id);
    for (auto f : body[FILES])
    {
        if (f[SID].is_null() || f[FID].is_null()) continue;

        // sid
        int sid = f[SID];

        // fid
        int fid = f[FID];

        if (_users(uid).strgs.N1() <= sid) continue;
        if (_users(uid).strgs(sid).files.N1() <= fid) continue;

        bool findFlag = false;
        for (int i = 0; i < shrd.itms.N1(); ++i)
        {   // TODO : need to change hash table, use map?
            if (shrd.itms(i).fid == fid &&
                shrd.itms(i).sid == sid &&
                shrd.itms(i).uid == uid)
            {
                findFlag = true;
                break;
            }
        }

        if (!findFlag)
        {
            shrd.AddItem(uid, sid, fid);
            addFlag = true;
        }
    }

    if (addFlag)
    {
        SaveShrd(album_id);
        SaveShrdList();

        setResultCode(jsn, 200);
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));

        for (int i = 0; i < shrd.mmbs.N1(); ++i)
        {
            //if (shrd.mmbs(i).uid != uid)
            {
                SendNotiFile(uid, shrd.mmbs(i).uid, album_id, CHG);
            }
        }
    }
    else
    {
        setResultCode(jsn, 400, "files empty or invalid file data!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }
}

void zbNet::ParseJsonPutShrdMember(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonPutShrdMember\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string albumStr = GetFirstParam(url);
    if (albumStr.empty())
    {
        setResultCode(jsn, 400, "need album_id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    if (!CheckDigit(albumStr))
    {
        setResultCode(jsn, 400, "invalid album_id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int album_id = stoi(albumStr);

    string userStr = GetFirstParam(url);
    if (userStr.empty())
    {
        setResultCode(jsn, 400, "need user_id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    if (!CheckDigit(userStr))
    {
        setResultCode(jsn, 400, "invalid user_id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int user_id = stoi(userStr);
    if (uid == user_id)
    {
        setResultCode(jsn, 400, "can not add, owner album!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    if (_shrds.N1() <= album_id || _shrds(album_id).state == zbShrdState::deleted)
    {
        setResultCode(jsn, 400, "album does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    auto& shrd = _shrds(album_id);
    for (int i = 0; i < shrd.mmbs.N1(); ++i)
    {
        if (shrd.mmbs(i).uid == user_id)
        {
            setResultCode(jsn, 400, "already exist!!");
            SendJsonRep(uid, kmStra(jsn.dump().c_str()));
            return;
        }
    }

    shrd.AddMember(user_id, zbShrdAuth::readonly);
    SaveShrd(album_id);
    SaveShrdList();

    setResultCode(jsn, 200);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));

    SendNotiMember(uid, user_id, album_id, ADD);
}

void zbNet::ParseJsonGetShrdMember(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetShrdMember\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    if (url.empty())
    {
        setResultCode(jsn, 400, "add member invalid url!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    if (_shrds.N1() < 1)
    {
        setResultCode(jsn, 400, "add member storage empty!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    int shrd_id = stoi(GetFirstParam(url));
    auto& shrd = _shrds(shrd_id);
    json members;
    for (int i = 0; i < shrd.mmbs.N1(); ++i)
    {
        members.clear();
        members[MEMBER_UID] = shrd.mmbs(i).uid;

        if (shrd.mmbs(i).auth == zbShrdAuth::readonly) members[MEMBER_AUTH] = "readonly";
        else if (shrd.mmbs(i).auth == zbShrdAuth::readwrite) members[MEMBER_AUTH] = "readwrite";
        else if (shrd.mmbs(i).auth == zbShrdAuth::admin) members[MEMBER_AUTH] = "admin";
        else members[MEMBER_AUTH] = "none";

        jsn[BODY][MEMBERS].push_back(members);
    }

    setResultCode(jsn, 200);
    // send json
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};

void zbNet::ParseJsonDelShrdAlbum(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonDelShrdAlbum\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    if (url.empty())
    {
        setResultCode(jsn, 400, "del shared album invalid url!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    if (_shrds.N1() < 1)
    {
        setResultCode(jsn, 400, "del shared album storage empty!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    string prm = GetFirstParam(url);
    if (prm.length() < 1 || !CheckDigit(prm))
    {
        setResultCode(jsn, 400, "del shared album invalid album_id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    int album_id = stoi(prm);

    if (_shrds.N1() <= album_id || _shrds(album_id).state == zbShrdState::deleted)
    {
        setResultCode(jsn, 400, "album does not exist!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    string file = GetFirstParam(url);
    if (file == FILE)
    {
        ParseJsonDelShrdAlbumFile(uid, jsna, album_id, body);
        return;
    }

    for (int i = 0; i < _shrds(album_id).mmbs.N1(); ++i)
    {
        if (uid != _shrds(album_id).mmbs(i).uid)
        {
            SendNotiAlbum(uid, _shrds(album_id).mmbs(i).uid, album_id, DEL);
        }
    }

    DeleteShrd(album_id);
    SaveShrdList();

    setResultCode(jsn, 200);

    // send json
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
}


void zbNet::ParseJsonDelShrdAlbumFile(int uid, kmStra& jsna, int album_id, json& body)
{
    print("*** ParseJsonDelShrdAlbumFile\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    if (body[FILES].is_null())
    {
        setResultCode(jsn, 400, "file array empty!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    bool deleteFlag = false;
    auto& shrd = _shrds(album_id);
    for (auto f : body[FILES])
    {
        if (f[SID].is_null() || f[FID].is_null()) continue;

        // sid
        int sid = f[SID];

        // fid
        int fid = f[FID];

        if (_users(uid).strgs.N1() <= sid) continue;
        if (_users(uid).strgs(sid).files.N1() <= fid) continue;

        for (int i = 0; i < shrd.itms.N1(); ++i)
        {   // TODO : need to change hash table, use map?
            if (shrd.itms(i).fid == fid &&
                shrd.itms(i).sid == sid &&
                shrd.itms(i).uid == uid)
            {
                shrd.itms.Erase(i);
                deleteFlag = true;
                break;
            }
        }
    }

    if (deleteFlag)
    {
        SaveShrd(album_id);
        SaveShrdList();

        setResultCode(jsn, 200);
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));

        for (int i = 0; i < shrd.mmbs.N1(); ++i)
        {
            if (shrd.mmbs(i).uid != uid)
            {
                SendNotiFile(uid, shrd.mmbs(i).uid, album_id, CHG);
            }
        }
    }
    else
    {
        setResultCode(jsn, 400, "files empty or invalid file data!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    }
}

void zbNet::ParseJsonDelPartner(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonDelPartner\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string prm = GetFirstParam(url);
    if (prm.length() < 1 || !CheckDigit(prm))
    {
        setResultCode(jsn, 400, "del partner invalid id");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    int album_id = GetAlreayExistUserOwnerAlbum(uid);
    if (album_id == -1)
    {
        setResultCode(jsn, 400, "does not exist shared album!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    int partner_id = stoi(prm);
    bool find = false;
    auto& shrd = _shrds(album_id);
    for (int index = 0; index < shrd.mmbs.N1(); ++index)
    {
        if (shrd.mmbs(index).uid == partner_id)
        {
            find = true;
            break;
        }
    }
    if (!find)
    {
        setResultCode(jsn, 400, "shared album does not exist or does not shared to me!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    shrd.mmbs.Erase(partner_id);
    SaveShrd(album_id);
    SaveShrdList();

    setResultCode(jsn, 200);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));

    SendNotiPartner(uid, partner_id, DEL);
}

void zbNet::ParseJsonDelShrdMember(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonDelShrdPartner\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    if (url.empty())
    {
        setResultCode(jsn, 400, "del shared album invalid url!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    string albumStr = GetFirstParam(url);
    if (!CheckDigit(albumStr))
    {
        setResultCode(jsn, 400, "invalid album id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    int album_id = stoi(albumStr);
    if (_shrds.N1() <= album_id)
    {
        setResultCode(jsn, 400, "del shared member, album id error!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    auto& shrd = _shrds(album_id);

    string userStr = GetFirstParam(url);
    if (!CheckDigit(userStr))
    {
        setResultCode(jsn, 400, "del shared member user id error");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int user_id = stoi(userStr);

    bool find = false;
    for (int i = 0; i < shrd.mmbs.N1(); ++i)
    {
        if (shrd.mmbs(i).uid == user_id)
        {
            shrd.mmbs.Erase(i);
            find = true;
            break;
        }
    }
    if (!find)
    {
        setResultCode(jsn, 400, "del shared member user does not exist");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    SaveShrd(album_id);
    SaveShrdList();

    setResultCode(jsn, 200);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));

    SendNotiMember(uid, user_id, album_id, DEL);
}

//#include <direct.h> // for _getdrive, _chdrive
void zbNet::ParseJsonGetMemoryInfo(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetMemoryInfo\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    /*
    char driveArray[30] = "";
    int driveCount = 0;
    int driveBackup = _getdrive();
    for (int i = 1; i <= 26; ++i)
    {
        if (_chdrive(i) == 0)
        {
            driveArray[driveCount++] = i + 'A' - 1;
        }
    }

    string drive;
    drive.assign(_path.EncWtoUtf8());
    drive = drive.substr(0, 2);
    */

    ULARGE_INTEGER uliAvailable;
    ULARGE_INTEGER uliTotal;
    ULARGE_INTEGER uliFree;
    GetDiskFreeSpaceExA(_path.P(), &uliAvailable, &uliTotal, &uliFree);

    int iAvailable = (int)(uliAvailable.QuadPart >> 20);
    int iTotal = (int)(uliTotal.QuadPart >> 20);
    int iFree = (int)(uliFree.QuadPart >> 20);

    jsn[BODY][TOTAL] = iTotal;
    jsn[BODY][USED] = iTotal - iFree;
    jsn[BODY][FREE] = iFree;

    setResultCode(jsn, 200);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
}

void zbNet::ParseJsonGetMyInfo(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetMyInfo\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    jsn[BODY][UID] = uid;
    jsn[BODY][NAME] = _users(uid).name.P();// string();

    setResultCode(jsn, 200);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
}

////////////////
// for profile

// parse json rep for command upload
void zbNet::ParseJsonUploadUser(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonUploadUser\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;
    setResultCode(jsn, 200);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};

// parse json rep for command upload
void zbNet::ParseJsonUploadShrd(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonUploadShrd\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string prm = GetFirstParam(url);
    if (prm.length() < 1 || !CheckDigit(prm))
    {
        setResultCode(jsn, 400, "del partner invalid id");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    int album_id = GetAlreayExistUserOwnerAlbum(uid);
    if (album_id == -1)
    {
        setResultCode(jsn, 400, "does not exist shared album!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    setResultCode(jsn, 200);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};

// parse json rep for command Profile
void zbNet::ParseJsonGetProfileUser(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetProfileUser\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string userStr = GetFirstParam(url);
    if (!CheckDigit(userStr))
    {
        setResultCode(jsn, 400, "invalid user id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int user_id = stoi(userStr);
    if (_users.N1() <= user_id)
    {
        setResultCode(jsn, 400, "invalid user id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    kmNetPtcFileRes resp = SendUserProfTo(uid, user_id, 8, true);
    int code = 200;
    string str = "";

    switch (resp)
    {
    case kmNetPtcFileRes::idwrong: code = 400; str = "idwrong"; break;
    case kmNetPtcFileRes::inqueue: code = 400; str = "inqueue"; break;
    case kmNetPtcFileRes::nonexistence: code = 400; str = "nonexistence"; break;
    case kmNetPtcFileRes::optwrong: code = 400; str = "optwrong"; break;
    case kmNetPtcFileRes::preackrejected: code = 400; str = "preackrejected"; break;
    case kmNetPtcFileRes::preacktimeout: code = 400; str = "preacktimeout"; break;
    case kmNetPtcFileRes::preackwrong: code = 400; str = "preackwrong"; break;
    case kmNetPtcFileRes::skipmax: code = 400; str = "skipmax"; break;
    case kmNetPtcFileRes::sndstatewrong: code = 400; str = "sndstatewrong"; break;
    default: break;
    }

    setResultCode(jsn, code, str);
    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
    return;
}

void zbNet::ParseJsonGetProfileAlbum(int uid, kmStra& jsna, string& url, json& body)
{
    print("*** ParseJsonGetProfileAlbum\n");

    // set json
    json jsn;

    jsn[CMD] = REP;
    if (!body.is_null()) jsn[ORIGIN] = body;

    string albumStr = GetFirstParam(url);
    if (!CheckDigit(albumStr))
    {
        setResultCode(jsn, 400, "invalid album id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }
    int album_id = stoi(albumStr);
    if (_shrds.N1() <= album_id)
    {
        setResultCode(jsn, 400, "invalid album id!!");
        SendJsonRep(uid, kmStra(jsn.dump().c_str()));
        return;
    }

    if (_shrds(album_id).itms.N1() > 0)
    {
        zbShrdItm& file = _shrds(album_id).itms(0);
        if (SendThumbShrd(uid, file.uid, file.sid, file.fid, true) == kmNetPtcFileRes::success)
        {
            SetFileInfoToJson(uid, file.uid, file.sid, file.fid, jsn[BODY]);
            setResultCode(jsn, 200);
        }
        else
        {
            setResultCode(jsn, 400, "get shared album thumb Fail!!");
        }
    }
    else
    {
        setResultCode(jsn, 400, "empty file!!");
    }

    // TODO
    /*
    kmNetPtcFileRes resp = SendShrdProfTo(uid, album_id, 10, true);
    int code = 200;
    string str = "";

    switch (resp)
    {
    case kmNetPtcFileRes::idwrong: code = 400; str = "idwrong"; break;
    case kmNetPtcFileRes::inqueue: code = 400; str = "inqueue"; break;
    case kmNetPtcFileRes::nonexistence: code = 400; str = "nonexistence"; break;
    case kmNetPtcFileRes::optwrong: code = 400; str = "optwrong"; break;
    case kmNetPtcFileRes::preackrejected: code = 400; str = "preackrejected"; break;
    case kmNetPtcFileRes::preacktimeout: code = 400; str = "preacktimeout"; break;
    case kmNetPtcFileRes::preackwrong: code = 400; str = "preackwrong"; break;
    case kmNetPtcFileRes::skipmax: code = 400; str = "skipmax"; break;
    case kmNetPtcFileRes::sndstatewrong: code = 400; str = "sndstatewrong"; break;
    default: break;
    }
    setResultCode(jsn, code, str);
    */

    SendJsonRep(uid, kmStra(jsn.dump().c_str()));
};


////////////////////////////////////
// noti message

// parse json to get shared
void zbNet::SendNotiPartner(int sourceUid, int targetUid, string action)
{
    print("*** SendNotiPartner\n");

    // set json
    json jsn;

    jsn[CMD] = NOTI;
    jsn[URL] = "/partner/" + action;
    jsn[BODY][PARTNER_ID] = sourceUid;
    jsn[BODY][NAME] = _users(sourceUid).name;

    kmStra stra(jsn.dump().c_str());

    AlarmData alarm;
    alarm.targetId = targetUid;
    alarm.data = stra;
    _jsnAlarm.Push(alarm);
    //SendJson(targetUid, stra);

    return;
}

void zbNet::SendNotiMember(int sourceUid, int targetUid, int album_id, string action)
{
    print("*** SendNotiMember\n");

    // set json
    json jsn;

    jsn[CMD] = NOTI;
    jsn[URL] = "/member/" + action;
    jsn[BODY][NAME] = _users(sourceUid).name;
    jsn[BODY][ALBUM_NAME] = _shrds(album_id).name;
    jsn[BODY][ALBUM_ID] = album_id;
    //jsn[BODY][ALBUM] = _shrds(album_id).ToJson(album_id);

    kmStra stra(jsn.dump().c_str());

    AlarmData alarm;
    alarm.targetId = targetUid;
    alarm.data = stra;
    _jsnAlarm.Push(alarm);
    //SendJson(targetUid, stra);

    return;
}

void zbNet::SendNotiAlbum(int sourceUid, int targetUid, int album_id, string action)
{
    print("*** SendNotiAlbum\n");

    // set json
    json jsn;

    jsn[CMD] = NOTI;
    jsn[URL] = "/album/" + action;
    jsn[BODY][NAME] = _users(sourceUid).name;
    jsn[BODY][ALBUM_NAME] = _shrds(album_id).name;
    jsn[BODY][ALBUM_ID] = album_id;
    //jsn[BODY][ALBUM] = _shrds(album_id).ToJson(album_id);

    kmStra stra(jsn.dump().c_str());

    AlarmData alarm;
    alarm.targetId = targetUid;
    alarm.data = stra;
    _jsnAlarm.Push(alarm);
    //SendJson(targetUid, stra);

    return;
}

void zbNet::SendNotiFile(int sourceUid, int targetUid, int album_id, string action)
{
    print("*** SendNotiFile\n");

    // set json
    json jsn;

    jsn[CMD] = NOTI;
    jsn[URL] = "/album/" + action;
    jsn[BODY][ALBUM_ID] = album_id;

    kmStra stra(jsn.dump().c_str());

    AlarmData alarm;
    alarm.targetId = targetUid;
    alarm.data = stra;
    _jsnAlarm.Push(alarm);
    //SendJson(targetUid, stra);

    return;
}

void zbNet::NotiUser(int uid)
{
    if (_users.N1() <= uid) return;

    for (int i = 0; i < _users.N1(); ++i)
    {
        if (uid != i)
        {
            SendNotiUser(uid, i, ADD);
        }
    }
}

void zbNet::SendNotiUser(int sourceUid, int targetUid, string action)
{
    print("*** SendNotiUser\n");

    // set json
    json jsn;

    jsn[CMD] = NOTI;
    jsn[URL] = "/user/" + action;
    jsn[BODY][UID] = sourceUid;

    kmStra stra(jsn.dump().c_str());

    AlarmData alarm;
    alarm.targetId = targetUid;
    alarm.data = stra;
    _jsnAlarm.Push(alarm);
    //SendJson(targetUid, stra);

    return;
}



//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// zbNetJsonAlarmBuf
//////////////////////////////////////////////////////////

//typedef std::unique_lock< Lock > _writeLock;
//typedef std::shared_lock< Lock > _readLock;

// create buffer
void zbNetJsnAlarmBuf::Init(void* net)
{
    //_writeLock r_rock(_alarmLock);
    kmLockGuard grd(_lck.Lock());

    _pNet = net;

    // create thread
    _thrd.Begin([](zbNetJsnAlarmBuf* alarms)
        {
            print("* alarm thread starts\n");
            while (1)
            {
                Sleep(1000);
                if (!alarms->Empty())
                {
                    alarms->sendNoti();
                }
            }
        }, this);
    _thrd.WaitStart();
};

void zbNetJsnAlarmBuf::Push(AlarmData data)
{
    //_writeLock r_rock(_alarmLock);
    kmLockGuard grd(_lck.Lock());

    _buf[_bufIdxE++] = data;
    if (_bufIdxE >= MAX_ALARM_BUFF) _bufIdxE = 0;
};

bool zbNetJsnAlarmBuf::Pop(AlarmData& data)
{
    //_writeLock r_rock(_alarmLock);
    kmLockGuard grd(_lck.Lock());

    if (_bufIdxS == _bufIdxE) return false;

    data = _buf[_bufIdxS++];
    if (_bufIdxS >= MAX_ALARM_BUFF) _bufIdxS = 0;
    return true;
};

bool zbNetJsnAlarmBuf::Empty()
{
    return (_bufIdxS == _bufIdxE) ? true : false;
};

void zbNetJsnAlarmBuf::sendNoti()
{
    zbNet* net = static_cast<zbNet*>(_pNet);

    int size = _bufIdxE - _bufIdxS;
    int startIdx = _bufIdxS;
    int endIdx = _bufIdxE;
    if (size < 0)
    {
        size += MAX_ALARM_BUFF;
    }

    int idx = 0;
    for (int i = 0; i < size; ++i)
    {
        idx = startIdx + i;
        if (idx >= MAX_ALARM_BUFF) idx -= MAX_ALARM_BUFF;
        if (net->IsConnected(_buf[idx].targetId))
        {
            AlarmData data;
            if (Pop(data))
            {
                net->SendJson(data.targetId, data.data);
            }
        }
    }
}

/*
void zbNetJsnAlarmBuf::printBuf()
{
    int size = _bufIdxE - _bufIdxS;
    int startIdx = _bufIdxS;
    int endIdx = _bufIdxE;
    if (size < 0)
    {
        size += MAX_ALARM_BUFF;
    }

    int idx = 0;
    for (int i = 0; i < size; ++i)
    {
        idx = startIdx + i;
        if (idx >= MAX_ALARM_BUFF) idx -= MAX_ALARM_BUFF;
    }
}
*/
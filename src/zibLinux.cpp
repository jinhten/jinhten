// include zbNet header
#include "zbNet.h"
#include <vector>

//////////////////////////////////////////////////////////////////
// window class 
class zibLinux
{
public:
    zbNet     _net;

    struct FileInfos
    {
        int type = 0;
        int state = 0;
        int date = 0;
        int dummy = 0;
        string name = "";
    };
    ///////////////////////////////
    // interface functions

    // create
/**
@brief  zibNet 의 초기화 작업 담당
*/
    void init(const zbNet::PathSet& pathSet) { CreateChild(pathSet); };

/**
@brief   Lan Network 을 통한 집서버 연결

@return  0 : 현재 단말에 등록된 zibServer가 없는 경우. (LAN에 기기가 없을 때)
             이미 연결된 동일한 mac address가 있는 경우.
             mac address로 Connect에 실패한 경우.
         1 : 등록된 mac address로 Connect에 성공한 경우.
*/
    int connectToLan()
    {
        // auto connection
        ConnectBroadcast();

        if (_net._users.N1() > 0) return 1;

        return 0;
    }

/**
@brief   등록된 User가 아닌 현재 zibServer와 연결 여부를 확인한다.

@return  0 : 연결 X
         1 : 연결 O
*/
    int checkConnect()
    {
        if (_net.GetIds().N() > 0) return 1;
        return 0;
    }

    int connectToWan() { return 1;} // TODO

/**
@brief  1. 파일 전송, File ID를 입력받아 파일 목록에서 파일 경로를 찾아 서버에 업로드
        2. 업로드 결과를 받아 파일 목록의 백업 상태를 업데이트.

@param  fid[in] : file.list에 저장된 file id

@return 1 : 성공
*/
    int uploadFile(int fid)
    {
        _net.SendFile(0, 0, fid);
        return 1;
    }

/**
@brief  1. 여러 파일 전송, File ID의 list를 입력받아 파일 목록에서 파일 경로를 찾아 서버에 업로드
        2. 업로드 결과를 받아 파일 목록의 백업 상태를 업데이트.

@param  fid[in] : file.list에 저장된 file id

@return 0 : file list size가 0
        1 : 성공
*/
    int uploadFiles(vector<int> vFid)
    {
        if (vFid.size() < 1) return 0;

        for (auto fid : vFid)
            _net.SendFile(0, 0, fid);

        return 1;
    }

/**
@brief   집서버에 storage List 를 요청하여 user 가 가진 storage 와 집서버가 가진 user 의 storage 가 일치하는지 확인

@return  ??
*/
    int validateFileList() {return 1;} // TODO


////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
    void quickSort(int first, int last, zbFiles& files, vector<int>& v)
    {
        int pivot;
        int i;
        int j;
        int temp;

        if (first < last)
        {
            pivot = first;
            i = first;
            j = last;

            while (i < j)
            {
                while (files(v[i]).date.GetInt() <= files(v[pivot]).date.GetInt() && i < last)
                {
                    i++;
                }
                while (files(v[j]).date.GetInt() > files(v[pivot]).date.GetInt())
                {
                    j--;
                }
                if (i < j)
                {
                    temp = v[i];
                    v[i] = v[j];
                    v[j] = temp;
                }
            }

            temp = v[pivot];
            v[pivot] = v[j];
            v[j] = temp;

            quickSort(first, j - 1, files, v);
            quickSort(j + 1, last, files, v);
        }
    }
////////////////////////////////////////
/**
@brief   안드로이드 내부 모든 저장소의 파일 목록을 시간순서로 정렬하여 Iterator 형태로 반환

@return  sortedFileList
*/
    vector<int> getTotalFileListDateSorted()
    {
        zbFiles& files = _net._users(0).strgs(0).files;
        vector<int> v;
        for (int i = 0; i < files.N1(); ++i)
        {
            v.push_back(i);
        }

        quickSort(0, files.N1()-1, files, v);
        return v;
    }

/**
@brief  fileID 를 입력받아 파일 목록에서 해당 파일의 정보를 반환

@param  fid[in] : file id
        infos[out] : fid, file type, name(path), backup state등의 file 정보

@return  0 : file info 획득 실패
         1 : file info 획득 성공
*/
    int getFileInfo(int fid, FileInfos& infos)
    {
        FileInfos ret;
        zbFiles& files = _net._users(0).strgs(0).files;
        if (fid >= files.N1()) return 0;

        zbFile& file = files(fid);

        ret.type = (int)file.type;
        ret.state = (int)file.state;
        ret.date = file.date.GetInt();

        char name[200] = {0,};
        wcstombs(name, file.name.P(), wcslen(file.name.P()));
        ret.name.assign(name);

        return 1;
    }

/**
@brief  file ID 를 입력받아 서버에  파일을 요청하여  응답 파일을 지정된 path 에 저장

@param fid[in] : file id
@param path[in] : 응답으로 받은 파일이 저장될 경로 , 입력되지 않으면 default path에 저장

@return 
*/
    int requestFile(int fid, string path) {return 1;} // TODO

/**
@brief   file ID 를 입력받아 서버에  thumbnail Media 를 요청하여  응답 파일을 지정된 path 에 저장

@param fid[in] : file id
@param path[in] : 응답으로 받은 파일이 저장될 경로 , 입력되지 않으면 default path에 저장

@return  
*/
    int requestThumbnail(int fid, string path) { return 1;} // TODO

/**
@brief   갤러리에 새로 추가된 파일이나 삭제된 파일을 검사하여  파일목록을 업데이트

@return  ??
*/
    int updateFileList()
    {
        _net.UpdateFile(false);
        return 1;
    }

/**
@brief   1. 파일리스트에 백업 상태가 backUpNo 인 파일을 전부 집서버에 전송
         2. 전송 확인 절차를 거쳐 전송이 확인되면 파일리스트의 백업 상태를 업데이트

@return  ??
*/
    int backUpAll()
    {
        _net.UpdateFile(); // default true
        return 1;
    }

/**
@brief  1. fileID 를 입력받아 device 의 (client) 파일 삭제
        2. 삭제가 완료되면 자신의 파일 목록을 업데이트하고 서버에 업데이트 사실을 알림
        3. 서버의 파일 목록 업데이트 확인

@param  fid[in] : file id

@return 
*/
    int deleteDeviceFile(int fid)
    {
        _net.DeleteFileClt(0, 0, fid);
        return 1;
    }

/**
@brief  1. fileID 를 입력받아 서버에 해당 파일의 삭제를 요청
        2. 삭제가 완료되면 삭제 완료 응답을 받아 자신의 파일 목록을 업데이트

@param  fid[in] : file id

@return 
*/
    int deleteServerFile(int fid)
    {
        _net.BanBkup(0, 0, fid);
        return 1;
    }

/**
@brief  fileID 를 입력받아 클라이언트에서 파일 백업이 가능한 상태로 변경

@param  fid[in] : file id

@return 
*/
    int allowFileBackUp(int fid)
    {
        _net.LiftBanBkup(0, 0, fid);
        return 1;
    }

/**
@brief  1. fileID 를 입력받아 서버에 해당 파일의 삭제를 요청
        2. 삭제가 완료되면 삭제 완료 응답을 받고 디바이스 내 파일도 삭제
        3. 파일리스트 업데이트

@param  fid[in] : file id

@return 
*/
    int deleteBothFile(int fid)
    {
        _net.DeleteFileBoth(0, 0, fid);
        return 1;
    }

////////////////////////////////////////
////////////////////////////////////////
    // set path
    void setRootPath(string& path)
    {
        wchar  path_name[512] = {0,};

        mbstowcs(path_name, path.c_str(), path.length());
        _net._path.SetStr(path_name);
    };

    void setSrcPath(string& path)
    {
        wchar  path_name[512] = {0,};

        mbstowcs(path_name, path.c_str(), path.length());
        _net._srcpath.SetStr(path_name);
    };

    void setDlPath(string& path)
    {
        wchar  path_name[512] = {0,};

        mbstowcs(path_name, path.c_str(), path.length());
        _net._dwnpath.SetStr(path_name);
    };

    ///////////////////////////////////
    // windows procedure functions
protected:
    // create child window
    virtual void CreateChild(const zbNet::PathSet& pathSet)
    {
        setlocale(LC_ALL, "");

        // init net
        _net.Init(this, cbRcvNetStt, pathSet);
    };

    // download files
    void DnloadFile()
    {
        const zbFiles& files = _net.GetFiles(0,0);

        for(int fid = (int)files.N1(); fid--; )
        {
            if(files(fid).state == zbFileState::bkuponly)
            {
                _net.ReqFile(0,0,fid,fid);
            }
        }
    };

    // upload bkupno files
    void UploadBkupno()
    {
        print("** update bkupno files\n");
        _net.SendBkupno();
    };

    // print pkeys only for nks svr
    void PrintPkeys()
    {
        print("******** pkeys\n");
        if(_net._mode == zbMode::nks) { _net._nks.Print(); _net.SaveNks(); }
        if(_net._mode == zbMode::svr) _net.GetPkey().Print();
        if(_net._mode == zbMode::clt)
        {
            for(int i = (int)_net._users.N1(); i--;) _net._users(i).key.Print();
        }
    };

    // test code only for test
    void TestCode()
    {
        print("** test code\n");

        if(_net._users.N1() == 0) return;

        _net.RequestAddr(_net.GetUser(0).key, _net.GetUser(0).mac);
    };    

    ////////////////////////////////////
    // callback functions for net
    static int cbRcvNetStt(void* lnx, uchar ptc_id, char cmd_id, void* arg)
    {
        return ((zibLinux*)lnx)->cbRcvNet(ptc_id, cmd_id, arg);
    };
    int cbRcvNet(uchar ptc_id, char cmd_id, void* arg)
    {
        switch(ptc_id)
        {
        //case 0: return cbRcvNetPtcBrdc(cmd_id, arg);
        case 1: return cbRcvNetPtcCnnt(cmd_id, arg);
        case 2: return cbRcvNetPtcData(cmd_id, arg); 
        //case 3: return cbRcvNetPtcLrgd(cmd_id, arg);
        case 4: return cbRcvNetPtcFile(cmd_id, arg);
        }
        return -1;
    };
    int cbRcvNetPtcCnnt(char cmd_id, void* arg)
    {    
        int id = _net.GetCnntingId(cmd_id);

        if(cmd_id == 0) print("* connection has been requested (id: %d)\n", id);
        else            print("* connection has been accepted  (id: %d)\n", id);

        _net.GetId(id).Print(); print("\n");

        return 1;
    };
    int cbRcvNetPtcData(char cmd_id, void* arg)
    {
        uchar data_id = _net.GetDataId();

        switch(data_id)
        {
        case 0: cbRcvInfo(); break; // receive info
        //case 1: UpdateTbl(); break; // receive reqregist
        //case 2: UpdateTbl(); break; // receive acceptregist
        default:break;
        }
        return 1;
    };
    int cbRcvNetPtcFile(char cmd_id, void* arg)
    {
        return 1;
    };
    void cbRcvInfo()
    {
        zbNetInfo& info = _net.GetLastRcvInfo(); // opposite's info

        // reqeust registration
        if(_net.FindUser(info.mac) < 0) // not registered
        if(_net._mode == zbMode::clt && info.mode == zbMode::svr)
        {
            _net.ReqRegist(info.src_id);
        }
    };

    ///////////////////////////////////////////////////
    // network functions

    // connect every network using broadcast
    void ConnectBroadcast()
    {
        static kmThread thrd; thrd.Begin([](zibLinux* lnx)
        {
            print("* connecting with udp broadcasting\n");

            kmAddr4s   addrs; 
            kmMacAddrs macs; 

            int n = lnx->_net.GetAddrsInLan(addrs, macs);
            
            for(int i = 0; i < n; ++i)
            {
                if(lnx->_net.FindId(macs(i)) > -1) continue; // check if already connected

                if(lnx->Connect(addrs(i)) < 0)
                {
                    print("* connecting failed (to %s)\n", addrs(i).GetStr().P());
                }
            }
        },this);
    };

    // connect with ip addr
    int Connect(kmAddr4 addr) { return _net.Connect(addr, _net._name, 500.f); };
};

/////////////////////////////////////////////////////////////////
void mainConvWC4to2(wchar* wc, ushort* c, const ushort& n) { for (ushort i = 0; i < n; ++i) { c[i] = (ushort)wc[i]; } };

/*
void zibCli(zibLinux* linuxNet)
{
    sleep(2);
    while (1)
    {
        int prot_id = 0;
        cout<<" =========================="<<endl;
        cout<<"Data(2), File(4) : ";
        cin>>prot_id;
        if (prot_id == 2)
            cout<<"Send Data"<<endl;
        else if (prot_id == 4)
            cout<<"Send File"<<endl;
        cin.clear();
        cin.ignore(INT_MAX, '\n');
    }
}
*/

/////////////////////////////////////////////////////////////////
// entry
int main() try
{
    zbNet::PathSet paths;
    paths.path = "/home/kktjin/backup/zibsvr";
    paths.srcpath = "/home/kktjin/backup/image";
    paths.dlpath = "/home/kktjin/backup/download";

    zibLinux linuxNet;
    linuxNet.init(paths);
    linuxNet.connectToLan();

    // for debug
    //std::thread cli(zibCli, &linuxNet);

    sleep(1);

    //linuxNet.uploadFile(0);
    //cout<<linuxNet.checkConnect()<<endl;
    //linuxNet.updateFileList();
    //linuxNet.backUpAll();
    //linuxNet.deleteBothFile(1);
    //linuxNet.deleteServerFile(5);
    linuxNet.allowFileBackUp(5);

    while (1) { sleep(100); }

    return 0;
}

catch(kmException e)
{
    print("* kmnet.cpp catched an exception\n");
    kmPrintException(e);
    system("pause");
    return 0;
}

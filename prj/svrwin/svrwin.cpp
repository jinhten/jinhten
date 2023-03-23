// include resource
#include "resource.h"

// include kmMat header
#include "../../inc/km7Wnd.h"

// include zbNet header
#include "../../inc/zbNet.h"

// define static member variables
int                 kmTiffTag::_endian           = 0;
ULONG_PTR           kmMdf::_gdiplus_token        = 0;
GdiplusStartupInput kmMdf::_gdiplus_startupinput = 0;

//////////////////////////////////////////////////////////////////
// window class 
class zbWnd: public kmWnd
{
public:
	zbNet     _net;

	// child window
	kmwMenu   _menu;	 // top menu
	kmwTable  _ids_tbl;  // ids table
	kmwTable  _usr_tbl;  // users table
	kmwTable  _stg_tbl;  // storage table

	kmwBox    _box;      // output box
	kmwEdt    _edt;      // input  box

	// window control
	kmWndMng  _mng;      // window managing for tray icon
	kmwMenu   _tsk_menu; // tray icon's menu in task bar;

	// additional functions
	bool      _svc_on     = false; // service for rerun
	bool      _autorun_on = false; 

	// destructor
	virtual ~zbWnd()
	{
		_mng.DelTrayIcon();
	};

	///////////////////////////////
	// interface functions

	// create window
	void Create() { CreateThread(1100,300,800,600,L"zibServer",0); };

	///////////////////////////////////
	// windows procedure functions
protected:
	// create child window
	virtual void CreateChild()
	{
		// create menu
		_menu   .Add(L"setting");
		_menu(0).Add(L"open zibsvr folder");		
		_menu(0).Add(L"set zibsvr folder");		
		_menu(0).AddSeparator();
		_menu(0).Add(L"mode : server");
		_menu(0).Add(L"mode : client");
		_menu(0).Add(L"mode : nks");

		_menu   .Add(L"connect");
		_menu(1).Add(L"connect new");
		_menu(1).Add(L"connect with pkey");		
		_menu(1).Add(L"connect with vkey");
		_menu(1).Add(L"reconnect all");

		_menu   .Add(L"sync");
		_menu(2).Add(L"up     : ctrl + u");
		_menu(2).Add(L"down : ctrl + d");
		_menu(2).AddSeparator();
		_menu(2).Add(L"view file list");
		_menu(2).Add(L"update table");

		_menu   .Add(L"option");
		_menu(3).Add(L"port : 60165");
		_menu(3).Add(L"port : 60166");
		_menu(3).Add(L"port : 60167");
		_menu(3).AddSeparator();
		_menu(3).Add(L"reset");

		_menu.Create(_hwnd, 1024);

		// get path from registry
		kmRegistry key(L"Software\\zibServer\\zibwin\\"); kmStrw path, name;

		if(key.Read(L"zibsvr path", path) == ERROR_SUCCESS) _net._path = path.cu();
		else
		{
			wchar buf[512]; GetCurrentDirectory(512, buf); _net._path = kmStrw(buf).cu();
		}
		print("* zibsvr path : %s\n", _net._path.P());

		// get name from registry
		if(key.Read(L"zibsvr name", name) == ERROR_SUCCESS) _net._name = name.cu();
		else
		{
			wchar buf[64]; DWORD n = 64; GetUserNameW(buf, &n); _net._name = kmStrw(buf).cu();
		}
		print("* zibsvr name : %s\n", _net._name.P());

		// get mode from registry
		if(key.Read(L"zibsvr mode", _net._mode) != ERROR_SUCCESS)
		{
			_net._mode = zbMode::clt; // default is client mode
		}
		switch(_net._mode)
		{
		case zbMode::svr : SetWinTxt(L"zibServer - server"); _menu(0)(3).SetCheck(1); break;
		case zbMode::clt : SetWinTxt(L"zibServer - client"); _menu(0)(4).SetCheck(1); break;
		case zbMode::nks : SetWinTxt(L"zibServer - nks"   ); _menu(0)(5).SetCheck(1); break;
		}

		// get port from registry
		if(key.Read(L"zibsvr port", _net._port) != ERROR_SUCCESS)
		{
			_net._port = KMNETDPORT; // default is client mode
		}
		switch(_net._port)
		{
		case 60165 : _menu(3)(0).SetCheck(1); break;
		case 60166 : _menu(3)(1).SetCheck(1); break;
		case 60167 : _menu(3)(2).SetCheck(1); break;
		}

		// set sourch path and download path only for client
		if(_net._mode == zbMode::clt)
		{	
			_net._srcpath = "D:/DCIM";
			_net._dwnpath = "D:/zibdown";
			_net._tmppath = "D:/tmp";

			if(!kmFile::Exist(L"D:/"))
			{
				_net._srcpath = "C:/DCIM";
				_net._dwnpath = "C:/zibdown";
				_net._tmppath = "C:/tmp";
			}
		}
		// set nks server address
		//_net.SetNksAddr(kmAddr4(10,114,75,46,60167));
		//_net.SetNksAddr(kmAddr4(10,114,75,48,60169));
		_net.SetNksAddr(kmAddr4(52,231,35,166,60165)); // azure
		//_net.SetNksAddr(kmAddr4(10,243,96,45,60167)); // notebook - wifi

		// create child windows
		_usr_tbl.Create(kmCwp(kmC(0,0),kmC(0,10),kmC(1,0),kmC(0,30)), this, 0); _usr_tbl.Hide();
		_stg_tbl.Create(kmCwp(kmC(0,0),kmC(0,10),kmC(1,0),kmC(0,30)), this, 1); _stg_tbl.Hide();
		_ids_tbl.Create(kmCwp(kmC(0,0),kmC(0,40),kmC(1,0),kmC(0,50)), this, 2); _ids_tbl.Hide();

		_box    .SetH(18).SetBkRgb(kmRgb(250,240,240)).SetLineSpace(3);
		_box    .Create(kmCwp(kmC(0,10),kmC(1,-90),kmC(1,-10),kmC(1,-35)), this, 3);
		_edt    .Create(kmCwp(kmC(0,10),kmC(1,-30),kmC(1,-10),kmC(1,-10)), this, 4);

		// init net
		_net.Init(this, cbRcvNetStt);

		UpdateTbl();

		const kmAddr4 brdc_addr = kmSock::GetBrdcAddr(_net._addr.GetPort());

		print("* my   address : %s\n", _net._addr       .GetStr().P());
		print("* brdc address : %s\n",  brdc_addr       .GetStr().P());
		print("* nks  address : %s\n", _net.GetNksAddr().GetStr().P());

		// auto connection
		const int usr_n = (int)_net._users.N1();

		print("* the number of users : %d\n", usr_n);

		if(usr_n == 0) {} //ConnectBroadcast();
		else
		{
			for(int uid = 0; uid < usr_n; ++uid) _net.Connect(uid);
		}

		// set timer
		SetTimer(1,2000);

		// set tray icon
		HICON icon = LoadIcon(GetModuleHandle(NULL), MAKEINTRESOURCE(IDI_ICON1));

		_mng.AddTrayIcon(_hwnd, L"zibWin for zibSvr", icon);

		// set svc and autorun
		if(key.Read(L"svc", _svc_on) != ERROR_SUCCESS)
		{
			_svc_on = false; key.Add(L"svc", _svc_on);
		}
		if(key.Read(L"autorun", _autorun_on) != ERROR_SUCCESS)
		{
			_autorun_on = false; key.Add(L"autorun", _autorun_on);
		}

		// creaet tray icon's menu
		_tsk_menu.Add(L"show window");
		_tsk_menu.Add(L"hide window");
		_tsk_menu.AddSeparator();
		_tsk_menu.Add(L"show console");
		_tsk_menu.Add(L"hide console");
		_tsk_menu.AddSeparator();
		_tsk_menu.Add(L"svc on");
		_tsk_menu.Add(L"autorun on");
		_tsk_menu.AddSeparator();
		_tsk_menu.Add(L"exit");

		_tsk_menu(6).SetCheck(_svc_on     ? 1:0);
		_tsk_menu(7).SetCheck(_autorun_on ? 1:0);

		if(_svc_on) StartSvc();
	};

	// create _tsk_menu
	void CreateTskMenu()
	{
		int x = 0, y = 0; kmT2(x,y) = GetCursorPosScn(); 

		_tsk_menu.CreatePopup(_hwnd, x, y, 2048);
	};

	// window procedure... wm_km_manage
	virtual kmePPType OnKmManage(WPARAM wp, LPARAM lp)
	{
		const uint id = (uint)wp;

		if(id == 1) // tray icon
			switch(lp)
			{
			case WM_LBUTTONDOWN : break;
			case WM_LBUTTONUP   : break;
			case WM_RBUTTONDOWN : break;
			case WM_RBUTTONUP   : CreateTskMenu(); break;
			}
		return PP_FINISH;
	};

	// window procedure... wm_command
	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control 
		ushort ntf = HIWORD(wp), id = LOWORD(wp);

		switch(id)
		{
			// control
		case 3: break; // _box
		case 4: if(ntf == NTF_EDT_ENTER) ParseTxt(); break; // _edt
			// menu... _menu
		case 1024 + 0: OpenPath();           break; // open zibsvr folder		
		case 1024 + 1: SetPath ();           break; // set zibsvr folder

		case 1024 + 2: SetMode(zbMode::svr); break; // set mode as server
		case 1024 + 3: SetMode(zbMode::clt); break; // set mode as client
		case 1024 + 4: SetMode(zbMode::nks); break; // set mode as nks server

		case 1024 + 5: ConnectNew();  break; 
		case 1024 + 6: ConnectPkey(); break;
		case 1024 + 7: ConnectVkey(); break;
		case 1024 + 8: ReconnectAll(); UpdateTbl(); break;

		case 1024 + 9: UploadFile(); break;
		case 1024 +10: DnloadFile(); break;
		case 1024 +11: if(_net.CheckFileList(0,0) > 0) _net.SaveFileList(0,0); 
			_net.PrintFileList(0,0);
			break;
		case 1024 +12: UpdateTbl(); break;

		case 1024 +13: SetPort(60165); break;
		case 1024 +14: SetPort(60166); break;
		case 1024 +15: SetPort(60167); break;

		case 1024 +16: ResetNet();     break;

			// popup menu of taskbar... _tsk_menu
		case 2048 + 0: Show       ();   break; // show window
		case 2048 + 1: Hide       ();   break; // hide window
		case 2048 + 2: ShowConsole();   break; // show console
		case 2048 + 3: HideConsole();   break; // hide console
		case 2048 + 4: ToggleSvc();     break; // toggle svc
		case 2048 + 5: ToggleAutorun(); break; // toggle autorun
		case 2048 + 6: Destroy();       break; // exit
		}
		return PP_FINISH;
	};
	virtual kmePPType OnTimer(WPARAM wp, LPARAM lp)
	{
		const int timer_id = (int) wp;

		switch(timer_id)
		{
		case 1: UpdateTbl(); break; // update table
		}
		return PP_FINISH;
	};

	// key procedure
	virtual kmePPType OnKeyDown(WPARAM wp, LPARAM lp)
	{
		int ret = 0;

		if(GetKeyState(VK_CONTROL) < 0)	switch(wp) // with control key
		{
		case 'A' : UploadAll();          break;
		case 'U' : UploadFile();         break;
		case 'D' : DnloadFile();         break;
		case 'T' : UpdateTbl();          break;
		case 'S' : UploadBkupno();       break;
		case 'Z' : TestCode();           break;
		case 'P' : PrintKeys();          break;
		case 'F' : _net.PrintFileList(); break;
		case 'M' : MakeTestFiles();      break;
		case 'C' : if(_net.CheckFileList(0,0) > 0) _net.SaveFileList(0,0); break;
		}
		else if(GetKeyState(VK_SHIFT) < 0) switch(wp) // with shift key
		{
		case 'R' : print_i(_net.GetPtcFile()._rcv_state); break;
		case 'S' : print_i(_net.GetPtcFile()._snd_state); break;
		case 'I' : _net.PrintIds(); break;
		case 'J' : _net._jsnbuf.Print(); break;
		}
		else if(GetKeyState('T') < 0) switch(wp)
		{
		case 'R' : TestRequestAddr(); break;
		case 'S' : TestSendSigNks();  break;
		case 'D' : TestSendSig();     break;
		case 'U' : TestNetPckSize();  break;

		case '0' : print_i(_net.RequestThumb(0,0,0)); break;
		case '1' : print_i(_net.RequestThumb(0,0,1)); break;
		case '2' : print_i(_net.RequestThumb(0,0,2)); break;
		case '3' : print_i(_net.RequestThumb(0,0,3)); break;
		case '4' : print_i(_net.RequestThumb(0,0,4)); break;
		case '5' : print_i(_net.RequestThumb(0,0,5)); break;
		}
		else if(GetKeyState('D') < 0) switch(wp)
		{
		case '0' : _net.DeleteFileClt(0,0,0); _net.PrintFileList(); break;
		case '1' : _net.DeleteFileClt(0,0,1); _net.PrintFileList(); break;
		case '2' : _net.DeleteFileClt(0,0,2); _net.PrintFileList(); break;
		case '3' : _net.DeleteFileClt(0,0,3); _net.PrintFileList(); break;
		case '4' : _net.DeleteFileClt(0,0,4); _net.PrintFileList(); break;
		case '5' : _net.DeleteFileClt(0,0,5); _net.PrintFileList(); break;
		}
		else if(GetKeyState('B') < 0) switch(wp)
		{
		case '0' : _net.BanBkup(0,0,0); _net.PrintFileList(); break;
		case '1' : _net.BanBkup(0,0,1); _net.PrintFileList(); break;
		case '2' : _net.BanBkup(0,0,2); _net.PrintFileList(); break;
		case '3' : _net.BanBkup(0,0,3); _net.PrintFileList(); break;
		case '4' : _net.BanBkup(0,0,4); _net.PrintFileList(); break;
		case '5' : _net.BanBkup(0,0,5); _net.PrintFileList(); break;
		}
		else if(GetKeyState('F') < 0) switch(wp)
		{
		case '0' : print_i(_net.RequestFile(0,0,0)); break;
		case '1' : print_i(_net.RequestFile(0,0,1)); break;
		case '2' : print_i(_net.RequestFile(0,0,2)); break;
		case '3' : print_i(_net.RequestFile(0,0,3)); break;
		case '4' : print_i(_net.RequestFile(0,0,4)); break;
		case '5' : print_i(_net.RequestFile(0,0,5)); break;
		case '6' : print_i(_net.RequestFile(0,0,6)); break;
		case '7' : print_i(_net.RequestFile(0,0,7)); break;
		case '8' : print_i(_net.RequestFile(0,0,8)); break;
		case '9' : print_i(_net.RequestFile(0,0,9)); break;
		}
		else if(GetKeyState('C') < 0) switch(wp)
		{
		case '0' : print_i(_net.RequestCache(0,0,0)); break;
		case '1' : print_i(_net.RequestCache(0,0,1)); break;
		case '2' : print_i(_net.RequestCache(0,0,2)); break;
		case '3' : print_i(_net.RequestCache(0,0,3)); break;
		case '4' : print_i(_net.RequestCache(0,0,4)); break;
		case '5' : print_i(_net.RequestCache(0,0,5)); break;
		}
		else if(GetKeyState('H') < 0) switch(wp)
		{
		case '0' : _net.DeleteShrd(0); _net.PrintShrdList(); break;
		case '1' : _net.DeleteShrd(1); _net.PrintShrdList(); break;
		case '2' : _net.DeleteShrd(2); _net.PrintShrdList(); break;
		case '3' : _net.DeleteShrd(3); _net.PrintShrdList(); break;
		case '4' : _net.DeleteShrd(4); _net.PrintShrdList(); break;
		case '5' : _net.DeleteShrd(5); _net.PrintShrdList(); break;
		}
		else if(GetKeyState('S') < 0) switch(wp)
		{
		case '0' : _net.SendFile(0,0,0,true); break;
		case '1' : _net.SendFile(0,0,1,true); break;
		case '2' : _net.SendFile(0,0,2,true); break;
		case '3' : _net.SendFile(0,0,3,true); break;
		case '4' : _net.SendFile(0,0,4,true); break;
		case '5' : _net.SendFile(0,0,5,true); break;
		case '6' : _net.SendFile(0,0,6,true); break;
		case '7' : _net.SendFile(0,0,7,true); break;
		case '8' : _net.SendFile(0,0,8,true); break;
		case '9' : _net.SendFile(0,0,9,true); break;
		}		
		else switch(wp) // general key
		{
		case VK_OEM_PLUS  : break;
		case VK_OEM_MINUS : break;
		}		
		return PP_FINISH;
	};

	// window procedure... wm_size
	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{
			_usr_tbl.Relocate(w,h);
			_stg_tbl.Relocate(w,h);
			_ids_tbl.Relocate(w,h);
			_box    .Relocate(w,h);
			_edt    .Relocate(w,h);
		}
		return PP_FINISH;
	};

	///////////////////////////////////////////////////
	// windows functions

	// destroy itself and svc
	void Destroy() { TernminateSvc(); kmWnd::Destroy(); };

	// toggle svc
	void ToggleSvc()
	{
		_tsk_menu(6).SetCheck(2);

		if(_tsk_menu(6).GetCheck()) StartSvc();
		else                        TernminateSvc();
	};

	// start svc
	void StartSvc()
	{
		if(kmPrc::Find(L"zibsvc.exe") == 0)
		{
			kmStrw path = kmPrc::GetPath(); path.Cutback(7) += L"svc.exe";

			kmPrc::Exec(path, SW_HIDE);

			print("* start svc\n");

			kmRegistry key(L"Software\\zibServer\\zibwin\\"); key.Add(L"svc", true);
		}
	};

	// terminate svc
	void TernminateSvc()
	{
		// terminate svc
		const uint pid = kmPrc::Find(L"zibsvc.exe");

		if(pid > 0) kmPrc::Terminate(pid); 

		print("* terminate svc\n"); 

		kmRegistry key(L"Software\\zibServer\\zibwin\\"); key.Add(L"svc", false);
	};

	// toggle autorun
	void ToggleAutorun()
	{
		_tsk_menu(7).SetCheck(2);

		if(_tsk_menu(7).GetCheck()) RegisterAutorun();
		else                        DeleteAutorun();
	};

	// register autorun
	void RegisterAutorun()
	{
		// get current exe file
		wchar buf[256]; ::GetModuleFileName(NULL, buf, 256);

		// register exe
		kmRegistry reg(L"Software\\Microsoft\\Windows\\CurrentVersion\\Run");
		kmStrw     path(buf), name(L"zibServer"), path0;

		if(reg.Read(name, path0) != ERROR_SUCCESS) reg.Add(name, path);
		else if(path != path0)                     reg.Add(name, path);

		print("* register autorun\n");

		kmRegistry key(L"Software\\zibServer\\zibwin\\"); key.Add(L"autorun", true);
	};

	// delete auto-run
	void DeleteAutorun()
	{	
		kmRegistry reg(L"Software\\Microsoft\\Windows\\CurrentVersion\\Run");

		LSTATUS ret = reg.Delete(kmStrw(L"zibServer"));

		if(ret == ERROR_SUCCESS) print("* delete autorun\n");

		kmRegistry key(L"Software\\zibServer\\zibwin\\"); key.Add(L"autorun", false);
	};

	// set mode
	void SetMode(zbMode mode)
	{
		switch(_net._mode = mode)
		{
		case zbMode::svr : _menu(0)(3).SetCheck(1); _menu(0)(4).SetCheck(0); _menu(0)(5).SetCheck(0); break;
		case zbMode::clt : _menu(0)(3).SetCheck(0); _menu(0)(4).SetCheck(1); _menu(0)(5).SetCheck(0); break;
		case zbMode::nks : _menu(0)(3).SetCheck(0); _menu(0)(4).SetCheck(0); _menu(0)(5).SetCheck(1); break;
		}

		kmRegistry(L"Software\\zibServer\\zibwin\\").Add(L"zibsvr mode", mode);

		switch(mode)
		{
		case zbMode::svr : SetWinTxt(L"zibServer - server"); break;
		case zbMode::clt : SetWinTxt(L"zibServer - client"); break;
		case zbMode::nks : SetWinTxt(L"zibServer - nks"   ); break;
		}
	};

	// set default port
	void SetPort(ushort port)
	{
		_net.SetPort(port);

		switch(port)
		{
		case 60165: _menu(3)(0).SetCheck(1); _menu(3)(1).SetCheck(0); _menu(3)(2).SetCheck(0);break;
		case 60166: _menu(3)(0).SetCheck(0); _menu(3)(1).SetCheck(1); _menu(3)(2).SetCheck(0);break;
		case 60167: _menu(3)(0).SetCheck(0); _menu(3)(1).SetCheck(0); _menu(3)(2).SetCheck(1);break;
		}
		kmRegistry(L"Software\\zibServer\\zibwin\\").Add(L"zibsvr port",port);

		print("* set port as %d\n", port);
	};

	// set path
	void SetPath()
	{
		kmStrw title(L"current : %s", _net._path.cuw().P());
		wchar  path_name[512] = {0,};

		if(kmFile::GetFolderName(_hwnd, path_name, title.P()))
		{
			_net._path = kmStrw(path_name).cu();
			kmRegistry(L"Software\\zibServer\\zibwin\\").Add(L"zibsvr path", _net._path.cuw());
			print("* zibsvr folder is set with %s", _net._path.P());
		}
	};

	// reset net
	void ResetNet()
	{
		_net.Reset();
	};

	// open explorer for path
	void OpenPath() { ShellExecuteA(NULL, "open", _net._path.ca().P(), NULL, NULL, SW_SHOWNORMAL); };

	// update tbl
	void UpdateTbl()
	{
		// init parameters
		const kmNetIds& ids = _net.GetIds();  const int ids_n = (int)ids.N1();
		const zbUsers&  usr = _net._users;    const int usr_n = (int)usr.N1();

		const int stg_n = _net.GetAllStrgN();

		// udpate usr_tbl		
		{
			kmLockGuard grd = _usr_tbl.Lock(); //==============lock-unlock

			// resize tbl
			_usr_tbl.GetCwp().b = kmC(0, 10 + (usr_n + 1)*25);

			// update item
			_usr_tbl.CreateItm(usr_n + 1, 5).SetFontH(16);

			_usr_tbl(0,0).SetStr(L"index");
			_usr_tbl(0,1).SetStr(L"name");
			_usr_tbl(0,2).SetStr(L"mac");
			_usr_tbl(0,3).SetStr(L"role");
			_usr_tbl(0,4).SetStr(L"src id");			

			for(int i = 0; i < usr_n; ++i)
			{	
				_usr_tbl(i+1,0).SetStr(kmStrw(L"%d", i));
				_usr_tbl(i+1,1).SetStr(usr(i).name.cuw());
				_usr_tbl(i+1,2).SetStr(usr(i).mac.GetStrw());
				_usr_tbl(i+1,3).SetStr(usr(i).GetRoleStr().cuw());
				_usr_tbl(i+1,4).SetStr(kmStrw(L"%d", usr(i).src_id));				
			}
			_usr_tbl.SetTwoTone(kmRgb(230,240,250), kmRgb(245,250,255)).UpdateItm();
		}
		if(usr_n > 0) _usr_tbl.Show(); // * Note that you must call Show() out of Lock()
		// * since it will call Enter() within itself
		// update strg_tbl		
		{
			kmLockGuard grd = _stg_tbl.Lock(); //==============lock-unlock

			// resize tbl
			_stg_tbl.GetCwp().t = kmC(0, _usr_tbl.GetCwp().b.d + 10);
			_stg_tbl.GetCwp().b = kmC(0, _stg_tbl.GetCwp().t.d + (stg_n +1)*25);

			// update item
			_stg_tbl.CreateItm(stg_n + 1, 6).SetFontH(16);

			_stg_tbl(0,0).SetStr(L"user");
			_stg_tbl(0,1).SetStr(L"strg");
			_stg_tbl(0,2).SetStr(L"path");
			_stg_tbl(0,3).SetStr(L"src path");
			_stg_tbl(0,4).SetStr(L"type");
			_stg_tbl(0,5).SetStr(L"file num");

			for(int uid = 0, i = 0; uid < usr_n; ++uid)
			{
				const zbStrgs& stg = usr(uid).strgs; const int stg_n = (int)stg.N1();

				for(int sid = 0; sid < stg_n; ++sid, ++i)
				{
					_stg_tbl(i+1,0).SetStr(usr(uid).name        .cuw());
					_stg_tbl(i+1,1).SetStr(stg(sid).name        .cuw());
					_stg_tbl(i+1,2).SetStr(stg(sid).path        .cuw());
					_stg_tbl(i+1,3).SetStr(stg(sid).srcpath     .cuw());
					_stg_tbl(i+1,4).SetStr(stg(sid).GetTypeStr().cuw());
					_stg_tbl(i+1,5).SetStr(kmStrw(L"%d",stg(sid).files.N1()));
				}
			}
			_stg_tbl.SetTwoTone(kmRgb(230,240,250), kmRgb(245,250,255)).UpdateItm();
		}
		if(stg_n > 0) _stg_tbl.Show();

		// update ids_tbl		
		{
			kmLockGuard grd = _ids_tbl.Lock(); //==============lock-unlock

			// resize tbl
			_ids_tbl.GetCwp().t = kmC(0, _stg_tbl.GetCwp().b.d + 10);
			_ids_tbl.GetCwp().b = kmC(0, _ids_tbl.GetCwp().t.d + (ids_n +1)*25);

			// update item
			_ids_tbl.CreateItm(ids_n + 1, 6).SetFontH(16);

			_ids_tbl(0,0).SetStr(L"src id");
			_ids_tbl(0,1).SetStr(L"name");
			_ids_tbl(0,2).SetStr(L"mac");
			_ids_tbl(0,3).SetStr(L"ip");
			_ids_tbl(0,4).SetStr(L"des id");
			_ids_tbl(0,5).SetStr(L"state");

			for(int i = 0; i < ids_n; ++i)
			{
				_ids_tbl(i+1,0).SetStr(kmStrw(L"%d", i));
				_ids_tbl(i+1,1).SetStr(kmStra(ids(i).name) .cuw());
				_ids_tbl(i+1,2).SetStr(ids(i).mac .GetStrw());
				_ids_tbl(i+1,3).SetStr(ids(i).addr.GetStrw());
				_ids_tbl(i+1,4).SetStr(kmStrw(L"%d", ids(i).des_id));
				_ids_tbl(i+1,5).SetStr(ids(i).GetStateStrw());
			}
			_ids_tbl.SetTwoTone(kmRgb(250,240,230), kmRgb(255,250,245));
			_ids_tbl.UpdateItm();
		}
		if(ids_n > 0) _ids_tbl.Show();

		// resize box
		_box.GetCwp().t = kmC(0, _ids_tbl.GetCwp().b.d + 10);

		// update window including relocate child windows
		RelocateChild();
	}

	// upload all file for test
	void UploadAll()
	{
		print("** upload all file\n");

		for(int i = 0; i < _net.GetFiles(0, 0).N1(); ++i)
		{
			auto ret = _net.SendFileTo(0,0,0,i,0, true);

			if(ret != kmNetPtcFileRes::success) break;
		}
	};

	// upload file
	void UploadFile()
	{
		print("** upload file\n");
		_net.UpdateFile(); _net.SendBkupno(); UpdateTbl();
	};

	// download files
	void DnloadFile()
	{
		const zbFiles& files = _net.GetFiles(0,0);

		for(int fid = (int)files.N1(); fid--; )
		{
			if(files(fid).state == zbFileState::bkuponly)
			{	
				_net.RequestFile(0,0,fid);
			}
		}
	};

	// upload bkupno files
	void UploadBkupno()
	{
		print("** update bkupno files\n");
		_net.SendBkupno(); UpdateTbl();
	};

	// print pkeys only for nks svr
	void PrintKeys()
	{
		print("******** keys\n");

		if(_net._mode == zbMode::nks) { _net._nks.Print(); _net.SaveNks(); }
		else
		{
			// print my key
			print("*** my key : %s\n", _net.GetPkey().GetStr().P());

			for(int i = 0; i < _net._vkeys.N1(); ++i)
			{
				print("*** vkey(%d) : %s\n", i, _net._vkeys(i).GetStr().P());
			}

			// print user's pkey
			for(int i = 0; i < _net._users.N1(); ++i)
			{
				print("*** user(%d) : %s\n", i, _net._users(i).key.GetStr().P());
			}
		}
	};

	////////////////////////////////////
	// box and edit functions

	// add string
	int Add(const kmStrw& str)
	{
		_box.Add(str).SetViewPosBottom().Invalidate();

		return _box.GetCurLineIdx();
	};

	// display string
	int Display(const kmStrw& str)
	{
		_box.Add(str).SetViewPosBottom().Invalidate();
		_box.UpdateWindow();

		return _box.GetCurLineIdx();
	};

	// update last string
	void UpdateStr(const kmStrw& str, int line_idx = -1)
	{		
		_box.UpdateStr(str, line_idx).SetViewPosBottom().Invalidate();		
	};

	// parse txt
	//   example  : json request /strg0/file?fid=all
	void ParseTxt()
	{
		const kmStrw& str = _edt.GetStr();

		if(str.Get(kmI(0, 4)) == L"json ") // json or url
		{
			kmStrw json = str.Get(kmI(5,end32));

			Display(kmStrw(L"%s:json > %s", _net._name.cuw().P(), json.P()));

			for(int i = 0; i < _net._users.N1(); ++i)
			{
				_net.SendJsonSync(i, json.cu());
			}
		}
		if(str.Get(kmI(0, 3)) == L"fun ") // fun
		{
			kmStrw fun = str.Get(kmI(4,end32));

			Display(kmStrw(L"%s:fun > %s", _net._name.cuw().P(), fun.P()));

			int idx = 0;

			kmStrw name = fun.GetSub(L'(',  idx);
			kmStrw arg  = fun.GetSub(L')',++idx);

			if(name == L"sendfileto")
			{
				int idx = 0;

				int dst_uid = arg.GetSub(L',',  idx).ToInt(); 
				int uid     = arg.GetSub(L',',++idx).ToInt();
				int sid     = arg.GetSub(L',',++idx).ToInt();
				int fid     = arg.GetSub(L',',++idx).ToInt();
				int opt     = arg.GetSub(L',',++idx).ToInt();

				_net.SendFileTo(dst_uid, uid, sid, fid, opt);
			}
			else if(name == L"sendownsvrprof")
			{
				int dst_uid = arg.ToInt();

				_net.SendOwnSvrProf(dst_uid);
			}
			else if(name == L"sendownprof")
			{
				int dst_uid = arg.ToInt();

				_net.SendOwnProf(dst_uid);
			}
			else if(name == L"sendmmbprof")
			{
				int idx = 0;

				int dst_uid = arg.GetSub(L',',   idx).ToInt();
				int uid     = arg.GetSub(L',', ++idx).ToInt();

				_net.SendMmbProf(dst_uid, uid);
			}
			else if(name == L"sendshrdproftosvr")
			{
				int idx = 0;

				int dst_uid = arg.GetSub(L',',   idx).ToInt();
				int hid     = arg.GetSub(L',', ++idx).ToInt();

				_net.SendShrdProfToSvr(dst_uid, hid);
			}
			else if(name == L"sendshrdproftoclt")
			{
				int idx = 0;

				int dst_uid = arg.GetSub(L',',   idx).ToInt();
				int hid     = arg.GetSub(L',', ++idx).ToInt();

				_net.SendShrdProfToClt(dst_uid, hid);
			}
			else if(name == L"maketestfiles")
			{
				int idx = 0;

				int file_n     = arg.GetSub(L',',   idx).ToInt();
				int size0_byte = arg.GetSub(L',', ++idx).ToInt();
				int inc_byte   = arg.GetSub(L',', ++idx).ToInt();

				MakeTestFiles(file_n, size0_byte, inc_byte);
			}
			else if(name == L"testnetpcksize")
			{
				int idx = 0;

				int src_id     = arg.GetSub(L',',   idx).ToInt();
				int start_byte = arg.GetSub(L',', ++idx).ToInt();
				int end_byte   = arg.GetSub(L',', ++idx).ToInt();
				int del_byte   = arg.GetSub(L',', ++idx).ToInt();

				_net.TestNetPckSize(src_id, start_byte, end_byte, del_byte);
			}
		}
		else if(str(0) == L'/') // command
		{
			Display(kmStrw(L"cmd > %s", str.P(1)));

			int idx = 1;

			kmStrw cmd = str.GetSub(L' ',idx);
			kmStrw prm = str.GetSub(L' ',idx);
			kmStrw val = str.GetSub(L' ',idx);

			if(cmd == L"set")
			{
				if     (prm == L"cnnt_tout_msec")    _net._cnnt_tout_msec    = val.ToFloat();
				else if(prm == L"precnnt_tout_msec") _net._precnnt_tout_msec = val.ToFloat();
				else if(prm == L"precnnt_try_n")     _net._precnnt_try_n     = val.ToInt();
				else if(prm == L"accs_tout_msec")    _net._accs_tout_msec    = val.ToFloat();
				else if(prm == L"accs_try_n")        _net._accs_try_n        = val.ToInt();
				else if(prm == L"blk_byte")          _net.SetFileBlkByte(val.ToInt());
			}
			else if(cmd == L"show")
			{
				if(prm == L"all")
				{
					Display(kmStrw(L"      cnnt_tout_msec    = %f", _net._cnnt_tout_msec));
					Display(kmStrw(L"      precnnt_tout_msec = %f", _net._precnnt_tout_msec));
					Display(kmStrw(L"      precnnt_try_n     = %d", _net._precnnt_try_n));
					Display(kmStrw(L"      accs_tout_msec    = %f", _net._accs_tout_msec));
					Display(kmStrw(L"      accs_try_n        = %d", _net._accs_try_n));
					Display(kmStrw(L"      blk_byte          = %d", _net.GetFileBlkByte()));
				}
				else if(prm == L"my")
				{
					Display(kmStrw(L"     ip  = %s", _net._addr.GetStrw().P()));
					Display(kmStrw(L"     mac = %s", _net._mac .GetStrw().P()));
				}
				else if(prm == L"shrd")
				{
					Display(kmStrw(L"     shrd n = %d", _net._shrds.N1()));

					_net.PrintShrdList();
				}
			}
			else if(cmd == L"preconnect")
			{
				kmStrw port_str = prm.Split(':');

				ushort port = (port_str.N1() > 1) ? port_str.ToInt():_net._port;

				kmAddr4 addr(prm.P(), port);

				print("** preconnect to %s\n", addr.GetStr().P());

				_net.Preconnect(addr);
			}
			else if(cmd == L"connect")
			{
				prm.Printf();

				kmStrw port_str = prm.Split(':');

				ushort port = (port_str.N1() > 1) ? port_str.ToInt():_net._port;

				kmAddr4 addr(prm.P(), port);

				print("** connect to %s\n", addr.GetStr().P());

				//_net.Connect(addr);
				_net.__ConnectNewSvrAsMember(addr);
			}
			else if(cmd == L"connectvkey")
			{
				kmStrw idx1_str = prm.Split('-');

				ushort idx0 = prm.ToInt(), idx1 = idx1_str.ToInt();

				kmNetKey vkey; vkey.SetVkey(idx0, idx1);

				vkey.Print();

				_net.ConnectNewSvrAsMember(vkey);
			}
		}
		else // text
		{
			Display(kmStrw(L"%s > %s", _net._name.cuw().P(), str.P()));

			for(int i = 0; i < _net._users.N1(); ++i)
			{
				_net.SendTxt(i, str.cu());
			}
		}
		_edt.Clear().Invalidate();
	};

	////////////////////////////////////
	// test codes

	void MakeTestFiles(int file_n = 100, int size0_byte = 1024*1024*2, int inc_byte = 127)
	{
		print("**** making test files : %d byte, %d files\n", size0_byte, file_n);

		kmStrw path = _net.GetPath(0,0,0).cuw(); // _net._srcpath;

		for(int i = 0; i < file_n; ++i)
		{
			kmStrw name(L"%s/%03d.txt", path.P(), i);

			kmFile file(name.P(), KF_NEW);

			kmStra data(size0_byte += inc_byte);

			data.SetVal(48 + (i%70));

			file.Write(data.P(), data.N1());
		}
		print("**** done\n");
	};

	// test code only for test
	void TestCode()
	{
		print("\n\n------------------------ test code\n");
		if(0)
		{	
			static kmThread thrd; thrd.Begin([](zbWnd* wnd)
				{
					wnd->ConnectNew();
				}, this);
		}
		if(0)
		{
			kmStra str(1024*3);

			for(int i = 0; i < str.N1(); ++i) str(i) = 'b';

			*str.End() = '\0';

			_net.SendTxt(0, str);

			return;
		}
		if(0)
		{
			//kmThread thrd[10];

			static kmThread thrd[10];

			for(int i = 0; i < 10; ++i)
			{
				thrd[i].Begin([](zbNet* net)
					{
						for(int i = 0; i < 10; ++i)
						{
							net->SendJsonSync(0,kmStra("get /userlist"));
						}
					}, &_net);
			}
			return;
		}		
		if(0)
		{
			for(int i = 10; i--;) print_x(kmfrand32());
			for(int i = 10; i--;) print_i(kmfrand(0,31));
		}
		if(0)
		{	
			print_i(std::endian::big);
			print_i(std::endian::little);
			print_i(std::endian::native);
		}
		if(0)
		{
			kmWork wrk(11, 23, 46.5f, short(32));

			print_i(wrk.Id());
			print_i(wrk.Byte());

			int a1{}; float a2{}; short a3{};

			wrk.Get(a1, a2, a3);

			print_i(a1);
			print_f(a2);
			print_i(a3);

			kmWork b(12, 34);

			int b1{};

			b.Get(b1);

			print_i(b.Byte());
			print_i(b1);
		}
		if(0)
		{
			string test("test");

			print("*** %s\n", test.c_str());

			kmStra str(test.c_str()); str.Printf();

			kmStra str2("abs"); str2 = test.c_str(); str2.Printf();
		}

		if(_net._users.N1() == 0) return;

		if(1)
		{
			kmNetKey vkey = _net.RequestVkey(0, 60*60*24, 123);

			vkey.Print();

			kmAddr4 addr; kmMacAddr mac; kmT2(addr, mac) = _net.RequestAddrToNks(vkey);

			addr.Print();
		}
		if(0)
		{
			zbShrd shrd = {zbShrdType::file, "shrd test", 0};

			shrd.AddMember(1,zbShrdAuth::readwrite);

			shrd.AddItem(0,0,0);
			shrd.AddItem(0,0,1);
			shrd.AddItem(0,0,3);
			shrd.AddItem(0,1,5);
			shrd.AddItem(1,2,3);

			_net.AddShrd(shrd);

			_net.PrintShrdList();
		}
		if(0)
		{
			kmStra jsna = _net._users(0).ToJson(0).dump().c_str();

			jsna.Printf();
		}
		if(0)
		{	
			zbUsers& usr   = _net._users;
			int      usr_n = (int)usr.N1();

			for(int uid = 0; uid < usr_n; ++uid)
			{
				zbStrg&  strg   = _net.GetStrg(uid, 0);
				zbFiles& files  = strg.files;
				int      file_n = (int)files.N1();

				for(int fid = 0; fid < file_n; ++fid)
				{
					files(fid).flg.thumb = 0;
				}
				strg.chknfid = 0;
			}
			print("*** clear thumbnail flag\n");
		}		
		if(0)
		{
			print_i(_net.IsConnected(0));
		}
		if(0)
		{
			print("** send signal to src_id(0) : %.2fmsec\n", _net.SendSig(0));
		}		
		if(0)
		{
			kmAddr4 addr = _net.RequestAddrToNks(_net.GetPkey(), _net._mac);

			print("*** key %s --> ip %s\n", _net.GetPkey().GetStr().P(),  addr.GetStr().P());
		}
		if(0)
		{
			print_i(_net.GetSock().GetSndBufByte());
			print_i(_net.GetSock().GetRcvBufByte());

			_net.GetSock().SetRcvBufByte(65536*8);
			_net.GetSock().SetSndBufByte(65536*8);

			print_i(_net.GetSock().GetSndBufByte());
			print_i(_net.GetSock().GetRcvBufByte());
		}
	};

	void TestRequestAddr()
	{
		if(_net._mode == zbMode::clt)
		{
			print("** test requetst addr : clt\n");

			if(_net._users.N1() < 1) return;

			zbUser& usr  = _net.GetUser(0);
			kmAddr4 addr = _net.RequestAddrToNks(usr.key, usr.mac);

			if(addr.IsInvalid())
				print("** failed to get addr\n");
			else
				print("*** key %s --> ip %s\n", usr.key.GetStr().P(),  addr.GetStr().P());

			int ret; kmAddr4 rcv_addr;

			kmT2(ret, rcv_addr) = _net.SendPreCnnt(addr, 50, 10);

			print("*** send precnnt : %d\n", ret);

			_net.Connect(addr);
		}
		else if(_net._mode == zbMode::svr)
		{
			print("** test requetst addr : svr\n");

			if(!_net.GetPkey().IsValid()) return;

			_net.GetPkey().Print();

			kmAddr4 addr = _net.RequestAddrToNks(_net.GetPkey(), _net._mac);

			if(addr.IsInvalid())
				print("** failed to get addr\n");
			else
				print("*** key %s --> ip %s\n", _net.GetPkey().GetStr().P(),  addr.GetStr().P());
		}
	};

	void TestSendSigNks()
	{
		float echo_msec = _net.SendSigToNks();

		if(echo_msec < 0) print("** test send sig to nks : timeout\n");
		else              print("** test send sig to nks : %.1fmsec\n", echo_msec);
	};

	void TestSendSig()
	{
		_net.TestNetCnd(0);
	};

	void TestNetPckSize()
	{
		_net.TestNetPckSize(0,1024, 1024 + 512, 8);
	};
	////////////////////////////////////
	// callback functions for net
	static int cbRcvNetStt(void* wnd, uchar ptc_id, char cmd_id, void* arg)
	{
		return ((zbWnd*)wnd)->cbRcvNet(ptc_id, cmd_id, arg);
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

		switch(cmd_id)
		{
		case 0 : // rcvreqcnnt
			print("* connection has been requested (id: %d)\n", id);
			_net.GetId(id).Print(); print("\n"); UpdateTbl();
			break;
		case 1 : // rcvaccept
			print("* connection has been accepted  (id: %d)\n", id);
			_net.GetId(id).Print(); print("\n"); UpdateTbl();
			break;
		}
		return 1;
	};
	int cbRcvNetPtcData(char cmd_id, void* arg)
	{
		uchar data_id; char* data; ushort byte, src_id; 

		kmT4(data_id, data, byte, src_id) = _net.GetData();		

		switch(data_id)
		{
		case 0: cbRcvInfo(); break; // receive info
		case 1: UpdateTbl(); break; // receive reqregist
		case 2: UpdateTbl(); break; // receive acceptregist
		case 9: // txt
		{
			Display(kmStra("%s:%d > %s", _net.GetId(src_id).name, src_id, data).cuw());
		}
		break;
		case 10: // json
		{
			//kmStrw jsnw = kmStra(data).EncAtoW(CP_UTF8);

			//Display(kmStrw(L"%s:%d:json > %s", _net.GetId(src_id).name, src_id, jsnw.P()));
		}
		break;
		}
		return 1;
	};
	int cbRcvNetPtcFile(char cmd_id, void* arg)
	{
		//if(cmd_id == 4 || cmd_id == -4) UpdateTbl(); 
		if(cmd_id == 4) UpdateTbl(); 
		return 1;
	};
	// * Note that when you receive rcvinfo, 
	// * you have to decide wheter to request register or not.
	void cbRcvInfo()
	{
		//zbNetInfo info = _net.GetRcvInfo(); // opposite's info
		//
		//if(info.mac.i64 == 0)
		//{
		//	print("**** 1. cbRcvInfo : mac is not valid\n"); return;
		//}
		//print("**** 1. cbRcvInfo : src_id(%d)\n", info.src_id);
		//
		//// reqeust registration
		//if(_net.FindUser(info.mac) < 0) // not registered
		//if(_net._mode == zbMode::clt && info.mode == zbMode::svr)
		//{	
		//	_net.EnqRequestRegist(info.src_id);
		//}
	};

	///////////////////////////////////////////////////
	// network functions

	// connect every network using broadcast
	//void ConnectBroadcast()
	//{
	//	static kmThread thrd; thrd.Begin([](zbWnd* wnd)
	//	{	
	//		int n = wnd->_net.ConnectNew();
	//
	//		print("* number of devices connected : %d\n", n);
	//
	//	},this);
	//};

	// connect to new device
	void ConnectNew()
	{
		static kmThread thrd; thrd.Begin([](zbNet* net)
			{
				zbNetInfos infos;

				int n = net->FindNewSvr(infos);

				if(n == 0) { print("*** there is no new devices\n"); return; };

				for(int i = 0; i < infos.N1(); ++i)
				{
					print("************** %d / %d\n", i + 1, n);

					print(" name : %s\n", infos(i).name.P());
					print(" mac  : %s\n", infos(i).mac.GetStr().P());
					print(" user : %d\n", infos(i).user_n);

					if(infos(i).user_n == 0) net->ConnectNewSvrAsOwner(infos(0));
				}		
			}, &_net);
	};

	// connect with pkey
	void ConnectPkey() {};

	// connect with vkey ...under construction
	void ConnectVkey() {};

	// connect with ip addr
	int Connect(kmAddr4 addr) { return _net.Connect(addr); };

	// reconnect
	void ReconnectAll()
	{
		for(int i = 0; i < _net._users.N1(); ++i)
		{
			int res = _net.Connect(i);

			if(res >= 0) print("** reconnect to uid(%d) : %d (0: last, 1: lan, 2: wan)\n", i, res);
			else         print("** reconnect to uid(%d) : failed\n", i);
		}
	};

	// connect with input
	void ConnectIn()
	{
		char bufin[128];

		cout << "> input ip (ex : 10.114.75.49) : ";
		cin  >> bufin;

		kmStra port_str = kmStra(bufin).Split(':');

		ushort port = (port_str.N1() > 1) ? port_str.ToInt():_net._port;

		kmAddr4 addr(bufin, port);

		cout << "* connect to " << addr.GetStr().P() << endl;		

		Connect(addr);
	};
};

/////////////////////////////////////////////////////////////////
// entry
int main() try
{	
	// set console's code page		
	SetConsoleOutputCP(CP_UTF8);; // UTF8 : CP_UTF8(65001), EUC-KR : 949

	// test code
	if(0)
	{
		system("pause");
		return 0;
	}

	// start window and sock
	//HideConsole();
	kmWnd ::Register(LoadIcon(GetModuleHandle(NULL), MAKEINTRESOURCE(IDI_ICON1)));
	kmSock::Startup();
	kmMdf ::Startup();

	// main process
	{
		zbWnd netwnd;

		netwnd.Create();

		while(!netwnd.IsCreated()) { Sleep(1000); }
		while( netwnd.IsCreated()) { Sleep(1000); }
	}
	//system("pause");

	kmMdf ::Shutdown(); // clean gdi pluse
	kmSock::Cleanup();  // end of winsock
	return 0;
}
catch(kmException e)
{
	print("* zibwin.cpp catched an exception\n");
	kmPrintException(e);
	system("pause");
	return 0;
}

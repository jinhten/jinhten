#ifndef __km7Wnd_H_INCLUDED_2018_11_12__
#define __km7Wnd_H_INCLUDED_2018_11_12__

/* Note ----------------------
* kmMat has been created by Choi, Kiwan
* This is version 7
* kmMat v7 is including the following
*   - km7Define.h
*   - km7Mat.h
*   - km7Wnd.h
*   - km7Dnn.h
*/

// base header
#include "km7Mat.h"
#include "imm.h"
#include "Tlhelp32.h" // for kmPrc

#pragma comment(lib,"imm32.lib")

// set define
#define POSTPROC(A) {if((A) == PP_FINISH) return 0;}

// user message
#define WM_KMWND               (WM_USER + 128)
#define WM_WIMG_LBUTTONDOWN    (WM_KMWND +  0)
#define WM_WIMG_RBUTTONDOWN    (WM_KMWND +  1)
#define WM_KM_UPDATE           (WM_KMWND +  2)
#define WM_KM_MANAGE           (WM_KMWND +  3)

// message for ntf
#define NTF_IMG_POSCLICK       1
#define NTF_IMG_MOUSEMOVE      2
#define NTF_BTN_MOUSEWHEEL     3
#define NTF_BTN_CLICKED        4
#define NTF_TBL_MOUSEWHEEL     5
#define NTF_TBL_CLICKED        6
#define NTF_AXES_UPDATE        7
#define NTF_IMGVIEW_POSCLICK   8
#define NTF_WMAT_DBLCLK        9
#define NTF_EDT_KILLFOCUS     10
#define NTF_EDT_ENTER         11

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class for window

/////////////////////////////////////////////////////////////////////
// class for window process

class kmPrc
{
public:
	// get full path of own process
	static kmStrw GetPath()
	{
		wchar buf[512]; ::GetModuleFileName(NULL, buf, 512);

		return kmStrw(buf);
	};

	// find process 
	//  return : process id (if found), 0 (not found)
	static uint Find(LPCTSTR name)
	{
		// get snapshot
		HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);

		if(snapshot == INVALID_HANDLE_VALUE) { print("* snapshot failed\n"); return 0; }

		// find process
		PROCESSENTRY32 pe32 = { sizeof(PROCESSENTRY32) };

		if(Process32First(snapshot, &pe32))	do
		{
			//printw(L"* process[%d] : %s\n", (int)pe32.th32ProcessID, pe32.szExeFile);

			if(wcscmp(pe32.szExeFile, name) == 0)
			{
				return pe32.th32ProcessID;
			}
		} while(Process32Next(snapshot, &pe32));

		return 0;
	};

	// find module
	//   return : handle of module (if found), 0 (not found)
	static HMODULE FindModule(LPCTSTR name)
	{
		// get snapshot
		HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, kmPrc::Find(name));

		if(snapshot == INVALID_HANDLE_VALUE) { print("* snapshot failed\n"); return 0; }

		// find process
		MODULEENTRY32 me32 = { sizeof(MODULEENTRY32) };

		if(Module32First(snapshot, &me32))	do
		{
			//printw(L"* module[%d] : %s : %s\n", (int)me32.hModule, me32.szExePath, me32.szModule);

			if(wcscmp(me32.szModule, name) == 0)
			{
				return me32.hModule;
			}
		} while(Module32Next(snapshot, &me32));

		return 0;
	};

	// exec program
	static HINSTANCE Exec(LPCTSTR path, int show = SW_SHOW)
	{
		return ::ShellExecute(NULL,L"open", path, NULL, NULL, show);
	};

	// terminate process
	static void Terminate(int process_id)
	{
		HANDLE handle = OpenProcess(PROCESS_TERMINATE, false, process_id);

		TerminateProcess(handle, 1); CloseHandle(handle);
	};
};

/////////////////////////////////////////////////////////////////////
// class for font

// font class.... wrapping LOGFONT
class kmFont
{
public:
	LOGFONT _font;

	kmFont()      { _font = {20,0,0,0,0,0,0,0,HANGUL_CHARSET,0,0,0,0,L"¸¼Àº °íµñ"}; };
	kmFont(int h) { _font = { h,0,0,0,0,0,0,0,HANGUL_CHARSET,0,0,0,0,L"¸¼Àº °íµñ"}; };

	kmFont(int h, const kmStrw& name)
	{
		_font = { h,0,0,0,0,0,0,0,HANGUL_CHARSET,0,0,0,0,L""};
		SetName(name);
	};

	// functions to set
	kmFont& SetH        (int h        ) { _font.lfHeight    = h;          return *this;};
	kmFont& SetW        (int w        ) { _font.lfWidth     = w;          return *this;};	
	kmFont& SetItalic   (uchar b = 1  ) { _font.lfItalic    = b;          return *this;};
	kmFont& SetUnderline(uchar b = 1  ) { _font.lfUnderline = b;          return *this;};	
	kmFont& SetBold     (int w = 1000 ) { _font.lfWeight    = w;          return *this;};
	kmFont& SetAngle    (int ang_deg10) { _font.lfEscapement = ang_deg10; return *this;};
	kmFont& SetName(const kmStrw& str)  { str.CopyTo(_font.lfFaceName);   return *this;};

	void Set(const kmStrw& str, int height, int bold = 0, uchar italic = 0, uchar underline = 0)
	{
		SetName     (str);
		SetH        (height);
		SetBold     (bold);
		SetItalic   (italic);
		SetUnderline(underline);
	};

	void Set(int height, int bold = 0, uchar italic = 0, uchar underline = 0)
	{	
		SetH        (height);
		SetBold     (bold);
		SetItalic   (italic);
		SetUnderline(underline);
	};

	// conversion operator... (LOGFONT) a
	operator LOGFONT () const { return  _font; };
	operator LOGFONT*()       { return &_font; };

	// functions to get
	LOGFONT  Get()          const { return _font;            };
	int      GetH()         const { return _font.lfHeight;   };
	int      GetW()         const { return _font.lfWidth;    };
	int      GetBold()      const { return _font.lfWeight;   };
	uchar    GetItalic()    const { return _font.lfItalic;   };
	uchar    GetUnderline() const { return _font.lfUnderline;};
	kmStrw   GetName()      const { return _font.lfFaceName; };
};
typedef kmMat1<kmFont> kmFonts;

/////////////////////////////////////////////////////////////////////
// graphic object function for kmWnd

// base class for graphic object
class kmgObj
{
public:
	uint   _mode = 1; // 1st bit: visible on/off
	kmRect _rt   = {0,0,0,0};
	kmRgb  _rgb  = 0;;

	// constructor
	kmgObj() {};
	kmgObj(kmRect rect, kmRgb rgb) : _rt(rect), _rgb(rgb) {};

	// destructor
	virtual ~kmgObj() {};

	// pure virtual member functions
	virtual void Draw(HDC hdc, int w, int h) = 0;
	
	///////////////////////////////////////////////////
	// general member functions
	int     GetW()    { return _rt.GetW(); };
	int     GetH()    { return _rt.GetH(); };
	kmRect& GetRect() { return _rt;        };
	kmRgb&  GetRgb()  { return _rgb;       };

	kmgObj& Set    (kmRect rt, kmRgb rgb) { _rt = rt; _rgb = rgb; return *this;};
	kmgObj& SetRect(kmRect rt )           { _rt  = rt;            return *this;};
	kmgObj& SetRgb (kmRgb  rgb)           { _rgb = rgb;           return *this;};

	// functions to set or get mode
	bool IsVisible() const { return GETBIT(_mode,0);};

	void SetVisible(bool on = true) { if(on) SETBIT(_mode,0); else CLEARBIT(_mode,0);};
};
typedef kmMat1<kmgObj*> kmgObjs;

// line class
class kmgLine : public kmgObj
{
public:	
	int _width = 0;
	int _style = 0;	

	// constructor
	kmgLine()  {};
	kmgLine(kmRect rt, kmRgb rgb) : kmgObj(rt, rgb) {};

	// vitual functions
	virtual void Draw(HDC hdc, int w = 0, int h = 0)
	{	
		if(IsVisible())
		{	
			HPEN   pen   = CreatePen(_style, _width, _rgb);
			HPEN   pen_o = (HPEN) SelectObject(hdc, pen);

			MoveToEx(hdc, _rt.l, _rt.t, NULL);
			LineTo  (hdc, _rt.r, _rt.b);

			// delete objects
			// * Note that deleting the object from GetStockObject() is not necessary,
			// * but it's not harmful according to the document of docs.microsoft.com			
			DeleteObject(SelectObject(hdc, pen_o  ));
		}
	};

	///////////////////////////////////////////////////
	// general member functions	
	int   GetWidth()  { return _width; };
	int   GetStyle()  { return _style; };	

	void Set(kmRect rt, kmRgb rgb, int width = 0, int style = 0)
	{
		_rt = rt; _rgb = rgb; _width = width; _style = style;
	};

	void SetWidth(int width) { _width = width; };
	void SetStyle(int style) { _style = style; };
};
typedef kmMat1<kmgLine> kmgLines;

// region class
class kmgReg : public kmgObj
{
public:
	// constructor
	kmgReg()  {};
	kmgReg(kmRect rt, kmRgb rgb) : kmgObj(rt, rgb) {};

	// vitual functions
	virtual void Draw(HDC hdc, int w, int h)
	{	
		if(IsVisible()) 
		{
			HBRUSH brush = CreateSolidBrush(_rgb);

			FillRect(hdc, _rt, brush);

			DeleteObject(brush);
		}
	};
};

// rect class
class kmgRect : public kmgObj
{
public:
	kmRgb _line_rgb   = 0;
	int   _line_width = 0;
	int   _line_style = 0;

	// constructor
	kmgRect()  {};
	kmgRect(kmRect rt, kmRgb rgb                ) : kmgObj(rt, rgb) {};
	kmgRect(kmRect rt, kmRgb rgb, kmRgb line_rgb) : kmgObj(rt, rgb) { _line_rgb = line_rgb; };

	// vitual functions
	virtual void Draw(HDC hdc, int w = 0, int h = 0)
	{	
		if(IsVisible())
		{
			// if _a > 0, bk is transparent
			HBRUSH brush   = (_rgb._a == 0) ? CreateSolidBrush(_rgb) : (HBRUSH) GetStockObject(NULL_BRUSH);
			HBRUSH brush_o = (HBRUSH) SelectObject(hdc, brush);

			// if _a > 0, line is transparent
			HPEN   pen   = (_line_rgb._a == 0) ? CreatePen(_line_style, _line_width, _line_rgb) : (HPEN) GetStockObject(NULL_PEN);
			HPEN   pen_o = (HPEN) SelectObject(hdc, pen);

			Rectangle(hdc, _rt.l, _rt.t, _rt.r, _rt.b);

			// delete objects
			// * Note that deleting the object from GetStockObject() is not necessary,
			// * but it's not harmful according to the document of docs.microsoft.com
			DeleteObject(SelectObject(hdc, pen_o  ));
			DeleteObject(SelectObject(hdc, brush_o));
		}
	};

	///////////////////////////////////////////////////
	// general member functions
	kmRgb GetLineRgb  ()  { return _line_rgb;   };
	int   GetLineWidth()  { return _line_width; };
	int   GetLineStyle()  { return _line_style; };

	void SetLine(kmRgb rgb, int width = 0, int style = 0)
	{
		_line_rgb = rgb; _line_width = width; _line_style = style;
	};
};

// string class with font and color
// * Note that the member of font is a pointer type 
// * since sharing a font is better for memory efficiency.
class kmgStr : public kmgObj
{
public:
	kmStrw  _str;
	kmFont* _pfont;
	kmRgb   _bk_rgb;
	int     _bk_mode;  // TRANSPARENT, OPAQUE
	uint    _fmt;      // DT_LEFT, DT_RIGHT, DT_CENTER...

	// constructor
	kmgStr()                                  { Init(); };
	kmgStr(const kmStrw& str)                 { Init(); _str = str;};
	kmgStr(const kmStrw& str, kmFont* pfont)  { Init(); _str = str; _pfont = pfont; };
	kmgStr(const kmStrw& str, kmFont* pfont, kmRgb rgb, kmRect rt, uint fmt = DT_LEFT)
	{
		Init();
		Set(str, pfont, rgb, rt, fmt);
	};
	kmgStr(const kmStrw& str, kmRgb rgb, kmRect rt, uint fmt = DT_LEFT)
	{
		Init();
		Set(str, rgb, rt, fmt);
	};

	// virtual functions
	virtual void Draw(HDC hdc, int w = 0, int h = 0)
	{
		if(_pfont == NULL) DrawStr(hdc);
		else
		{
			// select font
			HFONT font     = CreateFontIndirect(*_pfont);
			HFONT font_old = (HFONT) SelectObject(hdc, font);
		
			// draw text
			DrawStr(hdc);

			// release font
			SelectObject(hdc, font_old);
			DeleteObject(font);
		}
	};

	void DrawStr(HDC hdc) const
	{
		// set color and mode
		if(_bk_mode != TRANSPARENT) SetBkColor(hdc, _bk_rgb );
		SetTextColor(hdc, _rgb);
		::SetBkMode (hdc, _bk_mode);

		// draw text
		if(_str.N() > 0)DrawTextEx(hdc, _str, -1, _rt, _fmt, NULL);
	};

	///////////////////////////////////////////////////
	// general member functions

	// functions to init
	void Init()
	{
		_pfont   = NULL; 
		_rgb     = RGB_BLACK; 
		_bk_rgb  = RGB_WHITE;
		_bk_mode = TRANSPARENT;
		_rt      = {0,0,100,100};
		_fmt     = DT_LEFT; 
		_str     = L"";
	};

	// functions to set
	void Set(const kmStrw& str, kmFont* pfont, kmRgb rgb, kmRect rt, uint fmt = DT_LEFT)
	{
		_str     = str;
		_pfont   = pfont;
		_rgb     = rgb;
		_rt      = rt;
		_fmt     = fmt;
	};
	void Set(const kmStrw& str, kmRgb rgb, kmRect rt, uint fmt = DT_LEFT)
	{
		_str     = str;
		_rgb     = rgb;
		_rt      = rt;
		_fmt     = fmt;
	};
	void Set(kmFont* pfont, kmRgb rgb, kmRect rt, uint fmt = DT_LEFT)
	{
		_pfont   = pfont;
		_rgb     = rgb;
		_rt      = rt;
		_fmt     = fmt;
	};
	void SetStr   (kmStrw  str)   { _str     = str;  };
	void SetFont  (kmFont* pfont) { _pfont   = pfont;};
	void SetBkRgb (kmRgb   rgb)   { _bk_rgb  = rgb;  };
	void SetBkMode(int     mode)  { _bk_mode = mode; };
	void SetFormat(uint    fmt)   { _fmt     = fmt;  };

	// function to get
	kmStrw& GetStr()          { return _str;         };
	kmFont* GetFont()   const { return _pfont;       };
	kmRgb   GetBkRgb () const { return _bk_rgb;      };
	kmRgb   GetBkMode() const { return _bk_mode;     };
	uint    GetFormat() const { return _fmt;         };
	int64   GetLen()    const { return _str.GetLen();};

	// function to edit
	void Erase  (int64 idx) { _str.Erase(idx); };
	void Erase  (kmI   i1)  { _str.Erase(i1);  };
	void Insert (int64 idx, const wchar& val) { _str.Insert(idx, val); };
	void Replace(int64 idx, const wchar& val) { _str(idx) = val;       };

	// get width and height of string on screen
	// * Note that if nstr is less than 0, nstr will be set with the number of string.
	void GetStrSizePix(int& w, int& h, HDC hdc, int nstr = -1)
	{
		// init nstr
		const int strlen = int(_str.GetLen() - 1); // without null-terminating
		
		nstr = MIN(MAX(0, (nstr < 0) ? strlen : nstr), strlen);

		// get size
		SIZE size;

		if(_pfont == NULL) GetTextExtentPoint32W(hdc, _str, nstr, &size);
		else
		{
			// select font
			HFONT font     = CreateFontIndirect(*_pfont);
			HFONT font_old = (HFONT) SelectObject(hdc, font);
		
			// get size of string
			GetTextExtentPoint32W(hdc, _str, nstr, &size);

			// release font
			SelectObject(hdc, font_old);
			DeleteObject(font);
		}
		// set output
		w = size.cx; h = size.cy;
	};

	int GetStrWidth (HDC hdc, int nstr = -1) { int w, h; GetStrSizePix(w, h, hdc, nstr); return w; };
	int GetStrHegiht(HDC hdc, int nstr = -1) { int w, h; GetStrSizePix(w, h, hdc, nstr); return h; };

	int GetStrPosLeft (HDC hdc)
	{	
		// consider alignment		
		if     (_fmt & 0x01) return (_rt.r - GetStrWidth(hdc))/2;  // center
		else if(_fmt & 0x02) return  _rt.r - GetStrWidth(hdc);     // right
		return 0; // left
	};
	int GetStrPosRight(HDC hdc, int nstr = -1) { return GetStrPosLeft(hdc) + GetStrWidth(hdc, nstr); };
};
typedef kmMat1<kmgStr> kmgStrs;

// scroll bar class
class kmgBar : public kmgObj
{
public:
	int     _w    = 16;     // width of bar, pixel
	float   _ratt = 0.3f;   // ratio of top    of bar 
	float   _ratb = 0.7f;   // ratio of bottom of bar
	kmRect  _bar_rt;
	kmRgb   _bar_rgb;

	// set 
	kmgBar& SetW  (int w) { _w = w; return *this; };
	kmgBar& SetRgb(kmRgb bar_rgb, kmRgb bk_rgb)
	{
		_bar_rgb = bar_rgb; _rgb = bk_rgb; return *this;
	};

	// set ratio of bar postion
	//  ho : height of object (pixel)
	//  hv : height of view   (pixel)... (which is equal to bar's height)
	//  fb : distance from bottom of object to bottom of view (pixel)
	kmgBar& SetRat(int ho, int hv, int fb)
	{
		_ratt = float(ho - hv - fb)/ho;
		_ratb = float(ho      - fb)/ho;
		return *this;
	};

	// vitual functions
	virtual void Draw(HDC hdc, int w, int h)
	{	
		if(!IsVisible()) return;

		// init rt
		_rt     = kmRect(w - _w,  0, w, h);
		_bar_rt = kmRect(_rt.l + 2, int(h*_ratt), _rt.r - 2, int(h*_ratb));

		// draw		
		HBRUSH brush0 = CreateSolidBrush(_rgb);
		HBRUSH brush1 = CreateSolidBrush(_bar_rgb);

		FillRect(hdc,     _rt, brush0);
		FillRect(hdc, _bar_rt, brush1);

		DeleteObject(brush0);
		DeleteObject(brush1);
	};	
};

// axis class
class kmgAxis : public kmgObj
{
public:
	int        _type  = 0; // 0 : x-axis, 1: y-axis
	int        _dir   = 1; // 1 : forward, -1 : reverse
	float      _min   = 0;
	float      _max   = 1;
	kmFont*    _pfont = NULL;
	kmMat1f32  _tick;
	
	// constructor		
	kmgAxis() {};
	kmgAxis(int type, int dir = 1) { _type = type; _dir = dir;};

	// set axis
	void Set(int type, kmFont* pfont, int dir = 1) { _type = type; _pfont = pfont; _dir = dir; };

	// update tick
	void UpdateTick(int n, float min, float max)
	{
		_tick.Recreate(n);
		_tick.SetSpace125(min, max);
		_min = min; _max = max;
	};

	// get tick
	kmMat1f32& GetTick() { return _tick; };

	// get pix from val
	int GetPixFromVal(float val) const
	{
		const float rat = (_max == _min)? 0.5f : (val - _min)/(_max - _min);
		const float pix = rat*((_type == 0) ? _rt.GetW() : -_rt.GetH());
		return int((_dir > 0) ? pix : -pix);
	};
		
	// vitual functions
	virtual void Draw(HDC hdc, int w = 0, int h = 0)
	{	
		if(!IsVisible()) return;

		if(_type == 0) DrawXaxis(hdc);
		else           DrawYaxis(hdc);
	};

	// draw x-axis-x
	void DrawXaxis(HDC hdc) const
	{
		// init parameters
		const int h_font = (int) (_pfont->GetH()*1.4f);

		// draw axis
		kmRect xrt0 = {0, 0, 70, h_font};

		if(_dir > 0) xrt0.SetCen(_rt.l, _rt.b);
		else         xrt0.SetCen(_rt.r, _rt.b);

		xrt0.ShiftY(xrt0.GetH()/2 + 5);

		kmgStr xstr; xstr.Set(_pfont, kmRgb(0,0,0), xrt0, DT_CENTER | DT_TOP);
		
		for(int i = 0; i < _tick.N1(); ++i)
		{
			float x = _tick(i);

			if(x < _min || _max < x) continue;

			kmRect xrt = xrt0; xrt .ShiftX(GetPixFromVal(x));
			kmStrw strw;       strw.SetAxisFloat(x, _min, _max); //strw.SetFloat3(x);

			// draw xtick
			xstr.SetStr(strw);
			xstr.SetRect(xrt);
			xstr.Draw(hdc);
		}
	};

	// draw y-axis
	void DrawYaxis(HDC hdc) const
	{
		// init parameters
		const int h_font = (int) (_pfont->GetH()*1.4f);

		// draw axis
		kmRect yrt0 = {0, 0, 70, h_font}; 

		if(_dir > 0) yrt0.SetCen(_rt.l, _rt.b);
		else         yrt0.SetCen(_rt.l, _rt.t);

		yrt0.ShiftX(-yrt0.GetW()/2 - 5);

		kmgStr ystr; ystr.Set(_pfont, kmRgb(0,0,0), yrt0, DT_VCENTER | DT_RIGHT | DT_SINGLELINE);
		
		for(int i = 0; i < _tick.N1(); ++i)
		{
			float y  = _tick(i); if(y == 0) y = 0;

			if(y < _min || _max < y) continue;

			kmRect yrt = yrt0; yrt .ShiftY(GetPixFromVal(y));
			kmStrw strw;       strw.SetAxisFloat(y, _min, _max); //strw.SetFloat3(y);

			// draw ytick
			ystr.SetStr(strw);
			ystr.SetRect(yrt);
			ystr.Draw(hdc);
		}
	};
};

// image class
class kmgImg : public kmgObj
{
public:
	kmImg _img;
		
	// constructor
	kmgImg()  {};
	kmgImg(kmRect rt, kmImg img) : kmgObj(rt, 0), _img(img) {};

	// vitual functions
	virtual void Draw(HDC hdc, int w = 0, int h = 0)
	{	
		if(!IsVisible()) return;
		
		// get dc and memory dc... start drawing		
		HDC mdc = CreateCompatibleDC(hdc);

		// get bitmap handle		
		HBITMAP hbm = CreateCompatibleBitmap(hdc, _img.GetW(), _img.GetH());

		// get bitmap info
		BITMAP bm;
		GetObject(hbm, sizeof(BITMAP), &bm);

		// set img to bitmap		
		SetBitmapBits(hbm, (int)_img.GetByteFrame(),(void*) _img.Begin());
		
		// draw bitmap on memory dc
		HBITMAP hbm_ = (HBITMAP) SelectObject(mdc, hbm);
				
		// copy mdc to hdc
		//BitBlt(hdc, 0, 0, _win.w, _win.h, mdc, 0, 0, SRCCOPY);
		SetStretchBltMode(hdc, HALFTONE); // choose COLORNCOLOR for efficiency
		                                  // choose HALFTONE for quality
		StretchBlt(hdc, _rt.l, _rt.t, _rt.GetW(), _rt.GetH(),
			       mdc, 0, 0, _img.GetW(), _img.GetH(), SRCCOPY);

		// finish drawing
		SelectObject(mdc, hbm_);
		DeleteObject(hbm);
		DeleteDC    (mdc);
	};

	///////////////////////////////////////////////////
	// general member functions
	kmImg& GetImg() { return _img; };
};

/////////////////////////////////////////////////////////////////////
// class for window

// window input parameter class
class kmWin
{
public:	
	LPCTSTR		class_name;
	LPCTSTR		win_name;
	HINSTANCE	instance;
	HMENU		menu;
	HWND		parent;
	LPVOID		param;
	DWORD		style;
	DWORD		ex_style;
	int			x, y, w, h;

	// constructor 
	kmWin()		{ Init(); };

	// init
	void Init()
	{
		class_name = L"kmWnd";
		win_name   = NULL;
		instance   = GetModuleHandle(NULL);
		menu       = NULL;
		parent     = NULL;
		param      = NULL;
		style      = WS_OVERLAPPEDWINDOW | WS_VISIBLE | WS_CLIPCHILDREN;
		ex_style   = WS_EX_LEFT;
		x = 0; y = 0; w  = 320; h = 160;
	};

	// create window
	HWND Create()
	{
		return CreateWindowEx(ex_style, class_name, win_name, style, 
							  x, y, w, h, parent, menu, instance, param);
	};
};

// flag of kmWnd
union kmwflag
{
	// member
	uint val = 0;
	struct
	{
		uint is_sdm     : 1; // self-destroy mode on/off
		uint is_created : 1; // is fully created including child windows
	};

	// constuctor
	kmwflag() {};

	// compy constructor
	kmwflag(const uint& a) { val = a; };

	// assignment operator
	kmwflag& operator=(const uint& a) { val = a; return *this; };

	// conversion operator... (uint) a
	operator uint() const { return val; };

	// is functions
	bool IsSdm()     { return is_sdm     == 1; };
	bool IsCreated() { return is_created == 1; };
};

// window base class
class kmWnd
{
protected:	
	HWND     _hwnd = NULL;
	kmWin    _win;
	kmwflag  _flag;
	kmLock   _lck;      // for multi-thread safe

public:	
	// constructor
	kmWnd()	{};

	// destructor
	virtual ~kmWnd() {};
		
	// copy constructor    --> default
	// assignment operator --> default	

	// Init function
	void Init()	{ _hwnd  = NULL; _flag = 0; };

	// Destroy window
	void Destroy()	{ SendMessage(_hwnd, WM_CLOSE, 0, 0); };

	// Register window
	static void Register(HICON icon = NULL)
	{
		WNDCLASS wc;

		wc.cbClsExtra		= 0;
		wc.cbWndExtra		= 0;
		wc.hbrBackground	= (HBRUSH)GetStockObject(WHITE_BRUSH);
		wc.hCursor			= LoadCursor(NULL,IDC_ARROW);
		wc.hIcon			= (icon == NULL) ? LoadIcon(NULL,IDI_APPLICATION):icon;
		wc.hInstance		= GetModuleHandle(NULL);
		wc.lpfnWndProc		= (WNDPROC) (_Proc);
		wc.lpszClassName	= L"kmWnd";
		wc.lpszMenuName		= NULL;
		wc.style			= CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS;

		RegisterClass(&wc);
	};

	// get class name
	LPCSTR GetKmClass() { return typeid(*this).name() + 6; };

	// get point of kmWnd
	static kmWnd* GetWnd(HWND hwnd)
	{
		return (kmWnd*) GetWindowLongPtr(hwnd, GWLP_USERDATA);
	};

	/////////////////////////////////////////////////
	// window procedure
protected:	
	LRESULT static CALLBACK _Proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
	{
		// get user-data		
		kmWnd* wnd = GetWnd(hwnd);

		// * Note that WM_CREATE and WM_NCCREATE are unavailable
		// * because wnd will be null before the window is created.

		// call procuder function
		return (wnd != NULL)? wnd->Proc    (hwnd, msg, wp, lp):
		                      DefWindowProc(hwnd, msg, wp, lp);
	};

	virtual LRESULT Proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) 
	{
		// message procedure
		switch(msg)
		{
		case WM_DESTROY     : PostQuitMessage(0); return 0;		
		case WM_NCDESTROY   : if(_flag.IsSdm()) delete this; else Init(); return 0;
		
		case WM_PAINT       : POSTPROC(OnPaint      (wp, lp)); break;
		case WM_LBUTTONDOWN : POSTPROC(OnLButtonDown(wp, lp)); break;
		case WM_RBUTTONDOWN : POSTPROC(OnRButtonDown(wp, lp)); break;
		case WM_KEYDOWN     : POSTPROC(OnKeyDown    (wp, lp)); break;
		case WM_CHAR        : POSTPROC(OnChar       (wp, lp)); break;
		case WM_COMMAND     : POSTPROC(OnCommand    (wp, lp)); break;
		case WM_CONTEXTMENU : POSTPROC(OnContextMenu(wp, lp)); break;
		case WM_SIZE        : POSTPROC(OnSize       (wp, lp)); break;
		case WM_TIMER       : POSTPROC(OnTimer      (wp, lp)); break;
		case WM_MOUSEMOVE   : POSTPROC(OnMouseMove  (wp, lp)); break;
		case WM_MOUSEWHEEL  : POSTPROC(OnMouseWheel (wp, lp)); break;
		case WM_SETFOCUS    : POSTPROC(OnSetFocus   (wp, lp)); break;
		case WM_KILLFOCUS   : POSTPROC(OnKillFocus  (wp, lp)); break;
		case WM_NCPAINT     : POSTPROC(OnNcPaint    (wp, lp)); break;
		case WM_ERASEBKGND  : POSTPROC(OnEraseBkgnd (wp, lp)); break;
		case WM_KM_UPDATE   : POSTPROC(OnUpdate     (wp, lp)); break;
		case WM_DROPFILES   : POSTPROC(OnDropFiles  (wp, lp)); break;

		case WM_KM_MANAGE   : POSTPROC(OnKmManage   (wp, lp)); break;
		}
		return DefWindowProc(hwnd, msg, wp, lp);
	};		
	virtual kmePPType OnLButtonDown(WPARAM wp, LPARAM lp) { SetFocus(); return PP_DEFPROC; };
	virtual kmePPType OnRButtonDown(WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnKeyDown    (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnChar       (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnCommand    (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnContextMenu(WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnSize       (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnTimer      (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnMouseMove  (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnMouseWheel (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnSetFocus   (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnKillFocus  (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnNcPaint    (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnEraseBkgnd (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnUpdate     (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnDropFiles  (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };
	virtual kmePPType OnKmManage   (WPARAM wp, LPARAM lp) { return PP_DEFPROC; };

	virtual kmePPType OnPaint(WPARAM wp, LPARAM lp)
	{
		// init parameters
		PAINTSTRUCT ps;
		RECT        rt; GetClientRect(&rt);
		int         w = rt.right;
		int         h = rt.bottom;
	
		// get dc and memory dc... start drawing
		HDC hdc = BeginPaint(_hwnd, &ps);
		HDC mdc = CreateCompatibleDC(hdc);

		// attach bitmap to mdc
		HBITMAP hbm  = CreateCompatibleBitmap(hdc, w, h);
		HBITMAP hbm_ = (HBITMAP) SelectObject(mdc, hbm);

		// draw background
		FillRect(mdc, &rt, (HBRUSH) GetStockObject(WHITE_BRUSH));

		// draw graphic objects
		DrawGobjs(mdc, w, h);
	
		// copy mdc to hdc
		BitBlt(hdc, 0, 0, w, h, mdc, 0, 0, SRCCOPY);
		
		// finish drawing		
		SelectObject(mdc, hbm_);
		DeleteObject(hbm);
		DeleteDC    (mdc);
		EndPaint    (_hwnd, &ps);
	
		return PP_FINISH;
	};

public:
	virtual void DrawGobjs(HDC hdc, int w, int h) {};
	virtual void DrawGobjsLater() {};

	/////////////////////////////////////////////////
	// functions for creating
public:
	// create window with new thread
	// * Note that you have to set sdm(self-destroy mode) as 1 if you create the object with "new" command.
	void CreateThread(int x, int y, int w, int h, LPCTSTR win_name = NULL, uint sdm = 0)
	{		
		// init parameters
		_win.x = x;
		_win.y = y;
		_win.w = w;
		_win.h = h;
		_flag = (sdm > 0) ? 1:0;

		if(win_name != NULL) _win.win_name = win_name;
		
		// create message thread
		HANDLE hthr = (HANDLE)_beginthreadex(NULL, 0, ThCreate, this, 0, NULL);

		ASSERTA(hthr != NULL, "failed to create thread");
	};

	// create window 
	void Create(int x, int y, int w, int h, LPCTSTR win_name = NULL, uint sdm = 0)
	{
		// init parameters
		_win.x = x;
		_win.y = y;
		_win.w = w;
		_win.h = h;
		_flag = (sdm > 0) ? 1:0;

		if(win_name != NULL) _win.win_name = win_name;

		// create window
		Create();
	};

	static uint __stdcall ThCreate(void* arg) try
	{
		// get pointer of kmWnd
		kmWnd* wnd = (kmWnd*) arg;

		PRINTFA("* thread is running : %s\n", wnd->GetKmClass() );

		// create window
		wnd->Create();

		// message loop
		MSG msg;

		while(GetMessage(&msg, NULL, 0, 0))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		PRINTFA("* thread finished : %s\n", wnd->GetKmClass());
		return 0;
	}
	catch(kmException e)
	{
		PRINTFA("* kmWnd::ThCreate catched an exception\n");
		kmPrintException(e); //system("pause");
		return 0;
	};

protected:
	// create window... core
	void Create()
	{
		// create window
		_hwnd = _win.Create();

		ASSERTA(_hwnd != NULL, "[Create in 275]");

		// set pointer of this
		SetWindowLongPtr(_hwnd, GWLP_USERDATA, (LONG_PTR) this);

		// create child window
		CreateChild();

		// set flag
		_flag.is_created = 1; 
	};

	// create child window
	virtual void CreateChild() {};

	// invalidate child windows
	virtual void InvalidateChild() {};

	//////////////////////////////////////
	//  member fucntions
public:
	HWND    GetHwnd() { return _hwnd;  };
	int     GetX()    { return _win.x; };
	int     GetY()    { return _win.y; };
	int     GetH()    { return _win.h; };
	int     GetW()    { return _win.w; };
	kmWin&  GetWin()  { return _win;   };

	// * Note that width is (right - left) not (right - left + 1)
	// * and height is (bottom - top) not (bottom - top + 1)
	// * in the window system
	void GetRect(RECT* rect) 
	{
		rect->left   = _win.x;
		rect->top    = _win.y;
		rect->right  = _win.x + _win.w;
		rect->bottom = _win.y + _win.h;
	};
	kmRect GetRect()
	{		
		return kmRect(_win.x, _win.y, _win.x + _win.w, _win.y + _win.h);
	};

	// get the coordinates of a window's client area
	// * Note that rect.top and rect left is always zero.
	BOOL   GetClientRect(RECT* rect) { return ::GetClientRect(_hwnd, rect); };
	kmRect GetClientRect()
	{
		kmRect rt; GetClientRect((RECT*)rt); return rt;
	};

	int GetClientH() { RECT rect; GetClientRect(&rect); return rect.bottom; };
	int GetClientW() { RECT rect; GetClientRect(&rect); return rect.right;  };

	void GetClientWH(int* w, int* h)
	{
		RECT rect; GetClientRect(&rect);

		*w = rect.right;
		*h = rect.bottom;
	};

	// get the coordinates of window relative to the upper-left corner of the screen
	BOOL   GetWindowRect(RECT* rect) { return ::GetWindowRect(_hwnd, rect); };
	kmRect GetWindowRect()
	{
		kmRect rt; GetWindowRect((RECT*)rt); return rt;
	};

	int GetWindowH() { RECT rect; GetWindowRect(&rect); return rect.bottom; };
	int GetWindowW() { RECT rect; GetWindowRect(&rect); return rect.right;  };
	int GetWindowX() { RECT rect; GetWindowRect(&rect); return rect.left;   };
	int GetWindowY() { RECT rect; GetWindowRect(&rect); return rect.top;    };

	void GetWindowWH(int* w, int* h)
	{
		RECT rect; GetWindowRect(&rect);

		*w = rect.right;
		*h = rect.bottom;
	};

	// move window
	BOOL Move(int x, int y, int w = 0, int h = 0, BOOL repaint = TRUE)
	{
		if(w == 0) w = _win.w;
		if(h == 0) h = _win.h;

		_win.x = x; _win.y = y;	_win.w = w;	_win.h = h;
		
		return MoveWindow(_hwnd, x, y, w, h, repaint);
	};

	// get cursor position on client window
	kmT2<int, int> GetCursorPos() 
	{
		POINT pt; ::GetCursorPos(&pt); ScreenToClient(_hwnd, &pt);

		int x = pt.x; int y = pt.y;

		return kmT2(x, y);
	};

	// get cursor position on screen
	kmT2<int, int> GetCursorPosScn() 
	{
		POINT pt; ::GetCursorPos(&pt);

		int x = pt.x; int y = pt.y;

		return kmT2(x, y);
	};

	// set and get window text
	BOOL SetWinTxt (LPCWSTR str)              { return SetWindowText(_hwnd, str);};
	BOOL SetWinName(LPCWSTR str)              { return SetWinTxt    (       str);};	
	int  GetWinTxt (LPWSTR str, int size_str) { return GetWindowText(_hwnd, str, size_str);};
	int  GetWinName(LPWSTR str, int size_str) { return GetWinTxt    (       str, size_str);};

	void SetWinStr(const wchar_t* str, ...)
	{
		kmStrw buf;
		va_list args;
		va_start(args, str);
		{
			const int len = _vscwprintf(str, args) + 1;

			if(len > 0) { buf.Recreate(len); vswprintf_s(buf.P(), len, str, args); }
		}
		va_end(args);
		SetWinTxt(buf.P());
	};

	kmStrw GetWinStr(int64 size = 1024) 
	{
		kmStrw str(size);
		GetWinTxt(str.Begin(), (int)str.Size());
		return str;
	};

	BOOL Invalidate(            BOOL erase = FALSE) {return InvalidateRect(_hwnd, NULL, erase);};
	BOOL Invalidate(RECT* rect, BOOL erase = FALSE) {return InvalidateRect(_hwnd, rect, erase);};

	// invalidate including child windows
	BOOL InvalidateAll()
	{
		InvalidateChild(); return Invalidate();
	};

	BOOL UpdateWindow()	{ return ::UpdateWindow(_hwnd);	};

	// send message and wait for finishing message as calling directly procedure
	LRESULT SendMsg(UINT msg, WPARAM wp = 0, LPARAM lp = 0) { return SendMessage(_hwnd, msg, wp, lp); };

	// send message without waiting as adding message queue	
	LRESULT PostMsg(UINT msg, WPARAM wp = 0, LPARAM lp = 0) { return PostMessage(_hwnd, msg, wp, lp); };

	UINT_PTR SetTimer(int64 timer_id, uint tstep_msec, TIMERPROC timer_proc = NULL)
	{
		return ::SetTimer(_hwnd, timer_id, tstep_msec, timer_proc);
	};

	BOOL KillTimer(int64 timer_id)
	{
		return ::KillTimer(_hwnd, timer_id);
	};

	BOOL SetForeground()         { return ::SetForegroundWindow(_hwnd);};
	HWND SetFocus()	             { return ::SetFocus(_hwnd);           };
	BOOL Show(int cmd = SW_SHOW) { return ::ShowWindow(_hwnd, cmd);	   };
	BOOL Hide()                  { return ::ShowWindow(_hwnd, SW_HIDE);     };
	BOOL Restore()               { return ::ShowWindow(_hwnd, SW_RESTORE);  };
	BOOL Minimize()              { return ::ShowWindow(_hwnd, SW_MINIMIZE); };
	BOOL Maximize()              { return ::ShowWindow(_hwnd, SW_MAXIMIZE); };

	void SetWinStyle(DWORD rem_style, DWORD add_style)
	{
		_win.style &= ~rem_style;
		_win.style |=  add_style;
	};

	void SetWinStyleEx(DWORD rem_style, DWORD add_style)
	{
		_win.ex_style &= ~rem_style;
		_win.ex_style |=  add_style;
	};

	// check if window has been created
	// * Note that this doesn't guarantee child windows have been created
	BOOL IsCreated() { return IsWindow(_hwnd);  };

	// check if window has been created including child windows
	BOOL IsFullyCreated() { return _flag.IsCreated();};

	// wait for window to be created including child windows
	void WaitCreated() { while(!IsFullyCreated()) Sleep(1); };

	kmImg Capture()
	{
		HDC     hdc = GetDC(GetHwnd());
		HDC     mdc = CreateCompatibleDC(hdc);
		HBITMAP hbm = CreateCompatibleBitmap(hdc, GetW(), GetH());

		// select bitmap for mdc
		HBITMAP hbm_o = (HBITMAP) SelectObject(mdc, hbm);

		// copy hdc to mdc
		BitBlt(mdc,0,0,GetW(),GetH(),hdc,0,0,SRCCOPY);

		// copy bitmap to img
		kmImg      img(GetW(), GetH());
		BITMAPINFO bi = img.GetBmpInfo();

		GetDIBits(mdc, hbm, 0, GetH(), img.Begin(), &bi, DIB_RGB_COLORS);

		// finish the work
		DeleteObject(SelectObject(mdc, hbm_o)); // delete hbm
		DeleteDC    (mdc);         // delete dc obtained by CreateXXX()
		ReleaseDC(GetHwnd(), hdc); // relase dc obtained by GetDC()

		return img;
	};

	void CaptureCopy()
	{
		// open clipboard
		if(!OpenClipboard(GetHwnd())) PRINTFA(" open fail\n");
		EmptyClipboard();

		// get img and info
		kmImg      img = Capture();
		BITMAPINFO bi  = img.GetBmpInfo();

		// copy to global heap memory
		// * Note that you must use GlobalAlloc() for a clip board.
		// * If not, it isn't recommended since It has greater overhead
		int64 size_dib = img.Byte() + sizeof(BITMAPINFO);

		HGLOBAL hdib = GlobalAlloc(GMEM_MOVEABLE, size_dib); if(hdib == NULL) return;
		char*   pdib = (char*)GlobalLock(hdib);

		if(pdib != NULL)
		{
			memcpy(pdib, &bi, sizeof(BITMAPINFO)); pdib += sizeof(BITMAPINFO);
			memcpy(pdib, img.Begin(), img.Byte());

			GlobalUnlock(hdib);

			// set clip board
			SetClipboardData(CF_DIB, hdib);
			CloseClipboard();
		}

		// free global heap memory
		GlobalFree(hdib);
	};

	void RelocateChild()
	{
		SendMsg(WM_SIZE, SIZE_RESTORED, MAKELONG(_win.w, _win.h));
	};

	// enable drag and drop
	void EnableDragDrop(BOOL accepted = TRUE) { DragAcceptFiles(_hwnd, accepted); };

	// thread lock functions
	kmLock* Lock  () { return _lck.Lock  (); };
	void    Unlock() {        _lck.Unlock(); };
	kmLock& Enter () { return _lck.Enter (); };
	void    Leave () {        _lck.Leave (); };
};

// class for managing topmost window
class kmWndMng
{
	NOTIFYICONDATA _nid{}; // tray icon data in taskbar's status area
public : 
	
	// add tray icon
	BOOL AddTrayIcon(HWND hwnd, LPCTSTR tip, HICON icon)
	{
		_nid.cbSize           = sizeof(_nid);
		_nid.hWnd             = hwnd;		
		_nid.uFlags           = NIF_TIP | NIF_ICON | NIF_MESSAGE;
		_nid.uID              = 1;         // wparam
		_nid.uCallbackMessage = WM_KM_MANAGE;
		_nid.hIcon            = icon;

		SetTip(tip);

		return Shell_NotifyIcon(NIM_ADD, &_nid);
	};

	// delete tray icon
	BOOL DelTrayIcon() { return Shell_NotifyIcon(NIM_DELETE, &_nid); };

	// set tip
	void SetTip(LPCTSTR tip) { wcscpy_s(_nid.szTip, numof(_nid.szTip), tip); };
};

/////////////////////////////////////////////////////////////////////
// class for child window

// class for (child) window position calculation
class kmC
{
public:
	float r = 0.f; // ratio (0 : start position, 1: end position)	
	int   d = 0;   // delta (0 < : forward from r-pos, 0 > : backward from r-pos)
	
	// constructor
	kmC() {};
	kmC(int   delta)              { d =      delta;};
	kmC(float ratio)              {                 r =        ratio;};
	kmC(float ratio, int   delta) { d =      delta; r =        ratio;};
	kmC(int   ratio, int   delta) { d =      delta; r = (float)ratio;}; // avoid warning only
	kmC(int   ratio, float delta) { d = (int)delta; r = (float)ratio;}; // avoid warning only

	int GetPos(const int parent_size) { return int(float(parent_size)*r) + d; };
};

// class for child window position scheme... since 2020.11.24
class kmCwp
{
public:
	kmC l, t, r, b; // left, top, right, bottom

	kmCwp() {};
	kmCwp(kmC left, kmC top, kmC right, kmC bottom) { l = left; t = top; r = right; b = bottom;};

	kmCwp(kmeCwpType type, int c1, int c2, int c3, int c4)
	{
		switch(type)
		{
		case CWP_WHLT : l = c3;            t = c4;            r = c1 + c3;      b = c2 + c4;      break;
		case CWP_WHLB : l = c3;            t = kmC(1,-c2-c4); r = c1 + c3;      b = kmC(1,-c4);   break;
		case CWP_WHRT : l = kmC(1,-c1-c3); t = c4;            r = kmC(1,-c3);   b = c2 + c4;      break;
		case CWP_WHRB : l = kmC(1,-c1-c3); t = kmC(1,-c2-c4); r = kmC(1,-c3);   b = kmC(1,-c4);   break;
		case CWP_HLRT : l = c2;            t = c4;            r = kmC(1,-c3);   b = c1 + c4;      break;
		case CWP_HLRB : l = c2;            t = kmC(1,-c1-c4); r = kmC(1,-c3);   b = kmC(1,-c4);   break;
		case CWP_WLTB : l = c2;            t = c3;            r = c1 + c2;      b = kmC(1,-c4);   break;
		case CWP_WRTB : l = kmC(1,-c1-c2); t = c3;            r = kmC(1,-c2);   b = kmC(1,-c4);   break;
		case CWP_LRTB : l = c1;            t = c3;            r = kmC(1,-c2);   b = kmC(1,-c4);   break;
		}
	};

	kmCwp(kmeCwpType type, float c1, float c2, float c3, float c4)
	{
		switch(type)
		{
		case CWP_RAT : l = kmC(c1,0); t = kmC(c2,0); r = kmC(c1+c3,0); b = kmC(c2+c4,0); break;
		}
	};

	// get window position
	//   pw : width of parent window
	//   ph : width of parent window
	void Get(int& x, int& y, int& w, int& h, int pw, int ph)
	{
		x = l.GetPos(pw);
		y = t.GetPos(ph);
		w = r.GetPos(pw) - x;
		h = b.GetPos(ph) - y;
	};
	
	// get position of left, top, right, bottom
	int GetL(int pw) { return l.GetPos(pw); };
	int GetT(int ph) { return t.GetPos(ph); };
	int GetR(int pw) { return r.GetPos(pw); };
	int GetB(int ph) { return b.GetPos(ph); };

	// get rect of window from pw, ph
	kmRect GetRect(int pw, int ph) { return kmRect(GetL(pw),GetT(ph),GetR(pw),GetB(ph)); };

	// get width, height 
	// * Note that if pw (or ph) is zero, this will give only offset between r and l (or b and t)
	int GetW(int pw = 0) { return r.GetPos(pw) - l.GetPos(pw); };
	int GetH(int ph = 0) { return b.GetPos(ph) - t.GetPos(ph); };
};

// child window base class
class kmwChild  : public kmWnd
{
protected:
	kmCwp _cwp;

	//////////////////////////////////////
	// fucntions to create window
public:
	// create window 
	//  pwnd   : parent window
	//  id     : id of child window
	void Create(int x, int y, int w, int h, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		Create(kmCwp(x, y, x + w, y + h), pwnd, id, style);
	};

	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		// init parameters
		_win.style  = WS_CHILD | WS_VISIBLE | WS_CLIPCHILDREN | style;
		_win.parent = pwnd->GetHwnd();
		_win.menu   = (HMENU) id;
		_cwp        = cwp;

		// get x, y, w, h
		int x, y, w, h; cwp.Get(x, y, w, h, pwnd->GetClientW(), pwnd->GetClientH());

		// create window
		kmWnd::Create(x, y, w, h);
	};

	//////////////////////////////////////
	//  member fucntions
public:
	// resize and reposition when the size of parent window has changed.
	//  pw, ph : the size of client rect of parent window
	void Relocate(int pw, int ph)
	{
		_cwp.Get(_win.x, _win.y, _win.w, _win.h, pw, ph);

		Move(_win.x, _win.y, _win.w, _win.h);
	};

	HWND   GetParentHwnd()     { return ::GetParent(_hwnd);        };
	kmWnd* GetParent()         { return   GetWnd(GetParentHwnd()); };
	HWND   PassFocusToParent() { return   GetParent()->SetFocus(); };
	
	inline LRESULT PassMsgToParent(UINT msg, WPARAM wp, LPARAM lp)
	{
		return SendMessage(GetParentHwnd(), msg, wp, lp);
	};
	inline LRESULT SendMsgToParent(UINT msg, WPARAM wp, LPARAM lp)
	{
		return SendMessage(GetParentHwnd(), msg, wp, lp);
	};
	inline LRESULT PostMsgToParent(UINT msg, WPARAM wp, LPARAM lp)
	{
		return PostMessage(GetParentHwnd(), msg, wp, lp);
	};

	ushort GetId()  const { return (ushort) _win.menu; };
	kmCwp& GetCwp()       { return _cwp; };

	void SetCwp(const kmCwp& cwp) { _cwp = cwp; };

	///////////////////////////////////////////////
	// window procedures

	// key procedure
	virtual kmePPType OnKeyDown  (WPARAM wp, LPARAM lp) { PassMsgToParent(WM_KEYDOWN  , wp, lp); return PP_FINISH; };	
	virtual kmePPType OnDropFiles(WPARAM wp, LPARAM lp) { PassMsgToParent(WM_DROPFILES, wp, lp); return PP_FINISH; };
};

class kmwGrp : public kmGrp<kmwChild>
{
public:
	void Hide() { All([](kmwChild* a){ a->Hide(); }); };
	void Show() { All([](kmwChild* a){ a->Show(); }); };

	void Relocate(int pw, int ph)
	{
		All([=](kmwChild* a){ a->Relocate(pw, ph); });
	};
};

// table view control
class kmwTable : public kmwChild
{
public:
	kmMat2<kmgStr> _itm;
	kmMat2i32      _event;
	kmFont         _fnt;
	kmRgb          _bk_rgb = kmRgb(220,230,250);

	// members for mouse event
	int64 _i1      = -1;
	int64 _i2      = -1;
	int   _tracked = 0;
	kmRgb _rgb0    = kmRgbW;

	//////////////////////////////////////
	// fucntions to create window

	// create window without updating itm
	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		kmwChild::Create(cwp, pwnd, id, style);
	};

	//////////////////////////////////////////////
	// member functions

	kmwTable& CreateItm(int64 n1, int64 n2)
	{
		_itm.Recreate(n1, n2); 

		for(int64 i = 0; i < _itm.N(); ++i)
		{
			_itm(i).SetFormat(DT_CENTER | DT_VCENTER | DT_SINGLELINE);
			_itm(i).SetFont(&_fnt);
		}

		_event.Recreate(n1, n2); _event.SetVal(-1);

		return *this;
	};

	kmgStr& operator()(int64 i1, int64 i2) const { return _itm(i1,i2); };

	int64 N1() { return _itm.N1(); };
	int64 N2() { return _itm.N2(); };
	
	kmwTable& SetFontH(int h) { _fnt.SetH(h); return *this; };
	kmwTable& SetBkRgbCol(int64 i2, kmRgb rgb)
	{	
		for(int64 i1 = 0; i1 < _itm.N1(); ++i1) _itm(i1,i2).SetBkRgb(rgb);
		return *this;
	};
	kmwTable& SetBkRgbRow(int64 i1, kmRgb rgb)
	{	
		for(int64 i2 = 0; i2 < _itm.N2(); ++i2) _itm(i1,i2).SetBkRgb(rgb);
		return *this;
	};
	kmwTable& SetTwoTone(kmRgb rgb0, kmRgb rgb1, int tone_dim = 0)
	{
		for(int64 i2 = 0; i2 < _itm.N2(); ++i2)
		for(int64 i1 = 0; i1 < _itm.N1(); ++i1)
		{
			const int64 i = (tone_dim == 0) ? i1:i2;
			
			if(i%2 == 0) _itm(i1,i2).SetBkRgb(rgb0);
			else         _itm(i1,i2).SetBkRgb(rgb1);
		}
		return *this;
	};

	kmwTable& SetEvent  (int64 i1, int64 i2, ushort evnt_idx) { _event(i1,i2) = evnt_idx; return *this;};
	kmwTable& ClearEvent(int64 i1, int64 i2)                  { _event(i1,i2) = -1;       return *this;};
	kmwTable& ClearEvent()                                    { _event.SetVal(-1);        return *this;};

	kmwTable& UpdateItm()
	{
		if(_itm.N() == 0) return *this;

		// init parameters
		const int n1 = (int)_itm.N1(), n2 = (int)_itm.N2();
		const int h  = GetH()/n1,      w  = GetW()/n2;

		// set items
		for(int i2 = 0; i2 < n2; ++i2)
		for(int i1 = 0; i1 < n1; ++i1)
		{	
			_itm(i1,i2).SetRect(kmRect(i2*w + 1, i1*h + 1, (i2+1)*w - 1, (i1+1)*h - 1));
		}
		Invalidate(); return *this;
	};

	// return : kmT2( i1, i2 )
	kmT2<int64,int64> GetIdxFromPixelPos(int x, int y)
	{
		int64 i1 = -1, i2 = -1;

		for(int64 j2 = 0; j2 < _itm.N2(); ++j2)
		{
			for(int64 j1 = 0; j1 < _itm.N1(); ++j1)
			{
				if(_itm(j1,j2).GetRect().IsIn(x, y)) { i1 = j1; i2 = j2; break; };
			} 
			if(i1 > -1) break;
		}

		return kmT2(i1, i2);
	};

	//////////////////////////////////////////////
	// window proceduer
protected:
	virtual LRESULT Proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
	{
		// message procedure
		switch (msg)
		{	
		case WM_MOUSEMOVE:
			{
				// change a color of itm in cursor pos
				int64 i1, i2; kmT2(i1,i2) = GetIdxFromPixelPos(LOWORD(lp), HIWORD(lp));

				if(i1 >= 0) if(_i1 != i1 || _i2 != i2)
				{
					// recover selected itm
					if(_i1 >=  0) _itm(_i1, _i2).SetBkRgb(_rgb0);
					
					// vary color of selecting itm
					_rgb0 = _itm(_i1 = i1, _i2 = i2).GetBkRgb();

					_itm(_i1, _i2).SetBkRgb(_rgb0.Vary(8,(_event(_i1,_i2) >= 0) ? 32:0,0));
					 
					Invalidate();
				}

				// set mouse track for WM_MOUSELEAVE
				if(_tracked == 0)
				{
					TRACKMOUSEEVENT tme = {sizeof(tme), TME_LEAVE, GetHwnd(), 1};
					if(TrackMouseEvent(&tme)) _tracked = 1;
				}
			}
			return 0;

		case WM_MOUSELEAVE:
			if(_i1 >= 0)
			{
				_itm(_i1,_i2).SetBkRgb(_rgb0);
				_i1 = -1; _i2 = -1;
				Invalidate();
			}			
			_tracked = 0;			
			return 0;

		case WM_MOUSEWHEEL:	SendWheelMsg(wp); return 0;
		}
		return kmWnd::Proc(hwnd, msg, wp, lp);
	};

	void SendWheelMsg(WPARAM wp)
	{
		if(_i1 >= 0) if(_event(_i1,_i2) >=0)
		{
			WPARAM wp0 = MAKELONG(GetId(), NTF_TBL_MOUSEWHEEL);
			WPARAM lp0 = MAKELONG(_event(_i1,_i2), HIWORD(wp));
			PostMsgToParent(WM_COMMAND, wp0, lp0);
		}
	};

	virtual kmePPType OnKeyDown(WPARAM wp, LPARAM lp)
	{
		switch(wp)
		{
		case VK_UP   : SendWheelMsg(MAKELONG(0, 120)); break;
		case VK_DOWN : SendWheelMsg(MAKELONG(0,-120)); break;
		default      : PassMsgToParent(WM_KEYDOWN, wp, lp);
		}
		return PP_FINISH;
	};

	// draw gobj
	virtual void DrawGobjs(HDC hdc, int w, int h)
	{	
		// draw background
		HBRUSH brush = CreateSolidBrush(_bk_rgb);
		
		FillRect(hdc, kmRect(0,0,w,h), brush);

		DeleteObject(brush);

		// draw strings
		kmLockGuard grd = Enter(); //====================enter-leave

		for(int64 i2 = 0; i2 < _itm.N2(); ++i2)
		for(int64 i1 = 0; i1 < _itm.N1(); ++i1)
		{
			kmgStr& itm = _itm(i1,i2);

			brush = CreateSolidBrush( itm.GetBkRgb());

			FillRect(hdc, itm.GetRect(), brush);

			DeleteObject(brush);

			itm.Draw(hdc);
		}
	};

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{			
			UpdateItm();
		}
		return PP_FINISH;
	};

	virtual kmePPType OnLButtonDown(WPARAM wp, LPARAM lp)
	{			
		// get index of the table
		int64 i1,i2; kmT2(i1,i2) = GetIdxFromPixelPos(LOWORD(lp), HIWORD(lp));

		// check the event and send a message
		if(i1 > -1)
		{
			const int event_idx = _event(i1,i2);

			if(event_idx >= 0)
			{
				// combine wp, lp
				WPARAM wp0 = MAKELONG(  GetId(), NTF_TBL_CLICKED);
				WPARAM lp0 = MAKELONG(event_idx, 0);
	
				PassMsgToParent(WM_COMMAND, wp0, lp0);
			}
		}
		// set focus
		// * Note that a key down message won't be delivered properly without this.
		SetFocus();
	
		return PP_FINISH;
	};
};

// button control
class kmwBtn : public kmwChild
{
public:
	uint   _state   = 0; // 0 : normal, 1: hover, 2: clicked, 3: inactivated
	uint   _tracked = 0; // 0 : not tracked, 1: tracked

	kmgStr _str;
	kmFont _fnt;	
	kmRgb  _bk0_rgb = kmRgbW;             // normal
	kmRgb  _bk1_rgb = kmRgb(210,230,250); // hovering
	kmRgb  _bk2_rgb = kmRgb(100,150,250); // clicked
	kmRgb  _bk3_rgb = kmRgb(180,180,180); // inactivated
		
	//////////////////////////////////////
	// fucntions to create window
public:
	void Create(int x, int y, int w, int h, LPCTSTR str, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{		
		Create(kmCwp(x, y, x + w, y + h), str, pwnd, id, style);
	};

	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		Create(cwp, nullptr, pwnd, id, style);
	};

	void Create(kmCwp cwp, LPCTSTR str, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		// init font, fmt and rect
		_str.SetFont(&_fnt);
		_str.SetFormat(DT_CENTER | DT_VCENTER | DT_SINGLELINE);
		_fnt.SetH((int) (cwp.GetH(pwnd->GetH())*0.9f));
	
		// init
		if(str != nullptr) _str._str = str;
		
		// create window
		kmwChild::Create(cwp, pwnd, id, style);

		// set rect
		_str.SetRect(GetClientRect());
	};
	
	//////////////////////////////////////////////
	// member functions

	kmwBtn& SetFont  (kmFont& fnt) { _fnt     = fnt;      return *this; };
	kmwBtn& SetRgb   (kmRgb&  rgb) { _str.SetRgb(rgb);    return *this; };
	kmwBtn& SetBkRgb (kmRgb&  rgb) { _str.SetBkRgb(rgb);  return *this; }; // bk of string
	kmwBtn& SetBk0Rgb(kmRgb&  rgb) { _bk0_rgb = rgb;      return *this; };
	kmwBtn& SetBk1Rgb(kmRgb&  rgb) { _bk1_rgb = rgb;      return *this; };
	kmwBtn& SetBk2Rgb(kmRgb&  rgb) { _bk2_rgb = rgb;      return *this; };
	kmwBtn& SetFormat(uint    fmt) { _str.SetFormat(fmt); return *this; };
	kmwBtn& SetStr   (kmStrw& str) { _str.GetStr() = str; return *this; };

	kmwBtn& SetBkRgbTheme(const kmRgb& rgb)
	{
		_bk0_rgb = rgb;
		_bk1_rgb = rgb - kmRgb(20,20,20);
		_bk2_rgb = rgb - kmRgb(40,40,40);

		return *this;
	};

	kmwBtn& SetStr(const wchar_t* str, ...)
	{
		kmStrw& buf = _str._str;
		va_list args;
		va_start(args, str);
		{
			const int len = _vscwprintf(str, args) + 1;

			if(len > 0) { buf.Recreate(len); vswprintf_s(buf.P(), len, str, args); }
		}
		va_end(args);

		return *this;
	};

	kmwBtn& Activate  () { _state = 0; return *this; };
	kmwBtn& Inactivate() { _state = 3; return *this; };

	bool IsActivated() { return _state != 0; };
		
	// get str, font
	kmStrw& GetStr()  { return _str.GetStr(); } ;
	kmFont& GetFont() { return _fnt;};	

	//////////////////////////////////////////////
	// window proceduer
protected:
	virtual void DrawGobjs(HDC hdc, int w, int h)
	{	
		// draw background
		HBRUSH brush = 0;
		
		if     (_state == 1) brush = CreateSolidBrush(_bk1_rgb);
		else if(_state == 2) brush = CreateSolidBrush(_bk2_rgb);
		else if(_state == 3) brush = CreateSolidBrush(_bk3_rgb);
		else                 brush = CreateSolidBrush(_bk0_rgb);
				
		FillRect(hdc, kmRect(0,0,w,h), brush);

		DeleteObject(brush);

		// draw strings		
		if(_state == 2) _str.GetRect().ShiftY(1);
		
		_str.Draw(hdc);
		
		if(_state == 2) _str.GetRect().ShiftY(-1);
	};

	virtual LRESULT Proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
	{
		// inactivated
		if(_state == 3) return kmWnd::Proc(hwnd, msg, wp, lp);

		// message procedure
		switch (msg)
		{	
		case WM_LBUTTONDOWN:
			_state = 2; 
			SetCapture(GetHwnd());
			SetFocus();
			Invalidate();
			UpdateWindow();
			return 0;

		case WM_LBUTTONUP:
			_state = (_tracked == 0)? 0:1;
			ReleaseCapture();
			Invalidate();
			UpdateWindow();
			NotifyClicked();
			return 0;

		case WM_MOUSEMOVE:
			if(_state == 0) { _state = 1; Invalidate(); }
			if(_tracked == 0)
			{
				TRACKMOUSEEVENT tme = {sizeof(tme), TME_LEAVE, GetHwnd(), 1};
				if(TrackMouseEvent(&tme)) _tracked = 1;
			}
			return 0;

		case WM_MOUSELEAVE:
			_state   = 0;
			_tracked = 0;
			Invalidate();
			return 0;

		case WM_MOUSEWHEEL:
			NotifyMouseWheel(HIWORD(wp));
			return 0;
		}
		return kmWnd::Proc(hwnd, msg, wp, lp);
	};
	 
	////////////////////////////////////////////
	// notification functions

	void NotifyMouseWheel(short wh_stp)
	{		
		WPARAM wp0 = MAKELONG(GetId(), NTF_BTN_MOUSEWHEEL);
		WPARAM lp0 = MAKELONG(0, wh_stp);

		PostMessage(GetParentHwnd(), WM_COMMAND, wp0, lp0);
	};

	void NotifyClicked()
	{	
		WPARAM wp0 = MAKELONG(GetId(), NTF_BTN_CLICKED);

		PostMessage(GetParentHwnd(), WM_COMMAND, wp0, 0);
	};
};

// edit control (based on win-api)
class kmwEdit : public kmwChild
{	
	//////////////////////////////////////
	// fucntions to create window
public:
	void Create(int x, int y, int w, int h, LPCTSTR win_name, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		Create(kmCwp(x, y, x + w, y + h), win_name, pwnd, id, style);
	};

	void Create(kmCwp cwp, LPCTSTR win_name, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		// init parameters
		_win.class_name = L"edit";
		_win.win_name   = win_name;

		if(style == 0) style = WS_BORDER | ES_AUTOHSCROLL | ES_LEFT;

		// create window
		kmwChild::Create(cwp, pwnd, id, style);
	};

	// procedure for notification
	int ProcNtf(ushort ntf)
	{	
		switch(ntf)
		{
		case EN_CHANGE: return 1;
		case EN_UPDATE: break;
		}		
		return 0; // 0: skip parent's proc, 1: do parent's proc
	};

	//////////////////////////////////////
	// member functions

	LRESULT LimitText(uint num_txt)
	{
		return SendMsg(EM_LIMITTEXT, (WPARAM) num_txt, 0);
	};

	int GetNum()
	{	
		return GetWinStr(128).ToInt();
	};

	void SetStr(const wchar_t* str, ...)
	{
		kmStrw buf;
		va_list args;
		va_start(args, str);
		{
			const int len = _vscwprintf(str, args) + 1;

			if(len > 0) { buf.Recreate(len); vswprintf_s(buf.P(), len, str, args); }
		}
		va_end(args);

		SetWinTxt(buf.P());
	};

	kmStrw GetStr()
	{
		kmStrw str(1024);
		GetWinTxt(str.Begin(), (int)str.Size());
		return str;
	};
};

// static control
class kmwStt : public kmwChild
{
public:	
	kmgStr _str;
	kmFont _fnt;

	//////////////////////////////////////
	// fucntions to create window
public:
	void Create(int x, int y, int w, int h, LPCTSTR str, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		Create(kmCwp(x, y, x + w, y + h), pwnd, id, style);
	};

	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		Create(cwp, nullptr, pwnd, id, style);
	};

	void Create(kmCwp cwp, LPCTSTR str, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{	
		// init font and fmt
		_str.SetFont(&_fnt);
		_str.SetFormat(DT_LEFT | DT_VCENTER | DT_SINGLELINE);

		// init
		if(str != nullptr) _str._str = str;

		// create window
		kmwChild::Create(cwp, pwnd, id, style);

		// set rect
		_str.SetRect(GetClientRect());
	};

	//////////////////////////////////////////////
	// member functions

	// set str, font and colors
	kmwStt& SetFont  (kmFont  fnt) { _fnt = fnt;          return *this; };
	kmwStt& SetH     (int       h) { _fnt.SetH(h);        return *this; };
	kmwStt& SetRgb   (kmRgb   rgb) { _str.SetRgb(rgb);    return *this; };
	kmwStt& SetBkRgb (kmRgb   rgb) { _str.SetBkRgb(rgb);  return *this; };
	kmwStt& SetFormat(uint    fmt) { _str.SetFormat(fmt); return *this; };
	kmwStt& SetStr   (kmStrw& str) { _str.GetStr() = str; return *this; };

	kmwStt& SetStr(const wchar_t* str, ...)
	{
		kmStrw& buf = _str._str;
		va_list args;
		va_start(args, str);
		{
			const int len = _vscwprintf(str, args) + 1;

			if(len > 0) { buf.Recreate(len); vswprintf_s(buf.P(), len, str, args); }
		}
		va_end(args);

		return *this;
	};

	// get str, font
	kmStrw& GetStr()  { return _str.GetStr(); } ;
	kmFont& GetFont() { return _fnt;};
	
	//////////////////////////////////////////////
	// window proceduer
protected:
	virtual void DrawGobjs(HDC hdc, int w, int h)
	{	
		// draw background
		HBRUSH brush = CreateSolidBrush(_str.GetBkRgb());
						
		FillRect(hdc, kmRect(0,0,w,h), brush);

		DeleteObject(brush);

		// draw string
		_str.Draw(hdc);
	};
};

// multiline string box control
class kmwBox : public kmwChild
{
public:
	kmStrws _strs;
	kmFont  _fnt;
	kmRgb   _rgb    = kmRgbK;
	kmRgb   _bk_rgb = kmRgbW;	
	uint    _fmt    = DT_LEFT | DT_TOP;   // DT_LEFT, DT_RIGHT, DT_CENTER...
	int     _fb     = 0;                  // distance from bottom of object to bottom of view (pixel)
	int     _ho     = 0;                  // height of object (pixel)
	int     _ls     = 4;                  // line spacing (pixel)

	kmgBar  _bar;
		
	//////////////////////////////////////
	// fucntions to create window
	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		// create window
		kmwChild::Create(cwp, pwnd, id, style);
	};
public:
	//////////////////////////////////////////////
	// member functions

	// set str, font and colors
	kmwBox& SetH        (int        h) { _fnt.SetH(h);   return *this; };
	kmwBox& SetFormat   (uint     fmt) { _fmt    = fmt;  return *this; };
	kmwBox& SetFont     (kmFont   fnt) { _fnt    = fnt;  return *this; };	
	kmwBox& SetRgb      (kmRgb    rgb) { _rgb    = rgb;  return *this; };
	kmwBox& SetLineSpace(int ls_pixel) { _ls = ls_pixel; return *this; };
	kmwBox& SetBkRgb    (kmRgb    rgb)
	{
		_bk_rgb = rgb; _bar.SetRgb(rgb.Vary(30,30,30), rgb.Vary(10,10,10));
		return *this; 
	};
	
	// add strings
	kmwBox& Add(const kmStrw& str)
	{
		if(_strs.N() == 0) _strs.Recreate(0,16);
		_strs.PushBack(str);
		_ho = (_fnt.GetH() + _ls)*(int)_strs.N();
		return *this;
	};

	// udpate strings
	//    idx : string
	kmwBox& UpdateStr(const kmStrw& str, int idx = -1)
	{
		if(_strs.N() == 0) Add(str);
		else _strs(idx) = str;
		return *this;
	};

	// set view position
	//   fb : distance from bottom of object to bottom of view (pixel)
	kmwBox& SetViewPos(int fb)
	{
		const int hv = GetH();

		if(_ho < hv) _fb = 0;
		else         _fb = MIN(MAX(0, fb), _ho - hv); 
		
		return *this;
	};
	kmwBox& SetViewPosTop   ()        { return SetViewPos(_ho - GetH());    };
	kmwBox& SetViewPosBottom()        { return SetViewPos(0);               };
	kmwBox& ScrollUp  (int del_pixel) { return SetViewPos(_fb + del_pixel); };
	kmwBox& ScrollDown(int del_pixel) { return SetViewPos(_fb - del_pixel); };

	// get 
	kmFont& GetFont()       { return _fnt; };
	int     GetLineN()      { return (int)_strs.N(); };
	int     GetCurLineIdx() { return (int)_strs.N() - 1; };

	// get view position
	//  return value : distance from bottom of object to bottom of view (pixel)
	int GetViewPos() { return _fb; };
	
	//////////////////////////////////////////////
	// window proceduer
protected:
	virtual void DrawGobjs(HDC hdc, int w, int h)
	{	
		// draw background
		HBRUSH brush = CreateSolidBrush(_bk_rgb);
						
		FillRect(hdc, kmRect(0,0,w,h), brush);

		DeleteObject(brush);

		// draw strings
		if(_strs.N() > 0)
		{
			// select font
			HFONT font     = CreateFontIndirect(_fnt);
			HFONT font_old = (HFONT) SelectObject(hdc, font);

			// select color
			SetTextColor(hdc, _rgb);
			::SetBkMode (hdc, TRANSPARENT);

			// draw strings
			const int n  = (int) _strs.N();
			const int dh = _fnt.GetH() + _ls; _ho = dh*n;

			int t0 = MIN(h + _fb - _ho, 0);

			for(int i = 0; i < n; ++i, t0 += dh)
			{
				// set rect
				kmRect rt(10, t0, w - 10, t0 + dh);

				// draw string
				if( (0 < rt.b && rt.b < h) || (0 < rt.t && rt.t < h)) 
				{
					DrawTextEx(hdc, _strs(i), -1, rt, _fmt, NULL);
				}
			}

			// release font
			SelectObject(hdc, font_old);
			DeleteObject(font);

			// draw bar
			if(h < _ho) _bar.SetRat(_ho, h, _fb).Draw(hdc, w, h);
		}
	};

	virtual kmePPType OnMouseWheel (WPARAM wp, LPARAM lp)
	{
		const short wh_stp = HIWORD(wp); // wh_stp is 120 per a tick		
		const int   n = (int) _strs.N(), hv = GetH();

		if(hv < _ho)
		{
			_fb += (wh_stp>>2);
			_fb = MIN(MAX(0, _fb), _ho - hv);
			Invalidate();
			UpdateWindow();
		}
		else _fb = 0;

		return PP_FINISH; 
	};
};

// edit control
class kmwEdt : public kmwChild
{
public:	
	kmgStr  _str;
	kmFont  _fnt;
	kmgLine _line; // under line

	int    _caret_idx = 0;      // caret position
	int    _sel_idx   = -1;     // selected start position (if < 0, no selected)
	int    _ime_state = 0;      // ime state (0: not, 1: start, 2: ing)	
	int    _v_type    = 0;      // input type (0: text, 1: integer, 2: float)
	float  _v_min     = -1e10f;
	float  _v_max     =  1e10f;
		
	//////////////////////////////////////
	// fucntions to create window
public:
	void Create(int x, int y, int w, int h, LPCTSTR str, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{		
		Create(kmCwp(x, y, x + w, y + h), pwnd, id, style);
	};

	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		Create(cwp, nullptr, pwnd, id, style);
	};

	void Create(kmCwp cwp, LPCTSTR str, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		// init font and fmt
		_str.SetFont(&_fnt);
		_str.SetFormat(DT_LEFT | DT_VCENTER | DT_SINGLELINE);

		// init
		if(str != nullptr) _str._str = str;

		// create window
		kmwChild::Create(cwp, pwnd, id, style);

		// get rect
		kmRect rt = GetClientRect();

		// set rect
		_str.SetRect(kmRect(rt.l, rt.t, rt.r, rt.b-1));

		// init font size
		_fnt.SetH(rt.b - 2);

		// init caret_idx
		_caret_idx = (int)_str.GetLen() - 1;

		// init line
		_line.Set(kmRect(rt.l+1, rt.b-1, rt.r, rt.b-1), kmRgb(30,60,240), 1);
	};

	//////////////////////////////////////////////
	// member functions

	// set str, font and colors	
	kmwEdt& SetFont  (kmFont  fnt) { _fnt = fnt;          return *this; };
	kmwEdt& SetH     (int       h) { _fnt.SetH(h);        return *this; };
	kmwEdt& SetRgb   (kmRgb   rgb) { _str.SetRgb(rgb);    return *this; };
	kmwEdt& SetBkRgb (kmRgb   rgb) { _str.SetBkRgb(rgb);  return *this; };
	kmwEdt& SetFormat(uint    fmt) { _str.SetFormat(fmt); return *this; };
	kmwEdt& SetStr   (kmStrw& str) { _str.GetStr() = str; return *this; };

	kmwEdt& SetStr(const wchar_t* str, ...)
	{
		kmStrw& buf = _str._str;
		va_list args;
		va_start(args, str);
		{
			const int len = _vscwprintf(str, args) + 1;

			if(len > 0) { buf.Recreate(len); vswprintf_s(buf.P(), len, str, args); }
		}
		va_end(args);

		return *this;
	};

	// input type (0: text, 1: integer, 2: float)
	kmwEdt& SetType (int type)                 { _v_type = MIN(MAX(0, type), 2); return *this; };
	kmwEdt& SetRange(float v_min, float v_max) { _v_min = v_min; _v_max = v_max; return *this; };

	// get str, font
	kmStrw& GetStr()  { return _str.GetStr(); } ;
	kmFont& GetFont() { return _fnt;};

	float GetNum()
	{
		if     (_v_type == 1) return MIN(MAX(_v_min, _str.GetStr().ToInt()  ), _v_max);
		else if(_v_type == 2) return MIN(MAX(_v_min, _str.GetStr().ToFloat()), _v_max);
		return 0;
	};

	kmwEdt& Clear() { _str._str.SetStr(L""); MoveCaretFirst(); return *this;};

	//////////////////////////////////////////////
	// window proceduer
protected:
	virtual LRESULT Proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
	{
		// message procedure
		switch (msg)
		{
		case WM_IME_STARTCOMPOSITION : _ime_state = 1; break;
		case WM_IME_COMPOSITION : 
			if(lp & GCS_COMPSTR)
			{
				HIMC  himc = ImmGetContext(_hwnd);
				DWORD len  = ImmGetCompositionString(himc, GCS_COMPSTR, NULL, 0);
				wchar buf[8];
				if(len > 0)
				{
					ImmGetCompositionString(himc, GCS_COMPSTR, buf, len);
					if(_ime_state == 1)
					{
						if(InsertChar(buf[0])) _ime_state = 2;
						else ImmNotifyIME(himc, NI_COMPOSITIONSTR, CPS_CANCEL, 0);
					}
					else
					{
						ReplaceChar(buf[0]); 
					}
					Invalidate(); UpdateWindow();
				}
				else if(wp == 27) // backspace
				{
					DeleteBack(); Invalidate(); UpdateWindow();
				}
				ImmReleaseContext(_hwnd, himc);
			}
			else if(lp & GCS_RESULTSTR)
			{
				HIMC  himc = ImmGetContext(_hwnd);
				DWORD len  = ImmGetCompositionString(himc, GCS_RESULTSTR, NULL, 0);
				wchar buf[8];
				if(len > 0)
				{
					ImmGetCompositionString(himc, GCS_RESULTSTR, buf, len);
					ReplaceChar(buf[0]); _ime_state = 1;
					Invalidate(); UpdateWindow();
				}
				ImmReleaseContext(_hwnd, himc);
			}
			return 0;

		case WM_IME_ENDCOMPOSITION : break;
		}
		return kmWnd::Proc(hwnd, msg, wp, lp);
	};

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{	
			_line.GetRect().r = w;
			_str .GetRect().r = w;
		}
		return PP_FINISH;
	};

	virtual kmePPType OnKeyDown(WPARAM wp, LPARAM lp)
	{
		if(GetKeyState(VK_CONTROL) < 0) switch(wp) // with ctrl
		{
		case 'C' : CopyClipboard();  Invalidate(); UpdateWindow(); break;
		case 'V' : PasteClipboard(); Invalidate(); UpdateWindow(); break;
		case 'X' : CutClipboard();   Invalidate(); UpdateWindow(); break;
		}
		else if(GetKeyState(VK_SHIFT) < 0) switch(wp) // with shift
		{
		case VK_LEFT  : MoveCaretShiftLeft (); Invalidate(); UpdateWindow(); break;
		case VK_RIGHT : MoveCaretShiftRight(); Invalidate(); UpdateWindow(); break;
		case VK_HOME  : MoveCaretShiftFirst(); Invalidate(); UpdateWindow(); break;
		case VK_END   : MoveCaretShiftEnd  (); Invalidate(); UpdateWindow(); break;
		}
		else switch(wp) // single input
		{
		case VK_LEFT   : _sel_idx = -1; MoveCaretLeft (); Invalidate(); UpdateWindow(); break;
		case VK_RIGHT  : _sel_idx = -1; MoveCaretRight(); Invalidate(); UpdateWindow(); break;		
		case VK_HOME   : _sel_idx = -1; MoveCaretFirst(); Invalidate(); UpdateWindow(); break;
		case VK_END    : _sel_idx = -1; MoveCaretEnd();   Invalidate(); UpdateWindow(); break;
		case VK_ESCAPE : _sel_idx = -1;                   Invalidate(); UpdateWindow(); break; 
		case VK_DELETE :                DeleteChar();     Invalidate(); UpdateWindow(); break;
		default : 
			if(wp < '0' || 'Z' < wp) PassMsgToParent(WM_KEYDOWN, wp, lp);
		}
		return PP_FINISH;
	};

	virtual kmePPType OnChar(WPARAM wp, LPARAM lp)
	{
		// check ctrl-key
		// * Note that the return value of GetKeyState() is the followings
		// *  0x0000 : not pressed and not pressed before
		// *  0x0001 : not pressed and pressed before
		// *  0x8000 : pressed and not pressed before
		// *  0x8001 : pressed and pressed before
		if(GetKeyState(VK_CONTROL) < 0) return PP_FINISH;
		switch(wp)
		{
		case 0x08 : DeleteBack(); Invalidate(); UpdateWindow(); break; // backspace
		case 0x0A : break; // linefeed
		case 0x1B : break; // escape
		case 0x09 : break; // tap
		case 0x0D : NotifyEnter(); break; // carriage enter
	
		// dispalyable characters
		default : 
			if(_v_type == 1 && _v_min >= 0) // only for number
			{
				if(wp < 48 || 57 < wp) break;
			}
			else if(_v_type == 1) // only for number and '-'
			{
				if(wp != 45 && (wp < 48 || 57 < wp)) break;
			}
			else if(_v_type == 2) // only for number and '-' '.'
			{
				if(wp != 45 && wp != 46 && (wp < 48 || 57 < wp)) break;
			}
			InsertChar((wchar)wp); 
			Invalidate(); 
			UpdateWindow(); 
			break;
		}		
		return PP_FINISH;
	};

	virtual void DrawGobjs(HDC hdc, int w, int h)
	{	
		// draw background
		HBRUSH brush = CreateSolidBrush(_str.GetBkRgb());

		FillRect(hdc, kmRect(0,0,w,h), brush);

		DeleteObject(brush);

		// draw string
		_str.Draw(hdc);

		// draw caret
		if(GetFocus() == _hwnd)
		{	
			SetCaretPos(MIN(_str.GetStrPosRight(hdc, _caret_idx), _str._rt.r - 1), 0);
		}

		// invert selected string
		if(_sel_idx >= 0)
		{
			// get caret position
			int crt_pos = MIN(_str.GetStrPosRight(hdc, _caret_idx), _str._rt.r - 1);
			int sel_pos = MIN(_str.GetStrPosRight(hdc, _sel_idx)  , _str._rt.r - 1);

			kmRect sel_rt(sel_pos, 0, crt_pos, h-2);

			InvertRect(hdc, sel_rt);
		}

		// draw line
		_line.Draw(hdc);
	};

	virtual kmePPType OnLButtonDown(WPARAM wp, LPARAM lp)
	{
		// get clicked pos
		int x = LOWORD(lp), y = HIWORD(lp);

		// get caret_idx
		int nstr      = (int)_str.GetLen() - 1;
		int caret_idx = -1;
		HDC hdc      = GetDC(_hwnd);
		
		for(int i = 1, x0 = 0; i < nstr; ++i)
		{	
			int x1 = _str.GetStrPosRight(hdc, i);

			if(x0 <= x && x < x1)
			{
				caret_idx = (x - x0 < x1 - x) ? i-1 : i;
				break;
			}
			x0 = x1;
		}
		ReleaseDC(_hwnd, hdc);

		// update caret_idx
		_caret_idx = (caret_idx == -1) ? nstr : caret_idx;

		// set focus
		if(GetFocus() == _hwnd) { Invalidate(); UpdateWindow(); }
		else                      SetFocus();

		return PP_DEFPROC;
	};

	virtual kmePPType OnSetFocus(WPARAM wp, LPARAM lp)
	{	
		_line.SetWidth(2);
		CreateCaret(_hwnd, NULL, 1, _fnt.GetH() - 2);
		Invalidate();
		UpdateWindow();
		ShowCaret(_hwnd);
		return PP_FINISH; 
	};

	virtual kmePPType OnKillFocus(WPARAM wp, LPARAM lp)
	{
		CheckRange();
		_line.SetWidth(1);
		DestroyCaret();
		NotifyKillFocus();
		Invalidate();
		UpdateWindow();
		return PP_FINISH;
	};

	/////////////////////////////////////////////////
	// inner member functions
	
	void MoveCaretFirst() { _caret_idx = 0; };
	void MoveCaretEnd()   { _caret_idx = (int)_str.GetLen() - 1; };
	void MoveCaretRight() { if(_caret_idx < _str.GetLen()-1) ++_caret_idx; };
	void MoveCaretLeft()  { if(_caret_idx > 0) --_caret_idx; };
	
	void MoveCaretShiftLeft ()
	{
		if(_sel_idx < 0) _sel_idx = _caret_idx; MoveCaretLeft();
	};
	void MoveCaretShiftRight()
	{
		if(_sel_idx < 0) _sel_idx = _caret_idx; MoveCaretRight();
	};
	void MoveCaretShiftFirst()
	{
		if(_sel_idx < 0) _sel_idx = _caret_idx; MoveCaretFirst();
	};
	void MoveCaretShiftEnd()
	{
		if(_sel_idx < 0) _sel_idx = _caret_idx; MoveCaretEnd();
	};
	void DeleteBack()
	{
		if(_sel_idx < 0) 		
		{
			if(_caret_idx > 0) _str.Erase(--_caret_idx); 
		}
		else DeleteChar();
	};
	void DeleteChar()
	{
		if(_sel_idx < 0)
		{
			if(_caret_idx < _str.GetLen()-1) _str.Erase(_caret_idx); 
		}
		else
		{
			const int n   = abs(_caret_idx - _sel_idx);
			const int idx = MIN(_caret_idx,  _sel_idx);

			for(int i = n; i--;) _str.Erase(idx);

			_caret_idx = idx; _sel_idx = -1;
		}
	};

	void ReplaceChar(wchar wch) { _str.Replace(_caret_idx-1, wch); };
	bool InsertChar (wchar wch)
	{
		// insert char
		_str.Insert(_caret_idx++ , wch); 

		// check if string is out of the rect
		kmRect rt  = GetClientRect();
		HDC    hdc = GetDC(_hwnd);
		int    strw = _str.GetStrWidth(hdc);
		ReleaseDC(_hwnd, hdc);

		if(rt.r <= strw) { DeleteBack(); return false; }

		return true;
	};

	void CheckRange()
	{
		if(_v_type == 1)
		{
			const int v = _str.GetStr().ToInt();

			if     (v < _v_min) _str.GetStr().SetStr(L"%d",(int)_v_min);
			else if(v > _v_max) _str.GetStr().SetStr(L"%d",(int)_v_max);
		}
		else if(_v_type == 2)
		{
			const float v = _str.GetStr().ToFloat();

			if     (v < _v_min) _str.GetStr().SetStr(L"%.2f", _v_min);
			else if(v > _v_max) _str.GetStr().SetStr(L"%.2f", _v_max);
		}
	};

	// notify WM_KILLFOCUS to parent
	void NotifyKillFocus()
	{	
		WPARAM wp0 = MAKELONG(GetId(), NTF_EDT_KILLFOCUS);

		PostMessage(GetParentHwnd(), WM_COMMAND, wp0, 0);
	};

	// notify WM_CHAR with 'enter' to parent
	void NotifyEnter()
	{	
		WPARAM wp0 = MAKELONG(GetId(), NTF_EDT_ENTER);

		PostMessage(GetParentHwnd(), WM_COMMAND, wp0, 0);
	};

	// copy text to clip board
	void CopyClipboard()
	{
		if(_sel_idx < 0) return;

		if(!OpenClipboard(_hwnd)) return; //// open clip board

		const int n   = abs(_caret_idx - _sel_idx) + 1; // including null-terminated
		const int idx = MIN(_caret_idx,  _sel_idx);

		HANDLE clb = GlobalAlloc(GMEM_DDESHARE | GMEM_MOVEABLE, n*2);

		if(clb == 0) { return ; };

		wchar* data = (wchar*)GlobalLock(clb); //// lock

		if(data != NULL)
		{
			memcpy(data, _str.GetStr().P(idx), n*2);
			data[n-1] = L'\0';

			GlobalUnlock(clb); //// unlock

			EmptyClipboard();
			SetClipboardData(CF_UNICODETEXT, clb);
		}
		CloseClipboard(); //// close clip board
	};

	// cut text and copy to clip board
	void CutClipboard()
	{
		if(_sel_idx < 0) return;

		CopyClipboard(); 
		DeleteChar();
	};

	// paste text from clipboard
	void PasteClipboard()
	{
		if(_sel_idx >= 0) DeleteChar();

		if(!OpenClipboard(_hwnd)) return; //// open clip board
				
		if(IsClipboardFormatAvailable(CF_UNICODETEXT))
		{
			HANDLE clb = GetClipboardData(CF_UNICODETEXT);

			wchar* data = (wchar*)GlobalLock(clb); //// lock

			if(data != NULL)
			{
				for(int i = 0; data[i] != L'\0' && data[i] != L'\n'; ++i)
				{
					if(data[i] == L'\t') InsertChar(L' ');
					else                 InsertChar(data[i]);
				}
				GlobalUnlock(clb); //// unlock
			}
		}		
		CloseClipboard(); //// close clip board
	};
};

// class for edit table
class kmwEdtTbl :  public kmwChild
{
public:
	int   _n1  = 1;       // num of row
	int   _n2  = 1;	      // num of col
	float _rat = 0.5f;    // ratio of stt width and edt width

	kmMat2<kmwStt> _stt; // (_n1, _n2)
	kmMat2<kmwEdt> _edt; // (_n1, _n2)

	// constructor
	kmwEdtTbl() {};

	//////////////////////////////////////
	// fucntions to create window
public:
	// create window without updating itm
	void Create(kmCwp cwp, int n1, int n2, float rat, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		_n1 = n1; _n2 = n2; _rat = rat;

		kmwChild::Create(cwp, pwnd, id, style);
	};

	/////////////////////////////////////////////
	// function to create child 
protected:
	virtual void CreateChild()
	{
		// calc width and height ratio
		const float h     = 1.f/ _n1;
		const float w     = 1.f/ _n2;
		const float w_stt = w*       _rat;
		const float w_edt = w*(1.f - _rat);
	
		_edt.Recreate(_n1, _n2);
		_stt.Recreate(_n1, _n2);
	
		for(int i2 = 0; i2 < _n2; ++i2)
		for(int i1 = 0; i1 < _n1; ++i1)
		{	
			float t = h*i1, b = t + h, ls = w*i2, le = ls + w_stt;
			int   idx = i1 + _n1*i2;
	
			// cwp : ltrb
			_stt(i1,i2).Create(kmCwp(kmC(ls,4),kmC(t,5),kmC(ls + w_stt,-4),kmC(b,-3)), this);
			_edt(i1,i2).Create(kmCwp(kmC(le,4),kmC(t,6),kmC(le + w_edt,-4),kmC(b,-3)), this, idx);
		}
	};

	void InvalidateChild() 
	{
		for(int i2 = 0; i2 < _n2; ++i2)
		for(int i1 = 0; i1 < _n1; ++i1)
		{
			_stt(i1,i2).Invalidate();
			_edt(i1,i2).Invalidate();
		}
	};

	///////////////////////////////////////////////
	// window procedures

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{			
			for(int i2 = 0; i2 < _n2; ++i2)
			for(int i1 = 0; i1 < _n1; ++i1)
			{
				_stt(i1,i2).Relocate(w, h);
				_edt(i1,i2).Relocate(w, h);
			}
		}
		return PP_FINISH;
	};

	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control
		ushort ntf = HIWORD(wp), id = LOWORD(wp);

		if(ntf == NTF_EDT_KILLFOCUS) NotifyKillFocus(id);

		//switch(id) { case 0: break; }
		return PP_FINISH;
	};

	void NotifyKillFocus(ushort child_id)
	{	
		WPARAM wp0 = MAKELONG(GetId(), NTF_EDT_KILLFOCUS);
		WPARAM lp0 = MAKELONG(child_id, 0);

		PostMessage(GetParentHwnd(), WM_COMMAND, wp0, lp0);
	};

	//////////////////////////////////////////////
	// member functions
public:
	// set items... type (0: text, 1: integer, 2: float)
	void Set(int i1, int i2, LPCTSTR stt_str, LPCTSTR edt_str, int type, float v_min = 0, float v_max = 10.f)
	{
		_stt(i1,i2).SetStr(stt_str);		
		_edt(i1,i2).SetStr(edt_str).SetType(type).SetRange(v_min, v_max)
		           .SetFormat(DT_RIGHT | DT_VCENTER | DT_SINGLELINE);

		_stt(i1,i2).Show();
		_edt(i1,i2).Show();
	};

	// set functions
	kmwEdtTbl& SetH        (int      h) { for(int i = _n1*_n2; i--;) _edt(i).SetH     (  h); return *this; };
	kmwEdtTbl& SetFont     (kmFont fnt) { for(int i = _n1*_n2; i--;) _edt(i).SetFont  (fnt); return *this; };
	kmwEdtTbl& SetRgb      (kmRgb  rgb) { for(int i = _n1*_n2; i--;) _edt(i).SetRgb   (rgb); return *this; };
	kmwEdtTbl& SetBkRgb    (kmRgb  rgb) { for(int i = _n1*_n2; i--;) _edt(i).SetBkRgb (rgb); return *this; };
	kmwEdtTbl& SetFormat   (uint   fmt) { for(int i = _n1*_n2; i--;) _edt(i).SetFormat(fmt); return *this; };

	kmwEdtTbl& SetSttH     (int      h) { for(int i = _n1*_n2; i--;) _stt(i).SetH     (  h); return *this; };
	kmwEdtTbl& SetSttFont  (kmFont fnt) { for(int i = _n1*_n2; i--;) _stt(i).SetFont  (fnt); return *this; };
	kmwEdtTbl& SetSttRgb   (kmRgb  rgb) { for(int i = _n1*_n2; i--;) _stt(i).SetRgb   (rgb); return *this; };
	kmwEdtTbl& SetSttBkRgb (kmRgb  rgb) { for(int i = _n1*_n2; i--;) _stt(i).SetBkRgb (rgb); return *this; };
	kmwEdtTbl& SetSttFormat(uint   fmt) { for(int i = _n1*_n2; i--;) _stt(i).SetFormat(fmt); return *this; };

	// get functions
	kmwEdt& operator()(int i1, int i2) { return _edt(i1,i2); };
	kmwEdt& Edt       (int i1, int i2) { return _edt(i1,i2); };
	kmwStt& Stt       (int i1, int i2) { return _stt(i1,i2); };

	kmwEdt& operator()(int idx)        { return _edt(idx); };
	kmwEdt& Edt       (int idx)        { return _edt(idx); };
	kmwStt& Stt       (int idx)        { return _stt(idx); };
};

// class for menu item
class kmwMitem
{
public:
	HMENU            _hmenu = 0;           // if it is end-item, _hmenu is id
	uint             _flag  = MF_STRING;
	kmStrw           _str;
	kmMat1<kmwMitem> _item;                // child item
	HMENU            _parent = 0;          // hmenu of parent

	/////////////////////////////////////////////////
	// member functions	

	kmwMitem& Add(LPCWSTR str, uint flag = MF_STRING)
	{
		if(_item.Size() == 0) _item.Create(1, 16);
		else                  _item.PushBack();
		_item.End()->_str  = kmStrw(str);
		_item.End()->_flag = flag;

		return *this;
	};

	kmwMitem& AddSeparator() { return Add(NULL, MF_SEPARATOR); };

	// create menu
	void Create(HMENU hmenu, int& n, int id_s)
	{
		_parent = hmenu; 

		if     (_flag & MF_SEPARATOR) {}
		else if(_item.N1() == 0)
		{
			_hmenu = (HMENU) (id_s + n++);
		}
		else
		{
			_hmenu = CreatePopupMenu();
			_flag  |= MF_POPUP;

			for(int i = 0; i < _item.N1(); ++i)
			{
				_item(i).Create(_hmenu, n, id_s);
			}
		}
		Append(hmenu);
	};

	// append menu
	void Append(HMENU hmenu) { AppendMenu(hmenu, _flag, (int64)_hmenu, _str.P()); };

	// set check status... 0: uncheked, 1: check, 2: toggle
	kmwMitem& SetCheck(uint state)
	{
		if(_item.N() > 0)
		{
			for(int64 i = 0; i < _item.N(); ++i) _item(i).SetCheck(state);
		}
		else
		{
			if     (state == 0) _flag &= ~MF_CHECKED;
			else if(state == 1) _flag |=  MF_CHECKED;
			else                _flag ^=  MF_CHECKED;

			CheckMenuItem(_parent, (uint)_hmenu, _flag);
		}
		return *this;
	};

	// set status 
	kmwMitem& SetEnable  () { EnableMenuItem(_parent, (uint)_hmenu, MF_ENABLED ); return *this; };
	kmwMitem& SetGrayed  () { EnableMenuItem(_parent, (uint)_hmenu, MF_GRAYED  ); return *this; };
	kmwMitem& SetDisabled() { EnableMenuItem(_parent, (uint)_hmenu, MF_DISABLED); return *this; };

	// get check status
	bool GetCheck() { return (_flag & MF_CHECKED) > 0; };

	///////////////////////////////////
	// operators

	// operator.. a(i)
	kmwMitem& operator()(int64 i) const { return _item(i); };

	// void operator 
	kmwMitem operator+=(const kmwMitem& b) { return *this; };

	// get begin and end
	kmwMitem* Begin() const { return _item.Begin(); };	
	kmwMitem* End()   const { return _item.End();   };

	// get n
	int64 N() const { return _item.N(); };

	// get id
	int64 GetId() const { return (int64)_hmenu; };

	// calc number of end-item
	void CalcEndItemN(int& n) 
	{
		     if(_flag & MF_SEPARATOR) return;
		else if(_item.N1() == 0     ) n++;
		else for(int i = 0; i < _item.N1(); ++i)  _item(i).CalcEndItemN(n);
	};
};

// menu control
class kmwMenu : public kmwMitem
{
public:	
	int _n = 0; // total number of end-items

	/////////////////////////////////////////////////
	// member functions

	// create and attach menu
	void Create(HWND hwnd, int id_s = 1024)
	{
		_hmenu = CreateMenu();
		_n     = 0;

		for(int i = 0; i < _item.N1(); ++i)
		{
			_item(i).Create(_hmenu, _n, id_s);
		}
		// attach menu to window
		SetMenu(hwnd, _hmenu);
	};

	// create as popup menu
	void CreatePopup(HWND hwnd, int x, int y, uint id_s = 2048)
	{	
		_hmenu = CreatePopupMenu();
		_n     = 0;

		for(int i = 0; i < _item.N1(); ++i)
		{
			_item(i).Create(_hmenu, _n, id_s);
		}
		TrackPopupMenu(_hmenu, TPM_LEFTALIGN, x, y, 0, hwnd, NULL);
		DestroyMenu(_hmenu);
	};

	// create as popup menu
	void CreatePopup(HWND hwnd, LPARAM lp, uint id_s = 2048)
	{
		CreatePopup(hwnd, LOWORD(lp), HIWORD(lp), 0);
	};

	// get total number of end-items
	int GetTotalN() { return _n; };

	// calc total number of end-items before created
	int GetTotalNPre() { int n = 0; CalcEndItemN(n); return n; };

	// clear every item
	void Clear() { _n = 0; _item.Release(); }; 
};

// axes window
class kmwAxes : public kmwChild
{
public:
	kmMat1<kmMat1f32> _x;        // x data of graph
	kmMat1<kmMat1f32> _y;        // y data of graph
	kmMat1<kmRgb>     _rgb;	     // color  of garph
	kmMat1<int>       _state;    // 0 : invisible, 1 : visible, 2: highlight

	float _x_min, _x_max;
	float _y_min, _y_max;

	kmgRect _rct_zoom;           // graphic object

	kmMat1<kmMat1i32> _x_pix;    // pixel position of x data
	kmMat1<kmMat1i32> _y_pix;    // pixel position of y data

	// grid line
	int        _gridx_on = 0, _gridy_on = 0;
	kmMat1f32  _gridx,        _gridy;

	// graphic object array
	kmgLines   _glin_gridx; // grid line x
	kmgLines   _glin_gridy; // grid line y

	//////////////////////////////////////
	// fucntions to create window
public:
	void Create(int x, int y, int w, int h, kmWnd* pwnd, ushort id = 0)
	{
		kmwChild::Create(x, y, w, h, pwnd, id);
	};

	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0)
	{
		kmwChild::Create(cwp, pwnd, id);
	};

	virtual void CreateChild()
	{
		_rct_zoom.Set(kmRect(10,10,20,20), kmRgb(0,0,0,1));
		_rct_zoom.SetLine(kmRgb(0,0,0), 0, PS_DOT);
		_rct_zoom.SetVisible(false);
	};

	void Update(const kmMat1f32& x, const kmMat1f32& y, int64 idx = 0)
	{
		kmLockGuard grd = Lock(); //====================lock-unlock

		ASSERTA(x.IsEqualSizeDim(y), "[kmwAxes::Update in 2116]");

		_x(idx) = x; _y(idx) = y; Invalidate();
	};

	void Update(const kmMat1f32& y, int64 idx = 0)
	{
		kmMat1f32 x(y.N()); x.SetValInc(0, 1.f);

		Update(x, y, idx);
	};
	
	void UpdateY(const kmMat1f32& y, int64 idx = 0)
	{
		kmLockGuard grd = Lock(); //====================lock-unlock

		ASSERTA(_x(idx).IsEqualSizeDim(y), "[kmwAxes::UpdateY in 2123]");

		_y(idx) = y;  Invalidate();
	};

	void UpdateX(const kmMat1f32& x, int64 idx = 0)
	{
		kmLockGuard grd = Lock(); //====================lock-unlock

		ASSERTA(_y(idx).IsEqualSizeDim(x), "[kmwAxes::UpdateY in 2123]");

		_x(idx) = x;  Invalidate();
	};

	int64 Add(const kmMat1f32& x, const kmMat1f32& y, kmRgb rgb = kmRgbB, int state = 1)
	{
		Lock(); //------------------------lock

		// check if the size of x, y are the same
		ASSERTA(x.IsEqualSizeDim(y), "[kmwAxes::Add in 1191]");

		// create mat
		if(!_x    .IsCreated()) _x    .Create(0,16);
		if(!_y    .IsCreated()) _y    .Create(0,16);
		if(!_rgb  .IsCreated()) _rgb  .Create(0,16);
		if(!_x_pix.IsCreated()) _x_pix.Create(0,16);
		if(!_y_pix.IsCreated()) _y_pix.Create(0,16);
		if(!_state.IsCreated()) _state.Create(0,16);

		// push back
		kmMat1i32 dummy(x.N1(), x.Size());

		_x    .PushBack(x);
		_y    .PushBack(y);
		_rgb  .PushBack(rgb);
		_x_pix.PushBack(dummy);
		_y_pix.PushBack(dummy);
		_state.PushBack(state);

		Unlock(); //------------------------unlock

		// set range
		SetRangeFull();  Invalidate();

		return _x.N1() - 1;
	};

	int64 Add(const kmMat1f32& y, kmRgb rgb = kmRgbB, int state = 1)
	{
		kmMat1f32 x(y.N()); x.SetValInc(0, 1.f); 

		return Add(x, y, rgb, state);
	};

	void Erase(int64 idx) { Lock(); _x.Erase(idx); _y.Erase(idx); Unlock(); };
	void Clear()
	{
		kmLockGuard grd = Lock();  //==================lock-unlock

		_x    .Release(); _y    .Release(); _rgb  .Release(); 
		_x_pix.Release(); _y_pix.Release(); _state.Release();
	};

	void SetRangeX(float x_min, float x_max)
	{
		if(x_min < x_max) { _x_min = x_min; _x_max = x_max; }
		else              { _x_min = x_max; _x_max = x_min; }
	};

	void SetRangeY(float y_min, float y_max)
	{
		if(y_min < y_max) { _y_min = y_min; _y_max = y_max; }
		else              { _y_min = y_max; _y_max = y_min; }
	};
	
	// set range with rect_pixel
	void SetRange(kmRect rt_pix)
	{
		float x_min = GetAxisX((float) rt_pix.l);
		float x_max = GetAxisX((float) rt_pix.r);
		float y_min = GetAxisY((float) rt_pix.b);
		float y_max = GetAxisY((float) rt_pix.t);

		SetRangeX(x_min, x_max);
		SetRangeY(y_min, y_max);
	};

	// set range with full
	void SetRangeFull()
	{
		float x_max = FLOAT_MIN, x_min = FLOAT_MAX;
		float y_max = FLOAT_MIN, y_min = FLOAT_MAX;

		Enter(); //----------------------------enter

		for(int64 i = 0; i < _x.N1(); ++i) if(_state(i))
		{
			const float xi_max = _x(i).Max(); x_max = MAX(x_max, xi_max);
			const float xi_min = _x(i).Min(); x_min = MIN(x_min, xi_min);
		}
		for(int64 i = 0; i < _y.N1(); ++i) if(_state(i))
		{
			const float yi_max = _y(i).Max(); y_max = MAX(y_max, yi_max);
			const float yi_min = _y(i).Min(); y_min = MIN(y_min, yi_min);		
		}

		Leave(); //---------------------------leave

		const float x_gap = (x_min == x_max) ? 2.f:0;
		const float y_gap = (y_min == y_max) ? 2.f:fabs(y_max - y_min)*0.05f;

		SetRangeX(x_min - x_gap, x_max + x_gap);
		SetRangeY(y_min - y_gap, y_max + y_gap);
	};

	// set range with delta
	void SetRangeDelta(float delta, float x_pix, float y_pix)
	{
		const float x_size = GetAxisW();
		const float y_size = GetAxisH();
				
		float  x_del = delta*x_size;
		float  y_del = delta*y_size;

		float x_pos = GetAxisX(x_pix);
		float y_pos = GetAxisY(y_pix);
			
		float x_min = _x_min + x_del*( x_pos - _x_min)/x_size;
		float x_max = _x_max - x_del*(_x_max -  x_pos)/x_size;
		float y_min = _y_min + y_del*( y_pos - _y_min)/y_size;
		float y_max = _y_max - y_del*(_y_max -  y_pos)/y_size;

		SetRangeX(x_min, x_max);
		SetRangeY(y_min, y_max);
	};

	// set grid
	void SetGridXOn(int mode = 1) { _gridx_on = (mode == 0) ? 0:1; };
	void SetGridYOn(int mode = 1) { _gridy_on = (mode == 0) ? 0:1; };
	
	void SetGridX(const kmMat1f32& xtick) { Lock(); _gridx = xtick; _glin_gridx.Recreate(_gridx); Unlock(); };
	void SetGridY(const kmMat1f32& ytick) { Lock(); _gridy = ytick; _glin_gridy.Recreate(_gridy); Unlock(); };

	// set state
	void SetState(int64 idx, int state) { Lock(); _state(idx) = state; Unlock(); };

	// set visible
	void SetVisible(int64 idx, bool on = true) { Enter(); _state(idx) = (on)? 1:0; Leave(); };

	// update grid
	void UpdateGrid()
	{
		kmLockGuard grd =  Lock(); //=====================lock-unlock

		// init parameters
		kmRgb grid_rgb(180,180,180);
		// update grid x
		if(_gridx_on) for(int64 i = 0; i < _glin_gridx.N(); ++i)
		{
			const int x = (int) GetPixX(_gridx(i));

			_glin_gridx(i).Set(kmRect(x,0,x,GetH()), grid_rgb);
		}

		// update grid y		
		if(_gridy_on) for(int64 i = 0; i < _glin_gridy.N(); ++i)
		{
			const int y = (int) GetPixY(_gridy(i));

			_glin_gridy(i).Set(kmRect(0,y,GetW(),y), grid_rgb);
		}
		Invalidate();
	};

	// get ratio to transfer pixel to value
	float GetValRatX() { return (_x_max - _x_min)/GetW(); };
	float GetValRatY() { return (_y_max - _y_min)/GetH(); };

	// get ratio to transfer pixel to value
	float GetPixRatX() { return GetW()/(_x_max - _x_min); };
	float GetPixRatY() { return GetH()/(_y_max - _y_min); };

	// get pos in axis from pos in pixel
	float GetAxisX(float x_pix) { return x_pix*GetValRatX() + _x_min; };
	float GetAxisY(float y_pix) { return _y_max - y_pix*GetValRatY(); };

	// get pos in pixel from pos in axis
	float GetPixX(float x) { return (x - _x_min)*GetPixRatX(); };	
	float GetPixY(float y) { return (_y_max - y)*GetPixRatY(); };

	// get size of axis
	float GetAxisW() { return _x_max - _x_min; };
	float GetAxisH() { return _y_max - _y_min; };

	// get number of lines
	int64 GetN() { return _y.N(); };

	//////////////////////////////////////////////
	// window proceduer
protected:
	virtual LRESULT Proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
	{
		static short old_x_pix, old_y_pix;
		static int   mode = 0; // 0 : none, 1: rect zoom, 2: panning

		// message procedure
		switch (msg)
		{
		case WM_SETCURSOR: if(mode == 0) break; else return 0;
			
		case WM_LBUTTONDOWN:

			SetCapture(GetHwnd()); // capture mouse event

			old_x_pix= LOWORD(lp); old_y_pix = HIWORD(lp);

			if(wp & MK_CONTROL)
			{
				mode = 1; // rect zoom
				SetCursor(LoadCursor(NULL, IDC_CROSS));	

				_rct_zoom.SetRect(kmRect(old_x_pix, old_y_pix));
				_rct_zoom.SetVisible(true);
			}
			else
			{
				mode = 2; // panning
				SetCursor(LoadCursor(NULL, IDC_HAND ));
			}
			PassFocusToParent();
			return 0;

		case WM_LBUTTONUP:

			ReleaseCapture(); // release mouse event

			if(mode == 1) // rect zoom
			{
				_rct_zoom.SetVisible(false);

				SetRange(_rct_zoom._rt);

				InvalidateAll();
				NotifyUpdate();
			}
			mode = 0;
			SetCursor(LoadCursor(NULL, IDC_ARROW)); 
			return 0;

		case WM_MOUSEMOVE:

			if(wp & MK_LBUTTON)
			{
				const short x_pix = LOWORD(lp), y_pix = HIWORD(lp);

				if(mode == 1) // rect zoom
				{					
					_rct_zoom._rt.SetRB(x_pix, y_pix);
					InvalidateAll();
				}
				else if(mode == 2) // panning
				{
					float x_del = (x_pix - old_x_pix)*GetValRatX();
					float y_del = (y_pix - old_y_pix)*GetValRatY();

					old_x_pix = x_pix;
					old_y_pix = y_pix;
				
					SetRangeX(_x_min - x_del, _x_max - x_del);
					SetRangeY(_y_min + y_del, _y_max + y_del);
				
					InvalidateAll();
					NotifyUpdate();
				}
			}
			return 0;

		case WM_MOUSEWHEEL:
			// * Note that lp will give x,y in pixel relative to the upper left corner of the screen
			const short x_pix  = LOWORD(lp) - GetWindowX();
			const short y_pix  = HIWORD(lp) - GetWindowY();
			const short wh_stp = HIWORD(wp);

			float wh_del = wh_stp*(0.10f/120); // wh_stp is 120 per a tick

			SetRangeDelta(wh_del, x_pix, y_pix);

			InvalidateAll();
			NotifyUpdate();

			return 0;
		}
		return kmWnd::Proc(hwnd, msg, wp, lp);
	};

	virtual void DrawGobjs(HDC hdc, int w, int h)
	{	
		// update pix... _x_pix, _y_pix
		UpdatePix(); // inner lock-unlock

		kmLockGuard grd = Enter(); //=============================enter-leave

		// draw grid line
		if(_gridx_on) for(int64 i = 0; i < _glin_gridx.N(); ++i) _glin_gridx(i).Draw(hdc);
		if(_gridy_on) for(int64 i = 0; i < _glin_gridy.N(); ++i) _glin_gridy(i).Draw(hdc);

		// draw background		
		kmgRect(kmRect(0,0,w,h), kmRgb(250,250,250,1), kmRgb(100,100,100)).Draw(hdc);

		// draw axis
		if(_x_pix.N() > 0)
		{
			const float x_rat = GetPixRatX();
			const float y_rat = GetPixRatY();

			// set line type
			HPEN pen   = CreatePen(0, 1, kmRgb(150,150,150));
			HPEN pen_o = (HPEN) SelectObject(hdc, pen);
						
			// draw x-axis
			if(_y_min < 0 && _y_max > 0)
			{
				const int y_pix = int((_y_max - 0)*y_rat);

				MoveToEx(hdc, 0, y_pix, NULL);
				LineTo  (hdc, w, y_pix);
			}

			// draw y-axis
			if(_x_min < 0 && _x_max > 0)
			{
				const int x_pix = int((0 - _x_min)*x_rat);

				MoveToEx(hdc, x_pix, 0, NULL);
				LineTo  (hdc, x_pix, h);
			}

			// delete objects
			DeleteObject(SelectObject(hdc, pen_o));
		}

		// plot data
		for(int j = 0; j < _x_pix.N(); ++j) if(_state(j))
		{
			const int64 n_pix = _x_pix(j).N();

			// set line type
			HPEN pen   = CreatePen(0, 1, _rgb(j));
			HPEN pen_o = (HPEN) SelectObject(hdc, pen);

			// move the first point
			MoveToEx(hdc, _x_pix(j)(0), _y_pix(j)(0), NULL); 

			// draw lines
			for(int64 i = 1; i < n_pix; ++i)
			{
				LineTo(hdc, _x_pix(j)(i), _y_pix(j)(i));
			}

			// delete objects
			DeleteObject(SelectObject(hdc, pen_o));	
		}

		// draw rect for zoom
		{
			_rct_zoom.Draw(hdc);
		}
	};

	// update pixel position of data... _x, _y --> _x_pix, _y_pix
	void UpdatePix()
	{
		kmLockGuard grd = Lock(); //=================================lock-unlock

		const float x_rat = GetPixRatX();
		const float y_rat = GetPixRatY();

		for(int64 j = 0; j < _x.N(); ++j)
		{	
			int64 n_x = _x(j).N();
			int64 idx  = 0;

			float xm, x0, xp, ym, y0, yp;

			_x_pix(j).SetN1ToSize();
			_y_pix(j).SetN1ToSize();

			if(_x_pix(j).Size() < n_x) _x_pix(j).Recreate(n_x);
			if(_y_pix(j).Size() < n_x) _y_pix(j).Recreate(n_x);

			// update pixel 1st data
			for(int64 i = 0; i < 1; ++i)
			{				
				x0 = _x(j)(0); xp = _x(j)(1);
				y0 = _y(j)(0); yp = _y(j)(1);

				// check if outof range
				if(x0 < _x_min && xp < _x_min) continue;
				if(y0 < _y_min && yp < _y_min) continue;
				if(x0 > _x_max && xp > _x_max) continue;
				if(y0 > _y_max && yp > _y_max) continue;
				
				_x_pix(j)(idx) = (int) ((_x(j)(0) - _x_min  )*x_rat);
				_y_pix(j)(idx) = (int) ((_y_max   - _y(j)(0))*y_rat);

				++idx;
			}

			// update pixel data
			for(int64 i = 1; i < n_x-1; ++i)
			{
				xm = _x(j)(i-1); x0 = _x(j)(i); xp = _x(j)(i+1);
				ym = _y(j)(i-1); y0 = _y(j)(i); yp = _y(j)(i+1);

				// check if outof range
				if(xm < _x_min && x0 < _x_min && xp < _x_min) continue;
				if(ym < _y_min && y0 < _y_min && yp < _y_min) continue;
				if(xm > _x_max && x0 > _x_max && xp > _x_max) continue;
				if(ym > _y_max && y0 > _y_max && yp > _y_max) continue;

				int x_pix = (int) ((_x(j)(i) - _x_min  )*x_rat);
				int y_pix = (int) ((_y_max   - _y(j)(i))*y_rat);

				// check if it is the same with the previous pixel
				if(idx > 0)
				if(x_pix == _x_pix(j)(idx-1) && y_pix == _y_pix(j)(idx-1)) continue;

				_x_pix(j)(idx) = x_pix;
				_y_pix(j)(idx) = y_pix;

				++idx;
			}

			// update pixel the last data
			for(int64 i = n_x-1; i < n_x; ++i)
			{				
				xm = _x(j)(i-1); x0 = _x(j)(i);
				ym = _y(j)(i-1); y0 = _y(j)(i);

				// check if outof range
				if(xm < _x_min && x0 < _x_min) continue;
				if(ym < _y_min && y0 < _y_min) continue;
				if(xm > _x_max && x0 > _x_max) continue;
				if(ym > _y_max && y0 > _y_max) continue;
				
				_x_pix(j)(idx) = (int) ((_x(j)(i) - _x_min  )*x_rat);
				_y_pix(j)(idx) = (int) ((_y_max   - _y(j)(i))*y_rat);

				++idx;
			}
			_x_pix(j).SetN1(idx);
			_y_pix(j).SetN1(idx);;
		}
	};	

	////////////////////////////////////////////
	// notification functions

	void NotifyUpdate()
	{
		SendMsgToParent(WM_COMMAND, MAKELONG(GetId(), NTF_AXES_UPDATE), 0);
	};
};

// image window
class kmwImg : public kmwChild
{
public:
	kmImg _img;
	int   _sbmode = HALFTONE; // strechblt mode (BLACKONWHITE, WHITEONBLACK, COLORONCOLOR, HALFTONE)

	// graphic object array
	kmgObjs _gob;

	//////////////////////////////////////
	// fucntions to create window
public:
	void Create(int x, int y, int w, int h, const kmImg& img, kmWnd* pwnd, ushort id = 0)
	{
		_img = img;
		kmwChild::Create(x, y, w, h, pwnd, id);
	};

	void Create(kmCwp cwp, const kmImg& img, kmWnd* pwnd, ushort id = 0)
	{
		_img = img;
		kmwChild::Create(cwp, pwnd, id);
	};

	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0)
	{
		_img.Create(1,1,1); 
		_img(0) = 0;
		kmwChild::Create(cwp, pwnd, id);
	};

	void Update(const kmImg& img)
	{
		_img = img;
		Invalidate();
	};

	void Add(kmgObj* const gob)
	{
		if(!_gob.IsCreated()) _gob.Create(0,16);
		_gob.PushBack(gob);
	};

	// convert position on the view to postion on the image
	void ConvertPos(int& x, int& y, int x_i, int y_i) const
	{
		// convert to axis of image		
		x = (int) floorf(((float)x_i*_img.GetW())/_win.w);
		y = (int) floorf(((float)y_i*_img.GetH())/_win.h);
	};

	// set strechbit mode 
	// : BLACKONWHITE, WHITEONBLACK, COLORONCOLOR, HALFTONE
	// : choose COLORONCOLOR for efficiency
	// : choose HALFTONE for quality
	void SetSbMode(int sbmode) { _sbmode = sbmode; };

	// get _img
	kmImg& GetImg() { return _img; };

	////////////////////////////////////////////////////
	// window procedure
protected:	
	virtual kmePPType OnPaint(WPARAM wp, LPARAM lp)
	{
		// init parameters		
		PAINTSTRUCT ps;				

		// get dc and memory dc... start drawing
		HDC hdc = BeginPaint(_hwnd, &ps);
		HDC mdc = CreateCompatibleDC(hdc);

		// get bitmap handle		
		HBITMAP hbm = CreateCompatibleBitmap(hdc, _img.GetW(), _img.GetH());

		// get bitmap info
		BITMAP bm;
		GetObject(hbm, sizeof(BITMAP), &bm);

		// set img to bitmap		
		SetBitmapBits(hbm, (int)_img.GetByteFrame(),(void*) _img.Begin());
		
		// draw bitmap on memory dc
		HBITMAP hbm_ = (HBITMAP) SelectObject(mdc, hbm);
		
		// draw graphic objects of _gob
		for(int64 i = 0 ; i < _gob.N(); ++i) _gob(i)->Draw(mdc, _win.w, _win.h);

		// copy mdc to hdc
		//BitBlt(hdc, 0, 0, _win.w, _win.h, mdc, 0, 0, SRCCOPY);
		SetStretchBltMode(hdc, _sbmode);
		StretchBlt(hdc, 0, 0, _win.w     , _win.h,
			       mdc, 0, 0, _img.GetW(), _img.GetH(), SRCCOPY);

		// finish drawing
		SelectObject(mdc, hbm_);
		DeleteObject(hbm);
		DeleteDC    (mdc);
		EndPaint    (_hwnd, &ps);

		// call parent's drawgobj
		GetParent()->DrawGobjsLater();

		return PP_FINISH;
	};

	//virtual kmePPType OnLButtonDown(WPARAM wp, LPARAM lp)
	//{		
	//	lp = MAKELONG(LOWORD(lp) + GetX(), HIWORD(lp) + GetY());
	//
	//	PassMsgToParent(WM_LBUTTONDOWN, wp, lp);
	//
	//	return PP_FINISH;
	//};

	virtual kmePPType OnLButtonDown(WPARAM wp, LPARAM lp)
	{			
		// convert to axis of image
		int x = LOWORD(lp), y = HIWORD(lp), img_x, img_y;
	
		ConvertPos(img_x, img_y, x, y);
	
		// notify to the parent window
		NotifyPosClick(img_x, img_y);
	
		// set focus
		// * Note that a key down message won't be delivered properly without this.
		SetFocus();
	
		return PP_FINISH;
	};

	virtual kmePPType OnMouseMove(WPARAM wp, LPARAM lp)
	{
		// convert to axis of image
		int x = LOWORD(lp), y = HIWORD(lp), img_x, img_y;
	
		ConvertPos(img_x, img_y, x, y);
	
		// notify to the parent window
		NotifyMouseMove(img_x, img_y);
	
		return PP_FINISH;
	};

	////////////////////////////////////////////
	// notification functions

	void NotifyPosClick(int x, int y) // x, y is pixel position of img
	{
		// combine wp
		WPARAM wp0 = MAKELONG(GetId(), NTF_IMG_POSCLICK);
	
		// combine lp
		WPARAM lp0 = MAKELONG(x, y);
		
		SendMsgToParent(WM_COMMAND, wp0, lp0);
	};

	void NotifyMouseMove(int x, int y) // x, y is pixel position of img
	{
		// combine wp
		WPARAM wp0 = MAKELONG(GetId(), NTF_IMG_MOUSEMOVE);
	
		// combine lp
		WPARAM lp0 = MAKELONG(x, y);

		SendMsgToParent(WM_COMMAND, wp0, lp0);
	};
};

// matrix image window
class kmwMat : public kmwChild
{	
public:	
#ifdef KC7MAT
	kcMat3f32  _mat;                      // data of matrix (i1: y, i2: x, i3: z)
#else
	kmMat3f32 _mat;
#endif
	
	kmCMap     _cmap = kmCMap(CMAP_GREY); // color map
	float      _cmin = 0.f;               // min of colormap
	float      _cmax = 1.f;               // max of colormap
	float      _x0   = 0, _dx   = 1.f;    // info of _mat
	float      _y0   = 0, _dy   = 1.f;    // info of _mat
	float      _xmin = 0, _xmax = 0;      // info of displayed x-axis
	float      _ymin = 0, _ymax = 0;      // info of displayed y-axis
	int        _mode_aeq = 0;             // mode for axis equal
	int64      _z_idx = 0;                // z index or frame index

	// grid line
	int        _gridx_on = 0, _gridy_on = 0;
	kmMat1f32  _gridx,        _gridy;

	// guide line
	int        _gdlx_on = 0, _gdly_on = 0;
	float      _gdlx,        _gdly;

	// graphic object array
	kmMat1<kmgObj*> _gob;

	kmgImg          _gimg;       // main image... it will be added to _gob
	kmgRect         _grct_zoom;  // rect for zoom
	kmMat1<kmgLine> _glin_gridx; // grid line x
	kmMat1<kmgLine> _glin_gridy; // grid line y
	kmgLine         _glin_gdlx;  // guide line x
	kmgLine         _glin_gdly;  // guide line y

	// cuda members
#ifdef KC7MAT
private:	
	kcMat1<kmBgr> __cmap;
	kcMat2<kmBgr> __img;
#endif

	//////////////////////////////////////
	// fucntions to create window
public:
	void Create(kmCwp cwp, const kmMat3f32& mat, kmWnd* pwnd, ushort id = 0)
	{
		_mat = mat;

		kmwChild::Create(cwp, pwnd, id);
	};

	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0)
	{
		_mat.Recreate(1,1,1); _mat.SetZero(); //_mat(0) = 0;

		kmwChild::Create(cwp, pwnd, id);
	};

	/////////////////////////////////////////////
	// function to create child 
protected:
	virtual void CreateChild()
	{	
		// set graphic object
		_gimg.SetVisible(false);

		_grct_zoom.Set(kmRect(10,10,20,20), kmRgb(0,0,0,1));
		_grct_zoom.SetLine(kmRgb(0,0,0), 0, PS_DOT);
		_grct_zoom.SetVisible(false);

		// set __cmap
#ifdef KC7MAT
		__cmap = _cmap;
#endif
	};

	//virtual void InvalidateChild() { _wimg.InvalidateAll(); };

	///////////////////////////////////////////////
	// window procedures
protected:
	virtual LRESULT Proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
	{
		static short old_x_pix, old_y_pix;
		static int   mode = 0; // 0 : none, 1: rect zoom, 2: panning

		// message procedure
		switch (msg)
		{
		case WM_SETCURSOR: if(mode == 0) break; else return 0;

		case WM_LBUTTONDBLCLK: // double click... you must set class style with CS_DBLCLKS to use this message
			{
				const short x_pix = LOWORD(lp), y_pix = HIWORD(lp);
				const short x_idx = (short)GetIdxXfromPix((float)x_pix);
				const short y_idx = (short)GetIdxYfromPix((float)y_pix);

				SendMsgToParent(WM_COMMAND, MAKELONG(GetId(), NTF_WMAT_DBLCLK), MAKELONG(x_idx, y_idx));				
			}
			return 0;
			
		case WM_LBUTTONDOWN:
			SetFocus();
			SetCapture(GetHwnd()); // capture mouse event

			old_x_pix= LOWORD(lp); old_y_pix = HIWORD(lp);

			if(wp & MK_CONTROL)
			{
				mode = 1; // rect zoom
				SetCursor(LoadCursor(NULL, IDC_CROSS));	

				_grct_zoom.SetRect(kmRect(old_x_pix, old_y_pix));
				_grct_zoom.SetVisible(true);
			}
			else
			{
				mode = 2; // panning
				SetCursor(LoadCursor(NULL, IDC_HAND ));
			}
			return 0;

		case WM_LBUTTONUP:

			ReleaseCapture(); // release mouse event

			if(mode == 1) // rect zoom
			{
				_grct_zoom.SetVisible(false);

				SetAxis(_grct_zoom._rt);
				UpdatePos();				
				NotifyUpdate();
			}
			mode = 0;
			SetCursor(LoadCursor(NULL, IDC_ARROW)); 
			return 0;

		case WM_MOUSEMOVE:

			if(wp & MK_LBUTTON)
			{
				const short x_pix = LOWORD(lp), y_pix = HIWORD(lp);

				if(mode == 1) // rect zoom
				{					
					_grct_zoom._rt.SetRB(x_pix, y_pix);
					Invalidate();
				}
				else if(mode == 2) // panning
				{
					float x_del = (x_pix - old_x_pix)*GetValRatX();
					float y_del = (y_pix - old_y_pix)*GetValRatY();
					
					old_x_pix = x_pix;
					old_y_pix = y_pix;

					SetAxis(_xmin - x_del, _xmax - x_del, _ymin - y_del, _ymax - y_del);
					UpdatePos();
					NotifyUpdate();
				}
			}
			return 0;

		case WM_MOUSEWHEEL:
			// * Note that lp will give x,y in pixel relative to the upper left corner of the screen
			const short x_pix  = LOWORD(lp) - GetWindowX();
			const short y_pix  = HIWORD(lp) - GetWindowY();
			const short wh_stp = HIWORD(wp);

			float wh_del = wh_stp*(0.10f/120); // wh_stp is 120 per a tick
			
			SetAxisScale(wh_del, x_pix, y_pix);
			UpdatePos();
			NotifyUpdate();

			return 0;
		}
		return kmWnd::Proc(hwnd, msg, wp, lp);
	};

	// key procedure
	virtual kmePPType OnKeyDown(WPARAM wp, LPARAM lp)
	{
		switch(wp)
		{
		case VK_INSERT  : ChangeRangeCMin( 0.01f); UpdateImg(); UpdateWindow(); NotifyUpdate(); break;
		case VK_DELETE  : ChangeRangeCMin(-0.01f); UpdateImg(); UpdateWindow(); NotifyUpdate(); break;
		case VK_HOME    : ChangeRangeCMax( 0.01f); UpdateImg(); UpdateWindow(); NotifyUpdate(); break;
		case VK_END     : ChangeRangeCMax(-0.01f); UpdateImg(); UpdateWindow(); NotifyUpdate(); break;
		case VK_BACK    : SetRangeCAuto();         UpdateImg(); UpdateWindow(); NotifyUpdate(); break;

		case VK_PRIOR   : SetFrame(_z_idx + 1); UpdateImg(); UpdateWindow(); NotifyUpdate(); break;
		case VK_NEXT    : SetFrame(_z_idx - 1); UpdateImg(); UpdateWindow(); NotifyUpdate(); break;

		case VK_SPACE   : SetAxisFull(); NotifyUpdate(); break;

		default         : PassMsgToParent(WM_KEYDOWN, wp, lp);
		}
		return PP_FINISH;
	};

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{
			if(_mode_aeq == 1) SetAxisFull();
			UpdatePos();
		}
		return PP_FINISH;
	};
	
	// draw graphic objects
	virtual void DrawGobjs(HDC hdc, int w, int h)
	{
		kmLockGuard grd = Enter(); //==========================enter-leave

		// draw image
		_gimg.Draw(hdc);

		// draw grid line
		if(_gridx_on) for(int64 i = 0; i < _glin_gridx.N(); ++i) _glin_gridx(i).Draw(hdc);
		if(_gridy_on) for(int64 i = 0; i < _glin_gridy.N(); ++i) _glin_gridy(i).Draw(hdc);

		// draw guide line
		if(_gdlx_on) _glin_gdlx.Draw(hdc);
		if(_gdly_on) _glin_gdly.Draw(hdc);

		// draw extra graphic objects
		for(int64 i = 0 ; i < _gob.N(); ++i) _gob(i)->Draw(hdc, w, h);

		// draw rect for zoom
		_grct_zoom.Draw(hdc);
	};

	//////////////////////////////////////////////
	// member functions 
public:	
	
#ifdef KC7MAT
	// update mat with kcMat
	// * Note that you should call UpdateImg() if you want to update window
	void UpdateMat(const kcMat3f32& mat, float y0, float dy, float x0, float dx, cudaStream_t s = 0)
	{
		Lock(); //---------------------------lock
		if(mat.N() == 0) { _mat.Recreate(1,1,1); _mat.SetZero(s); }
		else             { _mat.RecreateIf(mat); _mat.CopyFrom(mat, s); }
		_x0 = x0, _dx = dx, _y0 = y0, _dy = dy;
		_z_idx = MIN(MAX(0,_z_idx),_mat.N3() - 1);
		Unlock(); //-------------------------unlock
	};

	// set mat with kcMat
	// * Note that you should ensure that mat is valid during use.
	void SetMat(const kcMat3f32& mat, float y0, float dy, float x0, float dx, cudaStream_t s = 0)
	{
		Lock(); //---------------------------lock
		if(mat.N() == 0) { _mat.Recreate(1,1,1); _mat.SetZero(s); }
		else             { _mat.Release();       _mat.Set(mat);   }
		_x0 = x0, _dx = dx, _y0 = y0, _dy = dy;
		_z_idx = MIN(MAX(0,_z_idx),_mat.N3() - 1);
		Unlock(); //-------------------------unlock
	};
#else
	// update mat
	// * Note that you should call UpdateImg() if you want to update window
	void UpdateMat(const kmMat3f32& mat, float y0, float dy, float x0, float dx)
	{
		Lock(); //---------------------------lock
		if(mat.N() == 0) { _mat.Recreate(1,1,1); _mat.SetZero(); }
		else             { _mat = mat; }
		_x0 = x0, _dx = dx, _y0 = y0, _dy = dy;
		_z_idx = MIN(MAX(0,_z_idx),_mat.N3() - 1);
		Unlock(); //-------------------------unlock
	};

	// set mat
	// * Note that you should ensure that mat is valid during use.
	void SetMat(const kmMat3f32& mat, float y0, float dy, float x0, float dx)
	{
		Lock(); //---------------------------lock
		if(mat.N() == 0) { _mat.Recreate(1,1,1); _mat.SetZero();}
		else             { _mat.Release();       _mat.Set(mat); }
		_x0 = x0, _dx = dx, _y0 = y0, _dy = dy;
		_z_idx = MIN(MAX(0,_z_idx),_mat.N3() - 1);
		Unlock(); //-------------------------unlock
	};
#endif

	// update cmap
	void UpdateCMap(kmeCMapType cmtype)
	{
		_cmap.Create(cmtype); 
#ifdef KC7MAT
		__cmap = _cmap; 
#endif
		Invalidate(); 
	};

#ifdef KC7MAT
	// update image from _mat 
	void UpdateImg(cudaStream_t s = 0)
	{	
		// * Note that if mattp_ is directly kmMat3, mattp_ will be set with return of Tp()
		// * instead of move-construction, which causes using invalid temporary memory.
		Lock(); //---------------------------lock

		kcMat2f32 mat = _mat.Mat2(_z_idx);

		__img.RecreateIf(mat.N2(), mat.N1());

		ConvertBgrTp(__img, mat, _cmin, _cmax, __cmap, s);

		_gimg.GetImg().RecreateIf(__img.N1(), __img.N2(), 1);

		__img.CopyToHost(_gimg.GetImg().P(), s);

		Unlock(); //-------------------------unlock

		_gimg.SetVisible();	Invalidate();
	};
#else
	// update image from _mat 
	void UpdateImg()
	{	
		// * Note that if mattp_ is directly kmMat3, mattp_ will be set with return of Tp()
		// * instead of move-construction, which causes using invalid temporary memory.		
		Enter();
		kmMat2f32 mattp_ = _mat.Mat2(_z_idx).Tp();
		kmMat3f32 mattp  = mattp_;
		Leave();

		Lock(); //---------------------------lock
		_gimg.GetImg().RecreateIf(mattp.N1(), mattp.N2(), 1);
		_gimg.GetImg().ConvertBgr(mattp, _cmin, _cmax, _cmap);		
		Unlock(); //-------------------------unlock
		
		_gimg.SetVisible();	Invalidate();
	};
#endif

	// update position of _wimg
	void UpdatePos()
	{
		const float l = (_x0     - _xmin)*GetW()/GetAxisW();
		const float t = (_y0     - _ymin)*GetH()/GetAxisH();
		const float r = (GetXe() - _xmin)*GetW()/GetAxisW();
		const float b = (GetYe() - _ymin)*GetH()/GetAxisH();

		_gimg.SetRect(kmRect((int)l,(int)t,(int)r,(int)b));
		Invalidate();
	};

	// set frame index
	void SetFrame(int64 frame_idx)
	{
		kmLockGuard grd = Enter();

		if(frame_idx < 0 || _mat.N3()-1 < frame_idx) return;

		_z_idx = frame_idx;
	};

	// set range
	void SetRangeC(float cmin, float cmax)
	{	
		_cmin = cmin; _cmax = MAX(_cmin + 1e-3f, cmax);

		//PRINTFA("[kmwMat::SetRangeC] cmin : %.1f, cmax : %.1f\n", _cmin, _cmax);
	};

	// set range auto
	void SetRangeCAuto()
	{
		kmLockGuard grd = Enter(); //==================enter-leave

		SetRangeC(Min(_mat.Mat2(_z_idx)), Max(_mat.Mat2(_z_idx))); 
	};

	// increase or decrease range cmin and cmax
	void ChangeRangeCMin(float rat) { _cmin += MAX(1e-3f, _cmax - _cmin)*rat; _cmin = MIN(_cmin, _cmax); };
	void ChangeRangeCMax(float rat) { _cmax += MAX(1e-3f, _cmax - _cmin)*rat; _cmax = MAX(_cmin, _cmax); };

	// set axis
	void SetAxis(float xmin, float xmax, float ymin, float ymax)
	{
		_xmin = MIN(xmin, xmax), _xmax = MAX(xmin, xmax);
		_ymin = MIN(ymin, ymax), _ymax = MAX(ymin, ymax);
	};

	// set axis with rect_pixel
	void SetAxis(const kmRect& rt_pix)
	{
		float xmin = GetAxisX((float) rt_pix.l);
		float xmax = GetAxisX((float) rt_pix.r);
		float ymin = GetAxisY((float) rt_pix.t);
		float ymax = GetAxisY((float) rt_pix.b);

		SetAxis(xmin, xmax, ymin, ymax);
		if(_mode_aeq == 1) SetAxisEqual();
	};

	// set axis with full size
	void SetAxisFull()
	{
		SetAxis(_x0, GetXe(), _y0, GetYe());
		if(_mode_aeq == 1) SetAxisEqual();
	};

	// set axis with axis equal
	void SetAxisEqual(int mode = 1)
	{
		_mode_aeq = mode;

		if(mode == 0) return;

		const int w_pix = GetW(), h_pix = GetH();
		const float dxp = (_xmax - _xmin)/w_pix;
		const float dyp = (_ymax - _ymin)/h_pix;

		if(dxp > dyp)
		{
			const float yhdel = dxp*h_pix*0.5f;
			const float ycen  = (_ymax + _ymin)*0.5f;
			
			SetAxis(_xmin, _xmax, ycen - yhdel, ycen + yhdel);
		}
		else
		{
			const float xhdel = dyp*w_pix*0.5f;
			const float xcen  = (_xmax + _xmin)*0.5f;

			SetAxis(xcen - xhdel, xcen + xhdel, _ymin, _ymax);
		}
	};

	// set axis wich scale change
	void SetAxisScale(float delta, float x_pix, float y_pix)
	{
		const float x_size = GetAxisW();
		const float y_size = GetAxisH();				
		const float x_del  = delta*x_size;
		const float y_del  = delta*y_size;
		const float x_pos  = GetAxisX(x_pix);
		const float y_pos  = GetAxisY(y_pix);			
		const float x_min  = _xmin + x_del*(x_pos -  _xmin)/x_size;
		const float x_max  = _xmax - x_del*(_xmax -  x_pos)/x_size;
		const float y_min  = _ymin + y_del*(y_pos -  _ymin)/y_size;
		const float y_max  = _ymax - y_del*(_ymax -  y_pos)/y_size;

		SetAxis(x_min, x_max, y_min, y_max);
	};

	// set grid
	void SetGridXOn(int mode = 1) { _gridx_on = (mode == 0) ? 0:1; };
	void SetGridYOn(int mode = 1) { _gridy_on = (mode == 0) ? 0:1; };
	
	void SetGridX(const kmMat1f32& xtick) { _gridx = xtick; _glin_gridx.Recreate(_gridx); };
	void SetGridY(const kmMat1f32& ytick) { _gridy = ytick; _glin_gridy.Recreate(_gridy); };

	// set guide line
	void SetGdlXOn(int mode = 1) { _gdlx_on = (mode == 0) ? 0:1; };
	void SetGdlYOn(int mode = 1) { _gdly_on = (mode == 0) ? 0:1; };
	
	void SetGdlX(float x) { _gdlx = x; };
	void SetGdlY(float y) { _gdly = y; };
		
	// update grid
	void UpdateGrid()
	{
		// init parameters
		kmRgb grid_rgb(180,180,255);

		// update grid x
		if(_gridx_on) for(int64 i = 0; i < _glin_gridx.N(); ++i)
		{
			const int x = (int) GetPixX(_gridx(i));

			_glin_gridx(i).Set(kmRect(x,0,x,GetH()), grid_rgb);
		}

		// update grid y		
		if(_gridy_on) for(int64 i = 0; i < _glin_gridy.N(); ++i)
		{
			const int y = (int) GetPixY(_gridy(i));

			_glin_gridy(i).Set(kmRect(0,y,GetW(),y), grid_rgb);
		}
		Invalidate();
	};

	// update guide line
	void UpdateGdl()
	{
		// init parameters
		kmRgb gdl_rgb(255,160,160);

		// update guide line x and y
		if(_gdlx_on) { const int x = (int)GetPixX(_gdlx); _glin_gdlx.Set(kmRect(x,0,x,GetH()), gdl_rgb); }		
		if(_gdly_on) { const int y = (int)GetPixY(_gdly); _glin_gdly.Set(kmRect(0,y,GetW(),y), gdl_rgb); }
		Invalidate();
	};

	// add gob
	void Add(kmgObj* const gob)
	{
		kmLockGuard grd = Lock(); //================lock-unlock

		if(!_gob.IsCreated()) _gob.Create(0,16);
		_gob.PushBack(gob);
	};

	// get x_end, y_end
	float GetXe() { kmLockGuard grd = Enter(); return _x0 + (_mat.N2() - 1)*_dx; };
	float GetYe() { kmLockGuard grd = Enter(); return _y0 + (_mat.N1() - 1)*_dy; };
	
	// get ratio to transfer pixel to value
	float GetValRatX() { return (_xmax - _xmin)/GetW(); };
	float GetValRatY() { return (_ymax - _ymin)/GetH(); };

	// get ratio to transfer pixel to value
	float GetPixRatX() { return GetW()/(_xmax - _xmin); };
	float GetPixRatY() { return GetH()/(_ymax - _ymin); };

	// get pos in axis from pos in pixel... pix --> pos
	float GetAxisX(float x_pix) { return x_pix*GetValRatX() + _xmin; };
	float GetAxisY(float y_pix) { return y_pix*GetValRatY() + _ymin; };

	// get pos in pixel from pos in axis... pos --> pix
	float GetPixX(float x) { return (x - _xmin)*GetPixRatX(); };
	float GetPixY(float y) { return (y - _ymin)*GetPixRatY(); };

	// get size of axis
	float GetAxisW() { return _xmax - _xmin; };
	float GetAxisH() { return _ymax - _ymin; };

	// get matrix index from pos in axis... pos --> idx
	float GetIdxX(float x) { return (x - _x0)/_dx; };
	float GetIdxY(float y) { return (y - _y0)/_dy; };

	// get pos in axis from matrix index... idx --> pos
	float GetAxisXfromIdx(int x_idx) { return x_idx*_dx + _x0; };
	float GetAxisYfromIdx(int y_idx) { return y_idx*_dy + _y0; };

	// get matrix index from pos in pixel... pix --> idx
	float GetIdxXfromPix(float x_pix) { return GetIdxX(GetAxisX(x_pix)); };
	float GetIdxYfromPix(float y_pix) { return GetIdxY(GetAxisY(y_pix)); };

	// get frame index
	int64 GetFrameIdx() const { return _z_idx; };

	// get range of cmap
	float GetRangeCMax() const { return _cmax; };
	float GetRangeCMin() const { return _cmin; };

	//////////////////////////////////////////////
	// inner member functions 
protected:
	// notification to parent
	void NotifyUpdate()
	{
		SendMsgToParent(WM_COMMAND, MAKELONG(GetId(), NTF_AXES_UPDATE), 0);
	};
};

// image with axes window
class kmwMatAxes : public kmwChild
{
public:
	kmwMat    _wmat;             // matrix image viewer including data
	kmwMenu   _mpop;             // pop-up menu

	kmFont    _font;
	kmStrw    _xtitle, _ytitle;
	kmgAxis   _xaxis,  _yaxis;

	//////////////////////////////////////
	// fucntions to create window
public:
	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0) { kmwChild::Create(cwp, pwnd, id);};
		
	/////////////////////////////////////////////
	// function to create child 
protected:
	virtual void CreateChild()
	{
		// create axes
		_wmat.Create(CalcWmatCwp(), this, 1);
		
		// create menu
		_mpop.Add(L"axis equal");
		_mpop.Add(L"grid-x on");
		_mpop.Add(L"grid-y on");

		// set graphical objects
		_xaxis.Set(0, &_font,  1);
		_yaxis.Set(1, &_font, -1);
	};

	virtual void InvalidateChild() { _wmat.InvalidateAll(); };

	///////////////////////////////////////////////
	// window procedures
protected:
	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{
			_wmat.Relocate(w, h);
			UpdateAxis();
		}
		return PP_FINISH;
	};

	virtual kmePPType OnKeyDown(WPARAM wp, LPARAM lp)
	{
		switch(wp)
		{
		case VK_INSERT : 
		case VK_DELETE :
		case VK_HOME   :
		case VK_END    :
		case VK_PRIOR  :
		case VK_NEXT   : _wmat.SendMsg(WM_KEYDOWN, wp, lp); break;
		default        : PassMsgToParent(WM_KEYDOWN, wp, lp);
		}
		return PP_FINISH;
	};

	virtual void DrawGobjs(HDC hdc, int w, int h)
	{
		// init parameter
		kmRect rt = {0, 0, w, h}, axes_rt = _wmat.GetRect();

		const int h_font = (int) (_font.GetH()*1.4f);
		
		// draw background
		FillRect(hdc, rt, (HBRUSH) GetStockObject(WHITE_BRUSH));

		// draw axes rect
		kmgRect(kmRect(axes_rt).SizeUp(1), kmRgbW, kmRgb(180,180,255)).Draw(hdc);
		
		// draw axis
		_xaxis.SetRect(axes_rt); _xaxis.Draw(hdc);
		_yaxis.SetRect(axes_rt); _yaxis.Draw(hdc);

		// draw x-title
		if(_xtitle.N() > 0)
		{
			kmRect ttl_rt = {0, h - h_font, w, h};

			kmgStr ttl; ttl.Set(&_font, kmRgb(0,0,0), ttl_rt, DT_BOTTOM | DT_CENTER | DT_SINGLELINE);

			ttl.SetStr(_xtitle);
			ttl.Draw(hdc);
		}

		// draw y-title
		if(_ytitle.N() > 0)
		{
			kmRect ttl_rt = {0, 0, h_font, h};

			_font.SetAngle(900);

			kmgStr ttl; ttl.Set(&_font, kmRgb(0,0,0), ttl_rt, DT_LEFT  | DT_VCENTER | DT_SINGLELINE);

			ttl.SetStr(_ytitle);
			ttl.Draw(hdc);

			_font.SetAngle(0);
		}
	};

	virtual kmePPType OnContextMenu(WPARAM wp, LPARAM lp) // right mouse click
	{		
		_mpop.CreatePopup(_hwnd, lp);
		
		return PP_FINISH;
	};

	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control
		ushort ntf = HIWORD(wp), id = LOWORD(wp);

		switch(id)
		{
		// notification
		case 1: // _wmat
			switch(ntf)
			{
			case NTF_AXES_UPDATE : // update x-axis and y-axiss
				UpdateAxis();
				InvalidateAll(); 
				UpdateWindow();
				break;
			case NTF_WMAT_DBLCLK : // double-click in wmat
				SendMsgToParent(WM_COMMAND,  MAKELONG(GetId(), NTF_WMAT_DBLCLK), lp);
				break;
			}
			break;

		// popup-menu
		case 2048 +  0: // axis equal
			_mpop(0).SetCheck(2); 
			SetAxisEqual(_mpop(0).GetCheck());
			UpdateAxis();
			InvalidateAll();
			UpdateWindow(); 
			break;

		case 2048 +  1: // grid-x on
			_mpop(1).SetCheck(2); 
			_wmat.SetGridXOn(_mpop(1).GetCheck());
			UpdateGrid();
			InvalidateAll();
			UpdateWindow(); 
			break;

		case 2048 +  2: // grid-y on
			_mpop(2).SetCheck(2); 
			_wmat.SetGridYOn(_mpop(2).GetCheck());
			UpdateGrid();
			InvalidateAll();
			UpdateWindow(); 
			break;
		}
		return PP_FINISH;
	};

	// notification to parent
	void NotifyUpdate()
	{
		SendMsgToParent(WM_COMMAND, MAKELONG(GetId(), NTF_AXES_UPDATE), 0);
	};

	//////////////////////////////////////
	// member functions
public:
	// update mat
	// * Note that you should call UpdateImg() if you want to update window
	void UpdateMat(const kmMat3f32& mat, float y0 = 0, float dy = 1, float x0 = 0, float dx = 1)
	{
		_wmat.UpdateMat(mat, y0, dy, x0, dx);
		if(_wmat._xmin == _wmat._xmax) _wmat.SetAxisFull();
	};
#ifdef KC7MAT
	// update mat with kcMat
	// * Note that you should call UpdateImg() if you want to update window
	void UpdateMat(const kcMat3f32& mat, float y0 = 0, float dy = 1, float x0 = 0, float dx = 1, cudaStream_t s = 0)
	{
		_wmat.UpdateMat(mat, y0, dy, x0, dx, s);
		if(_wmat._xmin == _wmat._xmax) _wmat.SetAxisFull();
	};

	void UpdateImg(cudaStream_t s = 0) { _wmat.UpdateImg(s); UpdateAxis(); };
#else
	void UpdateImg() { _wmat.UpdateImg(); UpdateAxis(); };
#endif

	void UpdateRangeC(float cmin, float cmax) { _wmat.SetRangeC(cmin, cmax);};
	void UpdateRangeCAuto()                   { _wmat.SetRangeCAuto();      };

	void UpdateAxis()
	{
		_wmat.UpdatePos();
		_xaxis.UpdateTick(5, _wmat._xmin, _wmat._xmax);
		_yaxis.UpdateTick(5, _wmat._ymin, _wmat._ymax);
		UpdateGrid();
		_wmat.UpdateGdl();
		InvalidateAll();
		NotifyUpdate();
	};

	void UpdateGrid()
	{
		_wmat.SetGridX(_xaxis._tick);
		_wmat.SetGridY(_yaxis._tick);
		_wmat.UpdateGrid();
	};

	void UpdateAxesCwp() {	_wmat.SetCwp(CalcWmatCwp()); };

	void UpdateGdl(){ _wmat.UpdateGdl(); };

	kmwMatAxes& SetTitleX(const kmStrw& title) { _xtitle = title; UpdateAxesCwp(); _wmat.Relocate(_win.w, _win.h); InvalidateAll(); return *this; };
	kmwMatAxes& SetTitleY(const kmStrw& title) { _ytitle = title; UpdateAxesCwp(); _wmat.Relocate(_win.w, _win.h); InvalidateAll(); return *this; };

	kmwMatAxes& SetGdlXOn() { _wmat.SetGdlXOn(); return *this; };
	kmwMatAxes& SetGdlYOn() { _wmat.SetGdlYOn(); return *this; };

	kmwMatAxes& SetGdlX(float x) { _wmat.SetGdlX(x); return *this; };
	kmwMatAxes& SetGdlY(float y) { _wmat.SetGdlY(y); return *this; };

	// add gobj to kmwMat
	void Add(kmgObj* const gob) { _wmat.Add(gob); };

	// set axis equal
	kmwMatAxes& SetAxisEqual(int mode = 1)
	{
		if(mode == 1) { _mpop(0).SetCheck(1); _wmat.SetAxisEqual(1);}
		else          { _mpop(0).SetCheck(0); _wmat.SetAxisEqual(0);}
		return *this;
	};

	// set frame index
	kmwMatAxes& SetFrame(int frame_idx) { _wmat.SetFrame(frame_idx); return *this; };

	// get _wmat
	kmwMat& GetWmat() { return _wmat; };

	// get frame index
	int64 GetFrameIdx() const { return _wmat.GetFrameIdx(); };

	// get range of cmap
	float GetRangeCMax() const { return _wmat.GetRangeCMax(); };
	float GetRangeCMin() const { return _wmat.GetRangeCMin(); };

	// calc cwp of _wmat as title and font size
	kmCwp CalcWmatCwp()
	{
		// get font size
		const float h = (float) _font.GetH();

		// get title info
		const float b_xttl = (_xtitle.N() == 0) ? 0:1.2f;
		const float b_yttl = (_ytitle.N() == 0) ? 0:1.2f;

		// calc gap		
		const int gap_l = int( h*(b_yttl + 2.2f));
		const int gap_b = int( h*(b_xttl + 1.0f));

		return kmCwp(kmC(0,gap_l),kmC(0,16),kmC(1,-16),kmC(1,-gap_b));
	};
};

// graph window
class kmwGraph : public kmwChild
{
public:
	kmwAxes   _axes;
	kmFont    _font;
	kmStrw    _xtitle, _ytitle;
	kmgAxis   _xaxis,  _yaxis;
	kmwMenu   _mpop;

	int       _scale = 0;   // 0: linear scale, 1: normalized dB scale

private: // hidden memebers
	kmMat1<kmMat1f32> __y0; // for scale conversion

	int       __mpop_n0;    // number of init mpop's memebers (including separator)
	int       __mpop_n1;    // number of end-items of init mpop's membersend items (excluding separator)
		
	//////////////////////////////////////
	// fucntions to create window
public:
	void Create(int x, int y, int w, int h, kmWnd* pwnd) { kmwChild::Create(x, y, w, h, pwnd);};
	void Create(kmCwp cwp,                  kmWnd* pwnd) { kmwChild::Create(cwp,        pwnd);};

	/////////////////////////////////////////////
	// function to create child 
protected:
	virtual void CreateChild()
	{
		// create axes
		_axes.Clear();
		_axes.Create(CalcAxesCwp(), this, 1);

		// set graphical objects
		_xaxis.Set(0, &_font);
		_yaxis.Set(1, &_font);

		// create popup memu
		_mpop.Clear();
		_mpop.Add(L"grid-x on")    .Add(L"grid-y on").AddSeparator();
		_mpop.Add(L"scale: linear").Add(L"scale: dB").AddSeparator();
		_mpop(3).SetCheck(1);

		__mpop_n0 = (int)_mpop.N();
		__mpop_n1 = (int)_mpop.GetTotalNPre();
	};

	virtual void InvalidateChild() { _axes.InvalidateAll(); };

	//////////////////////////////////////////////
	// window proceduer
protected:
	virtual kmePPType OnKeyDown(WPARAM wp, LPARAM lp)
	{
		switch(wp)
		{
		case VK_SPACE : 			
			_axes.SetRangeFull(); _axes.Invalidate(); UpdateAxis(); Invalidate(); UpdateWindow(); 
			break;

		default : PassMsgToParent(WM_KEYDOWN, wp, lp);
		}
		return PP_FINISH;
	};

	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control
		uint ntf = HIWORD(wp), id  = LOWORD(wp);
		
		switch(id)
		{
		case 1: // _axes
			switch(ntf)
			{
			case WM_LBUTTONDOWN : SetFocus(); break; // for onkeydown
			case NTF_AXES_UPDATE : // update x-axis and y-axiss
				UpdateAxis();
				Invalidate();
				break;
			}
			break;
		// popup-menu
		case 2048 + 0: // grid-x on
			_mpop(0).SetCheck(2);
			_axes.SetGridXOn(_mpop(0).GetCheck());
			UpdateGrid();
			UpdateWindow();
			break;

		case 2048 + 1: // grid-y on
			_mpop(1).SetCheck(2);
			_axes.SetGridYOn(_mpop(1).GetCheck());
			UpdateGrid();			
			UpdateWindow();
			break;

		case 2048 + 2: // linear scale
			if(_scale != 0)
			{
				_scale = 0; _mpop(3).SetCheck(1); _mpop(4).SetCheck(0); UpdateScale();
				
				_axes.SetRangeFull(); UpdateAxis(); InvalidateAll(); UpdateWindow();
			}
			break;

		case 2048 + 3: // dB scale
			if(_scale != 1)
			{	
				_scale = 1; _mpop(3).SetCheck(0); _mpop(4).SetCheck(1); UpdateScale();

				_axes.SetRangeFull(); UpdateAxis(); InvalidateAll(); UpdateWindow(); 
			}
			break;
		}
		const int64 mpop_idx = id - 2048 - __mpop_n1 + __mpop_n0;
		const int64 line_idx = id - 2048 - __mpop_n1;

		// popup-menu for visible lines
		if(line_idx >= 0 && line_idx < _axes.GetN())
		{
			_mpop(mpop_idx).SetCheck(2);
			_axes.SetVisible(line_idx, _mpop(mpop_idx).GetCheck());
			UpdateAxis();
			Invalidate();
		}
		return PP_FINISH;
	};

	virtual void DrawGobjs(HDC hdc, int w, int h)
	{
		// init parameter
		kmRect rt = {0, 0, w, h}, axes_rt = _axes.GetRect();

		const int h_font = (int) (_font.GetH()*1.4f);
		
		// draw background
		FillRect(hdc, rt, (HBRUSH) GetStockObject(WHITE_BRUSH));

		// draw axis
		_xaxis.SetRect(axes_rt); _xaxis.Draw(hdc);
		_yaxis.SetRect(axes_rt); _yaxis.Draw(hdc);

		// draw x-title
		if(_xtitle.N() > 0)
		{
			kmRect ttl_rt = {0, h - h_font, w, h};

			kmgStr ttl; ttl.Set(&_font, kmRgb(0,0,0), ttl_rt, DT_BOTTOM | DT_CENTER | DT_SINGLELINE);

			ttl.SetStr(_xtitle);
			ttl.Draw(hdc);
		}
		// draw y-title
		if(_ytitle.N() > 0)
		{
			kmRect ttl_rt = {0, 0, h_font, h};

			_font.SetAngle(900);

			kmgStr ttl; ttl.Set(&_font, kmRgb(0,0,0), ttl_rt, DT_LEFT  | DT_VCENTER | DT_SINGLELINE);

			ttl.SetStr(_ytitle);
			ttl.Draw(hdc);

			_font.SetAngle(0);
		}
	};

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{
			_axes.Relocate(w, h);
			UpdateAxis();
		}
		return PP_FINISH;
	};

	virtual kmePPType OnContextMenu(WPARAM wp, LPARAM lp) // right mouse click
	{		
		_mpop.CreatePopup(_hwnd, lp);
		
		return PP_FINISH;
	};

	//////////////////////////////////////
	// member functions
private:
	void AddMpop(LPCWSTR str = nullptr, int state = 1)
	{	
		_mpop.Add((str == nullptr)? kmStrw(L"line %d",_axes.GetN()).P() : str);
		_mpop.End()->SetCheck((state > 0)? 1:0); 
	};

	// get unit-changed data.. linear or dB
	const kmMat1f32 Yc(const kmMat1f32& y)
	{
		if(_scale == 0) return        y;    // linear scale
		else            return dBnorm(y);   // dB scale
	};

	void AddY0(const kmMat1f32& y)
	{
		if(_scale > 0) { Lock(); if(__y0.N() == 0) __y0.Recreate(0,6); __y0.PushBack(y); Unlock(); }		
	};

	void UpdateY0(const kmMat1f32& y, int64 idx = 0) { if(_scale > 0) { Lock(); __y0(idx) = y; Unlock(); } };

public:
	void UpdateY(const kmMat1f32& y, int64 idx = 0) { UpdateY0(y, idx); _axes.UpdateY(Yc(y), idx); UpdateAxis(); };
	void UpdateX(const kmMat1f32& x, int64 idx = 0) {                   _axes.UpdateX(    x, idx); UpdateAxis(); };
		
	void Update(const kmMat1f32& x, const kmMat1f32& y, int64 idx = 0) { UpdateY0(y, idx); _axes.Update(x, Yc(y), idx); UpdateAxis(); };
	void Update(                    const kmMat1f32& y, int64 idx = 0) { UpdateY0(y, idx); _axes.Update(   Yc(y), idx); UpdateAxis(); };

	// add x,y data
	// * Note that you must call this function after the window is created
	void Add(const kmMat1f32& x, const kmMat1f32& y, kmRgb rgb = kmRgbB, LPCWSTR str = nullptr, int state = 1) 
	{
		AddY0(y); _axes.Add(x, Yc(y), rgb, state); UpdateAxis(); AddMpop(str, state);
	};
	void Add(                    const kmMat1f32& y, kmRgb rgb = kmRgbB, LPCWSTR str = nullptr, int state = 1) 
	{ 
		AddY0(y); _axes.Add(   Yc(y), rgb, state); UpdateAxis(); AddMpop(str, state);
	};

	void Erase(int64 idx) { _axes.Erase(idx); UpdateAxis(); };
	void Clear()          { _axes.Clear();    UpdateAxis(); };
	
	void SetFont(const kmStrw& str, int height, int bold = 0, uchar italic = 0, uchar underline = 0)
	{
		_font.Set(str, height, bold, italic, underline);
	};

	void SetFont(int height, int bold = 0, uchar italic = 0, uchar underline = 0)
	{
		_font.Set(height, bold, italic, underline);
	};

	kmwGraph& SetRangeX(float x_min, float x_max) { _axes.SetRangeX(x_min, x_max); UpdateAxis(); return *this; };
	kmwGraph& SetRangeY(float y_min, float y_max) { _axes.SetRangeY(y_min, y_max); UpdateAxis(); return *this; };
	
	kmwGraph& SetTitleX(const kmStrw& title) { _xtitle = title; UpdateAxesCwp(); _axes.Relocate(_win.w, _win.h); InvalidateAll(); return *this; };
	kmwGraph& SetTitleY(const kmStrw& title) { _ytitle = title; UpdateAxesCwp(); _axes.Relocate(_win.w, _win.h); InvalidateAll(); return *this; };
	
	kmwGraph& SetVisible(int64 idx = 0, bool on = true) { _axes.SetVisible(idx, on); return *this; };
	
	void UpdateAxesCwp()
	{
		_axes.SetCwp(CalcAxesCwp());
	};

	void UpdateAxis()
	{	
		_xaxis.UpdateTick(5, _axes._x_min, _axes._x_max);
		_yaxis.UpdateTick(5, _axes._y_min, _axes._y_max);
		UpdateGrid();
	};

	void UpdateGrid()
	{
		_axes.SetGridX(_xaxis._tick);
		_axes.SetGridY(_yaxis._tick);
		_axes.UpdateGrid();
	};

	void UpdateScale()
	{
		kmLockGuard grd = Lock(); //===========================lock-unlock

		// init parameters
		const int64 n = GetN();

		if(_scale == 0) // linear
		{
			ASSERTFA(n == __y0.N(), "[kmwGraph] in 3841");

			for(int64 i = 0; i < n; ++i) _axes.UpdateY(__y0(i),i);
		}
		else // dB
		{
			__y0.Recreate(n);

			for(int64 i = 0; i < n; ++i)
			{
				__y0(i) = _axes._y(i); _axes.UpdateY(Yc(_axes._y(i)), i);
			}
		}
	};

	// calc cwp of _wmat as title and font size
	kmCwp CalcAxesCwp()
	{
		// get font size
		const float h = (float) _font.GetH();

		// get title info
		const float b_xttl = (_xtitle.N() == 0) ? 0:1.2f;
		const float b_yttl = (_ytitle.N() == 0) ? 0:1.2f;

		// calc gap		
		const int gap_l = int( h*(b_yttl + 3.2f));
		const int gap_b = int( h*(b_xttl + 1.0f));

		return kmCwp(kmC(0,gap_l),kmC(0,16),kmC(1,-16),kmC(1,-gap_b));
	};

	// get number of axes
	int64 GetN() { return _axes.GetN(); };
};

// image viewer window
class kmwImgView : public kmwChild
{
public:
	kmeImgType _type;                      // img_cmap, img_rgb
	kmMat3f32  _data;                      // for img_cmap
	kmImg      _imgs;                      // for img_rgb
	kmCMap     _cmap   = kmCMap(CMAP_JET); // color map

	float      _cmin   = 0.f;              // min of colormap
	float      _cmax   = 1.f;              // max of colormap
	int        _fidx   = 0;                // frame index
	int        _pstate = 0;                // play state

	// child window
	kmwImg   _wimg;	
	kmwBtn   _btn1, _btn2, _btn3, _btn4, _btn5;
	kmwStt   _stt1, _stt2, _stt3;
	kmwEdit  _edt1;	
	kmwMenu  _mpop;

	// constructor
	kmwImgView() {};

	///////////////////////////////////////////////	
	// functions for creating

	// create window with colormap type
	template<typename T>
	void Create(kmCwp cwp, const kmMat2<T>& mat, T min_v, T max_v, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		// update image
		Update(mat, min_v, max_v);

		// create window
		kmwChild::Create(cwp, pwnd, id, style);
	}

	template<typename T>
	void Create(kmCwp cwp, const kmMat3<T>& mat, T min_v, T max_v, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		// update image
		Update(mat, min_v, max_v);

		// create window
		kmwChild::Create(cwp, pwnd, id, style);
	}

	// create window with rgb type 
	void Create(kmCwp cwp, const kmImg& imgs, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		// update image
		Update(imgs);

		// create window
		kmwChild::Create(cwp, pwnd, id, style);
	};

	// create window without updating image
	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		kmwChild::Create(cwp, pwnd, id, style);
	};

	// update image
	void UpdateImg() { _wimg.Update(GetImg(_fidx));	};

	// update images of rgb type
	void Update(const kmImg& imgs) { _type = IMG_RGB; _imgs = imgs; UpdateImg(); };

	// update images of colormap type
	template<typename T> void Update(const kmMat3<T>& mat) { _type = IMG_CMAP; _data = mat; UpdateImg(); }
	template<typename T> void Update(const kmMat2<T>& mat) { Update(kmMat3<T>(mat)); }

	template<typename T> void Update(const kmMat3<T>& mat, T min_v, T max_v) { _cmin = min_v;  _cmax = max_v; Update(mat); }
	template<typename T> void Update(const kmMat2<T>& mat, T min_v, T max_v) { _cmin = min_v;  _cmax = max_v; Update(mat); }

	// update range of cmap type
	template<typename T> void UpdateRange(T min_v, T max_v)
	{
		_type = IMG_CMAP;  _cmin = min_v;  _cmax = max_v;
		UpdateImg(); 
		UpdateInfoRange(); 
	}

	// update cmap
	void UpdateCMap(kmeCMapType cmtype)	{ _cmap.Create(cmtype); };

	// add a gobject
	void Add(kmgObj* const gob) { _wimg.Add(gob); };

	/////////////////////////////////////////////
	// function to create child 
protected:
	virtual void CreateChild()
	{
		// create popup-menu
		_mpop   .Add(L"list1");
		_mpop   .Add(L"list2");
		_mpop   .Add(L"list3");
		_mpop(1).Add(L"list21");
		_mpop(1).Add(L"list22");

		// create window image		
		_wimg.Create(kmCwp(CWP_LRTB, 10, 10, 10, 40), GetImg(0), this, 1);		

		// create button
		_btn1.Create(kmCwp(CWP_WHLB, 25, 30,  30, 5), L"¢¸", this, 2);
		_btn2.Create(kmCwp(CWP_WHLB, 25, 30, 120, 5), L"¢º", this, 3);
		_btn3.Create(kmCwp(CWP_WHLB, 25, 30,   5, 5), L"¡ì", this, 4);
		_btn4.Create(kmCwp(CWP_WHLB, 25, 30, 145, 5), L"¡í", this, 5);
		_btn5.Create(kmCwp(CWP_WHLB, 25, 30, 170, 5), L"¡á", this, 6);

		// create edit
		_edt1.Create(kmCwp(CWP_WHLB, 30, 23, 50, 5), L"1", this, 7, ES_RIGHT | ES_NUMBER);		
		_edt1.LimitText(4);

		// create static text
		_stt1.SetStr(L"/ %lld", _data.N3());
		_stt1.GetFont().SetH(20);		
		_stt1.Create(kmCwp(CWP_WHLB, 35, 30, 85, 5), this, 8);

		_stt2.GetFont().SetH(20);
		_stt2.Create(kmCwp(CWP_WHLB, 160, 30, 210, 5), this, 9);

		_stt3.GetFont().SetH(20);
		_stt3.Create(kmCwp(CWP_WHLB, 180, 30, 380, 5), this, 10);

		UpdateInfoRange();
	};

	////////////////////////////////////////////////
	// window procedures

	virtual kmePPType OnKeyDown(WPARAM wp, LPARAM lp)
	{
		switch(wp)
		{
		case VK_LEFT  : DecFrame(); break;
		case VK_RIGHT : IncFrame(); break;
		case VK_UP    : IncRange(); break;
		case VK_DOWN  : DecRange(); break;
		default       : PassMsgToParent(WM_KEYDOWN, wp, lp);
		}
		return PP_FINISH;
	};

	virtual kmePPType OnContextMenu(WPARAM wp, LPARAM lp) // right mouse click
	{		
		_mpop.CreatePopup(_hwnd, lp);
		
		return PP_FINISH;
	};

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{			
			_btn1.Relocate(w, h);
			_btn2.Relocate(w, h);
			_btn3.Relocate(w, h);
			_btn4.Relocate(w, h);
			_btn5.Relocate(w, h);
			_wimg.Relocate(w, h);
			_stt1.Relocate(w, h);
			_stt2.Relocate(w, h);
			_stt3.Relocate(w, h);
			_edt1.Relocate(w, h);
		}
		return PP_FINISH;
	};

	virtual kmePPType OnTimer(WPARAM wp, LPARAM lp)
	{
		const int timer_id = (int) wp;

		switch(timer_id)
		{
		case 1: // play
			IncFrame();
			if(_fidx == _data.N3()-1) { KillTimer(1); _pstate = 0;}
			break;

		case 2: // rewind
			DecFrame();
			if(_fidx == 0) { KillTimer(2); _pstate = 0;}
			break;
		}
		return PP_FINISH;
	};

	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control
		ushort ntf = HIWORD(wp), id = LOWORD(wp);

		switch(id)
		{
		case 1: // _wimg
			switch(ntf)
			{
			case NTF_IMG_POSCLICK  : NotifyPosClick(lp); break;
			case NTF_IMG_MOUSEMOVE : UpdateInfoPos(LOWORD(lp), HIWORD(lp)); break;
			}
			break;
		// km controls
		case 2: DecFrame   (); UpdateWindow(); break;
		case 3: IncFrame   (); UpdateWindow(); break;
		case 4: RewindFrame(); break;
		case 5: PlayFrame  (); break;
		case 6: StopFrame  (); break;

		// conrols provided by ms
		case 7: if(_edt1.ProcNtf(ntf)) GotoFrame(); break;
					
		// popup-menu
		case 2048 +  0: PRINTFA("%s : list00\n", GetKmClass()); break;
		case 2048 +  1: PRINTFA("%s : list01\n", GetKmClass()); break;
		case 2048 +  2: PRINTFA("%s : list02\n", GetKmClass()); break;
		case 2048 +  3: PRINTFA("%s : list03\n", GetKmClass()); break;
		}
		return PP_FINISH;
	};

	////////////////////////////////////////////
	// notification functions

	void NotifyPosClick(LPARAM lp)
	{
		SendMsgToParent(WM_COMMAND, MAKELONG(GetId(), NTF_IMGVIEW_POSCLICK), lp);
	};

	//////////////////////////////////////////////
	// inner member functions 
protected:
	void UpdateFrame(int fidx)
	{
		_fidx = MIN(MAX(0, fidx), (int) _data.N3()-1);

		_edt1.SetStr(L"%lld", _fidx + 1); 		
	};

	void GotoFrame()
	{
		int idx   = _edt1.GetNum() - 1;
		_fidx     = MIN(MAX(0, idx), (int)_data.N3()-1);
		
		if(idx != _fidx) _edt1.SetStr(L"%lld", _fidx + 1);
		else             UpdateImg();
	};

	void IncFrame()
	{		
		if(_fidx < _data.N3()-1) UpdateFrame(++_fidx);
	};

	void DecFrame()
	{
		if(_fidx > 0) UpdateFrame(--_fidx);
	};
	
	void PlayFrame()
	{
		if(_pstate < 0) { KillTimer(2); _pstate = 0; }
		
		SetTimer(1, 100 / (uint) pow(2, (++_pstate)));
	};

	void RewindFrame()
	{
		if(_pstate > 0) { KillTimer(1); _pstate = 0; }
		
		SetTimer(2, 100 / (uint) pow(2, abs(--_pstate)));
	};

	void StopFrame()
	{
		if     (_pstate > 0) KillTimer(1);
		else if(_pstate < 0) KillTimer(2);
		_pstate = 0;
	};	

	kmImg GetImg(int fidx)
	{
		kmImg img;

		switch(_type)
		{
		case IMG_NULL:
			img.Create(1,1,1); *img.Begin() = 0;
			break;

		case IMG_CMAP:
			if(_data.Size() > 0)
			{
				img.CreateP(_data.N1(), _data.N2(), 1, _data.P1());
				img.ConvertBgr(_data.Mat2(fidx), _cmin, _cmax, _cmap);
			}
			else
			{
				img.Create(1,1,1); *img.Begin() = 0; 
			}
			break;

		case IMG_RGB:
			img = _imgs.GetFrame(fidx);
			break;
		}
		return img;
	};

	void IncRange()
	{
		_cmin *= 1.1f; _cmax *= 1.1f;
		UpdateImg();
		UpdateInfoRange();
	};

	void DecRange()
	{
		_cmin /= 1.1f; _cmax /= 1.1f;
		UpdateImg();
		UpdateInfoRange();
	};
	
	void UpdateInfoRange()
	{
		_stt2.SetStr(L"CMap: %.2f | %.2f ", _cmin, _cmax);
		_stt2.Invalidate();
	};
		
	void UpdateInfoPos(int x, int y)
	{
		if(_type == IMG_CMAP)
		{
			float v = _data(x, y, _fidx);
			_stt3.SetStr(L"Pt(%d, %d): %.2f", x, y, v);
		}
		else
		{
			kmRgb rgb = _imgs(x, y, _fidx);
			_stt3.SetStr(L"Pt(%d, %d): %d %d %d", x, y, rgb._r, rgb._g, rgb._b);
		}
		_stt3.Invalidate();
	};
};

// image viewer window
class kmwMatView : public kmwChild
{
public:	
	kmMat4f32  _mat;
	kmCMap     _cmap = kmCMap(CMAP_GREY); // color map
	float      _cmin = 0.f;               // min of colormap
	float      _cmax = 1.f;               // max of colormap
	int64      _i[4] = {0,};              // index of matrix
	
	// child window
	kmwImg   _wimg;	
	kmwStt   _stt[3]; // info
	kmwBtn   _btn[6]; // 1st, 2nd, 3rd, 4th index, low range, high range
	kmwMenu  _mpop;

	// constructor
	kmwMatView() {};

	///////////////////////////////////////////////	
	// function for creating
		
	// create window without updating image
	void Create(kmCwp cwp, kmWnd* pwnd, ushort id = 0, DWORD style = 0)
	{
		kmwChild::Create(cwp, pwnd, id, style);
	};
	
	// add a gobject
	void Add(kmgObj* const gob) { _wimg.Add(gob); };

	// update matrix
	void UpdateMat() 
	{
		// set info
		UpdateInfo();
		
		// update rnage
		UpdateRange(_mat.Min(), _mat.Max());

		// update index
		UpdateIdx(0,0,0,0);

		// updaet value
		UpdateVal();

		// update image
		UpdateImg();
	};

	// update cmap
	void UpdateCMap(kmeCMapType cmtype)	{ _cmap.Create(cmtype); };

	// update image
	void UpdateImg()
	{
		// update image
		_wimg.Update(GetImg()); _wimg.Invalidate();
	};

	/////////////////////////////////////////////
	// function to create child 
protected:
	virtual void CreateChild()
	{
		// create popup-menu
		_mpop.Add(L"list1");
		_mpop.Add(L"list2");
		
		// create window image		
		_wimg.Create(kmCwp(CWP_LRTB, 10, 10, 10, 40), GetImg(), this, 1);
		
		// create button
		int w = 45, x = 10;

		_stt[0].Create(kmCwp(CWP_WHLB, 150, 25, x, 5), this, 8);

		_btn[0].Create(kmCwp(CWP_WHLB, w, 25, x+=150, 5), this, 2); // 1st idx
		_btn[1].Create(kmCwp(CWP_WHLB, w, 25, x+=w  , 5), this, 3); // 2nd idx
		_btn[2].Create(kmCwp(CWP_WHLB, w, 25, x+=w  , 5), this, 4); // 3rd idx
		_btn[3].Create(kmCwp(CWP_WHLB, w, 25, x+=w  , 5), this, 5); // 4th idx
		
		_stt[1].Create(kmCwp(CWP_WHLB, 80, 25, x+=w,  5), this, 9);
		_stt[2].Create(kmCwp(CWP_WHLB, 50, 25, x+=80, 5), L"Range:",this, 10);

		_btn[4].Create(kmCwp(CWP_WHLB, w, 25, x+=50, 5), this, 6); // low  range
		_btn[5].Create(kmCwp(CWP_WHLB, w, 25, x+=w , 5), this, 7); // high range
				
		// set font size
		for(int i = 0; i < numof(_btn); ++i) _btn[i].GetFont().SetH(20);
		for(int i = 0; i < numof(_stt); ++i) _stt[i].GetFont().SetH(20);
	};

	////////////////////////////////////////////////
	// window procedures
	
	virtual kmePPType OnContextMenu(WPARAM wp, LPARAM lp) // right mouse click
	{		
		_mpop.CreatePopup(_hwnd, lp);
		
		return PP_FINISH;
	};

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{			
			for(int i = 0; i < numof(_btn); ++i) _btn[i].Relocate(w, h);
			for(int i = 0; i < numof(_stt); ++i) _stt[i].Relocate(w, h);
			_wimg.Relocate(w, h);			
		}
		return PP_FINISH;
	};
		
	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control
		ushort ntf = HIWORD(wp), id = LOWORD(wp);

		switch(id)
		{
		case 1: // _wimg
			switch(ntf)
			{
			case NTF_IMG_POSCLICK  : NotifyPosClick(lp); break;
			case NTF_IMG_MOUSEMOVE : 
				UpdateIdx(LOWORD(lp), HIWORD(lp), _i[2], _i[3]); 
				UpdateVal();				
				break;
			}
			break;
		// km controls
		case 2: // _btn[0]
		case 3: // _btn[1]
		case 4: // _btn[2]
		case 5: // _btn[3]
			if(ntf == NTF_BTN_MOUSEWHEEL)
			{
				_i[id-2] += ((short)HIWORD(lp))/120; 
				UpdateIdx();
				UpdateVal();
				UpdateImg();
			}
			else if(ntf == NTF_BTN_CLICKED)
			{
				_mat.Mat2(_i[2],_i[3]).PrintMat();
			}
			break;		
		case 6: // _btn[4]
			if(ntf == NTF_BTN_MOUSEWHEEL)
			{	
				float del = (0.05f/120.f)*short(HIWORD(lp));
				if(fabs(_cmin) < 0.05f) del = (del > 0) ? 0.1f:-0.1f;
				else                    del *= fabs(_cmin);
				UpdateRange(_cmin + del, _cmax); 
				UpdateImg();
			}
			break;
		case 7: // _btn[5]
			if(ntf == NTF_BTN_MOUSEWHEEL)
			{
				float del = (0.05f/120.f)*short(HIWORD(lp));
				if(fabs(_cmax) < 0.05f) del = (del > 0) ? 0.1f:-0.1f;
				else                    del *= fabs(_cmax);
				UpdateRange(_cmin, _cmax + del); 
				UpdateImg();				
			}
			break;
								
		// popup-menu
		case 2048 +  0: PRINTFA("%s : list00\n", GetKmClass()); break;
		case 2048 +  1: PRINTFA("%s : list01\n", GetKmClass()); break;
		}
		return PP_FINISH;
	};

	////////////////////////////////////////////
	// notification functions

	void NotifyPosClick(LPARAM lp)
	{
		SendMsgToParent(WM_COMMAND, MAKELONG(GetId(), NTF_IMGVIEW_POSCLICK), lp);
	};

	//////////////////////////////////////////////
	// inner member functions 
protected:

	// update index
	void UpdateIdx(int64 i1, int64 i2, int64 i3, int64 i4)
	{		
		_i[0] = i1; _i[1] = i2; _i[2] = i3; _i[3] = i4; UpdateIdx();
	};

	void UpdateIdx()
	{
		// check range and update btn
		const int64 n[4] = { _mat.N1(), _mat.N2(), _mat.N3(), _mat.N4() };

		for(int i = 0; i < 4; ++i)
		{
			_i[i] = MIN(MAX(0, _i[i]),n[i]-1);
			_btn[i].SetStr(L"%d", _i[i]);
			_btn[i].Invalidate();
		}
	};

	// update info of matrix
	void UpdateInfo()
	{
		_stt[0].SetStr(L"Mat(%d, %d, %d, %d) :", _mat.N1(), _mat.N2(), _mat.N3(), _mat.N4());
		_stt[0].Invalidate();
	};

	// update value of matrix
	void UpdateVal()
	{
		_stt[1].SetStr(L"[%.2f]", _mat(_i[0], _i[1], _i[2], _i[3]));
		_stt[1].Invalidate();
	};

	// update range
	void UpdateRange(float cmin, float cmax)
	{
		// check range and update
		_cmin = MIN(cmin, _cmax-0.01f); _cmax = MAX(_cmin+0.01f, cmax);
				
		// update btn
		_btn[4].SetStr(L"%.2f", _cmin); _btn[4].Invalidate();
		_btn[5].SetStr(L"%.2f", _cmax); _btn[5].Invalidate();
	};

	// get image
	kmImg GetImg()
	{
		kmImg img;

		if(_mat.N() == 0) { img.Create(1,1); img.SetZero();}
		else
		{
			img.CreateP(_mat.N1(), _mat.N2(), 1, _mat.P1());
			img.ConvertBgr(_mat.Mat2(_i[2], _i[3]), _cmin, _cmax, _cmap);
		}
		return img;
	};
};

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
// kmWnd example class

// message box
class kmWndMsgBox : public kmWnd
{
public:
	// child window
	kmwBox _box;

	///////////////////////////////
	// interface functions

	// create static window 
	static kmWndMsgBox* Create(kmStrw str, kmStrw name = L"message")
	{
		kmWndMsgBox* msgbox = new kmWndMsgBox;
		msgbox->Create(name.P(), 1);
		msgbox->WaitCreated();
		msgbox->Add(str).UpdateWindow();

		return msgbox;
	};

	// create window
	void Create(LPCTSTR win_name = NULL, uint mode = 0)
	{	
		if(!IsCreated()) CreateThread(400, 400, 600, 200, win_name, mode);
		WaitCreated();
	};

	// update str
	kmWndMsgBox& Add(const kmStrw& str)
	{
		_box.Add(str).SetViewPosBottom().Invalidate();
		return *this;
	};

	///////////////////////////////////
	// inner window functions
protected:
	// create child window
	virtual void CreateChild()
	{
		_box.SetH(18).SetBkRgb(kmRgb(250,240,240)).SetLineSpace(3);
		_box.Create(kmCwp(kmC(0,0),kmC(0,0),kmC(1,0),kmC(1,0)), this);
	}

	// window procedure... wm_size
	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{
			_box.Relocate(w,h);
		}
		return PP_FINISH;
	};
};

// matrix viewer... new version since 2021.9.17
class kmWndMat3f : public kmWnd
{
public:
	kmMat3f32 _mat;

	int      _i1 = 0;  // index of 1st dim
	int      _i2 = 0;  // index of 2nd dim
	int      _i3 = 0;  // index of 3rd dim

	// child window
	kmwMatAxes _waxes;
	kmwTable   _wtbl;
	kmwMenu    _menu;

	// constructor
	kmWndMat3f() {};

	//////////////////////////////////////
	// functions for creating

	// create window 	
	void Create(LPCTSTR win_name = NULL, uint mode = 0)
	{
		CreateThread(100, 100, 800, 600, win_name, mode);
	};

	void Create(int x, int y, int w, int h, LPCTSTR win_name = NULL, uint mode = 0)
	{
		CreateThread(x, y, w, h, win_name, mode);
	};

	// static fucntion to create window	
	static kmWndMat3f* Image(const kmMat3f32& mat, LPCTSTR name, bool iscopy = 0)
	{
		kmWndMat3f *wnd = new kmWndMat3f;
		wnd->Create(name, 1);
		wnd->WaitCreated();
		wnd->UpdateMat(mat, iscopy);		
		wnd->UpdateWindow();
		return wnd;
	};

	//////////////////////////////////////
	// functions for updating

	void UpdateMat(const kmMat3f32& mat, bool iscopy = 0)
	{
		// update waxes
		if(iscopy) _mat = mat; else _mat.Set(mat);

		_waxes.UpdateMat(_mat);
		_waxes.UpdateRangeCAuto();
		_waxes.UpdateImg();

		// update wtbl
		UpdateTbl();
	};
	
	/////////////////////////////////////////////
	// function to create child 
protected:
	virtual void CreateChild()
	{
		// create menu
		_menu.Add(L" File ");
		_menu.Add(L" Edit ");
		_menu.Add(L" Option ");
		_menu(0).Add(L"load mat data");
		_menu(0).Add(L"save mat data");
		_menu(1).Add(L"copy image");
		_menu(2).Add(L"set transpose");

		_menu.Create(this->_hwnd, 1024);

		// init parameters
		const float rat = 0.75f;

		// create axes
		_waxes.Create(kmCwp(5,5,kmC(rat,-5),kmC(1,-5)), this, 0);
		
		_waxes.SetTitleX(L"2nd dimension").SetTitleY(L"1st dimension").SetGdlXOn().SetGdlYOn();

		// create table
		_wtbl.Create(kmCwp(kmC(rat,5),10,kmC(1,-5),kmC(1,-10)), this, 1);

		_wtbl.CreateItm(9,2);
		_wtbl.SetFontH(15);
		_wtbl.SetTwoTone(kmRgb(230,240,250), kmRgb(245,250,255));
	};

	///////////////////////////////////////////////
	// window procedures

	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control
		ushort ntf = HIWORD(wp), id = LOWORD(wp);

		switch(id)
		{
		// notification
		case 0: // _wmataxes
			switch(ntf)
			{
			case NTF_AXES_UPDATE : // update axes
				
				_i3 = (int)_waxes.GetFrameIdx();
				
				UpdateTbl();				
				UpdateWindow();
				break;
			case NTF_WMAT_DBLCLK : // double-click in wmat
				short i1 = HIWORD(lp), i2 = LOWORD(lp);

				_i1 = MIN(MAX(0, i1), (int)_mat.N1() - 1);
				_i2 = MIN(MAX(0, i2), (int)_mat.N2() - 1);

				UpdateGdl();
				UpdateTbl();
				UpdateWindow();
				break;
			}
			break;
		}
		return PP_FINISH;
	};

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{
			_waxes  .Relocate(w, h);			
			_wtbl   .Relocate(w, h);
		}
		return PP_FINISH;
	};

	virtual kmePPType OnKeyDown(WPARAM wp, LPARAM lp)
	{
		switch(wp)
		{
		case VK_LEFT     : DecIdx2(); break;
		case VK_RIGHT    : IncIdx2(); break;
		case VK_UP       : DecIdx1(); break;
		case VK_DOWN     : IncIdx1(); break;
		}
		return PP_FINISH;
	};

	//////////////////////////////////////////////
	// inner member functions 
protected:
	// increase and decrease index
	void IncIdx1() { if(_i1 >= _mat.N1() - 1) return; ++_i1; UpdateGdl(); UpdateTbl(); UpdateWindow(); };
	void DecIdx1() { if(_i1 <  1            ) return; --_i1; UpdateGdl(); UpdateTbl(); UpdateWindow(); };
	void IncIdx2() { if(_i2 >= _mat.N2() - 1) return; ++_i2; UpdateGdl(); UpdateTbl(); UpdateWindow(); };
	void DecIdx2() { if(_i2 <  1            ) return; --_i2; UpdateGdl(); UpdateTbl(); UpdateWindow(); };
	void IncIdx3() { if(_i3 >= _mat.N3() - 1) return; ++_i3;              UpdateTbl(); UpdateWindow(); };
	void DecIdx3() { if(_i3 <  1            ) return; --_i3;              UpdateTbl(); UpdateWindow(); };

	// update gdlx and gdly	
	void UpdateGdl () { _waxes.SetGdlY((float)_i1).SetGdlX((float)_i2).UpdateGdl(); };

	// update wtalbe
	void UpdateTbl()
	{
		_wtbl(0,0).SetStr(kmStrw(L"1st number"));
		_wtbl(1,0).SetStr(kmStrw(L"2nd number"));
		_wtbl(2,0).SetStr(kmStrw(L"3rd number"));
		_wtbl(3,0).SetStr(kmStrw(L"1st index"));
		_wtbl(4,0).SetStr(kmStrw(L"2nd index"));
		_wtbl(5,0).SetStr(kmStrw(L"3rd index"));
		_wtbl(6,0).SetStr(kmStrw(L"value"));
		_wtbl(7,0).SetStr(kmStrw(L"cmap min"));
		_wtbl(8,0).SetStr(kmStrw(L"cmap max"));

		_wtbl(0,1).SetStr(kmStrw(L"%d"  , _mat.N1()));
		_wtbl(1,1).SetStr(kmStrw(L"%d"  , _mat.N2()));
		_wtbl(2,1).SetStr(kmStrw(L"%d"  , _mat.N3()));
		_wtbl(3,1).SetStr(kmStrw(L"%d"  , _i1));
		_wtbl(4,1).SetStr(kmStrw(L"%d"  , _i2));
		_wtbl(5,1).SetStr(kmStrw(L"%d"  , _i3));
		_wtbl(6,1).SetStr(kmStrw(L"%f"  , _mat(_i1,_i2,_i3)));
		_wtbl(7,1).SetStr(kmStrw(L"%.2f", _waxes.GetRangeCMin()));
		_wtbl(8,1).SetStr(kmStrw(L"%.2f", _waxes.GetRangeCMax()));

		_wtbl.UpdateItm();
	};
};

// matrix viewer
template<typename T>
class kmWndMatView : public kmWnd
{
public:
	kmMat1<kmMat4<T>> _mat;
	kmMat1<kmStrw>    _name;
	int64             _idx = 0;

	// child window
	kmwMatView _view;
	kmwMenu    _menu;
		
	// constructor
	kmWndMatView() { _mat.Create(0,16); _name.Create(0,16); };

	//////////////////////////////////////
	// functions for creating

	// create window 	
	void Create(uint mode = 0)
	{
		// create window
		CreateThread(100, 100, 600, 600, L"Mat Viewer", mode);
	};

	void Create(int x, int y, int w, int h, LPCTSTR win_name = NULL, uint mode = 0)
	{
		CreateThread(x, y, w, h, win_name, mode);
	};

	// static fucntion to create window
	template<template<typename> class M>
	static kmWndMatView<T>* Image(const kmMat1<M<T>>& mat, LPCWSTR name, bool iscopy = 0)
	{
		kmWndMatView<T> *wnd = new kmWndMatView<T>;

		for(int64 i = 0; i < mat.N(); ++i)
		{	
			if(mat(i).N() > 0) wnd->Add(mat(i), kmStrw(L"%s(%d)",name,i), iscopy);
		}
		wnd->Create(1);
		return wnd;
	}

	// static fucntion to create window
	template<template<typename> class M>
	static kmWndMatView<T>* Image(const M<T>& mat, LPCWSTR name, bool iscopy = 0)
	{
		kmWndMatView<T> *wnd = new kmWndMatView<T>;	

		wnd->Add(mat, name, iscopy);
		wnd->Create(1);
		return wnd;
	}

	//////////////////////////////////////
	// functions for updating
	void Add(const kmMat4<T>& mat, LPCWSTR name, bool iscopy = 0)
	{
		// check size
		if(mat.N() == 0 ) return;

		// set or copy matrix
		if(iscopy)   _mat.PushBack   (mat);
		else         _mat.PushBackSet(mat);

		// set name
		_name.PushBack(name);

		// set index
		_idx = _mat.N()-1;

		// update view
		UpdateView();
	};

	void Add(const kmMat3<T>& mat, LPCWSTR name) { kmMat4<T> mat4(mat); Add(mat4, name); };
	void Add(const kmMat2<T>& mat, LPCWSTR name) { kmMat4<T> mat4(mat); Add(mat4, name); };

	void UpdateView()
	{
		_view._mat.Set(_mat(_idx));
		_view.UpdateMat();
	};

	///////////////////////////////////////////////
	// function to create child
protected:
	virtual void CreateChild()
	{
		// create menu
		_menu   .Add(L"File");
		_menu   .Add(L"Edit");
		_menu   .Add(L"Mat");
		_menu(0).Add(L"Save as bmp");
		_menu(0).Add(L"Save as");
		_menu(0).AddSeparator();
		_menu(1).Add(L"Copy");
		_menu(1).Add(L"RTM on"); // real time monitoring

		for(int i = 0; i < _name.N(); ++i) _menu(2).Add(_name(i));

		_menu(2).End()->SetCheck(1);
		
		_menu.Create(this->_hwnd, 1024);

		// create image viewer
		_view.Create(kmCwp( CWP_RAT, 0.f, 0.f, 1.f, 1.f), this);

		// update window name
		UpdateWinName();
	};

	///////////////////////////////////////////////
	// window procedures

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{			
			_view.Relocate(w, h);
		}
		return PP_FINISH;
	};

	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control
		ushort ntf = HIWORD(wp), id = LOWORD(wp);

		switch(id)
		{			
		// menu
		case 1024 +  0: SaveAsBmp(); break; // file-save as bmp
		case 1024 +  1: SaveAs();    break; // file-save as
		case 1024 +  2: Copy  ();    break; // edit-copy
		case 1024 +  3: RtmOn();     break; // real-time monitoring
		}

		const int64 idx = (int64) id - _menu(2)(0).GetId(); 
		if( 0 <= idx && idx < _menu(2).N() )
		{
			// update menu's check state
			_menu(2).SetCheck(0);
			_menu(2)(idx).SetCheck(1);

			// update matrix
			_idx = idx;

			UpdateWinName();
			UpdateView();
		}
		return PP_FINISH;
	};

	virtual kmePPType OnTimer(WPARAM wp, LPARAM lp)
	{
		const int timer_id = (int) wp;

		switch(timer_id)
		{
		case 1: // play
			_view.UpdateImg();
			_view.UpdateWindow();
			break;
		}
		return PP_FINISH;
	};

	// menu procedures
	void UpdateWinName()
	{
		SetWinStr(L"Mat Viewer : %s",_menu(2)(_idx)._str.P());
	};

	void RtmOn()
	{
		_menu(1)(1).SetCheck(2);

		if(_menu(1)(1).GetCheck()) SetTimer (1, 100);
		else                       KillTimer(1);
	};

	void SaveAsBmp()
	{
		// get save file name
		wchar file_name[256] = L"";
		wchar filter   [256] = L"bmp files (*.bmp)\0*.bmp\0";
		BOOL  res =  kmFile::GetFileNameS(GetHwnd(), file_name, NULL, filter);
		
		// capture and save a image
		if(res) kmFile::WriteDib(_view._wimg._img, file_name);
	};

	void SaveAs()
	{
		// get save file name
		wchar file_name[256] = L"";
		wchar filter   [256] = L"bmp file (*.bmp)\0*.bmp\0png file (*.png)\0*.png";
		int   index          = 0;
		BOOL  res = kmFile::GetFileNameS(GetHwnd(), file_name, &index, filter);

		// capture and save image
		if(res)	kmFile::WriteDib(_view._wimg._img, file_name);		
	};

	void Copy()	{ _view._wimg.CaptureCopy(); };
};

#define IMAGEMAT_F32(A) kmWndMatView<float>::Image(A,L""#A"",true)

// white board window
class kmWb : public kmWnd
{
public:
	kmgStrs  _strs;
	kmFonts  _fnts;
	kmRgb    _bk_rgb;

	// constructor
	kmWb() { _strs.Create(0, 4); _fnts.Create(0, 4); };
		
	//////////////////////////////////////////////
	// window proceduer
protected:
	virtual void DrawGobjs(HDC hdc, int w, int h)
	{
		// init parameter
		RECT rt = {0, 0, w, h};

		// draw background
		HBRUSH brush = CreateSolidBrush(_bk_rgb);

		FillRect(hdc, &rt, brush);

		DeleteObject(brush);

		// draw strings
		for(int i = 0; i < _strs.N(); ++i) _strs(i).Draw(hdc);
	};

	///////////////////////////////////////////////
	// functions to create window
public:	
	void Create(int x, int y, int w, int h, kmRgb bk_rgb = RGB_WHITE, uint mode = 0)
	{
		// set parameters
		_bk_rgb = bk_rgb;
		
		// modify window style
		_win.style    &= ~(WS_THICKFRAME | WS_SYSMENU | WS_CAPTION);
		_win.style    |=  (WS_POPUP); 
		_win.ex_style &= ~(WS_EX_APPWINDOW);
		_win.ex_style |=  (WS_EX_TOPMOST | WS_EX_TOOLWINDOW);

		// create window
		CreateThread(x, y, w, h, L"kmWb", mode);
	};

protected:
	virtual void CreateChild()
	{
		// create popup-menu
	};

	///////////////////////////////////////////////
	// member functions
public:
	int Add(kmgStr& str, int fnt_id = -1)
	{
		if(fnt_id >= 0) str._pfont = _fnts.P(fnt_id);

		_strs.PushBack(str);

		return (int)(_strs.N() - 1);
	};

	int Add(kmFont& fnt)
	{
		_fnts.PushBack(fnt);

		return (int)(_fnts.N() - 1);
	};

	kmStrw& GetStr(int64 idx)
	{
		return _strs(idx)._str;
	};
};

// real time value display window
class kmWndRtv : public kmWb
{
public:	
	// child window
	kmwMenu  _mpop;
		
	///////////////////////////////////////////////
	// functions to create window
public:	
	void Create(int x, int y, int w, int h, uint mode = 0)
	{
		// set gobj
		kmFont fnt1(30); fnt1.SetItalic();
		kmFont fnt2(24);
		kmgStr str1_(L"", RGB_WHITE, kmRect(1, 1,139,35), DT_CENTER);
		kmgStr str2_(L"", RGB_GREEN, kmRect(1,35,140,70), DT_CENTER);

		// add gobj
		Add(str1_, Add(fnt1));
		Add(str2_, Add(fnt2));

		Update(0,0);

		// create window
		kmWb::Create(x, y, w, h, RGB_BLACK);
	};

	///////////////////////////////////////////////
	// functions to create child
protected:
	virtual void CreateChild()
	{
		// create popup-menu
		_mpop.Add(L"hide window");
		_mpop.Add(L"change unit");	
	};

	///////////////////////////////////////////////
	// window procedures
	
	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control
		ushort ntf = HIWORD(wp);

		switch(LOWORD(wp))
		{
		// control
				
		// popup-menu
		case 2048 +  0: Hide(); break;
		case 2048 +  1: PRINTFA("%s : list01\n", GetKmClass()); break;
		}
		return PP_FINISH;
	};
	
	virtual kmePPType OnContextMenu(WPARAM wp, LPARAM lp) // right mouse click
	{		
		_mpop.CreatePopup(_hwnd, lp);
		
		return PP_FINISH;
	};
	
	///////////////////////////////////////////////
	// member functions
public:
	void Update(float val_kpa, float val_rmi)
	{
		if(val_kpa == 0 || val_rmi < 0.1f)
		{
			GetStr(0).SetStr(L"  - kPa");
			GetStr(1).SetStr(L"RMI -  ");
		}
		else
		{
			GetStr(0).SetStr(L"%.1f kPa", val_kpa);
			GetStr(1).SetStr(L"RMI %.2f", val_rmi);
		}		
		Invalidate();
		Show();
	};
};

// graph window
class kmWndGraph : public kmWnd
{
public:
	// child window
	kmwGraph   _graph;	
	kmwMenu    _menu;

	//////////////////////////////////////
	// functions for creating

	// create window 	
	void Create(uint mode = 0)
	{
		// create window
		CreateThread(100, 100, 600, 600, L"kmGraph", mode);
	};

	void Create(int x, int y, int w, int h, LPCTSTR win_name = NULL, uint mode = 0)
	{
		CreateThread(x, y, w, h, win_name, mode);
	};

	// static fucntion to create window
	static kmWndGraph* Plot(const kmMat1f32& y, kmRgb rgb = kmRgbB, LPCWSTR str = nullptr, int state = 1)
	{
		kmWndGraph *wnd = new kmWndGraph;
		wnd->Create(1); wnd->WaitCreated();
		wnd->Add(y, rgb, str, state);
		return wnd;
	};

	static kmWndGraph* Plot(const kmMat1f32& x, const kmMat1f32& y, kmRgb rgb = kmRgbB, LPCWSTR str = nullptr, int state = 1)
	{
		kmWndGraph *wnd = new kmWndGraph;
		wnd->Create(1); wnd->WaitCreated();
		wnd->Add(x, y, rgb, str, state);
		return wnd;
	};

	// Add data
	void Add(                    const kmMat1f32& y, kmRgb rgb = kmRgbB, LPCWSTR str = nullptr, int state = 1) { _graph.Add(   y, rgb, str, state);};
	void Add(const kmMat1f32& x, const kmMat1f32& y, kmRgb rgb = kmRgbB, LPCWSTR str = nullptr, int state = 1) { _graph.Add(x, y, rgb, str, state);};

	void SetRangeX(float min, float max) { _graph.SetRangeX(min, max); };
	void SetRangeY(float min, float max) { _graph.SetRangeY(min, max); };

	void SetTitleX(const kmStrw& title) { _graph.SetTitleX(title); };
	void SetTitleY(const kmStrw& title) { _graph.SetTitleY(title); };

	///////////////////////////////////////////////
	// function to create child
protected:
	virtual void CreateChild()
	{
		// create menu		
		_menu      .Add(L"File");
		_menu      .Add(L"Edit");
		_menu(0)   .Add(L"Save as bmp");
		_menu(0)   .Add(L"Save as");
		_menu(1)   .Add(L"Copy");
		
		_menu.Create(this->_hwnd, 1024);
				
		// create image viewer		
		_graph.Create(kmCwp( CWP_LRTB, 10, 10, 10, 10), this); 
	};

	///////////////////////////////////////////////
	// window procedures

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{
			_graph.Relocate(w, h);
		}
		return PP_FINISH;
	};

	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control
		ushort ntf = HIWORD(wp), id = LOWORD(wp);

		switch(id)
		{			
		// menu
		case 1024 +  0: SaveAsBmp(); break; // file-save as bmp
		case 1024 +  1: SaveAs();    break; // file-save as
		case 1024 +  2: Copy  ();    break; // edit-copy
		}
		return PP_FINISH;
	};

	// menu procedures
	void SaveAsBmp()
	{
		// get save file name
		wchar file_name[256] = L"";
		wchar filter   [256] = L"bmp files (*.bmp)\0*.bmp\0";

		BOOL  res =  kmFile::GetFileNameS(GetHwnd(), file_name, NULL, filter);
		
		// capture and save a image
		if(res)
		{			
			kmImg img = _graph.Capture();
			kmFile::WriteDib(img, file_name);
		}
	};

	void SaveAs()
	{
		// get save file name
		wchar file_name[256] = L"";
		wchar filter   [256] = L"bmp file (*.bmp)\0*.bmp\0png file (*.png)\0*.png";
		int   index          = 0;

		BOOL  res = kmFile::GetFileNameS(GetHwnd(), file_name, &index, filter);

		// capture and save image
		if(res)
		{	
			kmImg img = _graph.Capture();
			kmFile::WriteDib(img, file_name);

			// save image... under construction
			//if(index == 2) kmFile::WritePng(img, file_name); // png
			//else           kmFile::WriteDib(img, file_name); // bmp
		}
	};

	void Copy()	{ _graph.CaptureCopy();	};
};

// image view window
class kmWndImage : public kmWnd
{
public:
	// child window
	kmwImgView _view;
	kmwMenu    _menu;

	//////////////////////////////////////
	// functions for creating

	// create window 	
	void Create(uint mode = 0)
	{
		// create window
		CreateThread(100, 100, 600, 600, L"kmImage", mode);
	};

	void Create(int x, int y, int w, int h, LPCTSTR win_name = NULL, uint mode = 0)
	{
		CreateThread(x, y, w, h, win_name, mode);
	};

	// functions to update data member
	// * It has to be callded before calling Create()
	template<typename T> void Update(const kmMat3<T>& mat, T min, T max) { _view.Update(mat, min, max);}
	template<typename T> void Update(const kmMat2<T>& mat, T min, T max) { _view.Update(mat, min, max);}
	template<typename T> void Update(const kmMat3<T>& mat)               { _view.Update(mat);}
	template<typename T> void Update(const kmMat2<T>& mat)               { _view.Update(mat);}

	void Update     (const kmImg& img)         { _view.Update(img);};
	void UpdateRange(float min_v, float max_v) { _view.UpdateRange(min_v, max_v);};
	void UpdateCMap (kmeCMapType cmtype)       { _view.UpdateCMap(cmtype);};

	// static fucntion to create window
	static kmWndImage* Image(const kmImg& img)
	{
		kmWndImage *wnd = new kmWndImage;
		wnd->Update(img);
		wnd->Create(1);

		return wnd;
	};

	template<typename T>
	static kmWndImage* Image(const kmMat2<T>& mat, T min, T max, kmeCMapType cmtype = CMAP_GREY)
	{
		kmWndImage *wnd = new kmWndImage;
		wnd->Update(mat, min, max);
		wnd->UpdateCMap(cmtype);
		wnd->Create(1);

		return wnd;
	}
	
	template<typename T>
	static kmWndImage* Image(const kmMat3<T>& mat, T min, T max, kmeCMapType cmtype = CMAP_GREY)
	{
		kmWndImage *wnd = new kmWndImage;
		wnd->Update(mat, min, max);
		wnd->UpdateCMap(cmtype);
		wnd->Create(1);

		return wnd;
	}

	///////////////////////////////////////////////
	// function to create child
protected:
	virtual void CreateChild()
	{
		// create menu
		_menu      .Add(L"File");
		_menu      .Add(L"Edit");
		_menu(0)   .Add(L"Save as bmp");
		_menu(0)   .Add(L"Save as");
		_menu(0)   .AddSeparator();
		_menu(1)   .Add(L"Copy");
		
		_menu.Create(this->_hwnd, 1024);
				
		// create image viewer
		_view.Create(kmCwp(CWP_RAT, 0.f, 0.f, 1.f, 1.f), this);
	};

	///////////////////////////////////////////////
	// window procedures

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{			
			_view.Relocate(w, h);
		}
		return PP_FINISH;
	};

	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control
		ushort ntf = HIWORD(wp), id = LOWORD(wp);

		switch(id)
		{			
		// menu
		case 1024 +  0: SaveAsBmp(); break; // file-save as bmp
		case 1024 +  1: SaveAs();    break; // file-save as
		case 1024 +  2: Copy  ();    break; // edit-copy
		}
		return PP_FINISH;
	};

	// menu procedures
	void SaveAsBmp()
	{
		// get save file name
		wchar file_name[256] = L"";
		wchar filter   [256] = L"bmp files (*.bmp)\0*.bmp\0";

		BOOL  res =  kmFile::GetFileNameS(GetHwnd(), file_name, NULL, filter);
		
		// capture and save a image
		if(res) kmFile::WriteDib(_view._wimg._img, file_name);
	};

	void SaveAs()
	{
		// get save file name
		wchar file_name[256] = L"";
		wchar filter   [256] = L"bmp file (*.bmp)\0*.bmp\0png file (*.png)\0*.png";
		int   index          = 0;
		BOOL  res = kmFile::GetFileNameS(GetHwnd(), file_name, &index, filter);

		// capture and save image
		if(res)	kmFile::WriteDib(_view._wimg._img, file_name);		
	};

	void Copy()	{ _view._wimg.CaptureCopy(); };
};

// image view window for D415
class kmWndImage2 : public kmWnd
{
public:	
	// child window
	kmwImgView _view_l, _view_r, _view_d, _view_rgb;
	kmwMenu    _menu;

	// gobj
	kmFont   _fnt;
	kmgStr   _str, _str_tracked;
	kmgRect  _rct, _rct_tracked;
	
	//////////////////////////////////////
	// functions for creating
public:
	// create window 	
	void Create(uint mode = 0)
	{
		// create window
		CreateThread(100,100, 1000, 800, L"kmImage", mode);
	};

	void Create(int x, int y, int w, int h, LPCTSTR win_name = NULL, uint mode = 0)
	{
		CreateThread(x, y, w, h, win_name, mode);
	}

	// functions to update data member
	// * It has to be callded before calling Create()
	void Update_L(const kmMat2f32& mat, float min_v, float max_v)
	{
		_view_l.Update(mat, min_v, max_v);
	};

	void Update_R(const kmMat2f32& mat, float min_v, float max_v)
	{
		_view_r.Update(mat, min_v, max_v);
	};

	void Update_D(const kmMat2f32& mat, float min_v, float max_v)
	{
		_view_d.Update(mat, min_v, max_v);
	};

	void Update_Rgb(const kmImg&     img) { _view_rgb.Update(img); };
	void Update_L  (const kmMat2f32& mat) {	_view_l  .Update(mat); };
	void Update_R  (const kmMat2f32& mat) { _view_r  .Update(mat); };
	void Update_D  (const kmMat2f32& mat) { _view_d  .Update(mat); };

	void UpdateRange_L(float min_v, float max_v) { _view_l.UpdateRange(min_v, max_v);};
	void UpdateRange_R(float min_v, float max_v) { _view_r.UpdateRange(min_v, max_v);};
	void UpdateRange_D(float min_v, float max_v) { _view_d.UpdateRange(min_v, max_v);};

	void UpdateCMap_L(kmeCMapType cmtype) { _view_l.UpdateCMap(cmtype);};
	void UpdateCMap_R(kmeCMapType cmtype) { _view_r.UpdateCMap(cmtype);};
	void UpdateCMap_D(kmeCMapType cmtype) { _view_d.UpdateCMap(cmtype);};

	///////////////////////////////////////////////
	// virtual functions for kmWnd
protected:
	virtual void CreateChild()
	{
		// create menu		
		_menu      .Add(L"menu0");
		_menu      .Add(L"menu1");
		_menu(0)   .Add(L"item0");
		_menu(0)   .AddSeparator();
		_menu(0)   .Add(L"item1");
		_menu(0)(2).Add(L"item0_A");
		_menu(0)(2).Add(L"item0_B");
		_menu(1)   .Add(L"item2" ); 
		_menu(1)   .Add(L"item3" ); 

		_menu.Create(this->_hwnd, 1024);
				
		// create image viewer
		_view_l  .Create(kmCwp( CWP_RAT, 0.0f, 0.0f, 0.5f, 0.5f), this, 1);
		_view_r  .Create(kmCwp( CWP_RAT, 0.5f, 0.0f, 0.5f, 0.5f), this, 2);
		_view_d  .Create(kmCwp( CWP_RAT, 0.0f, 0.5f, 0.5f, 0.5f), this, 3);
		_view_rgb.Create(kmCwp( CWP_RAT, 0.5f, 0.5f, 0.5f, 0.5f), this, 4);

		// set font
		_fnt.SetH(50);

		// add str of gobj
		_str.Set(L"", &_fnt, kmRgb(255,255,0), kmRect(100,100,1000,200));
		_str.SetVisible(false);

		_view_l.Add(&_str);
		_view_r.Add(&_str);

		// add rct of gobj
		_rct.Set(kmRect(1000,500,1000 + 128, 500 + 128), kmRgb(0,255,255,1));
		_rct.SetLine(kmRgb(255,0,0), 3);
		_rct.SetVisible(false);

		_view_l.Add(&_rct);
		_view_r.Add(&_rct);

		// add str_tracked
		_str_tracked.Set(L"", &_fnt, kmRgb(0,0,0), kmRect(100,100,1000,200));
		_str_tracked.SetVisible(false);

		_view_r.Add(&_str_tracked);

		// add rct_tracked
		_rct_tracked.Set(kmRect(1000,500,1000 + 128, 500 + 128), kmRgb(0,255,255,1));
		_rct_tracked.SetLine(kmRgb(0,0,255), 3);
		_rct_tracked.SetVisible(false);

		_view_r.Add(&_rct_tracked);
	};

	virtual kmePPType OnSize(WPARAM wp, LPARAM lp)
	{
		const int w = LOWORD(lp); _win.w = w;
		const int h = HIWORD(lp); _win.h = h;

		if(wp == SIZE_RESTORED || wp == SIZE_MAXIMIZED)
		{			
			_view_l  .Relocate(w, h);
			_view_r  .Relocate(w, h);
			_view_d  .Relocate(w, h);
			_view_rgb.Relocate(w, h);
		}
		return PP_FINISH;
	};

	virtual kmePPType OnCommand(WPARAM wp, LPARAM lp)
	{
		// get notification of control
		uint ntf = HIWORD(wp), id  = LOWORD(wp);
		
		switch(id)
		{
		case 1: // _view_l
		case 2: // _view_r
			switch(ntf)
			{
			case NTF_IMGVIEW_POSCLICK : TrackLeftRect(LOWORD(lp), HIWORD(lp)); break;
			}
			break;
		}
		return PP_FINISH;
	};

	void TrackLeftRect(int x, int y)
	{
		// update display
		_rct.SetRect(_rct.GetRect().SetCen(x, y));
		_rct.SetVisible();

		_str.SetRect(_str.GetRect().Set(x - 50, y + 50));
		_str.SetStr(kmStrw(L"(%d, %d)", x, y));
		_str.SetVisible();

		_view_l.UpdateImg();
		_view_r.UpdateImg();

		// track left rect on right image
		{
			// init parameters
			const kmRect rect = _rct.GetRect();
			const int    w = rect.GetW(), h = rect.GetH();
			const int    i = rect.l, j = rect.t;

			kmMem64 buf((2*w + 1) * (2*h + 1)*16*4);

			// set patch in left rect
			kmMat2f32 pat(&_view_l._data(i, j), w, h, _view_l._data.P1());

			// exec tracking the patch
			float p [] = {-40.f, 0.f};
			float x0[] = {(float) x, (float) y};

			//pat.TrackLK(p, x0, _view_r._data, buf, LK_XY);

			// set output
			const float x1 = x0[0] + p[0], y1 = x0[1] + p[1];

			_rct_tracked.SetRect(_rct_tracked.GetRect().SetCen((int) x1, (int) y1));
			_rct_tracked.SetVisible();

			_view_r.UpdateImg();

			// debug output
			PRINTFA("* (%d, %d) : delx,dely (%.2f, %.2f)\n", (int) x1, (int) y1, p[0], p[1]);
		}
	};
};

#endif /* __km7Wnd_H_INCLUDED_2018_11_12__ */

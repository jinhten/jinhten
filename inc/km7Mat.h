#ifndef __km7Mat_H_INCLUDED_2018_11_12__
#define __km7Mat_H_INCLUDED_2018_11_12__

/* Note ----------------------
* kmMat has been created by Choi, Kiwan
* This is version 7
* kmMat v7 is including the following
*   - km7Define.h
*   - km7Define.h -> km7Mat.h
*   - km7Define.h -> km7Mat.h -> km7Wnd.h
*   - km7Define.h -> km7Mat.h -> km7Dnn.h
*   - km7Define.h -> km7Mat.h -> kc7Mat.h
*/

// definition for OS
#ifdef _WIN64
#define _KM_OSWIN
#else
#define _KM_OSLINUX
#endif

#define _WINSOCKAPI_        // to prevent redefinition compile error if using winsock2.h

// base header
#include <cmath>
#include <windows.h>        // for time check.. QueryPeromanceCounter(..)
#include <stdio.h>          // for printf..
#include <assert.h>
#include <winbase.h>        // for OutputDebugString(..)
#include <typeinfo>         // for typeid()
#include <ctime>            // for time_t,struct tm
#include <process.h>        // for _beginthreadex()
#include <initializer_list>
#include <iostream>			// for cout
#include <io.h>             // for _access()
#include <commdlg.h>        // for OPENFILENAME
#include <atomic>			// for atomic<>
#include <bitset>           // for kmBit
#include <shlobj_core.h>    // for common dialog
#include <direct.h>         // for mkdir
#include <gdiplus.h>        // only for windows os

using namespace std;
using namespace Gdiplus;

#pragma comment(lib, "gdiplus.lib")

// define header
#include "km7Define.h"

///////////////////////////////////////////////////////////////
// kmf (km functions) .. .inline

// get the virtual table address of class T
template<class T> void* GetVfptr()     { T a; return (void*) *((int64*)&a); }
template<class T> void* GetVfptr(T& b) { T a; return (void*) *((int64*)&a); }

// swap endian
template<typename T> T kmfswapbyte(T a) { return a; }

inline ushort kmfswapbyte(ushort a) { return ENDIAN16(a); };
inline  short kmfswapbyte( short a) { return ENDIAN16(a); };
inline uint   kmfswapbyte(uint   a) { return ENDIAN32(a); };
inline  int   kmfswapbyte( int   a) { return ENDIAN32(a); };
inline uint64 kmfswapbyte(uint64 a) { return ENDIAN64(a); };
inline  int64 kmfswapbyte( int64 a) { return ENDIAN64(a); };

inline  float kmfswapbyte(float  a) { uint   b = ENDIAN32(*(uint  *)&a); return *(float*) &b; };
inline double kmfswapbyte(double a) { uint64 b = ENDIAN64(*(uint64*)&a); return *(double*)&b; };

// bit rotate (+ shift is right direction(>>))
template<typename T> T kmfbitrot(T val, int shift)
{
	constexpr int bits = sizeof(T)*8;
	if     (shift > 0) return (val>>  shift)  | (val<<(bits - shift));
	else if(shift < 0) return (val<<(-shift)) | (val>>(bits + shift));
	return val;
}

// get random integer between min and max
//uint kmrand(uint min, uint max) { return uint(GetTickCount64())%(max - min + 1) + min; };
//int  kmrand( int min,  int max) { return uint(GetTickCount64())%(max - min + 1) + min; };

// get random integer between min and max
inline uint kmfrand(uint min, uint max)
{
	uint64 cnt; QueryPerformanceCounter((LARGE_INTEGER*)&cnt); 
	return uint(cnt)%(int64(max) - min + 1) + min;
};
inline int  kmfrand(int min, int max)
{
	uint64 cnt; QueryPerformanceCounter((LARGE_INTEGER*)&cnt);
	return uint(cnt)%(int64(max) - min + 1) + min;
};
inline uint kmfrand32()
{
	uint64 cnt; QueryPerformanceCounter((LARGE_INTEGER*)&cnt);	
	return kmfbitrot(uint(cnt), kmfrand(0,31));
};

// usleep
inline void kmfusleep(int64 usec)
{
	LARGE_INTEGER lc1,lc2,lfreq; int64 time_usec;
	QueryPerformanceFrequency(&lfreq);
	QueryPerformanceCounter(&lc1);
	do
	{
		Sleep(0);
		QueryPerformanceCounter(&lc2);
		time_usec = (lc2.QuadPart-lc1.QuadPart)*1000000LL/lfreq.QuadPart;
	} while( time_usec < usec);
};

// set console position 
inline void kmfsetcursor(int x, int y)
{
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), COORD(x,y));
};

// get consol position
inline void kmfgetcursor(int& x, int& y)
{
	CONSOLE_SCREEN_BUFFER_INFO info;

	auto ret = GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info);

	if(ret == TRUE) { x = info.dwCursorPosition.X; y = info.dwCursorPosition.Y; }
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class for matrix

///////////////////////////////////////////////////////////////
// complex
template<typename T> class cmplx
{
public:
	T r = 0, i = 0;

	// constructor
	cmplx() {};
	cmplx(T r_, T i_)      { r = r_;  i = i_; };
	cmplx(const cmplx&  a) { r = a.r; i = a.i;};

	cmplx(const initializer_list<T>& val)
	{
		ASSERTFA(val.size() == 2, "[cmplx() in 58]");

		const int64 *pval = val.begin();
		r = *(pval++); i = *(pval);
	};

	// basic math functions		
	T     sq () const { return r*r + i*i; };
	T     ang() const { return atan2(i, r); };
	T     abs() const { return sqrt(r*r + i*i); };
	cmplx exp() const { return ::exp(r)*cmplx(cos(i), sin(i)); };

	// operator
	cmplx operator~() const { return cmplx( r,-i); }; // conjugate
	cmplx operator-() const { return cmplx(-r,-i); };

	cmplx& operator+=(const T& b) { r += b;         return *this; };
	cmplx& operator-=(const T& b) { r -= b;         return *this; };
	cmplx& operator*=(const T& b) { r *= b; i *= b; return *this; };
	cmplx& operator/=(const T& b) { r /= b; i /= b; return *this; };

	cmplx& operator+=(const cmplx& b) { r += b.r; i += b.i; return *this; };
	cmplx& operator-=(const cmplx& b) { r -= b.r; i -= b.i; return *this; };
	cmplx& operator*=(const cmplx& b) { *this = *this*b;    return *this;};
	cmplx& operator/=(const cmplx& b) { (*this *= ~b) /= (b.r*b.r + b.i*b.i); return *this;};
		
	cmplx operator+(const cmplx& b) const { return cmplx(r,i) += b; };
	cmplx operator-(const cmplx& b) const { return cmplx(r,i) -= b; };
	cmplx operator*(const cmplx& b) const { return cmplx(r*b.r - i*b.i, r*b.i + i*b.r) ; };
	cmplx operator/(const cmplx& b) const { return cmplx(r,i) /= b; };

	bool operator==(const cmplx& b) const { return r == b.r && i == b.i; };
	bool operator!=(const cmplx& b) const { return r != b.r || i != b.i; };

	// conversion operator...
	operator T() const { return abs(); };
};

template<typename T> T        abs(const cmplx<T>& a) { return a.abs(); };
template<typename T> T        ang(const cmplx<T>& a) { return a.ang(); };
template<typename T> T        sq (const cmplx<T>& a) { return a.sq (); };
template<typename T> cmplx<T> exp(const cmplx<T>& a) { return a.exp(); };

typedef cmplx<short>   cmplxi16;
typedef cmplx<int>     cmplxi32;
typedef cmplx<int64>   cmplxi64;
typedef cmplx<float>   cmplxf32;
typedef cmplx<double>  cmplxf64;

template<typename T> cmplx<T> operator+(const cmplx<T>& b, const T a) { return cmplx<T>(b.r+a, b.i+a); }
template<typename T> cmplx<T> operator-(const cmplx<T>& b, const T a) { return cmplx<T>(b.r-a, b.i-a); }
template<typename T> cmplx<T> operator*(const cmplx<T>& b, const T a) { return cmplx<T>(b.r*a, b.i*a); }
template<typename T> cmplx<T> operator/(const cmplx<T>& b, const T a) { return cmplx<T>(b.r/a, b.i/a); }

template<typename T> cmplx<T> operator+(const T a, const cmplx<T>& b) { return cmplx<T>(a+b.r, a+b.i); }
template<typename T> cmplx<T> operator-(const T a, const cmplx<T>& b) { return cmplx<T>(a-b.r, a-b.i); }
template<typename T> cmplx<T> operator*(const T a, const cmplx<T>& b) { return cmplx<T>(a*b.r, a*b.i); }
template<typename T> cmplx<T> operator/(const T a, const cmplx<T>& b) { return (a*~b)/sq(b); }

///////////////////////////////////////////////////////////////
// tempalte class for argument

template<typename T1, typename T2>
class kmT2
{	
public:
	T1 v1, *m1; T2 v2, *m2;

	// constructor
	// * Note that a1 and a2 must have objects because they're reference type
	// * and the optimizer can skip over definition of object.
	kmT2(T1& a1, T2& a2)
	{
		m1 = &a1; v1 = a1;
		m2 = &a2; v2 = a2;
	};
	kmT2& operator=(const kmT2& b)
	{
		*m1 = v1 = b.v1;
		*m2 = v2 = b.v2;
		return *this; 
	};
};

template<typename T1, typename T2, typename T3>
class kmT3
{
public:
	T1 v1, *m1; T2 v2, *m2; T3 v3, *m3;

	// constructor
	// * Note that a1, a2 and a3 must have objects because they're reference type
	// * and the optimizer can skip over definition of object.
	kmT3(T1& a1, T2& a2, T3& a3)
	{
		m1 = &a1; v1 = a1;
		m2 = &a2; v2 = a2;
		m3 = &a3; v3 = a3;
	};
	kmT3& operator=(const kmT3& b)
	{
		*m1 = v1 = b.v1;
		*m2 = v2 = b.v2;
		*m3 = v3 = b.v3;
		return *this; 
	};
};

template<typename T1, typename T2, typename T3, typename T4>
class kmT4
{
public:
	T1 v1, *m1; T2 v2, *m2; T3 v3, *m3; T4 v4, *m4;
	
	// constructor
	// * Note that a1, a2, a3 and a4 must have objects because they're reference type
	// * and the optimizer can skip over definition of object.
	kmT4(T1& a1, T2& a2, T3& a3, T4& a4)
	{
		m1 = &a1; v1 = a1;
		m2 = &a2; v2 = a2;
		m3 = &a3; v3 = a3;
		m4 = &a4; v4 = a4;
	};
	kmT4& operator=(const kmT4& b)
	{
		*m1 = v1 = b.v1;
		*m2 = v2 = b.v2;
		*m3 = v3 = b.v3;
		*m4 = v4 = b.v4;
		return *this; 
	};
};

// universal version with variadic template
// * Note that it has no limit for number of arguments,
// * but it is necessary to write template-types one bye one.
template<typename... Ts> class kmT {};

template<typename T, typename... Ts>
class kmT<T, Ts...>
{
public:	
	T _v, *_m;
	kmT<Ts...> _rest;

	kmT(T& a, Ts&... rest) : _v(a), _m(&a), _rest(rest...) {};

	kmT<T,Ts...>& operator=(const kmT<T,Ts...>& b)
	{
		*_m = _v = b._v; _rest = b._rest; return *this;
	};
};

//////////////////////////////////////////////////////////
// index class for kmArr or kmMat
class kmI
{
public:
	int64 s = 0;     // start index
	int64 e = end64; // end   index

	// constructor
	kmI() {};
	kmI(int64 s_)           {s = s_;  e = s_; };
	kmI(int64 s_, int64 e_) {s = s_;  e = e_; };
	kmI(const kmI& a)       {s = a.s; e = a.e;};

	kmI(const initializer_list<int64>& val)
	{	
		ASSERTFA(val.size() == 2, "[kmI() in 44]");

		const int64 *pval = val.begin();
		s = *(pval++); e = *(pval);
	};

	// algorithm
	int64 Len() const
	{
		ASSERTFA(e != end64, "[kmI() in 178]"); return e-s+1; 
	};

	int IsEqualSize(const kmI& b) const
	{
		return Len() == b.Len();
	};
};

//////////////////////////////////////////////////////////////
// state union 
union kmstate
{
	// members
private:
	// * Note that you should not directly access val 
	// * since it can be a different number even in the same state 
	// * depending on big endian or little endian.
	uint64 _val = 0; 
public:
	struct
	{
		uint is_created : 1; // memory was allocated
		uint is_pinned  : 1; // memory was pinned
		uint rsv2       : 1;
		uint rsv3       : 1;
		uint rsv4       : 1;
		uint rsv5       : 1;
		uint rsv6       : 1;
		uint rsv7       : 1;

		uint flg0       : 1; // for univeral purpose
		uint flg1       : 1;
		uint flg2       : 1;
		uint flg3       : 1;
	};
	// * Note that pinned state will ban Expend(), Move() and move constructor

	// constructor
	kmstate() {};

	// copy constructor	
	kmstate(const uint64& a)	{ _val = a; };

	// assignment operator
	kmstate& operator=(const uint64& a) { _val = a; return *this; };

	// conversion operator... (uint64) a
	operator uint64() const { return _val; };
};

//////////////////////////////////////////////////////////
// matrix base class
template<typename T> class kmArr
{
protected:
	T*	    _p     = nullptr; // pointer of data	
	kmstate _state = 0;       // state of memory allocation
	int64   _size  = 0;       // size of allocated memory (number of data)

	/////////////////////////////////////////////////
	// basic member functions
public:	
	// * Note that _state sholud be initialized seperately.
	virtual void Init() { _p = nullptr; _size = 0;}; 

	// construtor
	kmArr() {};
	kmArr(      int64 size) { Create(size);};
	kmArr(T* p, int64 size) { Set(p, size);};

	kmArr(const initializer_list<T>& val)
	{
		Create(val.size());
		const T *pval = val.begin();
		for(int64 i = 0; i < _size; ++i) *(_p + i) = *(pval++);
	};

	// destructor
	virtual ~kmArr() { Release(); };

	// copy constructor
	kmArr(const kmArr& b)
	{	
		Create(b._size);
		Copy  (b._p);
	};

	// move constructor
	kmArr(kmArr&& b) { Move(b); };
	
	// constructor for other type
	template<typename Y>
	kmArr(const kmArr<Y>& b)
	{
		Create(b.Size());
		CopyTCast(b.P());
	}

	// * Note that the assignment operator of the same type was separated from
	// * the assignment operators of different types, 
	// * because the complier considers there is no assigment operator of the same type
	// * if there is only the definition using tempalate.
	//
	// assignment operator
	kmArr& operator=(const kmArr& b) { RecreateIf(b); Copy(b._p); return *this; };

	// assignment operator for other type
	template<typename Y>
	kmArr& operator=(const kmArr<Y>& b) { RecreateIf(b); CopyTCast(b.P()); return *this; }

	// move assignment operator
	kmArr& operator=(kmArr&& b)
	{
		if(IsPinned()) *this = b; // * Note that this will call the assignment operator since b is an L-value.
		else           Move(b); 
		return *this; 
	};

	// allocate memory... core
	void Create(int64 size)
	{
		ASSERTA(!IsCreated(), "[kmArr::Create in 183] memory has already been created");
		ASSERTA(!IsPinned (), "[kmArr::Create in 184] memory is pinned");

		if(size == 0) { _size = 0; _state = 0; return; }

		_p = new T[size];

		ASSERTA(_p != 0, "[kmArr::Create in 146] %p != 0", _p);

		_size  = size;
		_state = 1;

		//PRINTFA("*************************** kmArr::Create(%d) : %p\n", size, _p);
	};

	// expand memory... core
	// * Note that expand must be call in case that kmArr was created not set.
	void Expand(int64 size)
	{	
		ASSERTA(IsCreated(), "[kmArr::Expand in 206] memory is not created");
		ASSERTA(!IsPinned(), "[kmArr::Expand in 207] memory is pinned");

		const int64 size_new = size + _size;

		T* p = new T[size_new];

		ASSERTA(_p != 0, "[kmArr::Expand in 160] %p != 0", _p);

		// * Note that "delete [] _p" will call the destructor of every elements.
		// * If T is kmat, it will release the memories of elements.
		// * To avoid this, std::move() and the move assignment operator have been used
		// * instead of CopyTo(p).
		// * std:move(A) means that A is rvalue.

		//CopyTo(p);
		for(int64 i = 0; i < _size; ++i) { *(p + i) = std::move(*(_p + i)); }

		//PRINTFA("******************* kmArr::Expand (%d) : %p --> %p\n", _state, _p, p);
		
		delete [] _p;
		
		_size  = size_new;
		_p     = p;
		_state = 1;
	};

	// release memory... core
	virtual void Release()
	{
		//PRINTFA("******************* kmArr::Release (%d) : %p\n", _state, _p);

		if(IsCreated() && _p) delete [] _p;
		_state.is_created = 0; 
		Init();
	};

	// set array... core
	void Set(T* p, int64 size)
	{
		ASSERTA(!IsCreated(), "[kmArr::Set in 181] memory has already been created");

		_size = size;
		_p    = p;
		PinMemory(); // * Note that Set() will automatically pin the memory.
	};	
		
	// copy from (a = b)... core
	void Copy(const T* b)	
	{
		ASSERTA(_size == 0 || _p != 0, "[kmArr::Copy in 190] %p != 0", _p);

		memcpy(_p, b, sizeof(T)*_size);
	};

	// copy from (a = b)... core
	void Copy(const kmArr& b)
	{
		ASSERTA(IsEqualN(b), "[kmArr::Copy in 149]");

		const int64 n = N();

		for(int64 i = 0; i < n; ++i) v(i) = b.v(i);
	};

	// copy to (b = a)... core
	void CopyTo(T* b) const
	{
		memcpy(b, _p, sizeof(T)*_size);
	};

	// copy from with casting(a = b)... core
	template <typename Y> void CopyTCast(const Y* b)
	{	
		ASSERTA(_p != 0, "[kmArr::CopyTCast in 145]");

		T *p = _p;

		for (int64 i = _size; i--;) *(p++) = (T) *(b++);
	}

	// copy form with casting(a = b)... core
	template <typename Y> void CopyTCast(const kmArr<Y>& b)
	{
		ASSERTA(IsEqualN(b), "[kmArr::CopyTCast in 155]");

		const int64 n = N();

		for(int64 i = 0; i < n; ++i) v(i) = (T) b.v(i);
	}
	
	// release and create
	void Recreate(int64 size) { Release();	Create(size); };

	template<typename Y> void Recreate(const kmArr<Y>& b) { Recreate(b.N()); }

	// recreate if
	int RecreateIf(int64 size) { if(size != _size) { Recreate(size); return 1; } return 0; };
		
	template<typename Y> int RecreateIf(const kmArr<Y>& b) { return RecreateIf(b.Size()); }

	// move (b --> a)... core
	void Move(kmArr& b)
	{
		ASSERTA(!IsPinned(), "[kmArr::Move in 298] memory is pinned");

		if(IsCreated()) Release();

		_p     = b._p;
		_size  = b._size;
		_state = b._state; b._state = 0;
	};

	// set array
	void Set(const kmArr& b)   { Set(b._p, b._size); };	

	// operator to get data
	T* P(int64 i1) const
	{
		if(i1 < 0) i1 += _size; // * Note that i1 = -1 is the same as End()

		ASSERTA(0 <= i1 && i1 < _size, "[kmArr::P in 212] %lld < %lld", i1, _size);

		return (_p + i1);
	};

	T* P() const { return _p; };

	T& operator()(int64 i1 = 0) const { return *P(i1); };
	T& a         (int64 i1 = 0) const { return *P(i1); };

	// * Note that you should use v(i) instead of a(i)
	// * if you want to access every elements with one index with N()
	// * in a member function
	virtual T& v (int64 i1 = 0) const { return *P(i1); };

	T* Begin() const { return _p;};
	T* End  () const { return _p + _size - 1; };

	// get class name	
	LPCSTR GetKmClass() const { return typeid(*this).name() + 6; };
	
	// restore virtual function pointer (__vfptr)	
	template<class Y> void RestoreVfptr()     { *((void**)this) = GetVfptr<Y>(); }
	template<class Y> void RestoreVfptr(Y& a) { *((void**)this) = GetVfptr<Y>(); }

	// restore the class including vfptr
	// * Note that this will clear and reset _p, _state, _size without release,
	// * which can cause a memory leak. So, you should use this very carefully.
	kmArr& Restore() { RestoreVfptr<kmArr<T>>(); _state = 0; Init(); return *this; };

	/////////////////////////////////////////////////
	// operator functions
	
	// conversion operator... (T*) a, (T) a
	operator T*() const { return  _p; };
	operator T () const { return *_p; };

	// conversion operator.. (kmArr<Y>) a
	template<typename Y> operator kmArr<Y>() const
	{
		kmArr<Y> b(_size); b.CopyTCast(_p);
		return b;
	}

	/////////////////////////////////////////////////
	// general member functions

	// pin of unpin memory
	void PinMemory  () { _state.is_pinned = 1; };
	void UnpinMemory() { _state.is_pinned = 0; };

	// get info	
	int64 Size()   const { return _size;          };
	int64 Byte()   const { return _size*sizeof(T);};
	int64 State()  const { return _state;         };

	const type_info& GetType() const { return typeid(T); };
	
	bool IsCreated() const { return _state.is_created == 1; };
	bool IsPinned () const { return _state.is_pinned  == 1; };

	template<typename Y> 
	static bool IsArr(kmArr<Y>* a) { return true; }
	static bool IsArr(void*     a) { return false;};
			
	// get the number of real elements
	virtual int64 N() const { return _size; };

	// get bytes of info members which is from _size to end
	virtual int64 GetInfoByte() const { return sizeof(*this) - 24;}; 

	// get the first pointer of info members
	void* GetInfoPt() const { return (void*) &_size; };
	
	// display member info
	virtual void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) print("[%s]\n", str);

		print("  _p     : %p\n"  , _p);
		print("  _state : created(%d), pinned(%d)\n", _state.is_created, _state.is_pinned);
		print("  _size  : %lld\n", _size);
	};

	// display dimension
	virtual void PrintDim(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		PRINTFA("  dim : %lld\n", _size);
	};

	// display member value
	void PrintVal(int64 idx) const
	{
		if     (IS_F(T)) PRINTFA("[%lld] %.3f\n", idx, v(idx));
		else if(IS_I(T)) PRINTFA("[%lld] %lld\n", idx, v(idx));
	};

	void  PrintVal(int64 s_idx, int64 e_idx) const
	{
		s_idx = MAX(s_idx, 0);
		e_idx = MIN(e_idx, N()-1);
		
		for(int64 i = s_idx; i <= e_idx; ++i) PrintVal(i);

		PRINTFA("\n");
	};

	void PrintVal() const {	PrintVal(0, N()-1); };

	// set value
	void SetVal(const T val)
	{
		for(int64 i = 0, n = N(); i < n; ++i) v(i) = val;
	};

	// set value in the block
	void SetVal(const T val, int64 i_s, int64 i_e)
	{
		// check idx
		i_s = MAX(i_s, 0);
		i_e = MIN(i_e, N() - 1);
		for (int64 i = i_s; i <= i_e; ++i) v(i) = val;
	};

	void SetVal(initializer_list<T>& val, int64 i_s = 0 )
	{
		const T      *pval = val.begin();
		const int64  i_ep = MIN(N(), val.size() + i_s);

		for(int64 i = i_s; i < i_ep; ++i) v(i) = *(pval++);
	};

	void SetValInc(T val_s, const T delta)
	{
		for(int64 i = 0, n = N(); i < n; ++i, val_s += delta) v(i) = val_s;
	};

	void SetSpaceLin(const T val_s, const T val_e)
	{		
		T delta = (_size < 2) ? 0 : (val_e - val_s) / (_size - 1);

		SetValInc(val_s, delta);
	};

	void SetRand(const T min, const T max)
	{
		// init rand
		static int seed = 1;
		if(seed) { srand(GetTickCount()); seed = 0; }

		const int64 n = N();

		for(int64 i = 0; i < n; ++i)
		{
			v(i) = (max - min)*((T) rand())/((T) RAND_MAX) + min;
		}
	};

	void SetSpace125(const T val_s, const T val_e)
	{
		const float n   = (float) N();
		const float del = (float) abs(val_e - val_s)/(n + 1.f);

		if(del == 0) { SetVal(0); return; }

		const float logdel   = log10f(del);
		const float logdel_n = floor(logdel);
		const float logdel_f = logdel - logdel_n;

		float step = 0;

		if     (logdel_f < 0.3010f) step = 2.f;
		else if(logdel_f < 0.6990f) step = 5.f;
		else                        step = 10.f;

		step *= powf(10.f, logdel_n);

		T val = (T) (step*(ceilf((float)val_s/step)));

		SetValInc(val, (T) step);
	};

	void SetZero() { memset(_p, 0, Byte()); };

	// swap
	void Swap(int64 i, int64 j)
	{
		T temp = v(i); v(i) = v(j); v(j) = temp;
	};

	// compare
	template<typename Y> bool IsEqualSize   (const kmArr<Y>& b) const { return _size == b.Size();}
	template<typename Y> bool IsEqualSizeDim(const kmArr<Y>& b) const { return _size == b.Size();}
	template<typename Y> bool IsEqualN      (const kmArr<Y>& b) const { return N() == b.N();}

	/////////////////////////////////////////////////
	// math member functions
	//
	// *  Note that general forms of math functions are the followings
	// *  - a.sort(...)    --> a = sort(a) 
	// *  - b.sort(a, ...) --> b = sort(a)

	// get sum
	T Sum() const
	{
		const int64 n = N(); T sum = 0;

		for(int i = 0; i < n; ++i) sum += v(i);

		return sum;
	};

	// get sum of square
	T SumSq() const
	{
		const int64 n = N(); T sum = 0;

		for(int i = 0; i < n; ++i) sum += v(i)*v(i);

		return sum;
	};

	// get mean
	T Mean() const { return  Sum()/(T) N(); };

	// get mean of sqaure
	T MeanSq() const { return SumSq()/(T) N(); };	

	// get median
	T Median() const
	{
		const int64 n = N();
	
		kmArr b = *this; b.Sort(1);
	
		return (n%2 == 0) ? (b(n/2-1) + b(n/2))/(T)2 : b(n/2);
	};

	// get min
	T Min() const
	{
		const int64 n = N(); T min_v = v(0);

		for(int i = 1; i < n; ++i) if(v(i) < min_v) min_v = v(i);

		return min_v;
	};

	// get index of min
	T Min(int64* idx)
	{
		const int64 n = N(); T min_v = v(0); *idx = 0;

		for(int i = 1; i < n; ++i) if(v(i) < min_v) {min_v = v(i); *idx = i;}

		return min_v;
	};

	// get max
	T Max() const
	{
		const int64 n = N(); T max_v = v(0);

		for(int i = 1; i < n; ++i) if(v(i) > max_v) max_v = v(i);

		return max_v;
	};

	// get index of max
	T Max(int64* idx)
	{
		const int64 n = N(); T max_v = v(0); *idx = 0;

		for(int i = 1; i < n; ++i) if(v(i) > max_v) {max_v = v(i); *idx = i;}

		return max_v;
	};

	// get max in the block
	T Max(int64 i_s, int64 i_e)
	{
		// check idx
		i_s = MAX(i_s, 0);
		i_e = MIN(i_e, N()-1);

		T max_v = v(i_s);

		for (int i = i_s; i <= i_e; ++i) if(v(i) > max_v) max_v = v(i);

		return max_v;
	};

	// get min of abs
	T MinAbs() const
	{
		const int64 n = N(); T min_v = abs(v(0));

		for(int i = 1; i < n; ++i) if(v(i) < min_v) min_v = abs(v(i));

		return min_v;
	};

	// get max of abs
	T MaxAbs() const
	{
		const int64 n = N(); T max_v = abs(v(0));

		for(int i = 1; i < n; ++i) if(v(i) > max_v) max_v = abs(v(i));

		return max_v;
	};

	// sort.. a.sort(order)	
	//   dir :  >= 0 (Ascending order), < 0(Descending order)
	void Sort(int dir = 1)
	{
		QuickSort(0, (int)N()-1, 1, dir);
	};

	// sort with b... a.sort(order, b)... b will be also sorted in order of a
	void Sort(kmArr& b, int dir = 1)
	{
		ASSERTA( N() <= b.N(), "[kmArr::Sort in 349] %lld <= %lld", N(), b.N());

		QuickSort(0, (int)N()-1, 1, dir, b);
	};

	// quick sort algorithm
	// int left		: start point
	// int right	: end point
	// int step		: step
	// int order	: >= 0 (Ascending order), < 0(Descending order)
	void QuickSort(int left, int right, int step, int dir)
	{
		if(left < right)
		{
			int j = left; const T pivot = v(left);

			for(int i = left + step; i <= right; i+= step)
			{
				const T vi = v(i);

				if((dir >= 0)? (vi < pivot) : (vi > pivot))
				{
					j+= step; v(i) = v(j); v(j) = vi;
				}
			}
			v(left) = v(j); v(j) = pivot;

			QuickSort(  left, j-step, step, dir);
			QuickSort(j+step,  right, step, dir);
		}
	}
	
	void QuickSort(int left, int right, int step, int dir, kmArr& b)
	{
		if(left < right)
		{
			int j = left; const T pivot = v(left);

			for(int i = left + step; i <= right; i+= step)
			{
				const T vi = v(i);

				if((dir >= 0)? (vi < pivot) : (vi > pivot))
				{
					j+= step; v(i) = v(j); v(j) = vi; b.Swap(i, j);
				}
			}
			v(left) = v(j); v(j) = pivot; b.Swap(left, j);

			QuickSort(  left, j-step, step, dir, b);
			QuickSort(j+step,  right, step, dir, b);
		}
	}
};

//////////////////////////////////////////////////////////
// 1D matrix class
template<typename T> class kmMat1 : public kmArr<T>
{
protected:
	// using for members of parents class
	using kmArr<T>::_p, kmArr<T>::_state, kmArr<T>::_size;

	// member variables
	int64 _n1 = 0;    // number of dim1

public:
	// using for functions of parents class
	using kmArr<T>::Expand;
	using kmArr<T>::Release;
	using kmArr<T>::IsPinned;
	using kmArr<T>::IsCreated;
	using kmArr<T>::GetKmClass;
	using kmArr<T>::Copy;
	using kmArr<T>::RestoreVfptr;

	/////////////////////////////////////////////////
	// basic member functions
public:	
	virtual void Init() { kmArr<T>::Init(); _n1 = 0; };

	// constructor
	kmMat1() {};
	kmMat1(      int64 n1)             { Create(n1);       };
	kmMat1(T* p, int64 n1)             { Set(p, n1);       };
	kmMat1(      int64 n1, int64 size) { Create(n1, size); };
	kmMat1(T* p, int64 n1, int64 size) { Set(p, n1, size); };

	kmMat1(const initializer_list<T>& val)
	{
		_state =0;
		Create(val.size());
		const T *pval = val.begin();
		for(int64 i = 0; i < _size; ++i) *(_p + i) = *(pval++);
	};

	// destructor
	virtual ~kmMat1() {};

	// copy constructor	
	kmMat1(const kmMat1& b) { Create(b.N1()); Copy(b.Begin()); };

	template<typename Y>
	kmMat1(const kmMat1<Y>& b) { Create(b.N1()); CopyTCast(b); }

	// move constructor
	kmMat1(kmMat1&& b) noexcept { Move(b); };

	// assignment operator
	kmMat1& operator=(const kmMat1& b) { RecreateIf(b); Copy(b._p); return *this; };
		
	template<typename Y>
	kmMat1& operator=(const kmMat1<Y>& b) { RecreateIf(b); CopyTCast(b); return *this; }

	// assignment operator for scalar
	kmMat1& operator=(const T b) { SetVal(b); return *this; };

	// move assignment operator
	kmMat1& operator=(kmMat1&& b)
	{
		if(IsPinned()) *this = b; else Move(b);
		return *this; 
	};

	// allocate memory
	void Create(int64 n1)	            { Create(n1, n1); };
	void Create(int64 n1, int64 size)
	{
		ASSERTA( n1 <= size, "[kmMat1::Create in 444] %lld <= %lld", n1, size);
		
		_n1 = n1; kmArr<T>::Create(size);
	};
	
	// release and create
	void Recreate(int64 n1)             { Release(); Create(n1); };
	void Recreate(int64 n1, int64 size) { Release(); Create(n1, size); };

	template<typename Y> void Recreate(const kmMat1<Y>& b) { Recreate(b.N1()); }

	// recreate if dimension is different
	int RecreateIf(int64 n1) { if(n1 != _n1) { Recreate(n1); return 1; } return 0; };

	template<typename Y> int RecreateIf(const kmMat1<Y>& b) { return RecreateIf(b.N1()); }

	// move 
	void Move(kmMat1& b) { kmArr<T>::Move(b); _n1 = b._n1; };

	// set array
	void Set(T* p, int64 n1)            { Set(p, n1, n1); };
	void Set(T* p, int64 n1, int64 size)
	{
		ASSERTA( n1 <= size, "[kmMat1::Set in 195]");

		_n1 = n1; kmArr<T>::Set(p, size);
	};

	// set array... a.Set(b)
	void Set(const kmMat1&    b) { Set(b.P(), b.N1(), b.Size()); };
	void Set(const kmArr<T>&  b) { Set(b.P(), b.Size());         };

	// operator to get data	
	T* P(int64 i1) const
	{
		if(i1 < 0) i1 += _n1; // * Note that i1 = -1 is the same as End()

		ASSERTA(0 <= i1 && i1 < _size , "[kmMat1::P in 951] %lld < %lld", i1, _size);

		return (_p + i1);
	};

	T* P() const { return _p; };

	T& operator()(int64 i1 = 0) const { return *P(i1); };
	T& a         (int64 i1 = 0) const { return *P(i1); };
	virtual T& v (int64 i1 = 0) const { return *P(i1); };

	// get linear interpolation
	float Li(float i1) const
	{
		const int64 i1n = MIN(MAX(0, (int)i1),_n1-2);
		const float i1r = i1 - float(i1n);

		return (1.f - i1r)*a(i1n)  + i1r*a(i1n + 1);
	};

	// operator to get data
	T* End () const { return _p + _n1 - 1; };
	T* End1() const { return _p + _n1;     };

	// restore the class including vfptr
	// * Note that this will clear and reset _p, _state, _size without release,
	// * which can cause a memory leak. So, you should use this very carefully.
	kmMat1<T>& Restore() { RestoreVfptr(*this); _state = 0; Init(); return *this; };
		
	/////////////////////////////////////////////////
	// general member functions

	// get mat1
	kmMat1<T> Mat1(kmI i1) const
	{
		i1.e = MIN(i1.e, _n1-1);

		return kmMat1<T>(P(i1.s), i1.Len());
	};

	// get info
	int64 N1() const { return _n1; };

	// get the number of real elements
	virtual int64 N() const { return _n1; };

	// get bytes of info members which is from _size to end
	virtual int64 GetInfoByte() const { return sizeof(*this) - 24;};

	// display member info
	virtual void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		kmArr<T>::PrintInfo();
		PRINTFA("  _n1    : %lld\n", _n1);
	};

	// display dimension
	virtual void PrintDim(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]", str);
		else               PRINTFA("[%p]", _p);

		PRINTFA(" %lld (%lld) \n", _n1, _size);
	};

	void PrintMat(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PrintDim(str);

		for(int64 i1 = 0; i1 < _n1  ; ++i1)
		{
			cout << i1 << " :\t" << a(i1) << endl;
		}
		cout << endl;
	};

	// compare
	template<typename Y> bool IsEqualSizeDim(const kmMat1<Y>& b) const { return (_n1 == b._n1); }

	// push back
	int64 PushBack()
	{
		if(_n1 == _size)
		{
			ASSERTA(IsCreated(),"[kmMat1::PushBack in 529]");
			Expand(_size);
		}
		return ++_n1 - 1;
	};
	int64 PushBack(const T& val)
	{
		PushBack();	*End() = val; return _n1 - 1;
	};

	// puch back using set if val is kmMat	
	int64 PushBackSet(const T& val)
	{
		PushBack(); End()->Set(val); return _n1 - 1;
	};

	// pop back
	T* PopBack()
	{
		if(_n1 == 0) return nullptr;

		--_n1;  return _p + _n1;
	};

	// insert
	kmMat1& Insert(int64 idx, const T& val)
	{
		// check idx
		ASSERTA(0 <= idx && idx <= _n1,"[kmMat1::Insert in 937]");

		if(idx == _n1) { PushBack(val); return *this; }

		// copy to tmp
		kmMat1 tmp0 = Mat1(kmI(idx, _n1 - 1)), tmp = tmp0;

		// pushback
		PushBack();

		// insert val
		a(idx) = val;

		//copy from tmp
		Mat1(kmI(idx + 1, _n1 - 1)) = tmp;

		return *this;
	};

	// erase one element
	kmMat1& Erase(int64 idx)
	{
		// check idx
		if(idx < 0) idx += _n1; // * Note that i1 = -1 is the same as End()

		ASSERTA(0 <= idx && idx < _n1,"[kmMat1::Insert in 937]");

		if(idx == _n1 - 1) { PopBack(); return *this; }

		// copy to temp
		kmMat1 tmp0 = Mat1(kmI(idx + 1, _n1 - 1)), tmp = tmp0;

		// copy from tmp
		Mat1(kmI(idx, _n1 - 2)) = tmp;

		// popback
		PopBack();

		return *this;
	};

	// erase block
	kmMat1& Erase(kmI i1)
	{
		// check idx
		i1.s = MAX(i1.s, 0);
		i1.e = MIN(i1.e, _n1 - 1);

		if(i1.e == _n1 - 1) { _n1 = i1.s; return *this; }
		
		// copy to temp
		kmMat1 tmp0 = Mat1(kmI(i1.e + 1, _n1 - 1)), tmp = tmp0;

		// decrease _n1
		_n1 = _n1 - i1.Len();

		// copy from tmp
		Mat1(kmI(i1.s, _n1 - 1)) = tmp;

		return *this;
	};

	// set n1
	void SetN1(int64 n1)
	{
		ASSERTA(n1 <= _size, "[kmMat1::SetN1 in 950] n1(%lld) <= _size(%lld)", n1, _size);
		_n1 = n1;
	};
	void IncN1(int64 dn1) { SetN1(_n1 + dn1); };
	void SetN1ToSize()    { _n1 = _size; };

	// pack... _size is forced to be _n1
	// * Note that this is recommened before WriteMat() to save capacity
	// * and to avoid errors caused by missing Restore().
	kmMat1& Pack() { _size = _n1; return *this; };

	// find
	int64 Find(const T& val) const
	{
		const int64 n = N();

		for(int64 i = 0; i < n; ++i)
		{
			if(v(i) == val) return i;
		}
		return -1; // not found
	};
};

//////////////////////////////////////////////////////////
// 2D matrix class
template<typename T> class kmMat2 : public kmMat1<T>
{
protected:
	// using for members of parents class
	using kmArr <T>::_p , kmArr <T>::_state, kmArr<T>::_size;
	using kmMat1<T>::_n1;
	
	// member variables
	int64 _p1 = 0;  // pitch of dim1
	int64 _n2 = 1;  // number of dim2

public:
	// using for functions of parents class
	using kmArr<T>::Expand;
	using kmArr<T>::Release;
	using kmArr<T>::IsPinned;
	using kmArr<T>::IsCreated;
	using kmArr<T>::GetKmClass;
	using kmArr<T>::SetVal;
	using kmArr<T>::Copy;
	using kmArr<T>::RestoreVfptr;

	/////////////////////////////////////////////////
	// basic member functions
public:
	virtual void Init() { kmMat1<T>::Init(); _p1 = 0; _n2 = 1; };

	// constructor
	kmMat2() {};
	kmMat2(int64 n) { Create(n, 1);};

	kmMat2(      int64 n1, int64 n2) { Create(n1, n2);};
	kmMat2(T* p, int64 n1, int64 n2) { Set(p, n1, n2);};

	kmMat2(      int64 n1, int64 n2, int64 p1) { CreateP(n1, n2, p1);};
	kmMat2(T* p, int64 n1, int64 n2, int64 p1) { SetP(p, n1, n2, p1);};

	kmMat2(      int64 n1, int64 n2, int64 p1, int64 size) { CreateP(n1, n2, p1, size);};
	kmMat2(T* p, int64 n1, int64 n2, int64 p1, int64 size) { SetP(p, n1, n2, p1, size);};

	kmMat2(const initializer_list<T>& val)
	{	
		Create(val.size(), 1);
		const T *pval = val.begin();
		for(int64 i = 0; i < _size; ++i) *(_p + i) = *(pval++);
	};

	// * Note that using Set() is not safe since an input variable can be right-value.
	kmMat2(const kmMat1<T>& a) { Create(a.N1(),1); Copy(a.Begin()); };

	// destructor
	virtual ~kmMat2() {};
	
	// copy constructor	
	kmMat2(const kmMat2& b)
	{		
		Create(b.N1(), b.N2());
		if(IsEqualSizeDimP(b)) Copy(b.Begin()); else Copy(b);
	};

	template<typename Y>
	kmMat2(const kmMat2<Y>& b)
	{
		Create(b.N1(), b.N2());
		CopyTCast(b);
	}

	// move constructor
	kmMat2(kmMat2&& b) { Move(b); };

	// assignment operator
	kmMat2& operator=(const kmMat2& b)
	{
		RecreateIf(b);
		if(IsEqualSizeDimP(b)) Copy(b.Begin()); else Copy(b);
		return *this;
	};

	template<typename Y>
	kmMat2& operator=(const kmMat2<Y>& b) { RecreateIf(b); CopyTCast(b); return *this; }

	// assignment operator for scalar
	kmMat2& operator=(const T b) { SetVal(b); return *this; };

	// move assignment operator
	kmMat2& operator=(kmMat2&& b)
	{
		if(IsPinned()) *this = b; else Move(b);
		return *this; 
	};
	
	// allocate memory
	void Create (int64 n1, int64 n2 = 1)         { CreateP(n1, n2, n1, n1*n2); };
	void Create (int64 n1, int64 n2, int64 size) { CreateP(n1, n2, n1, size ); };
	void CreateP(int64 n1, int64 n2, int64 p1)   { CreateP(n1, n2, p1, p1*n2); };
	void CreateP(int64 n1, int64 n2, int64 p1, int64 size)
	{	
		ASSERTA(p1*n2 <= size, "[kmMat2::CreateP in 261]");

		_n1 = n1; _n2 = n2; _p1 = p1; kmArr<T>::Create(size);
	};

	// release and create
	void Recreate (int64 n1, int64 n2 = 1)         { Release(); Create (n1, n2      ); };
	void Recreate (int64 n1, int64 n2, int64 size) { Release(); Create (n1, n2, size); };
	void RecreateP(int64 n1, int64 n2, int64 p1)   { Release(); CreateP(n1, n2, p1  ); };
	void RecreateP(int64 n1, int64 n2, int64 p1, int64 size)
	{
		Release(); CreateP(n1, n2, p1, size);
	};
	template<typename Y> void Recreate(const kmMat2<Y>& b) { Recreate(b.N1(), b.N2()); }

	// recreate if	
	int RecreateIf(int64 n1, int64 n2)
	{
		if(n1 != _n1 || n2 != _n2) { Recreate(n1, n2); return 1; } return 0;
	};	
	template<typename Y> int RecreateIf(const kmMat2<Y>& b) { return RecreateIf(b.N1(), b.N2()); }

	// move 
	void Move(kmMat2& b) { kmMat1<T>::Move(b); _n2 = b._n2; _p1 = b._p1; };

	// set array	
	void Set (T* p, int64 n1, int64 n2 = 1)         { SetP(p, n1, n2, n1, n1*n2); };
	void Set (T* p, int64 n1, int64 n2, int64 size) { SetP(p, n1, n2, n1, size ); };
	void SetP(T* p, int64 n1, int64 n2, int64 p1)   
	{ 
		SetP(p, n1, n2, p1, CalcMinSize(n1,n2,p1)); 
	};
	void SetP(T* p, int64 n1, int64 n2, int64 p1, int64 size)
	{
		const int64 min_size = CalcMinSize(n1,n2,p1);

		ASSERTA(min_size <= size, "[kmMat2::SetP in 284]");

		_n1 = n1; _n2 = n2; _p1 = p1; kmArr<T>::Set(p, size);
	};

	// calculate the minimum size.
	// * Note that (p - n) of the last element is actually not used.
	static int64 CalcMinSize(int64 n1, int64 n2, int64 p1)
	{
		return p1*n2 - (p1 - n1);
	};

	// set array... a.Set(b)
	void Set(const kmMat2<T>& a) { SetP(a.P(), a.N1(), a.N2(), a.P1(), a.Size()); };
	void Set(const kmMat1<T>& a) { Set (a.P(), a.N1(),      1, a.Size()); };
	void Set(const kmArr <T>& a) { Set (a.P(), a.Size(),    1, a.Size()); };
	
	// operator to get data
	T* P(int64 i1, int64 i2) const
	{
		const int64 idx = i1 + _p1*i2;

		ASSERTA(0 <= idx && idx < _size,"[kmMat2::P in 610] idx: %lld (%lld, %lld), _size: %lld", idx, i1, i2, _size);

		return _p + idx;
	};

	T* P(int64 i) const
	{
		if(i < 0) i += _n1*_n2; // * Note that i1 = -1 is the same as End()

		const int64 i2 = i/_n1; i-= i2*_n1;		

		return P(i, i2);
	};

	T* P() const { return _p; };

	T& operator()(int64 i1, int64 i2) const { return *P(i1, i2); };
	T& operator()(int64 i1          ) const { return *P(i1);     };
	T& operator()(                  ) const { return *_p;        };
	T& a         (int64 i1, int64 i2) const { return *P(i1, i2); };
	T& a         (int64 i1          ) const { return *P(i1);     };
	T& a         (                  ) const { return *_p;        };
	virtual T& v (int64 i1          ) const { return *P(i1);     };

	T* End() const { return P(_n1-1, _n2-1); };

	// restore the class including vfptr
	// * Note that this will clear and reset _p, _state, _size without release,
	// * which can cause a memory leak. So, you should use this very carefully.
	kmMat2& Restore() { RestoreVfptr(*this); _state = 0; Init(); return *this; };

	// get bi-linear interpolation	
	float Li(float i1, float i2) const
	{
		const int64 i1n = MIN(MAX(0, (int)i1),_n1-2);
		const int64 i2n = MIN(MAX(0, (int)i2),_n2-2);
		const float i1r = i1 - float(i1n);
		const float i2r = i2 - float(i2n);
		const float v00 = (1.f - i1r)*a(i1n, i2n    )  + i1r*a(i1n + 1, i2n    );
		const float v01 = (1.f - i1r)*a(i1n, i2n + 1)  + i1r*a(i1n + 1, i2n + 1);

		return (1.f - i2r)*v00  + i2r*v01;
	};

	// get linear interpolation	
	float Li(float i1, int64 i2) const
	{
		const int64 i1n = MIN(MAX(0, (int)i1),_n1-2);
		const float i1r = i1 - float(i1n);

		return (1.f - i1r)*a(i1n, i2)  + i1r*a(i1n + 1, i2);
	};
	float Li(float i1, int i2) const { return Li(i1, (int64)i2); };

	// get linear interpolation
	float Li(int64 i1, float i2) const
	{
		const int64 i2n = MIN(MAX(0, (int)i2),_n2-2);
		const float i2r = i2 - float(i2n);

		return (1.f - i2r)*a(i1, i2n)  + i2r*a(i1, i2n + 1);
	};
	float Li(int i1, float i2) const { return Li((int64)i1, i2); };

	// reshape
	void Reshape(int64 n1, int64 n2, int64 p1 = 0)
	{
		// init arguement
		if(p1 == 0) p1 = n1;

		// check size
		ASSERTA(p1*n2 <= _size, "[kmMat2::Reshape in 977]");

		// reset members
		_n1 = n1; _n2 = n2;	_p1 = p1;
	};
		
	/////////////////////////////////////////////////
	// general member functions

	// get mat1
	kmMat1<T> Mat1(int64 i2) const { return kmMat1<T>(P(0,i2), _n1);};

	// get mat2
	kmMat2<T> Mat2(kmI i1, kmI i2) const
	{
		i1.e = MIN(i1.e, _n1-1);
		i2.e = MIN(i2.e, _n2-1);

		return kmMat2<T>(P(i1.s, i2.s), i1.Len(), i2.Len(), _p1);
	};

	// get mat2 with linear interpolation
	kmMat2<T> Mat2Li(float i1s, float d1, int n1, kmI i2) const
	{
		if(i2.e == end64) i2.e = _n2 - 1;

		kmMat2<T> mat(n1, i2.Len());

		for(int64 j2 = i2.s; j2 <= i2.e; ++j2)
		for(int64 j1 =    0; j1 <  n1  ; ++j1)
		{
			mat(j1,j2) = Li(i1s + d1*j1,j2);
		}
		return mat;
	};

	// get flat matrix
	kmMat1<T> Flat() const
	{	
		ASSERTA(_n1 == _p1, "[kmMat2::Flat in 1310]");
	
		return kmMat1<T>(P(), N());
	};

	// set zero
	void SetZero() { SetVal(0); };

	// get info
	int64 N2() const { return _n2; };
	int64 P1() const { return _p1; };
	
	// get the number of real elements
	virtual int64 N() const { return _n1*_n2;};

	// get bytes of info members which is from _size to end
	virtual int64 GetInfoByte() const { return sizeof(*this) - 24;};

	// display member info
	virtual void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		kmMat1<T>::PrintInfo();
		PRINTFA("  _p1    : %lld\n", _p1);
		PRINTFA("  _n2    : %lld\n", _n2);
	};

	// display dimension
	virtual void PrintDim(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]", str);
		else               PRINTFA("[%p]", _p);

		PRINTFA(" %lld, %lld (%lld) \n", _n1, _n2, _size);
	};

	void PrintMat(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PrintDim(str);

		for(int64 i1 = 0; i1 < _n1  ; ++i1)
		{
			cout << i1 << " :\t";
			for(int64 i2 = 0; i2 < _n2; ++i2) cout << a(i1,i2) << " \t";
			cout << endl;
		}
		cout << endl;
	};

	// compare
	template<typename Y> bool IsEqualSizeDim (const kmMat2<Y>& b) const 
	{
		return _n1 == b.N1() && _n2 == b.N2();
	}

	template<typename Y> bool IsEqualSizeDimP(const kmMat2<Y>& b) const
	{
		return (_p1 == b.P1()) && IsEqualSizeDim(b);
	}

	template<typename Y> bool IsEqualSizeAll (const kmMat2<Y>& b) const 
	{
		return IsEqualSizeDimP(b) && IsEqualSize(b);
	}

	// get transpose matrix
	kmMat2 Tp() const
	{
		kmMat2 des(_n2, _n1);

		for(int64 i2 = 0; i2 < _n2; ++i2)
		for(int64 i1 = 0; i1 < _n1; ++i1)
		{
			des(i2,i1) = *P(i1,i2);
		}

		return des;	// * Note that des is the local variable,
		            // * so the return type cannot be reference type.
	};

	void Cat(const kmMat2& b, const kmMat2& c, const int dim = 1)
	{
		if(dim == 1)      // catenate as 1st dim
		{			
			const int64 n1  = b.N1() + c.N1();
			const int64 n2  = MAX(b.N2(), c.N2());
			const int64 bn1 = b.N1();

			Recreate(n1, n2);

			for(int64 i2 = 0; i2 < _n2; ++i2)
			for(int64 i1 = 0; i1 < _n1; ++i1)
			{
				*P(i1, i2) = (i1 < bn1) ? b(i1, i2) : c(i1 - bn1, i2);
			}
		}
		else if(dim == 2) // catenate along 2nd dim
		{			
			const int64 n1  = MAX(b.N1(), c.N1());
			const int64 n2  = b.N2() + c.N2();
			const int64 bn2 = b.N2();

			Recreate(n1, n2);

			for(int64 i2 = 0; i2 < _n2; ++i2)
			for(int64 i1 = 0; i1 < _n1; ++i1)
			{
				*P(i1, i2) = (i2 < bn2) ? b(i1, i2) : c(i1, i2 - bn2);
			}
		}
		else
		{
			PRINTFA("* [kmMat2::Cat] dim (%d) is not supported\n", dim);
		}
	};

	kmMat2 Repmat(const int64 r1, const int64 r2)
	{
		kmMat2 b(_n1*r1, _n2*r2);

		for(int64 j2 = 0; j2 < r2 ; ++j2)
		for(int64 j1 = 0; j1 < r1 ; ++j1)
		for(int64 i2 = 0; i2 < _n2; ++i2)
		for(int64 i1 = 0; i1 < _n1; ++i1)
		{
			b(i1 + j1*_n1, i2 + j2*_n2) = a(i1, i2);
		}

		return b;
	};

	/////////////////////////////////////////////////
	// math member functions

	// calc determinant
	T Det() const
	{
		// check size
		ASSERTA(_n1 == _n2   ,"[kmMat2::Det in 827]");		

		// calc determinant.. det
		T det = 0;

		if(_n1 == 1)
		{
			det = a(0,0);
		}
		else if(_n1 == 2)
		{
			det = a(0,0)*a(1,1) - a(1,0)*a(0,1);
		}

		return det;
	};

	// inverse matrix a = inv(b)... a.inv(b)
	kmMat2& Inv(const kmMat2& b)
	{
		// check size
		ASSERTA(b._n1 == b._n2   , "[kmMat2::Inv in 828]");
		ASSERTA(IsEqualSizeDim(b), "[kmMat2::Inv in 829]");

		// get det
		T det = b.Det();

		// check determiant
		ASSERTA(abs(det) > 0, "[kmMat2::Inv in 856]");

		if(_n1 == 1)
		{
			a(0,0) = (T) 1/det;
		}
		else if(_n1 == 2)
		{
			a(0,0) =  b(1,1)/det;
			a(1,1) =  b(0,0)/det;
			a(0,1) = -b(0,1)/det;
			a(1,0) = -b(1,0)/det;
		}
		return *this;
	};

	// inverse matrix b = inv(a)... b = a.inv()
	kmMat2 Inv() const
	{
		kmMat2 b(_n1, _n2);
		return b.Inv(*this);
	};

	// get diagonal matrix... a = diag(b)... a.Diag(b)
	kmMat2& Diag(const kmMat2& b)
	{
		// check size
		ASSERTA(b._n1 == b._n2   , "[kmMat2::Diag in 1160]");
		ASSERTA(IsEqualSizeDim(b), "[kmMat2::Diag in 1161]");

		// clear output matrix
		SetZero();

		// set diagonal terms
		for(int i = 0; i < _n1; ++i) a(i,i) = b(i,i);

		return *this;
	};

	// get diagnonal matrix... b = diag(a)... b = a.Diag();
	kmMat2 Diag() const
	{
		kmMat2 b(_n1, _n2);
		return b.Diag(*this);
	};

	// get sub element using bilinear interpolation
	T GetSub(float x, float y) const 
	{
		const int64 x0 = (int64) x, x1 = x0 + 1;
		const int64 y0 = (int64) y, y1 = y0 + 1;
		const float  xr  = x - (float) x0;
		const float  yr  = y - (float) y0;

		T vy0 = a(x0, y0)*(1.f - xr) + a(x1, y0)*xr;
		T vy1 = a(x0, y1)*(1.f - xr) + a(x1, y1)*xr;

		return vy0*(1.f - yr) + vy1*yr;
	};	
};

//////////////////////////////////////////////////////////
// 3D matrix class
template<typename T> class kmMat3 : public kmMat2<T>
{
protected:
	// using for members of parents class
	using kmArr <T>::_p , kmArr <T>::_state, kmArr<T>::_size;
	using kmMat1<T>::_n1;
	using kmMat2<T>::_n2, kmMat2<T>::_p1;

	// member variables
	int64 _p2 = 0;  // pitch of dim2
	int64 _n3 = 1;  // number of dim3

public:
	// using functions of parents class
	using kmArr<T>::Expand;
	using kmArr<T>::Release;
	using kmArr<T>::IsPinned;
	using kmArr<T>::IsCreated;
	using kmArr<T>::GetKmClass;
	using kmArr<T>::SetVal;
	using kmArr<T>::Copy;

	/////////////////////////////////////////////////
	// basic member functions
public:
	virtual void Init() { kmMat2<T>::Init(); _p2 = 0; _n3 = 1; };

	// constructor
	kmMat3() {};
	kmMat3(int64 n) { Create(n, 1, 1);};

	kmMat3(      int64 n1, int64 n2, int64 n3) { Create(n1, n2, n3);};
	kmMat3(T* p, int64 n1, int64 n2, int64 n3) { Set(p, n1, n2, n3);};

	kmMat3(      int64 n1, int64 n2, int64 n3, int64 p1) { CreateP(n1, n2, n3, p1);};
	kmMat3(T* p, int64 n1, int64 n2, int64 n3, int64 p1) { SetP(p, n1, n2, n3, p1);};

	kmMat3(      int64 n1, int64 n2, int64 n3, int64 p1, int64 p2) { CreateP(n1, n2, n3, p1, p2);};
	kmMat3(T* p, int64 n1, int64 n2, int64 n3, int64 p1, int64 p2) { SetP(p, n1, n2, n3, p1, p2);};

	kmMat3(      int64 n1, int64 n2, int64 n3, int64 p1, int64 p2, int64 size) { CreateP(n1, n2, n3, p1, p2, size);};
	kmMat3(T* p, int64 n1, int64 n2, int64 n3, int64 p1, int64 p2, int64 size) { SetP(p, n1, n2, n3, p1, p2, size);};
	
	kmMat3(const initializer_list<T>& val)
	{
		Create(val.size(), 1, 1);
		const T *pval = val.begin();
		for(int64 i = 0; i < _size; ++i) *(_p + i) = *(pval++);
	};

	// * Note that using Set() is not safe since an input variable can be right-value.
	kmMat3(const kmMat1<T>& a) { Create(a.N1(),      1, 1); Copy(a.Begin()); };
	kmMat3(const kmMat2<T>& a) { Create(a.N1(), a.N2(), 1); Copy(a); };

	// destructor
	virtual ~kmMat3() {};

	// copy constructor	
	kmMat3(const kmMat3& b)
	{		
		Create(b.N1(), b.N2(), b.N3());
		if(IsEqualSizeDimP(b)) Copy(b.Begin()); else Copy(b);
	};

	template<typename Y>
	kmMat3(const kmMat3<Y>& b)
	{
		Create(b.N1(), b.N2(), b.N3());
		CopyTCast(b);
	}

	// move constructor
	kmMat3(kmMat3&& b) { Move(b); };

	// assignment operator	
	kmMat3& operator=(const kmMat3& b)
	{
		RecreateIf(b);
		if(IsEqualSizeDimP(b)) Copy(b.Begin()); else Copy(b);
		return *this;
	};

	template<typename Y>
	kmMat3& operator=(const kmMat3<Y>& b) { RecreateIf(b); CopyTCast(b); return *this; }

	// assignment operator for scalar
	kmMat3& operator=(const T b) { SetVal(b); return *this; };
	
	// move assignment operator
	kmMat3& operator=(kmMat3&& b)	
	{
		if(IsPinned()) *this = b; else Move(b); 
		return *this; 
	};

	// allocate memory
	void Create (int64 n1, int64 n2, int64 n3 = 1)                 { CreateP(n1, n2, n3, n1, n2, n1*n2*n3); };
	void Create (int64 n1, int64 n2, int64 n3, int64 size)         { CreateP(n1, n2, n3, n1, n2, size    ); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 p1  )         { CreateP(n1, n2, n3, p1, n2, p1*n2*n3); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 p1, int64 p2) { CreateP(n1, n2, n3, p1, p2, p1*p2*n3); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 p1, int64 p2, int64 size)
	{
		ASSERTA(p1*p2*n3 <= size,"[kmMat3::CreateP in 778]");

		_n1 = n1; _n2 = n2; _n3 = n3; _p1 = p1; _p2 = p2; kmArr<T>::Create(size);
	};

	// release and create
	void Recreate (int64 n1, int64 n2, int64 n3 = 1)                 { Release(); Create (n1, n2, n3        ); };
	void Recreate (int64 n1, int64 n2, int64 n3, int64 size)         { Release(); Create (n1, n2, n3, size  ); };
	void RecreateP(int64 n1, int64 n2, int64 n3, int64 p1  )         { Release(); CreateP(n1, n2, n3, p1    ); };
	void RecreateP(int64 n1, int64 n2, int64 n3, int64 p1, int64 p2) { Release(); CreateP(n1, n2, n3, p1, p2); };
	void RecreateP(int64 n1, int64 n2, int64 n3, int64 p1, int64 p2, int64 size)
	{
		Release(); CreateP(n1, n2, n3, p1, p2, size);
	};
	template<typename Y> void Recreate(const kmMat3<Y>& b) { Recreate(b.N1(), b.N2(), b.N3()); }

	// recreate if dimension is different	
	int RecreateIf(int64 n1, int64 n2, int64 n3)
	{
		if(n1 != _n1 || n2 != _n2 || n3 != _n3) { Recreate(n1, n2, n3); return 1; } return 0;
	};	
	template<typename Y> int RecreateIf(const kmMat3<Y>& b) { return RecreateIf(b.N1(), b.N2(), b.N3()); }

	// move 
	void Move(kmMat3& b) { kmMat2<T>::Move(b); _n3 = b._n3; _p2 = b._p2; };

	// set array	
	void Set (T* p, int64 n1, int64 n2, int64 n3 = 1)         { SetP(p, n1, n2, n3, n1, n2);      };
	void Set (T* p, int64 n1, int64 n2, int64 n3, int64 size) { SetP(p, n1, n2, n3, n1, n2, size);};
	void SetP(T* p, int64 n1, int64 n2, int64 n3, int64 p1  ) { SetP(p, n1, n2, n3, p1, n2);      };
	void SetP(T* p, int64 n1, int64 n2, int64 n3, int64 p1, int64 p2)
	{
		SetP(p, n1, n2, n3, p1, p2, CalcMinSize(n1,n2,n3,p1,p2));
	};
	void SetP(T* p, int64 n1, int64 n2, int64 n3, int64 p1, int64 p2, int64 size)
	{
		const int64 min_size = CalcMinSize(n1,n2,n3,p1,p2);

		ASSERTA(min_size <= size, "[kmMat3::SetP in 800] %lld <= %lld", min_size, size);

		_n1 = n1; _n2 = n2; _n3 = n3; _p1 = p1; _p2 = p2; kmArr<T>::Set(p, size);
	};

	// calculate the minimum size.
	// * Note that (p - n) of the last element is actually not used.
	static int64 CalcMinSize(int64 n1, int64 n2, int64 n3, int64 p1, int64 p2)
	{
		return p1*p2*n3 - p1*(p2 - n2) - (p1 - n1);
	};

	// set array... a.Set(b)
	void Set(const kmMat3<T>& b) { SetP(b.P(), b.N1(), b.N2(), b.N3(), b.P1(), b.P2(), b.Size()); };
	void Set(const kmMat2<T>& b) { SetP(b.P(), b.N1(), b.N2(),      1, b.P1(), b.N2(), b.Size()); };
	void Set(const kmMat1<T>& b) { Set (b.P(), b.N1(),      1,      1, b.Size()); };
	void Set(const kmArr <T>& b) { Set (b.P(), b.Size(),    1,      1, b.Size()); };
	
	// operator to get data
	T* P(int64 i1, int64 i2 , int64 i3) const
	{
		const int64 idx = i1 + _p1*(i2 + _p2*i3);

		ASSERTA(0 <= idx && idx < _size, "[kmMat3::P in 816] %lld < %lld", idx, _size);

		return _p + idx;
	};

	T* P(int64 i1, int64 i) const
	{
		const int64 i3 = i/_n2; i-= i3*_n2;		

		return P(i1, i, i3);
	};

	T* P(int64 i) const
	{
		if(i < 0) i += _n1*_n2*_n3; // * Note that i1 = -1 is the same as End()

		const int64 n12 = _n1*_n2;
		const int64 n1  = _n1;

		const int64 i3  = i/n12; i-= i3*n12;
		const int64 i2  = i/n1;  i-= i2*n1;
		
		return P(i, i2, i3);
	};

	T* P() const { return _p; };
	
	T& operator()(int64 i1, int64 i2, int64 i3) const { return *P(i1, i2, i3);};
	T& operator()(int64 i1, int64 i2          ) const { return *P(i1, i2);    };
	T& operator()(int64 i1                    ) const { return *P(i1);        };
	T& operator()(                            ) const { return *_p;           };
	T& a         (int64 i1, int64 i2, int64 i3) const { return *P(i1, i2, i3);};
	T& a         (int64 i1, int64 i2          ) const { return *P(i1, i2);    };
	T& a         (int64 i1                    ) const { return *P(i1);        };
	T& a         (                            ) const { return *_p;           };
	virtual T& v (int64 i1                    ) const { return *P(i1);        };
		       
	T* End() const { return P(_n1-1, _n2-1, _n3-1); };

	// restore the class including vfptr
	// * Note that this will clear and reset _p, _state, _size without release,
	// * which can cause a memory leak. So, you should use this very carefully.
	kmMat3& Restore() { RestoreVfptr(*this); _state = 0; Init(); return *this; };

	// reshape
	void Reshape(int64 n1, int64 n2, int64 n3, int64 p1 = 0, int64 p2 = 0)
	{
		// init arguement
		if(p1 == 0) p1 = n1;
		if(p2 == 0) p2 = n2;

		// check size
		ASSERTA(p1*p2*n3 <= _size, "[kmMat3::Reshape in 1944]");

		// reset members
		_n1 = n1; _n2 = n2;	_n3 = n3; _p1 = p1; _p2 = p2;
	};
	
	/////////////////////////////////////////////////
	// general member functions

	// get mat1
	kmMat1<T> Mat1(int64 i23)          const { return kmMat1<T>(P(0,i23)  , _n1);};
	kmMat1<T> Mat1(int64 i2, int64 i3) const { return kmMat1<T>(P(0,i2,i3), _n1);};

	// get mat2
	kmMat2<T> Mat2(int64 i3) const { return kmMat2<T>(P(0,0,i3), _n1, _n2, _p1);};

	// get mat3
	kmMat3<T> Mat3(kmI i1, kmI i2, kmI i3) const
	{
		i1.e = MIN(i1.e, _n1-1);
		i2.e = MIN(i2.e, _n2-1);
		i3.e = MIN(i3.e, _n3-1);

		return kmMat3<T>(P(i1.s, i2.s, i3.s), i1.Len(), i2.Len(), i3.Len(), _p1, _p2);
	};

	// get info
	int64 N3() const { return _n3; };
	int64 P2() const { return _p2; };
	
	// get the number of real elements
	virtual int64 N() const { return _n1*_n2*_n3;};

	// get bytes of info members which is from _size to end
	virtual int64 GetInfoByte() const { return sizeof(*this) - 24;};
	
	// display member info
	virtual void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		kmMat2<T>::PrintInfo();
		PRINTFA("  _p2    : %lld\n", _p2);
		PRINTFA("  _n3    : %lld\n", _n3);		
	};

	// display dimension
	virtual void PrintDim(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]", str);
		else               PRINTFA("[%p]", _p);

		PRINTFA(" %lld, %lld, %lld (%lld) \n", _n1, _n2, _n3, _size);
	};

	void PrintMat(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PrintDim(str);
				
		const int64 n_mat = N()/(_n1*_n2);

		for(int64 im = 0; im < n_mat; ++im)
		{
			printf("[ %d / %d ]\n", im, n_mat);

			Mat2(im).PrintMat();
		}
		cout << endl;
	};

	// compare
	template<typename Y> bool IsEqualSizeDim(const kmMat3<Y>& b) const
	{
		return _n1 == b.N1() && _n2 == b.N2() && _n3 == b.N3();
	}

	template<typename Y> bool IsEqualSizeDimP(const kmMat3<Y>& b) const
	{
		return (_p1 == b.P1()) && (_p2 == b.P2()) && IsEqualSizeDim(b);
	}

	template<typename Y> bool IsEqualSizeAll(const kmMat3<Y>& b) const
	{
		return IsEqualSizeDimP(b) && IsEqualSize(b);
	}

	// get transpose matrix
	kmMat3 Tp() const
	{
		kmMat3 des(_n2, _n1, _n3);

		for(int64 i3 = 0; i3 < _n3; ++i3)
		for(int64 i2 = 0; i2 < _n2; ++i2)
		for(int64 i1 = 0; i1 < _n1; ++i1)
		{
			des(i2, i1, i3) = *P(i1, i2, i3);
		}

		return des;	// * Note that des is the local variable,
		            // * so the return type cannot be reference type.
	};

	// get swap matrix
	kmMat3 Swap13() const
	{
		kmMat3 des(_n3, _n2, _n1);

		for(int64 i3 = 0; i3 < _n3; ++i3)
		for(int64 i2 = 0; i2 < _n2; ++i2)
		for(int64 i1 = 0; i1 < _n1; ++i1)
		{
			des(i3, i2, i1) = *P(i1, i2, i3);
		}

		return des;	// * Note that des is the local variable,
		            // * so the return type cannot be reference type.
	};
};

//////////////////////////////////////////////////////////
// 4D matrix class
template<typename T> class kmMat4 : public kmMat3<T>
{
protected:
	// using for members of parents class
	using kmArr <T>::_p , kmArr <T>::_state, kmArr<T>::_size;
	using kmMat1<T>::_n1;
	using kmMat2<T>::_n2, kmMat2<T>::_p1;
	using kmMat3<T>::_n3, kmMat3<T>::_p2;

	// member variables
	int64 _p3 = 0;   // pitch of dim3
	int64 _n4 = 1;   // number of dim4
	
public:
	// using for functions of parents class
	using kmArr<T>::Expand;
	using kmArr<T>::Release;
	using kmArr<T>::IsPinned;
	using kmArr<T>::IsCreated;
	using kmArr<T>::GetKmClass;
	using kmArr<T>::SetVal;
	using kmArr<T>::RestoreVfptr;
	using kmArr<T>::Copy;

	/////////////////////////////////////////////////
	// basic member functions
public:
	virtual void Init() { kmMat3<T>::Init(); _p3 = 0; _n4 = 1; };

	// constructor
	kmMat4() {};
	kmMat4(int64 n) { Create(n, 1, 1, 1);};

	kmMat4(      int64 n1, int64 n2, int64 n3, int64 n4) { Create(n1, n2, n3, n4);};
	kmMat4(T* p, int64 n1, int64 n2, int64 n3, int64 n4) { Set(p, n1, n2, n3, n4);};

	kmMat4(      int64 n1, int64 n2, int64 n3, int64 n4, int64 p1) { CreateP(n1, n2, n3, n4, p1);};
	kmMat4(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1) { SetP(p, n1, n2, n3, n4, p1);};

	kmMat4(      int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2) { CreateP(n1, n2, n3, n4, p1, p2);};
	kmMat4(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2) { SetP(p, n1, n2, n3, n4, p1, p2);};

	kmMat4(      int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3) { CreateP(n1, n2, n3, n4, p1, p2, p3);};
	kmMat4(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3) { SetP(p, n1, n2, n3, n4, p1, p2, p3);};
												  
	kmMat4(      int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3, int64 size) { CreateP(n1, n2, n3, n4, p1, p2, p3, size);};
	kmMat4(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3, int64 size) { SetP(p, n1, n2, n3, n4, p1, p2, p3, size);};

	kmMat4(const initializer_list<T>& val)
	{
		Create(val.size(), 1, 1, 1);
		const T *pval = val.begin();
		for(int64 i = 0; i < _size; ++i) *(_p + i) = *(pval++);
	};	

	// * Note that using Set() is not safe since an input variable can be right-value.	
	kmMat4(const kmMat1<T>& a) { Create(a.N1(),      1,      1, 1); Copy(a.Begin()); };
	kmMat4(const kmMat2<T>& a) { Create(a.N1(), a.N2(),      1, 1); Copy(a); };
	kmMat4(const kmMat3<T>& a) { Create(a.N1(), a.N2(), a.N3(), 1); Copy(a); };

	// destructor
	virtual ~kmMat4() {};

	// copy constructor	
	kmMat4(const kmMat4& b)
	{		
		Create(b.N1(), b.N2(), b.N3(), b.N4());
		if(IsEqualSizeDimP(b)) Copy(b.Begin()); else Copy(b);
	};
	
	template<typename Y>
	kmMat4(const kmMat4<Y>& b)
	{
		Create(b.N1(), b.N2(), b.N3(), b.N4());
		CopyTCast(b);
	}

	// move constructor
	kmMat4(kmMat4&& b) { Move(b); };

	// assignment operator
	kmMat4& operator=(const kmMat4& b)
	{
		RecreateIf(b);
		if(IsEqualSizeDimP(b)) Copy(b.Begin()); else Copy(b);
		return *this;
	}

	template<typename Y>
	kmMat4& operator=(const kmMat4<Y>& b) { RecreateIf(b); CopyTCast(b); return *this; }

	// assignment operator for scalar
	kmMat4& operator=(const T b) { SetVal(b); return *this; };

	// move assignment operator
	kmMat4& operator=(kmMat4&& b)
	{
		if(IsPinned()) *this = b; else Move(b);
		return *this; 
	};

	// allocate memory
	void Create (int64 n1, int64 n2, int64 n3, int64 n4 = 1)                           { CreateP(n1, n2, n3, n4, n1, n2, n3, n1*n2*n3*n4); };
	void Create (int64 n1, int64 n2, int64 n3, int64 n4, int64 size)                   { CreateP(n1, n2, n3, n4, n1, n2, n3, size       ); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1  )                   { CreateP(n1, n2, n3, n4, p1, n2, n3, p1*n2*n3*n4); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2)           { CreateP(n1, n2, n3, n4, p1, p2, n3, p1*p2*n3*n4); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3) { CreateP(n1, n2, n3, n4, p1, p2, p3, p1*p2*p3*n4); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3, int64 size)
	{
		ASSERTA(p1*p2*p3*n4 <= size,"[kmMat4::CreateP in 1897]");

 		_n1 = n1; _n2 = n2; _n3 = n3; _n4 = n4; _p1 = p1; _p2 = p2; _p3 = p3; kmArr<T>::Create(size);
	};

	// release and create
	void Recreate (int64 n1, int64 n2, int64 n3, int64 n4 =1)                            { Release(); Create (n1, n2, n3, n4      );       };
	void Recreate (int64 n1, int64 n2, int64 n3, int64 n4, int64 size)                   { Release(); Create (n1, n2, n3, n4, size);       };
	void RecreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1  )                   { Release(); CreateP(n1, n2, n3, n4, p1    );     };
	void RecreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2)           { Release(); CreateP(n1, n2, n3, n4, p1, p2);     };
	void RecreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3) { Release(); CreateP(n1, n2, n3, n4, p1, p2, p3); };
	void RecreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3, int64 size)
	{
		Release(); CreateP(n1, n2, n3, n4, p1, p2, p3, size);
	};
	template<typename Y> void Recreate(const kmMat4<Y>& b) { Recreate(b.N1(), b.N2(), b.N3(), b.N4()); }

	// recreate if dimension is different
	int RecreateIf(int64 n1, int64 n2, int64 n3, int64 n4)
	{
		if(n1 != _n1 || n2 != _n2 || n3 != _n3 || n4 != _n4) { Recreate(n1,n2,n3,n4); return 1; }	return 0;
	};	
	template<typename Y> int RecreateIf(const kmMat4<Y>& b) { return RecreateIf(b.N1(), b.N2(), b.N3(), b.N4()); }

	// move
	void Move(kmMat4& b) { kmMat3<T>::Move(b); _n4 = b._n4; _p3 = b._p3; };

	// set array																									         
	void Set (T* p, int64 n1, int64 n2, int64 n3, int64 n4 = 1)                 { SetP(p, n1, n2, n3, n4, n1, n2, n3);      };
	void Set (T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 size)         { SetP(p, n1, n2, n3, n4, n1, n2, n3, size);};
	void SetP(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1  )         { SetP(p, n1, n2, n3, n4, p1, n2, n3);      };
	void SetP(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2) { SetP(p, n1, n2, n3, n4, p1, p2, n3);      };
	void SetP(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3) 
	{	
		SetP(p, n1, n2, n3, n4, p1, p2, p3, CalcMinSize(n1,n2,n3,n4,p1,p2,p3)); 
	};
	void SetP(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3, int64 size)
	{	
		const int64 min_size = CalcMinSize(n1,n2,n3,n4,p1,p2,p3);

		ASSERTA(min_size <= size, "[kmMat4::SetP in 1922] %lld <= %lld", min_size, size);

		_n1 = n1; _n2 = n2; _n3 = n3; _n4 = n4; _p1 = p1; _p2 = p2; _p3 = p3; kmArr<T>::Set(p, size);
	};
		
	// calculate the minimum size.
	// * Note that (p - n) of the last element is actually not used.
	static int64 CalcMinSize(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3)
	{
		return p1*p2*p3*n4 - p1*p2*(p3 - n3) - p1*(p2 - n2) - (p1 - n1);
	};

	// set array... a.Set(b)
	void Set(const kmMat4<T>& b) { SetP(b.P(), b.N1(), b.N2(), b.N3(), b.N4(), b.P1(), b.P2(), b.P3(), b.Size()); };
	void Set(const kmMat3<T>& b) { SetP(b.P(), b.N1(), b.N2(), b.N3(),      1, b.P1(), b.P2(), b.N3(), b.Size()); };
	void Set(const kmMat2<T>& b) { SetP(b.P(), b.N1(), b.N2(),      1,      1, b.P1(), b.N2(),     1 , b.Size()); };
	void Set(const kmMat1<T>& b) { Set (b.P(), b.N1(),     1,       1,      1, b.Size()); };
	void Set(const kmArr <T>& b) { Set (b.P(), b.Size(),   1,       1,      1, b.Size()); };
	
	// operator to get data
	T* P(int64 i1, int64 i2 , int64 i3, int64 i4) const
	{
		const int64 idx = i1 + _p1*(i2 + _p2*(i3 + _p3*i4));

		ASSERTA(0 <= idx && idx < _size, "[kmMat4::P in 1939] %lld < %lld", idx, _size);

		return _p + idx;
	};

	T* P(int64 i1, int64 i2, int64 i) const
	{
		const int64 i4 = i/_n3; i-= i4*_n3;

		return P(i1, i2, i, i4);
	};

	T* P(int64 i1, int64 i) const
	{
		const int64 n23 = _n2*_n3;
		const int64 n2  = _n2;

		const int64 i4 = i/n23; i-= i4*n23;
		const int64 i3 = i/n2;  i-= i3*n2;

		return P(i1, i, i3, i4);
	};

	T* P(int64 i) const
	{
		if(i < 0) i += _n1*_n2*_n3*_n4; // * Note that i1 = -1 is the same as End()

		const int64 n123 = _n1*_n2*_n3;
		const int64 n12  = _n1*_n2;
		const int64 n1   = _n1;

		const int64 i4 = i/n123; i-= i4*n123;
		const int64 i3 = i/n12;  i-= i3*n12;
		const int64 i2 = i/n1;   i-= i2*n1;

		return P(i, i2, i3, i4);
	};

	T* P() const { return _p; };
	
	T& operator()(int64 i1, int64 i2, int64 i3, int64 i4) const { return *P(i1, i2, i3, i4);};
	T& operator()(int64 i1, int64 i2, int64 i3          ) const { return *P(i1, i2, i3);    };
	T& operator()(int64 i1, int64 i2                    ) const { return *P(i1, i2);        };
	T& operator()(int64 i1                              ) const { return *P(i1);            };
	T& operator()(                                      ) const { return *_p;               };
	T& a         (int64 i1, int64 i2, int64 i3, int64 i4) const { return *P(i1, i2, i3, i4);};
	T& a         (int64 i1, int64 i2, int64 i3          ) const { return *P(i1, i2, i3);    };
	T& a         (int64 i1, int64 i2                    ) const { return *P(i1, i2);        };
	T& a         (int64 i1                              ) const { return *P(i1);            };
	T& a         (                                      ) const { return *_p;               };
	virtual T& v (int64 i1                              ) const { return *P(i1);            };
		       
	T* End() const { return P(_n1-1, _n2-1, _n3-1, _n4-1); };

	// restore the class including vfptr
	// * Note that this will clear and reset _p, _state, _size without release,
	// * which can cause a memory leak. So, you should use this very carefully.
	kmMat4& Restore() { RestoreVfptr(*this); _state = 0; Init(); return *this; };

	// reshape
	void Reshape(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1 = 0, int64 p2 = 0, int64 p3 = 0)
	{
		// init arguement
		if(p1 == 0) p1 = n1;
		if(p2 == 0) p2 = n2;
		if(p3 == 0) p3 = n3;

		// check size
		ASSERTA(p1*p2*p3*n4 <= _size, "[kmMat4::Reshape in 2280]");

		// reset members
		_n1 = n1; _n2 = n2;	_n3 = n3; _n4 = n4; _p1 = p1; _p2 = p2; _p3 = p3;
	};

	/////////////////////////////////////////////////
	// general member functions

	// get mat1
	kmMat1<T> Mat1(int64 idx)                    const { return kmMat1<T>(P(0,idx)     , _n1);};
	kmMat1<T> Mat1(int64 i2, int64 i3, int64 i4) const { return kmMat1<T>(P(0,i2,i3,i4), _n1);};

	// get mat2
	kmMat2<T> Mat2(int64 idx)          const { return kmMat2<T>(P(0,0,idx)  , _n1, _n2, _p1);};
	kmMat2<T> Mat2(int64 i3, int64 i4) const { return kmMat2<T>(P(0,0,i3,i4), _n1, _n2, _p1);};

	// get mat3
	kmMat3<T> Mat3(int64 idx) const
	{
		return kmMat3<T>(P(0,0,0,idx), _n1, _n2, _n3, _p1, _p2);
	};

	// get mat4
	kmMat4 Mat4(kmI i1, kmI i2, kmI i3, kmI i4) const
	{
		i1.e = MIN(i1.e, _n1-1);
		i2.e = MIN(i2.e, _n2-1);
		i3.e = MIN(i3.e, _n3-1);
		i4.e = MIN(i4.e, _n4-1);

		return kmMat4<T>(P(i1.s, i2.s, i3.s, i4.s), i1.Len(), i2.Len(), i3.Len(), i4.Len(), _p1, _p2, _p3);
	};

	// get info
	int64 N4() const { return _n4; };
	int64 P3() const { return _p3; };
	
	// get the number of real elements
	virtual int64 N() const { return _n1*_n2*_n3*_n4;};

	// get bytes of info members which is from _size to end
	virtual int64 GetInfoByte() const { return sizeof(*this) - 24;};
	
	// display member info
	virtual void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		kmMat3<T>::PrintInfo();
		PRINTFA("  _p3    : %lld\n", _p3);
		PRINTFA("  _n4    : %lld\n", _n4);		
	};

	// display dimension
	virtual void PrintDim(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]", str);
		else               PRINTFA("[%p]", _p);

		PRINTFA(" %lld, %lld, %lld, %lld (%lld) \n", _n1, _n2, _n3, _n4, _size);
	};

	// compare
	template<typename Y>
	bool IsEqualSizeDim(const kmMat4<Y>& b) const
	{
		return _n1 == b.N1() && _n2 == b.N2() && _n3 == b.N3() && _n4 == b.N4();
	}

	template<typename Y>
	bool IsEqualSizeDimP(const kmMat4<Y>& b) const
	{
		return _p1 == b.P1() && _p2 == b.P2() && _p3 == b.P3() && IsEqualSizeDim(b);
	}

	template<typename Y>
	bool IsEqualSizeAll(const kmMat4<Y>& b) const
	{
		return IsEqualSizeDimP(b) && IsEqualSize(b);
	}

	// get transpose matrix
	kmMat4 Tp() const
	{
		kmMat4 des(_n2, _n1, _n3, _n4);

		for(int64 i4 = 0; i4 < _n4; ++i4)
		for(int64 i3 = 0; i3 < _n3; ++i3)
		for(int64 i2 = 0; i2 < _n2; ++i2)
		for(int64 i1 = 0; i1 < _n1; ++i1)
		{
			des(i2, i1, i3, i4) = *P(i1, i2, i3, i4);
		}
		return des;	// * Note that des is the local variable,
		            // * so the return type cannot be reference type.
	};
};

// define type for kmMat
typedef kmArr <char>			kmArr0i8;
typedef kmMat1<char>			kmMat1i8;
typedef kmMat2<char>			kmMat2i8;
typedef kmMat3<char>			kmMat3i8;
typedef kmMat4<char>			kmMat4i8;

typedef kmArr <uchar>			kmArr0u8;
typedef kmMat1<uchar>			kmMat1u8;
typedef kmMat2<uchar>			kmMat2u8;
typedef kmMat3<uchar>			kmMat3u8;
typedef kmMat4<uchar>			kmMat4u8;

typedef kmArr <short>			kmArr0i16;
typedef kmMat1<short>			kmMat1i16;
typedef kmMat2<short>			kmMat2i16;
typedef kmMat3<short>			kmMat3i16;
typedef kmMat4<short>			kmMat4i16;

typedef kmArr <ushort>			kmArr0u16;
typedef kmMat1<ushort>			kmMat1u16;
typedef kmMat2<ushort>			kmMat2u16;
typedef kmMat3<ushort>			kmMat3u16;
typedef kmMat4<ushort>			kmMat4u16;

typedef kmArr <int>				kmArr0i32;
typedef kmMat1<int>				kmMat1i32;
typedef kmMat2<int>				kmMat2i32;
typedef kmMat3<int>				kmMat3i32;
typedef kmMat4<int>				kmMat4i32;

typedef kmArr <uint>			kmArr0u32;
typedef kmMat1<uint>			kmMat1u32;
typedef kmMat2<uint>			kmMat2u32;
typedef kmMat3<uint>			kmMat3u32;
typedef kmMat4<uint>			kmMat4u32;

typedef kmArr <int64>			kmArr0i64;
typedef kmMat1<int64>			kmMat1i64;
typedef kmMat2<int64>			kmMat2i64;
typedef kmMat3<int64>			kmMat3i64;
typedef kmMat4<int64>			kmMat4i64;

typedef kmArr <float>			kmArr0f32;
typedef kmMat1<float>			kmMat1f32;
typedef kmMat2<float>			kmMat2f32;
typedef kmMat3<float>			kmMat3f32;
typedef kmMat4<float>			kmMat4f32;

typedef kmArr <double>			kmArr0f64;
typedef kmMat1<double>			kmMat1f64;
typedef kmMat2<double>			kmMat2f64;
typedef kmMat3<double>			kmMat3f64;
typedef kmMat4<double>			kmMat4f64;

typedef kmArr <float>			kmArr0f32;
typedef kmMat1<float>			kmMat1f32;
typedef kmMat2<float>			kmMat2f32;
typedef kmMat3<float>			kmMat3f32;
typedef kmMat4<float>			kmMat4f32;

typedef kmArr <cmplxf32>		kmArr0c32;
typedef kmMat1<cmplxf32>		kmMat1c32;
typedef kmMat2<cmplxf32>		kmMat2c32;
typedef kmMat3<cmplxf32>		kmMat3c32;
typedef kmMat4<cmplxf32>		kmMat4c32;

typedef kmArr <f32xy>			kmArr0f32xy;
typedef kmMat1<f32xy>			kmMat1f32xy;
typedef kmMat2<f32xy>			kmMat2f32xy;
typedef kmMat3<f32xy>			kmMat3f32xy;
typedef kmMat4<f32xy>			kmMat4f32xy;

typedef kmArr <f32yz>			kmArr0f32yz;
typedef kmMat1<f32yz>			kmMat1f32yz;
typedef kmMat2<f32yz>			kmMat2f32yz;
typedef kmMat3<f32yz>			kmMat3f32yz;
typedef kmMat4<f32yz>			kmMat4f32yz;

typedef kmArr <f32zx>			kmArr0f32zx;
typedef kmMat1<f32zx>			kmMat1f32zx;
typedef kmMat2<f32zx>			kmMat2f32zx;
typedef kmMat3<f32zx>			kmMat3f32zx;
typedef kmMat4<f32zx>			kmMat4f32zx;

typedef kmArr <f32xyz>			kmArr0f32xyz;
typedef kmMat1<f32xyz>			kmMat1f32xyz;
typedef kmMat2<f32xyz>			kmMat2f32xyz;
typedef kmMat3<f32xyz>			kmMat3f32xyz;
typedef kmMat4<f32xyz>			kmMat4f32xyz;

///////////////////////////////////////////////////////////////
// memory block class for manual memory allocation
template<typename T> class kmMem : public kmMat1<T>
{
protected:
	// using for parents class
	using kmArr <T>::_p , kmArr <T>::_state, kmArr<T>::_size;
	using kmMat1<T>::_n1;

	using kmArr<T>::Byte;
	using kmArr<T>::GetKmClass;
	/////////////////////////////////////////////////
	// basic member functions
public:	
	virtual void Init() { kmMat1<T>::Init(); };

	// constructor
	kmMem() {};
	kmMem(      int64 byte) { Create(byte);};
	kmMem(T* p, int64 byte) { Set(p, byte);};

	// destructor
	virtual ~kmMem() {};
		
	// allocate memory
	void Create(int64 byte) { kmMat1<T>::Create(0, Byte2Size(byte)); };
	
	// release and create
	void Recreate(int64 byte) { kmMat1<T>::Release(); Create(byte); };

	// set array
	void Set(T* p, int64 byte) { kmMat1<T>::Set(p, 0, Byte2Size(byte)); };

	/////////////////////////////////////////////////
	// general member functions

	static int64 Size2Byte(int64 size) {return size*sizeof(T);};
	static int64 Byte2Size(int64 byte) {return (byte - 1)/(int64)sizeof(T) + 1;};

	int64 GetIdx      () const { return _n1; };
	int64 GetByteLeft () const { return (_size - GetIdx())*sizeof(T); };
	int64 GetByteUsed () const { return GetIdx()*sizeof(T); };
	float GetMbyteLeft() const { return GetByteLeft()/(1024.f*1024.f); };
	float GetMbyteUsed() const { return GetByteUsed()/(1024.f*1024.f); };

	void PrintMemState(LPCSTR str = nullptr)
	{
		if(str != nullptr) PRINTFA("* %s : ",str);
		else               PRINTFA("* kmMem : ");
		PRINTFA("%.1f / %lld MB\n", GetMbyteUsed(), Byte()>>20);
	};
	
	// get memory block
	void* GetMem(int64 byte)
	{
		// calc size
		int64 size = Byte2Size(byte); 

		// check size
		ASSERTA(_n1 + size <= _size, "[kmMem::GetMem in 451] over memory");

		// update index
		int64 idx = _n1; _n1 += size;
				
		return (void*) (_p + idx);
	};

	// reset allocated memory block
	void Reset(const int64 idx = 0) { _n1 = idx; };

	// clear allocated memory block
	void Clear() { memset(_p, 0, _size*sizeof(T)); };

	// allocate memory to kmArr and kmMat
	template<typename Y> void Give(kmArr <Y>& a, int64 n ) { a.Set((Y*)this->GetMem(n *sizeof(Y)), n );}
	template<typename Y> void Give(kmMat1<Y>& a, int64 n1) { a.Set((Y*)this->GetMem(n1*sizeof(Y)), n1);}

	// allocate memory to kmMat2
	template<typename Y> 
	void Give(kmMat2<Y>& a, int64 n1, int64 n2, int64 p1 = 0)
	{
		if(p1 == 0) p1 = n1;

		a.SetP((Y*)this->GetMem(p1*n2*sizeof(Y)), n1, n2, p1);
	}

	// allocate memory to kmMat3
	template<typename Y> 
	void Give(kmMat3<Y>& a, int64 n1, int64 n2, int64 n3, int64 p1 = 0, int64 p2 = 0)
	{
		if(p1 == 0) p1 = n1; if(p2 == 0) p2 = n2;

		a.SetP((Y*)this->GetMem(p1*p2*n3*sizeof(Y)), n1, n2, n3, p1, p2);
	}

	// allocate memory to kmMat4
	template<typename Y> 
	void Give(kmMat4<Y>& a, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1 = 0, int64 p2 = 0, int64 p3 = 0)
	{
		if(p1 == 0) p1 = n1; if(p2 == 0) p2 = n2; if(p3 == 0) p3 = n3;

		a.SetP((Y*)this->GetMem(p1*p2*p3*n4*sizeof(Y)), n1, n2, n3, n4, p1, p2, p3);
	}
};

typedef kmMem<char>    kmMem8;
typedef kmMem<short>   kmMem16;
typedef kmMem<int>     kmMem32;
typedef kmMem<int64>   kmMem64;

///////////////////////////////////////////////////////////////
// bit flag class

class kmBit
{
public:
	uchar* _ptr  = nullptr;
	uchar  _mask = 0;

	// constructor
	kmBit(uchar* ptr, int idx) { _ptr = ptr; SetIdx(idx); };

	// bit  1: set bit as 1
	// bit  0: set bit as 0
	// bit -1: set bit as ~bit
	kmBit& operator=(const int bit)
	{
		if     (bit >  0) *_ptr |=  _mask;
		else if(bit == 0) *_ptr &= ~_mask;
		else              *_ptr ^=  _mask;
		return *this;
	};

	// conversion operator... (uchar) a	
	operator uchar() const { return ((*_ptr)&_mask) ? 1:0; };

	// get bool
	int Get() const { return ((*_ptr)&_mask) ? 1:0; };
	
	// set index
	void SetIdx(int idx) { _mask = uchar(0b10000000)>>idx; };

	// print binary format
	void Print() { cout << "* " << (uint)*_ptr << " : " << bitset<8>(*_ptr) << endl; };	
};

class kmMat1bit: public kmMat1u8
{
public:
	int64 _bit_n = 0;

	// constructor
	kmMat1bit() {};
	kmMat1bit(int64 bit_n) { Create(bit_n); };

	kmMat1bit(const initializer_list<uchar>& val)
	{
		const int64 bit_n = val.size();
		_state =0; Create(bit_n);
		const uchar *pval = val.begin();
		for(int64 i = 0; i < bit_n; ++i) (*this)(i) = (int)*(pval++);
	};

	// destructor
	virtual ~kmMat1bit() {};

	// create matrix
	void Create(int64 bit_n) { _bit_n = bit_n; kmMat1u8::Create(u8_n(bit_n)); };

	// recreate matrix
	void Recreate(int64 bit_n) { Release(); Create(bit_n); };

	// set every bit as 1
	void SetOne() { SetVal(0b11111111); };

	// operator
	kmBit operator()(int64 bit_i) const
	{
		return kmBit(P(u8_i(bit_i)), bit_i & 0b111);
	};

	// get bit_n
	int64 BitN() const { return _bit_n; };

	// true  : if all bits are 1
	// false : if any bits are 0
	bool IsAll() const
	{
		const int64 n = _n1 - 1;

		for(int64 i = 0; i < n; ++i) if(a(i) != 0b11111111) return false;

		const int last_bit_n = int(_bit_n & 0b111);

		if(last_bit_n > 0)
		{
			const uchar mask = 0b11111111 << (8 - last_bit_n);
			if( (*End()&mask) != mask) return false;
		}
		return true;
	};

	// true  : if any bits are 1
	// false : if all bits are 0
	bool IsAny() const
	{
		const int64 n = _n1 - 1;

		for(int64 i = 0; i < n; ++i) if(a(i) != 0) return true;

		const int last_bit_n = int(_bit_n & 0b111);

		if(last_bit_n > 0)
		{
			const uchar mask = 0b11111111 << (8 - last_bit_n);
			if( (*End()&mask) != 0) return true;
		}
		return false;
	};

	// true  : if all bits are 0
	// false : if any bits are 1
	bool IsNone() const { return !IsAny(); };
		
	// print functions
	void PrintVal() const
	{
		cout << "* 0b "; 
		for(int64 i = 0; i < _n1; ++i) cout << bitset<8>(*P(i)) << " "; 
		cout << endl;
	}

protected:
	// inline inner functions
	inline int64 u8_n(int64 bit_n) const { return ((bit_n - 1)>>3) + 1; };
	inline int64 u8_i(int64 bit_i) const { return bit_i>>3;             };
};

///////////////////////////////////////////////////////////////
// block buffer class

// memory block buffer class
class kmMat1blk : public kmMat1i8
{
public:
	uint  _blk_byte = 0;
	uint  _blk_n    = 0;

	// constructor
	kmMat1blk() {};
	kmMat1blk(           int64 byte, uint blk_byte, uint blk_n) { Create(  byte, blk_byte, blk_n); };
	kmMat1blk(char* ptr, int64 byte, uint blk_byte, uint blk_n) { Set(ptr, byte, blk_byte, blk_n); };

	// create
	void Create(int64 byte, uint blk_byte, uint blk_n)
	{
		kmMat1i8::Create(byte); SetBlk(byte, blk_byte, blk_n);
	};

	// recreate
	void Recreate(int64 byte, uint blk_byte, uint blk_n)
	{
		kmMat1i8::Recreate(byte); SetBlk(byte, blk_byte, blk_n);
	};

	// set
	void Set(char* ptr, int64 byte, uint blk_byte, uint blk_n)
	{
		kmMat1i8::Set(ptr, byte); SetBlk(byte, blk_byte, blk_n);
	};

	// set block info
	void SetBlk(int64 byte, uint blk_byte, uint blk_n)
	{
		ASSERTA( byte > (int64)(blk_n - 1)*(int64)blk_byte, "[kmMat1Blk::SetBlk in 2935]");

		_blk_byte = blk_byte; _blk_n = blk_n; 
	};

	// get block ptr
	char* GetBlkPtr(uint iblk)
	{
		ASSERTA( iblk < _blk_n, "[kmMat1Blk::GetBlkPtr in 2937]");

		return P(_blk_byte*iblk); 
	};

	// get block byte
	uint GetBlkByte(uint iblk)
	{
		ASSERTA( iblk < _blk_n, "[kmMat1Blk::GetBlkByte in 2940]");

		return (iblk == _blk_n - 1)? uint(Byte() - _blk_byte*iblk) : _blk_byte;
	};
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class for string

template<typename T> class kmStr : public kmMat1<T>
{
protected:
	// using for members of parents class
	using kmArr <T>::_p , kmArr <T>::_state, kmArr<T>::_size;
	using kmMat1<T>::_n1;

public:
	// using for functions of parents class
	using kmMat1<T>::IsPinned;
	using kmMat1<T>::IsCreated;
	using kmMat1<T>::Create;
	using kmMat1<T>::Release;
	using kmMat1<T>::Init;
	using kmMat1<T>::a;
	using kmMat1<T>::P;
	using kmMat1<T>::Move;
	using kmArr <T>::RestoreVfptr;
	using kmArr <T>::GetKmClass;
	using kmArr <T>::SetZero;

public:
	// construtor
	kmStr() {};
	kmStr(int64 size) : kmMat1<T>(size) { SetZero(); };

	kmStr(const char* str, ...)
	{
		// if str is null
		if(str == nullptr) { Create(1); *_p = '\0'; return; }

		va_list args;
		va_start(args, str);
		{
			const int len = _vscprintf(str, args) + 1;

			if(len > 0) { Create(len); vsprintf_s(_p, len, str, args); }
		}
		va_end(args);
	};

	kmStr(const char8* str, ...)
	{
		// if str is null
		if(str == nullptr) { Create(1); *_p = '\0'; return; }

		va_list args;
		va_start(args, str);
		{
			const int len = _vscprintf((char*)str, args) + 1;

			if(len > 0) { Create(len); vsprintf_s(_p, len, (char*)str, args); }
		}
		va_end(args);
	};

	kmStr(const wchar* str, ...)
	{
		// if str is null
		if(str == nullptr) { Create(1); *_p = L'\0'; return; }

		va_list args;
		va_start(args, str);
		{
			const int len = _vscwprintf(str, args) + 1;

			if(len > 0) { Create(len); vswprintf_s(_p, len, str, args); }
		}
		va_end(args);
	};

	// destructor
	~kmStr() {};

	// copy constructor
	kmStr(const kmStr& str)
	{
		Create(str.GetLen());
		Copy  (str._p);
	};

	// move constructor
	kmStr(kmStr&& str) noexcept { Move(str); };

	// assignment operator
	kmStr& operator=(const kmStr& str)
	{
		Recreate(str.GetLen());
		Copy    (str._p);
		return *this;
	};

	// move assignment operator
	kmStr& operator=(kmStr&& str) noexcept
	{
		if(IsPinned()) *this = str;
		else           Move(str); 
		return *this; 
	};

	// assignment operator for LPCTSTR
	kmStr& operator=(const T* str)
	{
		Recreate(GetStrLen(str));
		Copy    (str);
		return *this;
	};

	// recreate for str
	void Recreate(int64 size)
	{
		if(size > _size || size + 128 < _size) { Release(); Create(size); }
	};

	// restore the class including vfptr
	// * Note that this will clear and reset _p, _state, _size without release,
	// * which can cause a memory leak. So, you should use this very carefully.
	kmStr<T>& Restore() { RestoreVfptr(*this); _state = 0; Init(); return *this; };

	/////////////////////////////////////////////////
	// operator functions

	// operator... a == b
	bool operator==(const kmStr& b) const
	{
		if(b.GetLen() != GetLen()) return false;
		return Compare(b._p) == 0;
	};

	// operator... a != b
	bool operator!=(const kmStr& b) const
	{
		if(b.GetLen() != GetLen()) return true;
		return Compare(b._p) != 0;
	};

	// compare with b.. return 0 : the same, not 0 : different
	int Compare(const T* b)  const
	{
		if(typeid(T) == typeid(wchar_t)) return wcscmp((wchar*)_p, (wchar*)b);
		                                 return strcmp(( char*)_p, ( char*)b);
	};

	// operator... a += b
	kmStr<T>& operator+=(const kmStr& b)
	{
		const int64 an = GetLen(), bn = b.GetLen();

		if(an + bn - 1 > _size) Expand(bn - 1);

		memcpy(P(an-1), b.P(), bn*sizeof(T)); 
		
		_n1 = an + bn - 1;

		return *this;
	};

	// operator... a += b
	kmStr<T>& operator+=(const T* b)
	{
		const int64 an = GetLen(), bn = GetStrLen(b);

		if(an + bn - 1 > _size) Expand(bn - 1);

		memcpy(P(an-1), b, bn*sizeof(T)); 

		_n1 = an + bn - 1;

		return *this;
	};

	// operator... c = a + b
	kmStr<T> operator+(const kmStr& b) const
	{
		kmStr<T> c(*this); return c += b;
	};

	// operator... c = a + b
	kmStr<T> operator+(const T* b) const 
	{
		kmStr<T> c(*this); return c += b;
	};

	// expand for string
	void Expand(int64 size)
	{
		ASSERTA(IsCreated(), "[kmStr::Expand in 3143] memory is not created");
		ASSERTA(!IsPinned(), "[kmStr::Expand in 3144] memory is pinned");

		const int64 size_new = size + _size;

		T* p = new T[size_new];

		ASSERTA(_p != 0, "[kmstr::Expand in 3150] %p != 0", _p);
				
		CopyTo(p);		

		delete [] _p;

		_size  = size_new;
		_p     = p;
		_state = 1;
	};

	// conversion operator
	operator T*() const { return static_cast<T*>(_p); };
	
	/////////////////////////////////////////////////
	// member functions

	// copy str data... core
	void Copy(const T* str)
	{
		ASSERTA(_p != 0, "[kmStr::Copy in 1191]");
		
		const int64 size = GetStrLen(str);

		ASSERTA(size <= _size, "[kmStr::Copy in 646]");

		memcpy(_p, str, sizeof(T)*size);
	};

	// copy to... core
	void CopyTo(T* str) const
	{
		ASSERTA(_p != 0, "[kmStr::CopyTo in 2835]");
		
		memcpy(str, _p, sizeof(T)*GetLen());
	};

	// print string
	void Printf() const
	{
		if(IS_C16(T)) wprintf(L"%s\n", _p);
		else           printf( "%s\n", _p);
	};

	// get string from keyboard
	void Scanf() const
	{
		if(IS_C16(T)) wscanf_s(L"%ls", _p, _size);
		else          scanf_s ( "%s" , _p, _size);
	};

	// set string
	void SetStr(const char* str, ...)
	{
		va_list args;
		va_start(args, str);
		{
			const int len = _vscprintf(str, args) + 1;

			if(len > 0) { Recreate(len); vsprintf_s(_p, len, str, args); }
		}
		va_end(args);
	};

	void SetStr(const char8* str, ...)
	{
		va_list args;
		va_start(args, str);
		{
			const int len = _vscprintf((char*)str, args) + 1;

			if(len > 0) { Recreate(len); vsprintf_s(_p, len, (char*)str, args); }
		}
		va_end(args);
	};
	
	void SetStr(const wchar_t* str, ...)
	{
		va_list args;
		va_start(args, str);
		{
			const int len = _vscwprintf(str, args) + 1;

			if(len > 0) { Recreate(len); vswprintf_s(_p, len, str, args); }
		}
		va_end(args);
	};

	void SetAxisFloat(float v, float vmin, float vmax)
	{
		if(fabs(v) < 1e-9f) { *this = L"0"; return; }

		int   vdec = (int)floor(log10(fabs(v))), vdec3 = 3*(vdec/3);
		float ddec = log10(fabs(vmax - vmin));

		if(vdec > 3)
		{
			if     (ddec - vdec3 < -1.5f) this->SetStr(L"%.3fe%d", v*pow(0.1f, vdec3), vdec3);
			else if(ddec - vdec3 < -0.5f) this->SetStr(L"%.2fe%d", v*pow(0.1f, vdec3), vdec3);
			else if(ddec - vdec3 <  0.5f) this->SetStr(L"%.1fe%d", v*pow(0.1f, vdec3), vdec3);
			else                          this->SetStr(L"%.0fe%d", v*pow(0.1f, vdec3), vdec3);
		}
		else if(vdec < -3)
		{
			vdec3 = 3*((vdec-2)/3);

			if     (ddec - vdec3 < -1.5f) this->SetStr(L"%.3fe%d", v*pow(0.1f, vdec3), vdec3);
			else if(ddec - vdec3 < -0.5f) this->SetStr(L"%.2fe%d", v*pow(0.1f, vdec3), vdec3);
			else if(ddec - vdec3 <  0.5f) this->SetStr(L"%.1fe%d", v*pow(0.1f, vdec3), vdec3);
			else                          this->SetStr(L"%.0fe%d", v*pow(0.1f, vdec3), vdec3);
		}
		else
		{	
			if     (ddec < -2.5f) this->SetStr(L"%.4f",v);
			else if(ddec < -1.5f) this->SetStr(L"%.3f",v);
			else if(ddec < -0.5f) this->SetStr(L"%.2f",v);
			else if(ddec <  0.5f) this->SetStr(L"%.1f",v);
			else                  this->SetStr(L"%.0f",v);
		}
	};

	void SetFloat3(const float val)
	{
		if(fabs(val) < 1e-9f) { *this = L"0"; return; }
		
		int dec = (int)floor(log10(fabs(val))), dec3 = 3*(dec/3);

		if(dec3 == 0)
		{
			if     (fabs(floor(val      + 1e-4f) - val      ) < 1e-4f) this->SetStr(L"%.0f", val);
			else if(fabs(floor(val*10.f + 1e-4f) - val*10.f ) < 1e-3f) this->SetStr(L"%.1f", val);
			else if(fabs(floor(val*100.f+ 1e-4f) - val*100.f) < 1e-2f) this->SetStr(L"%.2f", val);
			else                                                       this->SetStr(L"%.3f", val);
		}
		else
		{
			if(dec < -3) dec3 = 3*((dec-2)/3);

			this->SetStr(L"%.1fe%d", val*pow(0.1f, dec3), dec3);
		}
	};

	// set str... core
	void Set(const wchar* str, int n = 0)
	{
		ASSERTA(_state == 0, "[kmStr::Set in 779] %lld == 0", _state);

		_p = (wchar*)(void*)str;

		_n1 = _size = (n == 0) ? wcslen(str) + 1 : n;
	};
	void Set(const char* str, int n = 0)
	{
		ASSERTA(_state == 0, "[kmStr::Set in 787] %lld == 0", _state);

		_p = (char*)(void*)str;

		_n1 = _size = (n == 0) ? strlen(str) + 1 : n; 
	};

	// get a part of string
	// * Note that this will give a new allocated object unlike Mat1().
	// * you should not use Mat1() since it doesn't guarantee null-terminating
	kmStr<T> Get(kmI i1) const
	{
		if(i1.e < 0) i1.e += _n1; i1.e = MIN(i1.e, _n1-2);
		const int n = (int)i1.Len() + 1;

		kmStr<T> b(n);

		memcpy(b._p, _p + i1.s, sizeof(T)*(n - 1));

		b(-1) = T('\0');

		return b;
	};

	// get a part of string from end position
	kmStr<T> GetEnd(int n) const
	{
		kmStr<T> b(n + 1);

		memcpy(b._p, _p + GetLen() - n - 1, sizeof(T)*(n));

		b(-1) = T('\0');

		return b;
	};

	// cut back with number of text
	kmStr<T>& Cutback(int cut_n)
	{
		int64 n = GetLen();

		ASSERTA(cut_n < n, "[kmStr::Cutback in 3272]");

		(*this)(n - 1 - cut_n) = T('\0');

		_n1 = n - cut_n;

		return *this;
	};
	kmStr<T>& Cutback(int64 cut_n) { return Cutback(int(cut_n)); };

	// cut back (from 0 to ch's pos - 1) as finding char in reverse order
	kmStr<T>& CutbackRvrs(T ch)
	{
		const int idx = FindRvrs(ch); if(idx < 0) return *this;
	
		 (*this)(idx) = T('\0'); _n1 = idx + 1;

		return *this;
	};

	// split as finding char in revers order
	//   return : rear part (from char'pos + 1 to end) 	
	kmStr<T> SplitRvrs(T ch)
	{
		const int idx = FindRvrs(ch); if(idx < 0) return kmStr<T>();
		const int n   = (int)GetLen();

		kmStr<T> rear(n - idx - 1);

		memcpy(rear.P(), P(idx + 1), rear.N1()*sizeof(T));

		if(idx >= 0) { (*this)(idx) = T('\0'); _n1 = idx + 1; }

		return rear;
	};

	// split as finding char 
	//   return : rear part (from char'pos + 1 to end) 	
	kmStr<T> Split(T ch)
	{
		const int idx = Find(ch); if(idx < 0) return kmStr<T>();
		const int n   = (int)GetLen();

		kmStr<T> rear(n - idx - 1);

		memcpy(rear.P(), P(idx + 1), rear.N1()*sizeof(T));

		if(idx >= 0) { (*this)(idx) = T('\0'); _n1 = idx + 1; }

		return rear;
	};

	// find and replace
	int Replace(T org, T rep) const
	{
		const int idx = Find(org); if(idx < 0) return 0;
		(*this)(idx) = rep;
		return 1;
	};

	// find and replace in reverse order
	int ReplaceRvrs(T org, T rep) const
	{
		const int idx = FindRvrs(org); if(idx < 0) return 0;
		(*this)(idx) = rep;
		return 1;
	};

	// get string length from const T*
	// * Note that the return of GetStrLen includes null-ternminating characters,
	// * while strlen and wsclen don't include it.
	static int64 GetStrLen(const char*    str) { if(str == nullptr) return 0; return strlen(str) + 1; };
	static int64 GetStrLen(const wchar_t* str) { if(str == nullptr) return 0; return wcslen(str) + 1; };

	// get string length including null-terminating character
	// * Note that it can be more than (number of characters + 1) if including Korean
	// * Actually, it is the nunber of bytes including null-terminateing character if it's stra.
	int64 GetLen() const { return GetStrLen(_p); };

	// pack string... change n1 as getlen()
	kmStr<T>& Pack() { _n1 = GetLen(); return *this; };

	// conversion function
	int ToInt()
	{
		if(IS_C16(T)) return _wtoi((const wchar_t*) _p);
		else          return  atoi((const  char  *) _p);
	};

	float ToFloat()
	{
		if(IS_C16(T)) return (float)_wtof((const wchar_t*) _p);
		else          return (float) atof((const  char  *) _p);
	};

	// find the character from istart (0 <= : position, -1 : not found)
	int Find(T ch, int istart = 0) const
	{
		int n = (int)GetLen() - 1, i = istart;
		for(; i < n; ++i) if(a(i) == ch) break;

		return (i < n) ? i:-1;
	};

	// find the character in reverse order	(0 <= : position, -1 : not found)
	int FindRvrs(T ch, int istart = end32 - 1) const
	{
		int n = (int)GetLen() - 1, i = MIN(n, istart + 1);
		for(; i--;) if(a(i) == ch) break;

		return i;
	};

	// find the first alphabet from istart (only for wchar)
	int FindAlpha(int istart = 0) const
	{
		int n = (int)GetLen() - 1, i = istart;
		for(; i < n; ++i) if(iswalpha(a(i))) break;

		return (i < n) ? i:-1;
	};

	// find the first alphabet from istart in reverse order (only for wchar)
	int FindAlphaRvrs(int istart = end32 - 1) const
	{
		int n = (int)GetLen() - 1, i = MIN(n, istart + 1);
		for(; i--;) if(iswalpha(a(i))) break;

		return i;
	};

	// find the first alphabet or number from istart (only for wchar)
	int FindAlnum(int istart = 0) const
	{
		int n = (int)GetLen() - 1, i = istart;
		for(; i < n; ++i) if(iswalnum(a(i))) break;

		return (i < n) ? i:-1;
	};

	// find the first alphabet from istart in reverse order (only for wchar)
	int FindAlnumRvrs(int istart = end32 - 1) const
	{
		int n = (int)GetLen() - 1, i = MIN(n, istart + 1);
		for(; i--;) if(iswalnum(a(i))) break;

		return i;
	};

	// find the first non-character from istart (only for wchar)
	int FindNonAlpha(int istart = 0) const
	{
		int n = (int)GetLen() - 1, i = istart;
		if(!iswalpha(a(i++))) return -1;
		for(; i < n; ++i) if(!iswalpha(a(i))) break;

		return i;
	};

	// find the first non-alphabet and non-number from istart (only for wchar)
	int FindNonAlnum(int istart = 0) const
	{
		int n = (int)GetLen() - 1, i = istart;
		if(!iswalnum(a(i++))) return -1;
		for(; i < n; ++i) if(!iswalnum(a(i))) break;

		return i;
	};

	// find word (only for wchar)
	//   idx : input is istart to find,
	//         ouput is iend (end position of the word + 1)
	kmStr<T> FindWord(int& idx) const
	{
		const int is = FindAlnum(idx);

		if(is < 0) { idx = (int)_n1; return kmStr<T>(); }

		const int ie = FindNonAlnum(is);

		idx = (ie < 0) ? (int)_n1 : ie;

		return Get(kmI(is,ie-1));
	};
	kmStr<T> FindWord() const { int idx = 0; return FindWord(idx); };

	// get sub
	kmStr<T> GetSub(T ch, int& idx) const
	{
		const int n = (int)GetLen();

		int is = idx; for(; is < n; ++is) if((*this)(is) != ch) break;

		if(is > n - 2) return kmStr<T>();

		idx = Find(ch, is + 1);

		if(idx < 0) idx = n-1;

		return Get(kmI(is,idx-1));
	};

	// remove space
	void RemoveSpace()
	{
		const int n = (int)GetLen();

		for(int i = 0, j = 0; i < n; ++i)
		{
			const T ch = (*this)(i);

			if(ch != T(' ')) (*this)(j++) = ch;
		}
		Pack();
	};

	///////////////////////////////////////
	// encoding functions
	//
	// * Note that the followings for len
	// *  ANIS, UTF-8    : len is size in bytes. (multi-byte from 1 to 4 bytes per ch)
	// *  UTF-16 (wchar) : len is size in characters. (2 bytes per ch)
	// *  every len is including null-terminated.

	// encode wchar to ansi only for kmStrw	
	//   str_n : size to convert in characters incluing null-terminated
	//           if it's -1, it will encode to null-terminated.
	kmStr<char> EncWtoA(int str_n = -1) const
	{
		int len = WideCharToMultiByte(CP_ACP, 0, (wchar*)P(), str_n, NULL, 0, NULL, NULL);
	
		kmStr<char> stra(len);
	
		WideCharToMultiByte(CP_ACP, 0, (wchar*)P(), str_n, stra.P(), len, NULL, NULL);
	
		return stra;
	};

	// encode wchar to utf-8 only for kmStrw	
	//   str_n : size to convert in characters incluing null-terminated
	//           if it's -1, it will encode to null-terminated.
	kmStr<char> EncWtoU(int str_n = -1) const
	{
		int len = WideCharToMultiByte(CP_UTF8, 0, (wchar*)P(), str_n, NULL, 0, NULL, NULL);

		kmStr<char> stra(len);

		WideCharToMultiByte(CP_UTF8, 0, (wchar*)P(), str_n, stra.P(), len, NULL, NULL);

		return stra;
	};

	// encode ansi to wchar only for kmStra	
	//   byte  : size to convert in bytes incluing null-terminated
	//           if it's -1, it will encode to null-terminated.
	kmStr<wchar> EncAtoW(int byte = -1) const
	{
		int len = MultiByteToWideChar(CP_ACP, 0, (char*)P(), byte, NULL, 0);

		kmStr<wchar> strw(len);

		MultiByteToWideChar(CP_ACP, 0, (char*)P(), byte, strw.P(), len);

		return strw;
	};

	// encode utf-8 to wchar only for kmStra	
	//   byte  : size to convert in bytes incluing null-terminated
	//           if it's -1, it will encode to null-terminated.
	kmStr<wchar> EncUtoW(int byte = -1) const
	{
		int len = MultiByteToWideChar(CP_UTF8, 0, (char*)P(), byte, NULL, 0);

		kmStr<wchar> strw(len);

		MultiByteToWideChar(CP_UTF8, 0, (char*)P(), byte, strw.P(), len);

		return strw;
	};;
	
	// convert from ansi to widechar (utf-16)
	inline kmStr<wchar> caw() const { return EncAtoW(); };

	// convert from utf-8 to widechar (utf-16)
	inline kmStr<wchar> cuw() const { return EncUtoW(); };

	// convert to ansi
	kmStr<char> ca() const
	{
		if     (IS_C16(T)) return EncWtoA(); 
		else if(IS_C08(T)) return EncUtoW().EncWtoA();
		return kmStr<char>();
	};

	// convert to utf-8
	kmStr<char> cu() const
	{
		if     (IS_C16(T)) return EncWtoU(); 
		else if(IS_C08(T)) return EncAtoW().EncWtoU();
		return kmStr<char>();	
	};
};
typedef kmStr<char>				kmStra;
typedef kmStr<wchar>			kmStrw;
typedef kmMat1<kmStra>          kmStras;
typedef kmMat1<kmStrw>          kmStrws;

// kmStr only for utf-8... under contruction
class kmStru : public kmStra
{
public:
	kmStru() {};
	kmStru(int64 size)                  : kmStra(size) {};
	kmStru(const kmStru&  str)          : kmStra((kmStru)str) {};
	kmStru(      kmStru&& str) noexcept : kmStra((kmStru)str) {};
	kmStru(const kmStra&  str)          : kmStra(        str) {};
	kmStru(      kmStra&& str)          : kmStra(        str) {};

	kmStru& operator=(const kmStru&  a)          { return *(kmStru*)&kmStra::operator=((kmStra)a); };
	kmStru& operator=(const kmStru&& a) noexcept { return *(kmStru*)&kmStra::operator=((kmStra)a); };
	kmStru& operator=(const kmStra&  a)          { return *(kmStru*)&kmStra::operator=(a); };
	kmStru& operator=(const kmStra&& a)          { return *(kmStru*)&kmStra::operator=(a); };

	operator  kmStra() { return *( kmStra*)this; };	

	kmStrw cw() const { return EncUtoW(); };
	kmStra ca() const { return EncUtoW().EncWtoA(); };
};
typedef kmMat1<kmStru>          kmStrus;

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class for queue

// queue class
template<typename T> class kmQue1 : public kmArr<T>
{
protected:
	// using for members of parents class
	using kmArr <T>::_p , kmArr <T>::_state, kmArr<T>::_size;	

	// member variables
	int64 _n1 = 0; // number of dim1
	int64 _s1 = 0; // start index of dim1

public:
	// using for functions of parents class	
	using kmArr<T>::Release;
	using kmArr<T>::IsPinned;
	using kmArr<T>::IsCreated;
	using kmArr<T>::GetKmClass;
	using kmArr<T>::SetVal;
	using kmArr<T>::Copy;
	using kmArr<T>::RestoreVfptr;
	using kmArr<T>::Create;
	using kmArr<T>::Recreate;

	/////////////////////////////////////////////////
	// basic member functions
public:
	virtual void Init() { kmArr<T>::Init(); _n1 = _s1 = 0; };

	// constructor
	kmQue1() {};	
	kmQue1(int64 size) { Create(size); };

	// destructor
	virtual ~kmQue1() {};

	// copy constructor	
	kmQue1(const kmQue1& b) { Create(b.N1()); Copy(b.Begin()); };

	// assignment operator
	kmQue1& operator=(const kmQue1& b) { RecreateIf(b); Copy(b._p); return *this; };
	
	// operator to get data	
	T* P(int64 i1) const
	{
		if(i1 < 0) i1 += _n1; // * Note that i1 = -1 is the same as End()

		ASSERTA(0 <= i1 && i1 < _n1 , "[kmQue1::P in 3405] %lld < %lld", i1, _n1);

		return _p + _s1 + i1;
	};

	T* P() const { return _p; };

	T& operator()(int64 i1 = 0) const { return *P(i1); };
	T& a         (int64 i1 = 0) const { return *P(i1); };
	virtual T& v (int64 i1 = 0) const { return *P(i1); };

	// operator to get data
	T* End () const { return _p + _s1 + _n1 - 1; };
	T* End1() const { return _p + _s1 + _n1;     };

	// restore the class including vfptr
	// * Note that this will clear and reset _p, _state, _size without release,
	// * which can cause a memory leak. So, you should use this very carefully.
	kmQue1<T>& Restore() { RestoreVfptr(*this); _state = 0; Init(); return *this; };

	// expand memory	
	void Expand(int64 size)
	{
		const int64 size_new = _size + size - _s1;

		T* p = _p;

		if(size - _s1 > 0)
		{
			p = new T[size_new];
			ASSERTA(_p != 0, "[kmQue1::Expand in 3432] %p != 0", _p);
		}

		// * Note that "delete [] _p" will call the destructor of every elements.
		// * If T is kmat, it will release the memories of elements.
		// * To avoid this, std::move() and the move assignment operator have been used
		// * instead of CopyTo(p).
		// * std:move(A) means that A is rvalue.
		for(int64 i = 0; i < _n1; ++i) { *(p + i) = std::move(*(_p + _s1 + i)); }

		if(size - _s1 > 0) delete [] _p;

		_size  = size_new;
		_p     = p;
		_s1    = 0;
		_state = 1;
	};

	// move to the first element
	void MoveTo1st()
	{
		if(_s1 == 0) return;

		for(int64 i = 0; i < _n1; ++i) { *(_p + i) = std::move(*(_p + _s1 + i)); }

		_s1 = 0;
	};

	/////////////////////////////////////////////////
	// general member functions

	// get info
	int64 N1() const { return _n1; };

	// get the number of real elements
	virtual int64 N() const { return _n1; };

	// enqueue
	int64 Enqueue()
	{
		if(_s1 + _n1 == _size) // to guarantee extra space for more than current _n1
		{
			if(_n1 < _size/2) MoveTo1st();
			else              Expand(MAX(16, _n1 - _s1));
		}
		return _n1++;
	}
	int64 Enqueue(const T& val)
	{
		Enqueue();  *End() = val;  return _n1 - 1;
	};

	// Dequeue
	T* Dequeue()
	{
		if     (_n1 == 0) return nullptr;
		else if(_n1 == 1) { T* pval = _p + _s1; _s1 = _n1 = 0; return pval; }
		
		--_n1; return _p + _s1++;
	};
};

// main queue class ... this can add elements which have different sizes.
// * Note that kmQue is designed for a struct not using deep copy.
// * So, you should be careful when using a class like a kmMat.
template<typename T> class kmQue
{
public:
	kmMat1<T*> _pobj; // pointer queue
	kmMem8     _obj;  // object queue (memory storage)

	/////////////////////////////////////////////////
	// basic member functions
public:	
	virtual void Init() { _pobj.Init(); _obj.Init(); };

	// constructor
	kmQue() {};	
	kmQue(int64 size, int64 obj_byte) { Create(size, obj_byte); };

	// destructor
	virtual ~kmQue() {};

	// copy constructor	
	//kmQue(const kmQue& b) {};

	// move constructor
	//kmQue(kmQue&& b) {};

	// assignment operator
	//kmQue& operator=(const kmQue& b) { return *this; };

	// move assignment operator
	//kmQue& operator=(kmQue&& b) { return *this; };

	// allocate memory
	void Create(int64 size, int64 obj_byte)
	{
		_pobj.Create(0, size);
		_obj .Create(obj_byte);
	};

	// release memory
	void Release()
	{
		_pobj.Release();
		_obj .Release();
	};

	/////////////////////////////////////////////////
	// general functions

	// push back
	template<class Y> int64 PushBack(const Y& obj)
	{
		// get memory from _obj
		Y* pobj = (Y*)_obj.GetMem(sizeof(Y));

		// copy object to _obj
		memcpy((void*) pobj, &obj, sizeof(Y));

		// pushback pointer of stored object
		return _pobj.PushBack((T*) pobj);
	}

	// push back
	template<class Y> int64 PushBack(const Y* obj, const int64 byte)
	{
		// get memory from _obj
		Y* pobj = (Y*)_obj.GetMem(byte);

		// copy object to _obj
		memcpy((void*) pobj, obj, byte);

		// pushback pointer of stored object
		return _pobj.PushBack((T*) pobj);
	}

	// pop back
	T* PopBack()
	{
		_obj.Reset(_obj.GetIdx() - Byte(_pobj.N()-1));

		return *_pobj.PopBack();
	};

	// get size of obj(idx)
	int64 Byte(int64 idx) const
	{
		int64 n = _pobj.N(), byte = 0;

		if(idx < n-1)
		{
			byte = (int64) _pobj(idx + 1) - (int64) _pobj(idx);
		}
		else if(idx == n-1)
		{
			byte = (int64) _obj.P(_obj.GetIdx()) - (int64) _pobj(idx);
		}
		return byte;
	};

	// get number of objects
	int64 N() const { return _pobj.N(); };

	// get object
	T* Begin() const { return *_pobj.Begin();};
	T* End  () const { return *_pobj.End();  };

	// print info of que
	void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);
				
		_pobj.PrintInfo("_pobj");

		for(int64 i = 0; i < N(); ++i)
		{
			PRINTFA("  [%d] %lld, %d byte\n", i, _pobj(i), Byte(i));
		}
		PRINTFA("\n");

		_obj.PrintInfo("_obj");
	};

	/////////////////////////////////////////////////
	// operator

	T* operator()(int64 i) const { return _pobj(i); };
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class for image

// rect class : compatible to RECT
class kmRect
{
public:
	int l, t, r, b; // left, top, right, bottom

	// constructor
	kmRect()                               { Set( 0,  0,  0,  0); };
	kmRect(int l_, int t_)                 { Set(l_, t_, l_, t_); };
	kmRect(int l_, int t_, int r_, int b_) { Set(l_, t_, r_, b_); };
	kmRect(RECT rt)                        { Set(rt.left, rt.top, rt.right, rt.bottom); };

	// input method
	kmRect& Set(const RECT rect) { *((RECT*) this) = rect; return *this;};
	kmRect& Set(int l_, int t_, int r_, int b_)
	{
		l = l_; t = t_; r = r_; b = b_; return *this;
	};

	kmRect& Set(int l_, int t_)
	{
		const int w = GetW(), h = GetH();

		l = l_; t = t_; r = l_ + w; b = t_ + h;

		return *this;
	};

	kmRect& SetCen(int x, int y)
	{
		const int wh = GetW()/2, hh = GetH()/2;

		l = x - wh; r = x + wh;
		t = y - hh; b = y + hh;

		return *this;
	};

	kmRect& SetRB(int r_, int b_) { r = r_; b = b_; return *this; };

	kmRect& ShiftX(int del_x) { l += del_x; r += del_x; return *this; };
	kmRect& ShiftY(int del_y) { t += del_y; b += del_y; return *this; };

	kmRect& Shift(int del_x, int del_y) { ShiftX(del_x); ShiftY(del_y); return *this;};

	kmRect& SizeUpX(int del_s) { l -= del_s; r += del_s;       return *this; };
	kmRect& SizeUpY(int del_s) { t -= del_s; b += del_s;       return *this; };
	kmRect& SizeUp (int del_s) { SizeUpX(del_s); SizeUpY(del_s); return *this;};

	// conversion operator... (RECT) a
	//operator RECT () const { RECT rect = {_l, _t, _r, _b}; return rect; };
	operator RECT () const { return *((RECT*) this); };
	operator RECT*() const { return   (RECT*) this;  };

	//////////////////////////////////////////////////
	// general functions

	// * Note that width is (right - left) not (right - left + 1)
	// * and height is (bottom - top) not (bottom - top + 1)
	// * in the window system
	int GetW() const { return (r - l); };
	int GetH() const { return (b - t); };

	int GetCenX() const { return (r + l)/2; };
	int GetCenY() const { return (b + t)/2; };

	bool IsIn(int x, int y)
	{
		if(l <= x && x <= r && t <= y && y <= b) return true;
		return false;
	};
};

// rgb class : 4byte data type (0xrrggbbaa), compatible to COLORREF
// * Note that Intel's byte order is little endian.
// * So, the byte order (or address order) is 0xrrggbbaa, 
// * but the word order is 0xaabbggrr.
class kmRgb
{
public:		
	uchar _r, _g, _b, _a;

	// constructor
	kmRgb()                                   { Set(0, 0, 0, 0);};	
	kmRgb(const int rgb)                      { Set(rgb);       };
	kmRgb(uchar r, uchar g, uchar b)          { Set(r, g, b);   };
	kmRgb(uchar r, uchar g, uchar b, uchar a) { Set(r, g, b, a);};

	// input method
	void Set(const int rgb) { *((int*)this) = rgb; };
	void Set(uchar r, uchar g, uchar b, uchar a = 0)
	{
		_b = b; _g = g; _r = r; _a = a;
	};

	// conversion operator... (COLORREF) a	
	operator COLORREF() const { return *((COLORREF*) this); };
		
	// operator
	kmRgb& operator+=(const kmRgb& b) { _b += b._b; _g += b._g; _r += b._r; _a += b._a; return *this; };
	kmRgb& operator-=(const kmRgb& b) { _b -= b._b; _g -= b._g; _r -= b._r; _a -= b._a; return *this; };
	kmRgb  operator+ (const kmRgb& b) { return kmRgb(_r + b._r, _g + b._g, _b + b._b, _a + b._a);};
	kmRgb  operator- (const kmRgb& b) { return kmRgb(_r - b._r, _g - b._g, _b - b._b, _a - b._a);};
	
	// member fucntions
	COLORREF GetBgr() const
	{
		return (COLORREF) kmRgb(_b, _g, _r, _a);
	};

	// vary color
	kmRgb Vary(uchar dr, uchar dg, uchar db)
	{
		kmRgb rgb = *this;

		rgb._r += (_r < 128) ? dr:-dr;
		rgb._g += (_g < 128) ? dg:-dg;
		rgb._b += (_b < 128) ? db:-db;

		return rgb;
	};
};

#define kmRgbR kmRgb(255,  0,  0)
#define kmRgbG kmRgb(  0,255,  0)
#define kmRgbB kmRgb(  0,  0,255)
#define kmRgbY kmRgb(255,255,  0)
#define kmRgbM kmRgb(255,  0,255)
#define kmRgbC kmRgb(  0,255,255)
#define kmRgbW kmRgb(255,255,255)
#define kmRgbK kmRgb(  0,  0,  0)

// bgr class : 4byte data type (0xbbggrraa), compatible to Bitmap file
class kmBgr
{
public:
	uchar _b, _g, _r, _a;

	// constructor
	kmBgr()                                   { Set(0, 0, 0, 0);};
	kmBgr(const int bgr)                      { Set(bgr);       };
	kmBgr(uchar b, uchar g, uchar r)          { Set(b, g, r);   };
	kmBgr(uchar b, uchar g, uchar r, uchar a) { Set(b, g, r, a);};

	// input method
	void Set(const int   bgr) { *((int*)this) = bgr; };
	void Set(const kmRgb rgb) { Set(rgb._b, rgb._g, rgb._b, rgb._a); };
	void Set(uchar b, uchar g, uchar r, uchar a = 0)
	{
		_b = b; _g = g; _r = r; _a = a;
	};

	// conversion operator... (kmRgb) a
	operator kmRgb() const { return kmRgb(_r, _g, _b, _a); };

	// conversion operator... (COLORREF) a	
	operator COLORREF() const { return (COLORREF)kmRgb(_r,_g,_b,_a); };

	// operator
	kmBgr& operator+=(const kmBgr& b) { _b += b._b; _g += b._g; _r += b._r; _a += b._a; return *this; };

	// assigne operator
	kmBgr& operator=(const kmRgb& rgb)
	{
		Set(rgb);		
		return *this;
	};
};

// colormap class
// * Note that the type of kmCMpa is kmBgr instead of kmRgb, 
// * since a bitmap is compatible to kmBgr not kmRgb
class kmCMap : public kmMat1<kmBgr>
{
public:
	// constructor
	kmCMap() {};
	kmCMap(kmeCMapType cmtype) { Create(cmtype); };

	// member functions
	virtual void Create(int64 n) { kmMat1<kmBgr>::Create(n);};
	
	void Create(kmeCMapType cmtype)
	{
		if(_size != 256) Recreate(256);

		switch(cmtype)
		{
		case CMAP_JET : ////////////////// jet colormap

			for(int i=0; i<32; ++i)
			{				
				P(i    )->Set( 128+i*4, 0,       0);
				P(i+224)->Set(       0, 0, 252-i*4);
			}
			for(int i=0; i<64; ++i)
			{
				P(i+ 32)->Set(	   255,     i*4,   0);
				P(i+ 96)->Set( 252-i*4,     255, i*4);
				P(i+160)->Set(	     0, 252-i*4, 255);
			}			
			break;

		case CMAP_GREY : ////////////////// grey colormap

			for(int i=0; i<256; ++i)
			{
				P(i)->Set(i,i,i);
			}
			break;
		}
	};	
};

// image class for RGB
// * Note that the type of kmImg is kmBgr instead of kmRgb, 
// * since a bitmap is compatible to kmBgr not kmRgb
class kmImg : public kmMat3<kmBgr>
{
	/////////////////////////////////////////////////
	// basic member functions
public:

	// constructor
	kmImg() {};
	kmImg(          int64 n1, int64 n2, int64 n3 = 1)                             : kmMat3(   n1, n2, n3)               {};
	kmImg(kmBgr* p, int64 n1, int64 n2, int64 n3 = 1)                             : kmMat3(p, n1, n2, n3)               {};
	kmImg(          int64 n1, int64 n2, int64 n3, int64 p1)                       : kmMat3(   n1, n2, n3, p1)           {};
	kmImg(kmBgr* p, int64 n1, int64 n2, int64 n3, int64 p1)                       : kmMat3(p, n1, n2, n3, p1)           {};
	kmImg(          int64 n1, int64 n2, int64 n3, int64 p1, int64 p2)             : kmMat3(   n1, n2, n3, p1, p2)       {};
	kmImg(kmBgr* p, int64 n1, int64 n2, int64 n3, int64 p1, int64 p2)             : kmMat3(p, n1, n2, n3, p1, p2)       {};
	kmImg(          int64 n1, int64 n2, int64 n3, int64 p1, int64 p2, int64 size) : kmMat3(   n1, n2, n3, p1, p2, size) {};
	kmImg(kmBgr* p, int64 n1, int64 n2, int64 n3, int64 p1, int64 p2, int64 size) : kmMat3(p, n1, n2, n3, p1, p2, size) {};

	/////////////////////////////////////////////////
	// general member functions

	// Convert to BGR image... b (matrix) --> a (BGR image)
	template<typename T> 
	void ConvertBgr(const kmMat3<T>& b, T min_v, T max_v, const kmCMap& cmap = kmCMap(CMAP_JET))
	{		
		// check size		
		ASSERTA(_n1 == b.N1() && _n2 == b.N2(), "[kmImg::ConvertBGR in 1231]");
		ASSERTA(max_v >= min_v                , "[kmImg::ConvertBGR in 1232]");

		// assign colors to elements
		const float coff = 255.f/float(max_v - min_v);

		for(int i3 = 0; i3 < _n3; ++i3)
		for(int i2 = 0; i2 < _n2; ++i2)
		for(int i1 = 0; i1 < _n1; ++i1)
		{
			T val = MIN(MAX(min_v, b(i1, i2, i3)), max_v);

			int idx = int(float(val - min_v)*coff);

			*P(i1, i2, i3) = cmap(MIN(MAX(0, idx),255));
		}
	}

	template<typename T>
	void ConvertBgr(const kmMat2<T>& b, T min_v, T max_v, const kmCMap& cmap = kmCMap(CMAP_JET))
	{
		ConvertBgr(kmMat3<T>(b), min_v, max_v, cmap);
	}	

	// get frame
	kmImg GetFrame(const int64 idx) const 
	{
		kmImg img(P(0,0,idx), _n1, _n2, 1, _p1);

		return img;
	};

	// get info
	int GetW() const { return (int)_n1; };
	int GetH() const { return (int)_n2; };

	int64 GetByteFrame() const { return _p1*_n2*sizeof(kmBgr); };

	/////////////////////////////////////////////////
	// member functions for DIB

	BITMAPFILEHEADER GetBmpFileHeader() const
	{
		const int64 size_fh = sizeof(BITMAPFILEHEADER);
		const int64 size_ih = sizeof(BITMAPINFOHEADER);

		BITMAPFILEHEADER fh = {0,};

		fh.bfType    = 0x4d42;
		fh.bfSize    = size_fh;
		fh.bfOffBits = size_fh + size_ih;

		return fh;
	};

	BITMAPINFOHEADER GetBmpInfoHeader() const
	{
		BITMAPINFOHEADER ih = {0,};

		ih.biSize          = sizeof(BITMAPINFOHEADER);
		ih.biWidth         =    (LONG) _n1;
		ih.biHeight        = -1*(LONG) _n2; // +bottom-up, -up-down
		ih.biPlanes        = 1;
		ih.biBitCount      = 32; 
		ih.biCompression   = BI_RGB;
		ih.biSizeImage     = 0; 
		ih.biXPelsPerMeter = 0;
		ih.biYPelsPerMeter = 0;
		ih.biClrUsed       = 0;
		ih.biClrImportant  = 0;

		return ih;
	};

	BITMAPINFO GetBmpInfo() const
	{
		BITMAPINFO bi = {0,};
		
		bi.bmiHeader = GetBmpInfoHeader();

		return bi;
	};
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class of time 

// date class... 8byte
class kmDate
{
protected:
	ushort _year{};
	uchar  _mon{}, _date{}, _hour{}, _min{}, _sec{}, _day{};

public:
	kmDate() {};
	kmDate(time_t time) { Set(time); };

	kmDate& Set(time_t time)
	{
		tm t; localtime_s(&t, &time);

		_year = t.tm_year + 1900; _mon = t.tm_mon + 1; _date = t.tm_mday;
		_hour = t.tm_hour;        _min = t.tm_min;      _sec = t.tm_sec;
		_day  = t.tm_wday;

		return *this;
	};
	kmDate& Set(kmStra& str) // yyyy:mm:dd hh:mm:ss
	{
		_year = str.ToInt(); 
		_mon  = str.Get(kmI( 5, 6)).ToInt();
		_date = str.Get(kmI( 8, 9)).ToInt();
		_hour = str.Get(kmI(11,12)).ToInt();
		_min  = str.Get(kmI(14,15)).ToInt();
		_sec  = str.Get(kmI(17,18)).ToInt();
		
		return *this;
	};
	kmDate& SetCur() { return Set(time(NULL)); };

	bool operator==(const kmDate& b) const { return *(int64*)this == *(int64*)&b; };
	bool operator!=(const kmDate& b) const { return *(int64*)this != *(int64*)&b; };

	kmStrw GetStrw() const
	{
		return kmStrw(L"%4d%02d%02d%02d%02d%02d",_year, _mon, _date, _hour, _min, _sec); 
	};
	kmStra GetStr() const
	{
		return kmStra("%4d%02d%02d%02d%02d%02d",_year, _mon, _date, _hour, _min, _sec); 
	};
	kmStrw GetStrwPt() const
	{
		return kmStrw(L"%4d-%02d-%02d %02d:%02d:%02d",_year, _mon, _date, _hour, _min, _sec); 
	};
	kmStra GetStrPt() const
	{
		return kmStra("%4d-%02d-%02d %02d:%02d:%02d",_year, _mon, _date, _hour, _min, _sec); 
	};
	int64 GetInt() const // this is not seconds
	{
		int64 date = _sec + _min*100 + _hour*10000 + _date*1000000;
		date += _mon *int64(100000000);
		date += _year*int64(10000000000);

		return date;
	};
	kmStrw GetDay() const
	{
		kmStrw str;
		switch(_day)
		{
		case 0: str.SetStr(L"Sun"); break; case 1: str.SetStr(L"Mon"); break;
		case 2: str.SetStr(L"Tue"); break; case 3: str.SetStr(L"Wed"); break;
		case 4: str.SetStr(L"Thu"); break; case 5: str.SetStr(L"Fri"); break;
		case 6: str.SetStr(L"Sat"); break;
		}
		return str;
	};

	// get time passed in seconds
	int64 GetPassSec() const
	{	
		return (int64)time(NULL) - sec();
	};

	// seconds since 1970-1-1 00:00:00 UTC
	int64 sec() const 
	{	
		tm t = {_sec, _min, _hour, _date, _mon - 1, _year - 1900,};

		return (int64)mktime(&t); // t is local time
	};
};

// gps class... 24 byte
class kmGps
{
public:
	double _lat_deg = 0;     // latitude  (deg)
	double _lng_deg = 0;     // longitude (deg)
	float  _alt_m   = 0;     // altitude  (m)
	char   _lat_ref = 'N';   // latitude ref 'N' or 'S'
	char   _lng_ref = 'E';   // latitude ref 'E' or 'W'

	bool operator==(const kmGps& b) const
	{
		return _lat_deg == b._lat_deg && _lng_deg == b._lng_deg && _alt_m == b._alt_m &&
			   _lat_ref == b._lat_ref && _lng_ref == b._lng_ref; 
	};
	bool operator!=(const kmGps& b) const
	{
		return _lat_deg != b._lat_deg || _lng_deg != b._lng_deg || _alt_m != b._alt_m ||
			   _lat_ref != b._lat_ref || _lng_ref != b._lng_ref;
	};

	void Print()
	{
		printw(L"* gps : %s\n", GetStrw().P());
	};
	kmStrw GetStrw()
	{
		return kmStrw(L"%7.3f%c %7.3f%c %3.0fm",
			          _lat_deg, _lat_ref, _lng_deg, _lng_ref, _alt_m);
	};
};

class kmCounter
{
protected:
	int64 _cnt = 0;

public:
	void Start() { QueryPerformanceCounter((LARGE_INTEGER*)&_cnt); };
	void Reset() { Start(); };
	void Stop () { _cnt = 0;};

	bool IsStarted() { return _cnt != 0; };

	// time interval in sec
	double GetTime() const
	{	
		int64 cnt; QueryPerformanceCounter  ((LARGE_INTEGER*)&cnt);
		int64 frq; QueryPerformanceFrequency((LARGE_INTEGER*)&frq);

		return double(cnt - _cnt)/frq;
	};
	double sec () const { return GetTime(); };
	double msec() const { return GetTime()*1e3; };
	double usec() const { return GetTime()*1e6; };
};

class kmTimer
{
protected:
	int   _state   = 0; // 0: not started, 1: started, 2: paused
	int64 _s_cnt   = 0; // start count
	int64 _p_cnt   = 0; // pause count

public:
	// constructor
	kmTimer() {};

	kmTimer(const int state)
	{
		_state = MIN(MAX(0,state),2);
		if(_state > 0) _s_cnt = _p_cnt = GetCnt();
	};

	// update count
	static int64 GetFreq() { int64 a; QueryPerformanceFrequency((LARGE_INTEGER*)&a); return a; };
	static int64 GetCnt () { int64 a; QueryPerformanceCounter  ((LARGE_INTEGER*)&a); return a; };

	static double GetTime_sec (int64 del_cnt) { return (double)del_cnt/GetFreq();     };
	static double GetTime_msec(int64 del_cnt) { return (double)del_cnt/GetFreq()*1e3; };
	static double GetTime_usec(int64 del_cnt) { return (double)del_cnt/GetFreq()*1e6; };
	
	//////////////////////////////////////
	// member fucntions
	inline int GetState    () { return  _state;       };
	inline int IsNotStarted() { return (_state == 0); };
	inline int IsStarted   () { return (_state == 1); };
	inline int IsPaused    () { return (_state == 2); };

	void Start() {                   _s_cnt = GetCnt(); _state = 1;  };
	void Stop () { if(_state == 1)   _p_cnt = GetCnt(); _state = 0;  };
	void Pause() { if(_state == 1) { _p_cnt = GetCnt(); _state = 2; }};

	void Continue()
	{
		if     (_state == 0) Start();
		else if(_state == 2) // paused
		{
			// adjust _s_cnt to remove holding time.
			_s_cnt += (GetCnt() - _p_cnt);

			// set state
			_state = 1;
		}
	};

	// if state is 0, it will return zero.
	// otherwise, it will return the time from starting to now except holding time.
	double GetTime()
	{	
		// calc time_cnt
		int64 time_cnt = 0;

		switch(_state)
		{
		case 1: time_cnt = GetCnt() - _s_cnt; break;
		case 2: time_cnt = _p_cnt   - _s_cnt; break;
		}

		// calc time_msec
		return GetTime_msec(time_cnt);
	};
	inline double GetTime_sec()  { return GetTime()*1e-3;};
	inline double GetTime_msec() { return GetTime();     };
	inline double GetTime_usec() { return GetTime()*1e3; };

	inline double sec()  { return GetTime()*1e-3;};
	inline double msec() { return GetTime();     };
	inline double usec() { return GetTime()*1e3; };

	void Printf(const char* str, ...)
	{
		char buf[256];
		va_list args; // variable argument list

		va_start  (args, str);
		vsprintf_s(buf, sizeof(buf), str, args);
		va_end    (args);

		PRINTFA("* %s : %.3f msec\n", buf, GetTime());
	};

	// wait for t_usec with Sleep(sleep_msec)
	void Wait(float t_usec, uint sleep_msec = 0)
	{
		Start(); while(GetTime_usec() < t_usec) Sleep(sleep_msec);
	};
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class for file control

class kmFileInfo
{
public:
	kmStrw name {}; // file name only
	uint   attrb{};
	int64  byte {};
	kmDate date {}; // created date

	// cosntructor
	kmFileInfo() {};
	kmFileInfo(const _wfinddata64_t& fd) { Init(fd);   };
	kmFileInfo(const wchar* path)        { Init(path); };

	// init
	void Init(const _wfinddata64_t& fd)
	{
		name.SetStr(fd.name);
		attrb =     fd.attrib;
		byte  =     fd.size;
		date.Set   (fd.time_write);		
	};

	// Init from path
	void Init(const wchar* path)
	{
		_wfinddata64_t fd; 
		intptr_t handle = _wfindfirst64(path, &fd);

		if(handle == -1) return;

		Init(fd);
	};

	// member functions
	bool IsDir    () const { return attrb & _A_SUBDIR; };	
	bool IsHidden () const { return attrb & _A_HIDDEN; };
	bool IsNormal () const { return !IsDir() && !IsHidden(); };
	bool IsRealDir() const
	{
		return IsDir() && !(name == L"." || name == L"..");
	};
	void Print()
	{
		printw(L" %s %10lld byte %s\n", date.GetStrwPt().P(), byte, name.P()); 
	};
};

class kmFileList : public kmMat1<kmFileInfo>
{
public:
	// constructor
	kmFileList() {};
	kmFileList(const wchar* path) { GetFileList(path); };

	// get file list in path (ex. path = L"c:\data\*.*")
	void GetFileList(const wchar* path)
	{	
		// init
		this->Recreate(0,32);

		// find first
		_wfinddata64_t fd; 
		intptr_t handle = _wfindfirst64(path, &fd);

		if(handle == -1) return;

		// find next
		for(int ret = 1; ret != -1;)
		{
			PushBack(kmFileInfo(fd));

			ret = _wfindnext64(handle, &fd);
		}
		_findclose(handle);
	};

	void Print()
	{
		for(int i = 0; i < N1(); ++i) P(i)->Print();
	};
};

class kmFile
{
protected:
	FILE* _file;

	/////////////////////////////////////////////////
	// basic member functions
public:
	// constructor
	kmFile() { _file = 0; };
	kmFile(const char * name, kmeFile mode = KF_READ) { _file = 0; Open(name, mode); };
	kmFile(const wchar* name, kmeFile mode = KF_READ) { _file = 0; Open(name, mode); };

	// destructor
	~kmFile() { Close(); };

	// get class name
	LPCSTR GetKmClass() const {return "kmFile";};

	//////////////////////////////////////////////////
	// general member functions

	// open file
	void Open(const char* name, kmeFile mode = KF_READ)
	{
		ASSERTA(!IsOpen(), "[kmFile::Open in 1830]");
		
		// get fmode
		const char* fmode = NULL;
	
		switch(mode)
		{
		case KF_READ       : fmode = "rb" ; break;
		case KF_NEW        : fmode = "wb+"; break;
		case KF_ADD        : fmode = "ab+"; break;
		case KF_MODIFY     : fmode = "rb+"; break;
		case KF_READ_TXT   : fmode = "r" ;  break;
		case KF_NEW_TXT    : fmode = "w+";  break;
		case KF_ADD_TXT    : fmode = "a+";  break;
		case KF_MODIFY_TXT : fmode = "r+";  break;
		default            : fmode = "rb" ; break;
		}
	
		// open file
		errno_t err = fopen_s(&_file, name, fmode);
	
		if(err)
		{
			// error message
			char buf[128]; strerror_s(buf, sizeof(buf), err);
	
			PRINTFA("* file (%s) cannot open with mode (%s)\n  %d : %s\n", 
				    name, fmode, err, buf);
			throw KE_CANNOT_OPEN;
		}
	};

	// open file for wide character vergion
	void Open(const wchar* name, kmeFile mode = KF_READ)
	{
		ASSERTA(!IsOpen(), "[kmFile::Open in 1830]");
		
		// get fmode
		const wchar_t* fmode = NULL;

		switch(mode)
		{
		case KF_READ   : fmode = L"rb" ; break;
		case KF_NEW    : fmode = L"wb+"; break;
		case KF_ADD    : fmode = L"ab+"; break;
		case KF_MODIFY : fmode = L"rb+"; break;
		default:         fmode = L"rb" ; break;
		}

		// open file
		errno_t err = _wfopen_s(&_file, name, fmode);

		if(err)
		{
			// error message
			wchar buf[128]; _wcserror_s(buf, numof(buf), err);

			PRINTFW(L"* file (%s) cannot open with mode (%s)\n  %d : %s\n", 
				    name, fmode, err, buf);
			throw KE_CANNOT_OPEN;
		}
	};

	// close file
	void Close() { if(IsOpen()) { fclose(_file); _file = 0; } };

	// check file
	int IsOpen() { return (_file == 0)? 0:1; };

	// get file size byte
	int64 GetByte()
	{
		const int64 cur_pos = GetPos(), byte = SeekEnd().GetPos(); Seek(cur_pos);		
		return byte;
	};

	// seek file position
	//kmFile& Seek   (int offset = 0) { fseek(_file, offset, SEEK_SET); return *this; };
	//kmFile& SeekEnd(int offset = 0) { fseek(_file, offset, SEEK_END); return *this; };
	//kmFile& SeekCur(int offset = 0) { fseek(_file, offset, SEEK_CUR); return *this; };

	kmFile& Seek   (int64 offset = 0) { _fseeki64(_file, offset, SEEK_SET); return *this; };
	kmFile& SeekEnd(int64 offset = 0) { _fseeki64(_file, offset, SEEK_END); return *this; };
	kmFile& SeekCur(int64 offset = 0) { _fseeki64(_file, offset, SEEK_CUR); return *this; };

	// get file position
	int64 GetPos() const { return _ftelli64(_file); };

	// flush
	void Flush() { fflush(_file); };

	// get windows os's file handle
	HANDLE GetHandle() { return (HANDLE)_get_osfhandle(_fileno(_file)); };

	//////////////////////////////////////////////////
	// static file functions

	static int Exist(const char*  name, bool show_err = false)
	{
		errno_t err = _access_s (name, 0);
		if(show_err) switch(err)
		{
		case EACCES: print("[kmFile::Exist] access denied\n");               break;
		case ENOENT: print("[kmFile::Exist] file name or path not found\n"); break;
		case EINVAL: print("[kmFile::Exist] invalide parameter\n");          break;
		}
		return (err == 0)? 1:0;
	};
	static int Exist(const wchar* name, bool show_err = false)
	{
		errno_t err = _waccess_s (name, 0);
		if(show_err) switch(err)
		{
		case EACCES: print("[kmFile::Exist] access denied\n");               break;
		case ENOENT: print("[kmFile::Exist] file name or path not found\n"); break;
		case EINVAL: print("[kmFile::Exist] invalide parameter\n");          break;
		}
		return (err == 0)? 1:0;
	};

	static int Remove(const wchar* name) { return _wremove(name); };
	static int Remove(const  char* name) { return   remove(name); };

	static int Rename(const wchar* cur_name, const wchar* new_name) { return _wrename(cur_name, new_name); };
	static int Rename(const  char* cur_name, const  char* new_name) { return   rename(cur_name, new_name); };

	static int MakeDir(const wchar* name) { return _wmkdir(name); };
	static int MakeDir(const  char* name) { return _mkdir (name); };

	static int RemoveDir(const wchar* name) { return _wrmdir(name); };
	static int RemoveDir(const  char* name) { return _rmdir (name); };

	static BOOL SetAttrb(const wchar* name, int attrb) { return SetFileAttributesW(name, attrb); };
	static BOOL SetAttrb(const  char* name, int attrb) { return SetFileAttributesA(name, attrb); };

	static BOOL SetHidden(const wchar* name) { return SetAttrb(name, FILE_ATTRIBUTE_HIDDEN); };
	static BOOL SetHidden(const  char* name) { return SetAttrb(name, FILE_ATTRIBUTE_HIDDEN); };
	static BOOL SetNormal(const wchar* name) { return SetAttrb(name, FILE_ATTRIBUTE_NORMAL); };
	static BOOL SetNormal(const  char* name) { return SetAttrb(name, FILE_ATTRIBUTE_NORMAL); };

	// make the directory including sub-folder
	static int MakeDirs(const kmStrw& path)
	{
		if(kmFile::Exist(path.P())) return 1;

		if(path.ReplaceRvrs(L'/', L'\0') == 0) return 0;

		if(MakeDirs(path) == 1)
		{
			path(path.GetLen()-1) = L'/';

			MakeDir(path.P());
		}
		return 1;
	};

	// remove the directory and all files and sub-directories
	static int RemoveDirAll(const wchar* name)
	{
		// get file list
		kmFileList flst(kmStrw(L"%s/*.*", name));
	
		int files_n = (int)flst.N1();
	
		// remove sub-directories and all files
		for(int i = 0; i < files_n; ++i)
		{
			const kmFileInfo& file = flst(i);
	
			kmStrw fullpath(L"%s/%s", name, file.name.P());
	
			if     ( file.IsRealDir()) RemoveDirAll(fullpath.P());
			else if(!file.IsDir())     Remove      (fullpath.P());
		}
	
		// remove the dir
		return RemoveDir(name);
	};

	// remove all files and sub-directories in the directory
	static void RemoveAll(const wchar* name)
	{
		// get file list
		kmFileList flst(kmStrw(L"%s/*.*", name));
	
		int files_n = (int)flst.N1();
	
		// remove sub-directories and all files
		for(int i = 0; i < files_n; ++i)
		{
			const kmFileInfo& file = flst(i);
	
			kmStrw fullpath(L"%s/%s", name, file.name.P());
	
			if     ( file.IsRealDir()) RemoveDirAll(fullpath.P());
			else if(!file.IsDir())     Remove      (fullpath.P());
		}
	};

	//////////////////////////////////////////////////
	// read and write functions

	// write str
	void WriteStr(const char* str, ...)
	{
		ASSERTA(IsOpen(),"[kmFile::WriteStr in 1852]");

		va_list args; // variable argument list

		va_start(args, str);
		vfprintf(_file, str, args);
		va_end(args);
	};

	// write data
	template<typename Y> void Write(const Y* str, const size_t cnt = 1)
	{		
		ASSERTA(IsOpen(), "[kmFile::Write in 1865]");
		
		fwrite((void*) str, sizeof(Y), cnt, _file);
	}

	// read data	
	template<typename Y> int64 Read(Y* str, const size_t cnt = 1)
	{		
		ASSERTA(IsOpen(), "[kmFile::Read in 1874]");

		return fread((void*) str, sizeof(Y), cnt, _file);
	}

	// read data with swapping endian
	template<typename Y> int64 ReadSwap(Y* str, const size_t cnt = 1)
	{		
		ASSERTA(IsOpen(), "[kmFile::Read in 1874]");

		int64 ret = fread((void*) str, sizeof(Y), cnt, _file);

		for(int64 i = 0; i < ret; ++i) str[i] = kmfswapbyte(str[i]);

		return ret;
	}

	template<typename Y> kmFile& operator<<(Y& data) { Write(&data); return *this; }
	template<typename Y> kmFile& operator>>(Y& data) { Read (&data); return *this; }

	//////////////////////////////////////////////////
	// member functions to read and write kmMat

	// write kmMat
	template<typename Y> void WriteMat(const kmArr<Y>* b)
	{
		ASSERTA(IsOpen(), "[kmFile::WriteMat in 2066]");

		// write header
		kmStra hd(b->GetKmClass()); if(hd.GetLen() > 63) hd(63) = '\0';
		fwrite((void*)hd, 1, 64, _file);

		// write info of kmArr
		fwrite(b->GetInfoPt(), b->GetInfoByte(), 1, _file);

		// write data
		if(kmArr<Y>::IsArr(b->Begin()))
		{	
			for(int64 n = b->N(), i = 0; i < n; ++i) WriteMat(&b->v(i));
		}
		else
		{
			int64 size = sizeof(Y);
			fwrite(     &size,    8,         1, _file);
			fwrite(b->Begin(), size, b->Size(), _file);
		}
	}

	// read kmMat
	template<typename Y> void ReadMat(kmArr<Y>* b)
	{
		ASSERTA(IsOpen(), "[kmFile::ReadMat in 2084]");

		// read header
		kmStra hd(64);
		fread((void*)hd, 1, 64, _file);	hd(63) = '\0';
		
		// get correct header
		kmStra hd0(b->GetKmClass()); if(hd.GetLen() > 63) hd(63) = '\0';
				
		// check header
		if(hd != hd0)
		{
			PRINTFA("[kmFile::Read] the type of kmFile(%s) is different from (%s)\n",
				    (char*)hd, (char*)hd0);
			throw KE_WRONG_CONFIG;
		}
		// release b
		b->Release();
		
		// read info of kmMat
		fread(b->GetInfoPt(), b->GetInfoByte(), 1, _file);

		// recreate mat
		b->Create(b->Size());

		// read data
		if(kmArr<Y>::IsArr(b->Begin()))
		{
			for(int64 n = b->N(), i = 0; i < n; ++i) ReadMat(&b->v(i));
		}
		else
		{
			int64 size;
			fread(&size, 8, 1, _file);

			if(size == sizeof(Y)) fread(b->Begin(), sizeof(Y), b->Size(), _file);
			else if(size < sizeof(Y)) // in case that class Y was bigger than before
			{
				print("[kmFile::Read, %s] size of class was bigger than before\n", b->GetKmClass());

				for(int i = 0; i < b->Size(); ++i) fread(b->P(i), size, 1, _file);
			}
			else // in case that class Y was smaller than before
			{
				print("[kmFile::Read, %s] size of class was smaller than before\n", b->GetKmClass());

				char* buf = new char[size];

				for(int i = 0; i < b->Size(); ++i)
				{
					fread(buf, size, 1, _file);
					memcpy(b->P(i), buf, sizeof(Y));
				}
				delete[] buf;
			}
		}
	}

	// dummy functions for a input which isn't a kmArr-based class
	void WriteMat(const void* b)
	{
		PRINTFA("[kmFile::WriteMat] called with a not kmArr\n");
	};
	void ReadMat(const void* b)
	{
		PRINTFA("[kmFile::ReadMat] called with a not kmArr\n");
	};

	//////////////////////////////////////////////////
	// member functions to read and write kmQue

	// write kmQue
	template<typename Y> void WriteQue(const kmQue<Y>* b)
	{
		ASSERTA(IsOpen(), "[kmFile::WriteQue in 3286]");

		// write header
		kmStra hd(16);  hd.SetVal(0);
		hd.SetStr("kmQue");

		fwrite((void*)hd, 1, 16, _file);

		// write _pobj
		WriteMat(&(b->_pobj));
		
		// write _obj
		WriteMat(&(b->_obj));
	}

	// read kmQue
	// * Note that if Y is a class with virtual function,
	// * you should reset a virtual funtion pointer (__vfptr).
	template<typename Y> void ReadQue(kmQue<Y>* b)
	{
		ASSERTA(IsOpen(), "[kmFile::ReadQue in 3309]");

		// read header
		kmStra hd(16);

		fread(hd, 1, 16, _file);

		// get correct header
		kmStra hd0(16);  hd0.SetVal(0);
		hd0.SetStr("kmQue");

		// check header
		if(hd != hd0)
		{
			PRINTFA("[kmFile::ReadQue] the type of kmFile(%s) is different from (%s)\n",
				    (char*)hd, (char*)hd0);
			throw KE_WRONG_CONFIG;
		}

		// read _pobj
		ReadMat(&(b->_pobj));

		// read _obj
		ReadMat(&(b->_obj));

		// re-arrange address of _obj		
		const int64 offset = (int64) b->_obj.Begin() - (int64) b->_pobj(0);

		for(int64 i = 0; i < b->N(); ++i)
		{
			b->_pobj(i) = (Y*)(void*)((int64) b->_pobj(i) + offset);
		}
	}
	
	//////////////////////////////////////////////////
	// member functions to read and write DIB file

	// write DIB... core
	void WriteDib(const kmImg& img)
	{
		// get headers
		BITMAPFILEHEADER fh = img.GetBmpFileHeader();
		BITMAPINFOHEADER ih = img.GetBmpInfoHeader();

		// write header
		Write(&fh);
		Write(&ih);

		// write
		Write(&img(0), img.Size());
	};

	// write DIB 
	template<typename T>
	static void WriteDib(const kmImg& img, const T* file_name)
	{
		kmFile file(file_name, KF_NEW); file.WriteDib(img);	file.Close();
	}

	// read DIB... core
	void ReadDib(kmImg& img)
	{
		// get headers
		BITMAPFILEHEADER fh;
		BITMAPINFOHEADER ih;

		// write header
		Read(&fh);
		Read(&ih);

		// init parameters
		const int64 n1 = (int64) abs(ih.biWidth );
		const int64 n2 = (int64) abs(ih.biHeight);

		// create img
		img.Recreate(n1, n2);

		// write		
		Read(&img(0), img.Size());
	};

	// read DIB
	template<typename T>
	static void ReadDib(kmImg& img, const T* file_name)
	{
		kmFile file(file_name, KF_READ); file.ReadDib(img);	file.Close();
	}

	//////////////////////////////////////////////////
	// get filename using openfilename dialog box

	// get file name for save
	static BOOL GetFileNameS(HWND hwnd, LPWSTR file_name, 
		                    int* index = NULL, LPWSTR filter = NULL, LPWSTR title = NULL)
	{
		// set openfilename
		OPENFILENAME ofn = {0,};	
		
		ofn.lStructSize = sizeof(OPENFILENAME);
		ofn.hwndOwner   = hwnd;
		ofn.lpstrTitle  = title;
		ofn.lpstrFilter = filter;
		ofn.lpstrFile   = file_name;
		ofn.nMaxFile    = 256;

		// open dialog box
		BOOL res = ::GetSaveFileName(&ofn);

		// set fileter index
		if(index != NULL) *index = ofn.nFilterIndex;

		return res;
	};

	// get file name for open
	static BOOL GetFileName(HWND hwnd, LPWSTR file_name, 
		                    int* index = NULL, LPWSTR filter = NULL, LPWSTR title = NULL)
	{
		// set openfilename
		OPENFILENAME ofn = {0,};	
		
		ofn.lStructSize = sizeof(OPENFILENAME);
		ofn.hwndOwner   = hwnd;
		ofn.lpstrTitle  = title;
		ofn.lpstrFilter = filter;
		ofn.lpstrFile   = file_name;
		ofn.nMaxFile    = 256;

		// open dialog box
		BOOL res = ::GetOpenFileName(&ofn);

		// set fileter index
		if(index != NULL) *index = ofn.nFilterIndex;

		return res;
	};

	// get folder name 
	static BOOL GetFolderName(HWND hwnd, LPWSTR path_name, LPWSTR title = nullptr)
	{
		BROWSEINFO brinfo{};

		brinfo.hwndOwner = hwnd;
		brinfo.lpszTitle = title;
		brinfo.ulFlags   = BIF_NEWDIALOGSTYLE | BIF_RETURNONLYFSDIRS;

		LPITEMIDLIST pitemlist = ::SHBrowseForFolder(&brinfo);

		return ::SHGetPathFromIDList(pitemlist, path_name);
	};
};

// file block buffer class
class kmFileBlk : public kmFile
{
public:
	int64    _byte        = 0;
	uint     _blk_byte    = 0;
	uint     _blk_n       = 0;
	int64    _writed_byte = 0;

	// constructor
	kmFileBlk() {};

	// open to read
	template<typename T>
	void OpenToRead(const T* name, uint blk_byte)
	{
		kmFile::Open(name, KF_READ);
		
		_byte     = kmFile::GetByte();
		_blk_byte = blk_byte;
		_blk_n    = uint((_byte - 1)/blk_byte +1);
	};
	
	// open to write
	template<typename T>
	void OpenToWrite(const T* name, int64 byte, uint blk_byte, uint blk_n)
	{
		kmFile::Open(name, KF_NEW);

		_byte = byte; _blk_byte = blk_byte; _blk_n = blk_n;
	};

	// get members
	uint  GetBlkByte() { return _blk_byte; };
	uint  GetBlkN   () { return _blk_n;    };
	int64 GetByte   () { return _byte;     };

	// read block 
	void ReadBlk(char* ptr, uint iblk)
	{
		Seek(iblk*(int64)_blk_byte); Read(ptr, GetBlkByte(iblk));
	};

	// write block
	void WriteBlk(uint iblk, char* ptr)
	{
		Seek(iblk*(int64)_blk_byte); Write(ptr, GetBlkByte(iblk));
	};

	// get block byte
	uint GetBlkByte(uint iblk)
	{
		ASSERTA(iblk < _blk_n, "[kmFileBlk::GetBlkByte in 1624] %d < %d", iblk, _blk_n);
		
		return (iblk == _blk_n - 1)? uint(_byte - _blk_byte*(int64)iblk) : _blk_byte;
	};
};

// class for log file
class kmLog : public kmFile
{	
	/////////////////////////////////////
	// member variables
protected:

	char _file_name[256]; // file name including path

	////////////////////////////////////
	// member functions
public:
	// constructor
	kmLog() : kmFile() { _file_name[0] = 0; }
	kmLog(const char* file_name) : kmFile()
	{
		SetFileName(file_name);		
	};

	// destructor
	~kmLog() {}

	// set file name
	void SetFileName(const char* file_name)
	{
		ASSERTA(strlen(file_name) < 128, "kmLog::SetFileName");

		strcpy_s(_file_name, file_name);
	};

	// check state 
	int IsFileNameValid() { return (_file_name[0]==0) ? 0:1; }
	
	// write string into log-files
	void Write(const char* str, ...)
	{
		ASSERTA(IsFileNameValid(), "kmLog::Write");

		// open file
		Open(_file_name, KF_ADD);

		// get time stamp			
		char time_str[32];
		GetTimeStr(time_str, sizeof(time_str));

		kmFile::WriteStr(time_str);
				
		// write str		
		va_list ap; // variable argument list

		va_start(ap, str);		
		vfprintf(_file, str, ap);
		va_end(ap);

		// write
		kmFile::WriteStr("\r\n");

		// close file
		Close();
	};

	// write Console text on the log file
	void WriteConsoleText(int readlineNum)
	{
		HANDLE	hStdout;
		hStdout = GetStdHandle(STD_OUTPUT_HANDLE);

		CONSOLE_SCREEN_BUFFER_INFO BufInfo;
		GetConsoleScreenBufferInfo(hStdout, &BufInfo);

		if (readlineNum > BufInfo.dwCursorPosition.Y)
			readlineNum = BufInfo.dwCursorPosition.Y;

		COORD	dwReadCoord;
		dwReadCoord.X = 0;
		dwReadCoord.Y = BufInfo.dwCursorPosition.Y - readlineNum;

		DWORD	nLength = BufInfo.dwSize.X; // One Line 80
		LPSTR	lpCharacter = (LPSTR)malloc(nLength * sizeof(CHAR));
		DWORD	lpNumberOfCharsRead;

		if(lpCharacter == NULL) return;

		// open file
		Open(_file_name, KF_ADD);

		// write to file
		kmFile::WriteStr("----------------------\n");

		while (dwReadCoord.Y < BufInfo.dwCursorPosition.Y) 
		{
			memset(lpCharacter, 0, nLength * sizeof(CHAR));
			ReadConsoleOutputCharacterA(hStdout, lpCharacter, nLength, dwReadCoord, &lpNumberOfCharsRead);

			fwrite(lpCharacter, sizeof(CHAR), lpNumberOfCharsRead, _file);
			kmFile::WriteStr("\n");
			
			dwReadCoord.Y++;
		}
		kmFile::WriteStr("----------------------\n");

		// close file
		Close();

		free(lpCharacter);
	};

private:
	// get string of time stamp
	void GetTimeStr(char* str, int str_size)
	{
		// get time
		time_t time_now = time(NULL);

		// convert time into struct tm
		tm time;
		localtime_s(&time, &time_now);

		// convert struct to str
		sprintf_s (str, str_size,
				"%d-%02d-%02d %02d:%02d:%02d> ",
				time.tm_year+1900, time.tm_mon+1, time.tm_mday,
				time.tm_hour,      time.tm_min,   time.tm_sec);
	};
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class for ini file control

#define ReadKey(section, key)              Read(section, ""#key"", key)
#define ReadKeyDefault(section, key, val0) Read(section, ""#key"", key, val0)

class kmIni
{
protected:
	const char* _file  = nullptr;
	int         _nsize = 64;     // string buffer size

public:
	kmIni(const char* file, int buf_size = 64) { Open(file, buf_size); };

	void Open(const char* file, int buf_size = 64)
	{
		if (_access(file, 0) == 0)
		{
			_file  = file;
			_nsize = buf_size;
		}
		else
		{
			PRINTFA("[kmIni::Open] file(%s) cannot be found\n", file);
			throw KE_CANNOT_OPEN;
		}
	};

	// read string
	void Read(LPCSTR section, LPCSTR key, LPSTR str)
	{
		// read ini file
		GetPrivateProfileStringA(section, key, NULL, str, _nsize, _file);
		
		// check if there is the key
		if (str[0] == NULL)
		{
			PRINTFA("[kmIni::Read] key (%s) cannot be found in %s\n", key, _file);
			throw KE_CANNOT_FIND;
		}
	};

	// read float	
	void Read(LPCSTR section, LPCSTR key, float& val)
	{
		// init string
		kmStra str(_nsize); str.SetStr("");

		// read ini file
		Read(section, key, str);

		// str --> float
		val = str.ToFloat();
	};

	// read float with default value
	void Read(LPCSTR section, LPCSTR key, float& val, const float default_val)
	{
		// init string
		kmStra str(_nsize); str.SetStr("");

		// read ini file
		Read(section, key, str);

		// str --> float
		val = (str(0) == NULL) ? default_val : str.ToFloat();
	};

	// read int
	void Read(LPCSTR section, LPCSTR key, int& val)
	{
		// init string
		kmStra str(_nsize); str.SetStr("");

		// read ini file
		Read(section, key, str);

		// str --> int
		val = str.ToInt();
	};

	// read int with default value
	void Read(LPCSTR section, LPCSTR key, int& val, const int default_val)
	{
		// init string
		kmStra str(_nsize); str.SetStr("");

		// read ini file
		Read(section, key, str);

		// str --> int
		val = (str(0) == NULL) ? default_val : str.ToInt();
	};
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class of registry
class kmRegistry 
{
public:
	HKEY   _hkey = 0;
	kmStrw _path;

	kmRegistry() {};
	kmRegistry(const kmStrw& path) { Open(path); };

	virtual ~kmRegistry() { if(_hkey != 0) Close(); };

	///////////////////////////////////////////
	// member functions
	void Open(const kmStrw& path)
	{	
		_path = path;

		DWORD dwdesc; wchar buf[128] = {0};

		//LSTATUS res = RegOpenKeyEx(HKEY_CURRENT_USER, (LPCWSTR)_path, 0, 
		//	                       KEY_ALL_ACCESS, &_hkey);

		LSTATUS res = RegCreateKeyEx(HKEY_CURRENT_USER, (LPCWSTR)_path, 0, buf, 
			           REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &_hkey, &dwdesc);

		if(res != ERROR_SUCCESS)
		{
			printw(L"* kmReg : fail to open (%s)\n", (LPCWSTR)_path);
		}
	};

	void Add(const kmStrw& name, const kmStrw& val)
	{
		DWORD dwdesc; wchar buf[128] = {0};

		LSTATUS res = RegCreateKeyEx(HKEY_CURRENT_USER, (LPCWSTR)_path, 0, buf, 
			           REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &_hkey, &dwdesc);

		if(res == ERROR_SUCCESS)
		{
			RegSetValueEx(_hkey, (LPCWSTR) name, 0, REG_SZ, (BYTE*)val.P(), (DWORD) val.Byte());
		}
		else { printw(L"* kmReg : fail to add key"); }
	};

	// add with template
	// * Note that T must be less than or equal to 8byte
	template<typename T> void Add(const kmStrw& name, T val)
	{
		DWORD dwdesc; wchar buf[128] = {0};

		LSTATUS res = RegCreateKeyEx(HKEY_CURRENT_USER, (LPCWSTR)_path, 0, buf, 
			REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &_hkey, &dwdesc);

		if(res == ERROR_SUCCESS)
		{
			uint64 val64 = (uint64)val;
			RegSetValueEx(_hkey, (LPCWSTR) name, 0, REG_QWORD, (BYTE*)&val64, 8);
		}
		else { printw(L"* kmReg : fail to add key"); }
	};

	// delete
	LSTATUS Delete(const kmStrw& name)
	{
		return RegDeleteValueW(_hkey, name.P());
	};

	// read
	LSTATUS Read(const kmStrw& name, kmStrw& val)
	{	
		wchar buf[128]; DWORD dwbyte = sizeof(buf), dwtype;

		LSTATUS res = RegQueryValueEx(_hkey, (LPCWSTR) name, 0, &dwtype, (LPBYTE)buf, &dwbyte);

		if(res == ERROR_SUCCESS) val = buf;

		return res;
	};

	// read with template
	// * Note that T must be less than or equal to 8byte
	template<typename T> LSTATUS Read(const kmStrw& name, T& val)
	{	
		uint64 val64; DWORD dwbyte = 8, dwtype;

		LSTATUS res = RegQueryValueEx(_hkey, (LPCWSTR) name, 0, &dwtype, (LPBYTE)&val64, &dwbyte);

		if(res == ERROR_SUCCESS) val = (T)val64;

		return res;
	};

	LSTATUS Close() { return RegCloseKey(_hkey); };
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class of thread

typedef unsigned (__stdcall *kmThreadFun)(void *);

// * Note that you sholud very carefully use a capture of lambda function,
// * since the captured variables cannot be available during thread's running.
// * So, it is recommanded using static or global for variables to be captured
// * One more thing, lambda function cannot capture any local variable
// * if it is called in any other thread.
class kmThread
{
protected:
	HANDLE _h      = 0;
	void*  _lambda = nullptr; // pointer of lambdafun
	void*  _arg0   = nullptr; // addtional argument0
	void*  _arg1   = nullptr; // addtional argument1
	void*  _arg2   = nullptr; // addtional argument2

	////////////////////////////////////////////
	// static functions
	template<class L> static
	unsigned __stdcall _kmThreadRun(void* arg)
	{	
		// get a thread object
		kmThread* const thrd = (kmThread*) arg;

		// exec lambda function
		(*(L*)(thrd->_lambda))();

		// terminate the thread
		thrd->Close(); _endthreadex(0); return 0;
	}

	template<class L, class A> static
	unsigned __stdcall _kmThreadRunArg(void* arg)
	{	
		// get a thread object
		kmThread* const thrd = (kmThread*) arg;

		// exec lambda function
		(*(L*)(thrd->_lambda))((A)thrd->_arg0);

		// terminate the thread
		thrd->Close(); _endthreadex(0); return 0;
	}

	template<class L, class A, class B> static
	unsigned __stdcall _kmThreadRunArg2(void* arg)
	{
		// get a thread object
		kmThread* const thrd = (kmThread*) arg;

		// exec lambda function
		(*(L*)(thrd->_lambda))((A)thrd->_arg0, (B)thrd->_arg1);

		// terminate the thread
		thrd->Close(); _endthreadex(0); return 0;
	}

	template<class L, class A, class B, class C> static
	unsigned __stdcall _kmThreadRunArg3(void* arg)
	{
		// get a thread object
		kmThread* const thrd = (kmThread*) arg;

		// exec lambda function
		(*(L*)(thrd->_lambda))((A)thrd->_arg0, (B)thrd->_arg1, (C)thrd->_arg2);

		// terminate the thread
		thrd->Close(); _endthreadex(0); return 0;
	}
	////////////////////////////////////////////
	// member functions
public:
	// constructors
	kmThread() {};
	kmThread(kmThreadFun cbfun, void* arg = NULL) { Begin(cbfun, arg); };

	// constructor with lambda function
	// * Note that these constructors have been removed 
	// * since there is a risk that thread members can be unavailable.
	// * So you should use kmThread object as member or static or global variable.
	// * Later, Let's apply self-destroy mode (sdm) to use thread without an object
	//
	//template<class L>                   kmThread(L lambdafun)   { Begin(lambdafun);}
	//template<class L, class A>          kmThread(L l, A a)      { Begin(l, a);     }
	//template<class L, class A, class B> kmThread(L l, A a, B b) { Begin(l, a, b);  }

	// destructor
	~kmThread() { Close(); };

	// begin a thread... core
	void Begin(kmThreadFun cbfun, void* arg = NULL)
	{
		if(_h == 0)
		{
			_h = (HANDLE) _beginthreadex(NULL, 0, cbfun, arg, 0, NULL);

			if(_h == 0) { PRINTFA("* kmThread: launching failed.\n"); throw KE_THREAD_FAILED; }
		}
		else PRINTFA("* kmThread: the same thread is already running.\n");
	};

	// begin a thread with lambda function
	template<class L> void Begin(L lambdafun)
	{
		_lambda = (void*)&lambdafun;
	
		Begin(_kmThreadRun<L>, (void*)this);
	}

	// begin a thread with lambda function
	template<class L, class A> void Begin(L lfun, A arg0)
	{
		_lambda = (void*)&lfun;
		_arg0   = (void*)arg0;
	
		Begin(_kmThreadRunArg<L,A>, (void*)this);
	}

	// begin a thread with lambda function
	template<class L, class A, class B> void Begin(L lfun, A arg0, B arg1)
	{
		_lambda = (void*)&lfun;
		_arg0   = (void*) arg0;
		_arg1   = (void*) arg1;
	
		Begin(_kmThreadRunArg2<L,A,B>, (void*)this);
	}

	// begin a thread with lambda function
	template<class L, class A, class B, class C> void Begin(L lfun, A arg0, B arg1, C arg2)
	{
		_lambda = (void*)&lfun;
		_arg0   = (void*) arg0;
		_arg1   = (void*) arg1;
		_arg2   = (void*) arg2;

		Begin(_kmThreadRunArg3<L,A,B,C>, (void*)this);
	}

	// wait for the thread finishing
	DWORD Wait(DWORD msec = INFINITE) { return WaitForSingleObject(_h, msec); };

	// wait for the thread starting
	void WaitStart() { while(!IsRunning()) { Sleep(0); } };

	// suspend the thread
	DWORD Suspend() { return SuspendThread(_h); };

	// resume the thread
	DWORD Resume() { return ResumeThread(_h); };

	// check status of thread
	BOOL IsRunning() { return _h != 0; };

	// set thread priority
	//   THREAD_MODE_BACKGOUND_BEING   : 0x00010000
	//   THREAD_MODE_BACKGOUND_END     : 0x00020000
	//   THREAD_PRIORITY_TIME_CRITICAL : 15
	//   THREAD_PRIORITY_ABOVE_NORMAL  : 1
	//   THREAD_PRIORITY_NORMAL        : 0
	//   THREAD_PRIORITY_BELOW_NORMAL  : -1
	//   THREAD_PRIORITY_LOWEST        : -2
	//   THREAD_PRIORITY_IDLE          : -15
	BOOL SetPriority(int priority) { return SetThreadPriority(_h, priority); };

	// get thread priority
	int GetPriority() { return GetThreadPriority(_h); };

	// close the handle
	BOOL Close()
	{
		if(_h == 0) return FALSE;

		BOOL ret = CloseHandle(_h);
		
		if(ret) _h = 0;

		return ret;
	};
};

// critical section class
// * Note that this class is just an example. Use kmLock instead.
class __kmCs
{
protected:
	CRITICAL_SECTION _cs;
public:
	// constructor
	__kmCs() { InitializeCriticalSection(&_cs); };

	// destructor
	~__kmCs() { DeleteCriticalSection(&_cs); };

	// enter critical section
	void Enter() { EnterCriticalSection(&_cs); };

	// leave critical section
	void Leave() { LeaveCriticalSection(&_cs); };
};

// atomic critical section class
// * Note that this class is just an example. Use kmLock instead.
class __kmCsat
{
protected:
	atomic_flag _lck = ATOMIC_FLAG_INIT;	
public:
	// enter critical section
	void Enter() noexcept {	while(_lck.test_and_set(memory_order_acquire));};

	// leaver critical section	
	void Leave() noexcept { _lck.clear(memory_order_release); };
};

// atomic mutex (mutual exclusion) class
// * Note that char or bool is slower than int or short.
class kmLock
{
protected:
	atomic<ushort> _lck = {0}, _cnt = {0};
public:	
	// constructor
	kmLock() {};

	// copy-construtor and assignment operator will be banned
	kmLock           (const kmLock&) = delete;
	kmLock& operator=(const kmLock&) = delete;

	// lock
	// * Note that this will ban both locking and entering of another thread.
	kmLock* Lock() noexcept
	{
		ushort b = 0;		
		while(!_lck.compare_exchange_strong(b, 1, memory_order_acquire)) b = 0;
		while(_cnt > 0);
		return this;
	};

	// unlock
	void Unlock() noexcept { _lck.store(0, memory_order_release); };

	// enter
	// * Note that this will ban only locking of another thread, not entering.	
	kmLock& Enter() noexcept
	{
		while(_lck == 1);
		++_cnt;
		if(_lck == 1) { --_cnt; Enter(); }
		return *this;
	};

	// leave 
	void Leave() noexcept { if(_cnt > 0) --_cnt; };

	// Is functions
	bool IsLock() const { return (_lck == 1); };

	// get functions
	ushort GetCount() const noexcept { return _cnt; };
	ushort GetLock () const noexcept { return _lck; };
};

class kmLockGuard
{
protected:
	kmLock* _plck = nullptr;
	int     _type = 0;      // 0 : lock, 1 : enter
public:
	
	// constructor
	kmLockGuard(kmLock* plck) { _type = 0; _plck = plck; }; // lock
	kmLockGuard(kmLock&  lck) { _type = 1; _plck = &lck; }; // enter

	// destructor
	~kmLockGuard() { if(_type == 0) _plck->Unlock(); else _plck->Leave(); };

	// copy-construtor and assignment operator will be banned
	kmLockGuard           (const kmLockGuard&) = delete;
	kmLockGuard& operator=(const kmLockGuard&) = delete;
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// loop functions for lambda function
//
// * Note that const on "const M<T>&" is only for compatibility.
// * The elements of "const M<T>&" can be changed.

template<template<typename> class M, typename T, class L>
void foreach(const M<T>& a, L lambdafun)
{
	for(int64 i = a.N(); i--;) lambdafun(a(i)); 
}

template<template<typename> class M, typename T1, typename T2, class L>
void foreach(const M<T1>& a, const M<T2>& b, L lambdafun)
{
	for(int64 i = a.N(); i--;) lambdafun(a(i), b(i));
}

template<template<typename> class M, typename T1, typename T2, typename T3, class L>
void foreach(const M<T1>& a, const M<T2>& b, const M<T3>& c, L lambdafun)
{
	for(int64 i = a.N(); i--;) lambdafun(a(i), b(i), c(i));
}

template<template<typename> class M, typename T1,typename T2, typename T3, typename T4, class L>
void foreach(const M<T1>& a, const M<T2>& b, const M<T3>& c, const M<T4>& d, L lambdafun)
{
	for(int64 i = a.N(); i--;) lambdafun(a(i), b(i), c(i), d(i));
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class of work

// class for work 
//  bytes of arguement buffer(_buf) is 32
class kmWork
{
protected:
	int _id{}, _byte{}, _ptr{}; char _buf[32]{};

public:
	// constructor
	kmWork() {};
	template<typename... Ts> 
	kmWork(int id, Ts... args) : _id(id) { Set(args...); }

	// set arguments
	template<typename T, typename... Ts> 
	void Set(T arg, Ts... args)
	{
		if(_byte + sizeof(T) <= sizeof(_buf))
		{
			*(T*)(_buf + _byte) = arg; _byte += sizeof(T);
		}
		else print("* [kmWork::Set] buffer was over\n");

		Set(args...);
	}
	void Set() {}; // to finish a recursve call

	// set operator
	template<typename T>
	kmWork& operator<<(const T& arg) { Set(arg); return *this; }

	// set id
	kmWork& SetId(int id) { _id = id; return *this; };

	// get id or byte
	int Id  () { return _id; };
	int Byte() { return _byte; };

	// get arguments
	template<typename... Ts, typename T> 
	void Get(T& arg, Ts&... args) { _ptr = 0; _Get(arg); _Get(args...); }

	// get operator
	template<typename T> 
	kmWork& operator>>(T& arg) { _Get(arg); return *this; }

	// begin to get argument using operator>>
	kmWork& Begin() { _ptr = 0; return *this; };

private:
	template<typename... Ts, typename T> 
	void _Get(T& arg, Ts&... args)
	{
		arg = *(T*)(_buf + _ptr); _ptr += sizeof(T);

		_Get(args...);
	}
	void _Get() {}; // to finish a recursive call
};

// class for works including thread and mutex
class kmWorks : public kmQue1<kmWork>
{
protected:
	kmThread _thrd;
	kmLock   _lck;
	void*    _lfun = nullptr;
	void*    _arg  = nullptr;
public:
	// create thread for work
	// example>
	//	wrks.Create([](kmWork& wrk, kmClass* cls)
	//	{	 	
	//		short arg0; float arg1;
	//		
	//		switch(wrk.Id())
	//		{
	//		case 0: wrk.Get(arg0);       cls->fun1(arg0);       break;
	//		case 1: wrk.Get(arg0, arg1); cls->fun2(arg0, arg1); break;	
	//		}	
	//	}, this);
	template<class L, class A> void Create(L lfun, A arg)
	{
		// create queue
		kmQue1<kmWork>::Recreate(16);

		// set lambda funtions
		_lfun = (void*)&lfun;
		_arg  = (void*) arg;

		// create thread
		_thrd.Begin([](kmWorks* wrks)
		{
			print("* work thread starts\n");
			while(1)
			{
				if(wrks->N1() > 0)
				for(int i = (int)wrks->N1(); i--;)
				{
					wrks->_lck.Lock();   //////// lock
					kmWork wrk = *wrks->Dequeue();
					wrks->_lck.Unlock(); //////// unlock

					(*(L*)(wrks->_lfun))(wrk, (A)wrks->_arg);
				}
				Sleep(1);
			}
		}, this);
		_thrd.WaitStart();
	}

	// enqueue work
	template<typename... Ts>
	kmWorks& Enqueue(int id, Ts... args)
	{
		_lck.Lock();   ///////////////////////// lock
		kmQue1<kmWork>::Enqueue(kmWork(id, args...));
		_lck.Unlock(); /////////////////////////  unlock
		return *this;
	}
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class of group
template<typename T>
class kmGrp
{
protected:
	kmMat1<T*> _member;

public:
	// constructor
	kmGrp() { _member.Create(0,8); };	

	// operator()
	T& operator()(int64 idx) { return *_member(idx); };

	// add members
	template<typename ... Ts>
	kmGrp& Add(T& a, Ts& ... as) { _member.PushBack((T*)&a); Add(as...); return *this; }
	void   Add() {}; // to finish a recursive call.

	// call lambdafun for every members
	//  * ex : kmGrp<kmwChild> _grp; ... ; _grp.All([](kmwChild* a){ a->Hide(); });
	template<class L>
	void All(L lambdafun) { foreach(_member, lambdafun); }
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// operation functions for kmMat

/////////////////////////////////////////////////////
// operator+=

// general form of operator+=... a+=b
template<template<typename> class M, typename T> 
M<T>& operator+=(M<T>& a, const M<T>& b)
{
	ASSERTFA(a.IsEqualSizeDim(b), "[operator+= in 3453]");

	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		a(i) += b(i);
	}
	return a;
}

// general form of operator+=... a+=b
template<template<typename> class M, typename T> 
M<T>& operator+=(M<T>& a, const T b)
{
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		a(i) += b;
	}
	return a;
}
/////////////////////////////////////////////////////
// operator-=

// general form of operator-=... a-=b
template<template<typename> class M, typename T> 
M<T>& operator-=(M<T>& a, const M<T>& b)
{	
	ASSERTFA(a.IsEqualSizeDim(b), "[operator-= in 3469]");
		
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		a(i) -= b(i);
	}
	return a;
}

// general form of operator-=... a-=b
template<template<typename> class M, typename T> 
M<T>& operator-=(M<T>& a, const T b)
{
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		a(i) -= b;
	}
	return a;
}

/////////////////////////////////////////////////////
// operator *= ...subtraction and assign

// general form of operator*=... a*=b
template<template<typename> class M, typename T> 
M<T>& operator*=(M<T>& a, const M<T>& b)
{
	ASSERTFA(a.IsEqualSizeDim(b), "[operator*= in 3485]");

	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		a(i) *= b(i);
	}
	return a;
}

// general form of operator*=... a*=b
template<template<typename> class M, typename T> 
M<T>& operator*=(M<T>& a, const T b)
{
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		a(i) *= b;
	}
	return a;
}

/////////////////////////////////////////////////////
// operator /= 

// general form of operator/=... a/=b
template<template<typename> class M, typename T> 
M<T>& operator/=(M<T>& a, const M<T>& b)
{
	ASSERTFA(a.IsEqualSizeDim(b), "[operator/= in 3501]");

	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		a(i) /= b(i);
	}
	return a;
}

// general form of operator/=... a/=b
template<template<typename> class M, typename T> 
M<T>& operator/=(M<T>& a, const T b)
{
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		a(i) /= b;
	}
	return a;
}

/////////////////////////////////////////////////////
// operator + ...addition

// general form of operator+... c = a + b
template<template<typename> class M, typename T>
M<T> operator+(const M<T>& a, const M<T>& b)
{
	ASSERTFA(a.N() == b.N(), "[operator+ in 3509]");

	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = a(i) + b(i);
	}
	return c;
}

// general form of operator+... c = a + b
template<template<typename> class M, typename T>
M<T> operator+(const M<T>& a, const T b)
{	
	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = a(i) + b;
	}
	return c;
}

// general form of operator+... c = a + b
template<template<typename> class M, typename T>
M<T> operator+(const T b, const M<T>& a)
{	
	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = a(i) + b;
	}
	return c;
}

// kmMat2 operator+... c = a + b
template<typename T> 
kmMat2<T> operator+(const kmMat2<T>& a, const kmMat2<T>& b)
{	
	const int64 n1 = a.N1(), n2 = a.N2();

	ASSERTFA(a.IsEqualSizeDim(b), "[operator+ in 1443]");

	kmMat2<T> c(n1,n2);
	
	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{
		c(i1, i2) = a(i1, i2) + b(i1, i2);
	}
	return c;
}

// kmMat2 operator+... c = a + b
template<typename T>
kmMat2<T> operator+(const kmMat2<T>& a, const T b)
{	
	const int64 n1 = a.N1(), n2 = a.N2();

	kmMat2<T> c(n1, n2);
				
	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{
		c(i1, i2) = a(i1, i2) + b;
	}
	return c;
}

// kmMat2 operator+... c = b + a
template<typename T>
kmMat2<T> operator+(const T b, const kmMat2<T>& a)
{	
	const int64 n1 = a.N1(), n2 = a.N2();	

	kmMat2<T> c(n1, n2);

	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{
		c(i1, i2) = b + a(i1, i2);
	}
	return c;
}

/////////////////////////////////////////////////////
// operator * ...mutification

// general form of operator*... c = a * b
template<template<typename> class M, typename T>
M<T> operator*(const M<T>& a, const T b)
{	
	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = b*a(i);
	}
	return c;
}

// general form of operator*... c = b * a
template<template<typename> class M, typename T>
M<T> operator*(const T b, const M<T>& a)
{	
	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = b*a(i);
	}
	return c;
}

// kmMat2 operator*... c = a * b
template<typename T>
kmMat2<T> operator*(const kmMat2<T>& a, const T b)
{	
	const int64 n1 = a.N1(), n2 = a.N2();

	kmMat2<T> c(n1, n2);

	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{
		c(i1, i2) = a(i1, i2)*b;
	}
	return c;
}

// kmMat2 operator*... c = b * a
template<typename T>
kmMat2<T> operator*(const T b, const kmMat2<T>& a)
{
	const int64 n1 = a.N1(), n2 = a.N2();

	kmMat2<T> c(n1, n2);

	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{
		c(i1, i2) = a(i1, i2)*b;
	}
	return c;
}

// kmMat2 operator*... c = a * b
template<typename T>
kmMat2<T> operator*(const kmMat2<T>& a, const kmMat2<T>& b)
{		
	const int64 an1 = a.N1(), an2 = a.N2();
	const int64 bn1 = b.N1(), bn2 = b.N2();

	ASSERTFA(an2 == bn1, "[operator* in 1691]");

	kmMat2<T> c(an1, bn2);
				
	for(int64 i2 = 0; i2 < bn2; ++i2)
	for(int64 i1 = 0; i1 < an1; ++i1)
	{
		T sum = 0; 
		for(int64 i = 0; i  < an2; ++i ) sum += a(i1, i)*b(i, i2);
		c(i1, i2) = sum;
	}
	return c;		
}

// kmMat3 operator*... c = a * b
template<typename T>
kmMat3<T> operator*(const kmMat3<T>& a, const kmMat3<T>& b)
{		
	const int64 an1 = a.N1(), an2 = a.N2(), an3 = a.N3();
	const int64 bn1 = b.N1(), bn2 = b.N2(), bn3 = b.N3();

	ASSERTFA(an2 == bn1, "[operator* in 3621]");
	ASSERTFA(an3 == bn3, "[operator* in 3622]");

	kmMat3<T> c(an1, bn2, an3);
	
	for(int64 i3 = 0; i3 < an3; ++i3)
	for(int64 i2 = 0; i2 < bn2; ++i2)
	for(int64 i1 = 0; i1 < an1; ++i1)
	{
		T sum = 0;
		for(int64 i = 0; i  < an2; ++i ) sum += a(i1, i, i3)*b(i, i2, i3);
		c(i1, i2, i3) = sum;
	}
	return c;
}

// kmMat4 operator*... c = a * b
template<typename T>
kmMat4<T> operator*(const kmMat4<T>& a, const kmMat4<T>& b)
{		
	const int64 an1 = a.N1(), an2 = a.N2(), an3 = a.N3(), an4 = a.N4();
	const int64 bn1 = b.N1(), bn2 = b.N2(), bn3 = b.N3(), bn4 = b.N4();

	ASSERTFA(an2 == bn1, "[operator* in 3643]");
	ASSERTFA(an3 == bn3, "[operator* in 3644]");
	ASSERTFA(an4 == bn4, "[operator* in 3645]");

	kmMat4<T> c(an1, bn2, an3, an4);
	
	for(int64 i4 = 0; i4 < an4; ++i4)
	for(int64 i3 = 0; i3 < an3; ++i3)
	for(int64 i2 = 0; i2 < bn2; ++i2)
	for(int64 i1 = 0; i1 < an1; ++i1)
	{
		T sum = 0; 
		for(int64 i = 0; i  < an2; ++i ) sum += a(i1, i, i3, i4)*b(i, i2, i3, i4);
		c(i1, i2, i3, i4) = sum;
	}
	return c;
}

/////////////////////////////////////////////////////
// operator - ...subtraction

// general form of operator-... c = a - b
template<template<typename> class M, typename T>
M<T> operator-(const M<T>& a, const M<T>& b)
{	
	ASSERTFA(a.IsEqualSizeDim(b), "[operator- in 3476]");

	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = a(i) - b(i);
	}
	return c;
}

// general form of operator-... c = a - b
template<template<typename> class M, typename T>
M<T> operator-(const M<T>& a, const T b)
{	
	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = a(i) - b;
	}
	return c;
}

// general form of operator-... c = b - a
template<template<typename> class M, typename T>
M<T> operator-(const T b, const M<T>& a)
{	
	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = b - a(i);
	}
	return c;
}

// kmMat2 operator-... c = a - b
template<typename T>
kmMat2<T> operator-(const kmMat2<T>& a, const kmMat2<T>& b)
{	
	const int64 n1 = a.N1(), n2 = a.N2();

	ASSERTFA(a.IsEqualSizeDim(b), "[operator- in 2157]");

	kmMat2<T> c(n1, n2);
	
	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{
		c(i1, i2) = a(i1, i2) - b(i1, i2);
	}
	return c;
}

// kmMat2 operator-... c = a - b
template<typename T>
kmMat2<T> operator-(const kmMat2<T>& a, const T b)
{		
	const int64 n1 = a.N1(), n2 = a.N2();

	kmMat2<T> c(n1, n2);

	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{
		c(i1, i2) = a(i1, i2) - b;
	}
	return c;
}

// kmMat2 operator-... c = b - a
template<typename T>
kmMat2<T> operator-(const T b, const kmMat2<T>& a)
{	
	const int64 n1 = a.N1(), n2 = a.N2();

	kmMat2<T> c(n1, n2);

	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{
		c(i1, i2) = b - a(i1, i2);
	}
	return c;
}

/////////////////////////////////////////////////////
// operator / ...division

// general form of operator/... c = a / b
template<template<typename> class M, typename T>
M<T> operator/(const M<T>& a, const T b)
{	
	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = a(i) / b;
	}
	return c;
}

// general form of operator/... c = b / a
template<template<typename> class M, typename T>
M<T> operator/(const T b, const M<T>& a)
{	
	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = b / a(i);
	}
	return c;
}

// kmMat2 operator/... c = a / b
template<typename T>
kmMat2<T> operator/(const kmMat2<T>& a, const T b)
{
	// init parameters		
	const int64 n1 = a.N1(), n2 = a.N2();
		
	// subtract matrix
	kmMat2<T> c(n1, n2);
				
	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{		
		c(i1, i2) = a(i1, i2) / b;
	}
	return c;
}

// kmMat2 operator/... c = b / a
template<typename T>
kmMat2<T> operator/(const T b, const kmMat2<T>& a)
{
	// init parameters		
	const int64 n1 = a.N1(), n2 = a.N2();	
		
	// divide matrix
	kmMat2<T> c(n1, n2);
				
	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{		
		c(i1, i2) = b / a(i1, i2);
	}
	return c;
}

/////////////////////////////////////////////////////
// operator ^ ...power

// general form of operator^... c = a^b ... c = pow(a, b)
template<template<typename> class M, typename T>
M<T> operator^(const M<T>& a, const T b)
{	
	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = pow(a(i), b);
	}
	return c;
}

// kmMat2 operator^... c = a^b ... c = pow(a, b)
template<typename T>
kmMat2<T> operator^(const kmMat2<T>& a, const T b)
{	
	const int64 n1 = a.N1(), n2 = a.N2();

	kmMat2<T> c(n1, n2);

	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{
		c(i1, i2) = pow(a(i1, i2), b);
	}
	return c;
}

/////////////////////////////////////////////////////
// operator Mul ...mutification with each element

// general form of operator+... c = a .* b
template<template<typename> class M, typename T>
M<T> Mul(const M<T>& a, const M<T>& b)
{
	ASSERTFA(a.N() == b.N(), "[Mul in 3872]");

	M<T> c(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		c(i) = a(i) * b(i);
	}
	return c;
}

// kmMat2 multify each elements ... c = a .* b
template<typename T>
kmMat2<T> Mul(const kmMat2<T>& a, const kmMat2<T>& b)
{	
	const int64 n1 = a.N1(), n2 = a.N2();

	ASSERTFA(a.IsEqualSizeDim(b), "[Mul in 3888]");
	
	kmMat2<T> c(n1, n2);
				
	for(int64 i2 = 0; i2 < n2; ++i2)
	for(int64 i1 = 0; i1 < n1; ++i1)
	{
		c(i1, i2) = a(i1, i2) * b(i1, i2);
	}
	return c;
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// math functions for kmMat

// max function... ... general form for kmArr, kmMat1, kmMat2, kmMat3
template<template<typename> class M, typename T>
T Max(const M<T>& a)
{		
	T b = a(0);
	for(int64 n = a.N(), i = 1; i < n; ++i) if(b < a(i)) b = a(i);
	return b;
}

// min function... ... general form for kmArr, kmMat1, kmMat2, kmMat3
template<template<typename> class M, typename T>
T Min(const M<T>& a)
{		
	T b = a(0);
	for(int64 n = a.N(), i = 1; i < n; ++i) if(b > a(i)) b = a(i);
	return b;
}

// sin function... general form for kmArr, kmMat1, kmMat2, kmMat3
template<template<typename> class M, typename T>
M<T> Sin(const M<T>& a)
{		
	M<T> b(a);
	for(int64 n = a.N(), i = 0; i < n; ++i) b(i) = sin(b(i));
	return b;
}

// cos function... general form for kmArr, kmMat1, kmMat2, kmMat3
template<template<typename> class M, typename T>
M<T> Cos(const M<T>& a)
{		
	M<T> b(a);
	for(int64 n = a.N(), i = 0; i < n; ++i) b(i) = cos(b(i));
	return b;
}

// sigmoid function... general form for kmArr, kmMat1, kmMat2, kmMat3
template<template<typename> class M, typename T>
M<T> Sigmoid(const M<T>& a)
{		
	M<T> b(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		b(i) = (T) 1/((T) 1 + exp(-b(i)));
	}
	return b;
}

// rectified linear unit function (ReLU)... general form for kmArr, kmMat1, kmMat2, kmMat3
template<template<typename> class M, typename T>
M<T> ReLU(const M<T>& a)
{	
	M<T> b(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		b(i) = MAX(0, b(i));
	}
	return b;
}

// natural exponential... general form for kmArr, kmMat1, kmMat2, kmMat3
template<class T>
T Exp(const T& a)
{		
	T b(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		b(i) = exp(b(i));
	}
	return b;
}

// natural logarithm... general form for kmArr, kmMat1, kmMat2, kmMat3
template<class T>
T Log(const T& a)
{	
	T b(a);
	for(int64 n = a.N(), i = 0; i < n; ++i)
	{
		b(i) = log(b(i));
	}
	return b;
}

// general form... for kmArr, kmMat1, kmMat2, kmMat3
template<template<typename> class M, typename T>
M<T> Softmax(const M<T>& a)
{
	M<T> exp_a = Exp(a - a.Max());

	return exp_a/exp_a.Sum();
}

// mean squared error... general form for kmArr, kmMat1, kmMat2, kmMat3
template<template<typename> class M, typename T>
T Mse(const M<T>& y, const M<T>& t)
{
	M<T> se = (y - t)^2.f;

	return se.Mean();
}

// cross entropy error... general form for kmArr, kmMat1, kmMat2, kmMat3
template<template<typename> class M, typename T>
T Cee(const M<T>& y, const M<T>& t)
{
	M<T> cee = Mul(t, Log(y + (T)1e-12));

	return -cee.Sum();
}

// next power2
template<typename T>
T NextPow2(const T a)
{
	return (T) round(pow(2, floor(log2((double)a))));
}

// hypot
template<template<typename> class M, typename T>
M<T> Hypot(const M<T>& a, const M<T>& b)
{
	M<T> c(a);

	foreach(c, a, b, [](T& z, T x, T y) { z = hypot(x, y); });

	return c;
}

// log compression
template<template<typename> class M, typename T>
M<T> LogCompression(const M<T>& a, const M<T>& b)
{
	M<T> c(a);

	foreach(c, a, b, [](T& z, T x, T y) { z = log10(hypot(x, y) + T(1)); });
	
	return c;
}

// dB scale... min value : -200dB
template<template<typename> class M, typename T>
M<T> dB(const M<T>& a)
{	
	M<T> b(a);

	for(int64 i = 0, n = b.N(); i < n; ++i) b(i) = T(20)*log10(abs(b(i)) + T(1.e-10));

	return b;
}

// normalized.. with min and max
template<template<typename> class M, typename T>
M<T> Normalize(const M<T>& a)
{
	M<T> b(a);

	const T amax = a.Max(), amin = a.Min();
	const T rat  = (amax == amin) ? T(1):T(1)/(amax - amin);
	
	for(int64 i = 0, n = b.N(); i < n; ++i) b(i) = (b(i) - amin)*rat;

	return b;
}

// normalized dB... min value : -200dB
template<template<typename> class M, typename T>
M<T> dBnorm(const M<T>& a)
{	
	M<T> b(a);
	
	T max_b = b.MaxAbs();

	if(max_b == 0) b.SetZero();
	else
	{
		T inv_max = T(1)/max_b;

		for(int64 i = 0, n = b.N(); i < n; ++i)
		{
			b(i) = T(20)*log10(abs(b(i))*inv_max + T(1.e-10));
		}
	}	
	return b;
}

// calc contrast using sobel
template<typename T>
T ContrastSobel(const kmMat2<T>& a)
{
	const int64 n1 = a.N1(), n2 = a.N2();

	if(n1 < 3 || n2 < 3) return 0;

	T sum = 0;
	for(int64 i2 = 1; i2 < n2 - 1; ++i2)
	for(int64 i1 = 1; i1 < n1 - 1; ++i1)
	{	
		T gx  = a(i1-1, i2-1) + 2.f*a(i1-1,i2) + a(i1-1,i2+1);
		  gx -= a(i1+1, i2-1) + 2.f*a(i1+1,i2) + a(i1+1,i2+1);

		T gy  = a(i1-1, i2-1) + 2.f*a(i1,i2-1) + a(i1+1,i2-1);
		  gy -= a(i1-1, i2+1) + 2.f*a(i1,i2+1) + a(i1+1,i2+1);

		sum += gx*gx + gy*gy;
	}
	return sum/((n1 - 2)*(n2 - 2));
}

// calc contrast using SML (sum-modifed Laplacian)
template<typename T>
T ContrastSml(const kmMat2<T>& a)
{
	const int64 n1 = a.N1(), n2 = a.N2();

	if(n1 < 3 || n2 < 3) return 0;

	T sum = 0;
	for(int64 i2 = 1; i2 < n2 - 1; ++i2)
	for(int64 i1 = 1; i1 < n1 - 1; ++i1)
	{	
		T gx  = a(i1-1, i2) - 2.f*a(i1,i2) + a(i1+1,i2);
		T gy  = a(i1, i2-1) - 2.f*a(i1,i2) + a(i1,i2+1);

		sum += gx*gx + gy*gy;
	}
	return sum/((n1 - 2)*(n2 - 2));
}

// LU decomposition...  [lu] = [a]
// * Note that lu is the compressed L and U matrix.
// * Pivoting is not implemented yet.
template<typename T> 
kmMat2<T> DecomposeLU(const kmMat2<T>& a)
{
	// check 
	ASSERTFA( a.N1() == a.N2(),"DecomposeLU in 1354");

	// init parameters
	const int64 n = a.N1();

	// create and init output
	kmMat2<T> lu = a;

	// main loop
	for(int64 j = 0; j < n - 1; ++j)
	{
		// check singularity
		if(lu(j,j) == 0) 
		{
			PRINTFA("* LU is singular at(%lld,%lld)\n",j,j);
			throw KE_DIVIDE_BY_ZERO;
		}

		for(int64 i = j + 1; i < n; ++i)
		{
			// get L(i,j)
			// L(i,j) = (A(i,j) - sum( L(i,k)*U(k,j), k=0:j-1))/U(j,j)
			{
				T sum = 0;
				for(int64 k = 0; k < j; ++k)
				{
					sum += lu(i,k)*lu(k,j);
				}
				lu(i,j) = (lu(i,j) - sum)/lu(j,j);
			}
			// get U(j+1,i)
			// U(j+1,i) = A(j+1,i) - sum( L(j+1,k)*U(k,i), k=0:j)
			{
				T sum = 0;
				for(int64 k = 0; k < j+1; ++k)
				{
					sum += lu(j+1,k)*lu(k,i); 
				}
				lu(j+1,i) -= sum;
			}
		}
	}
	return lu;
}

// x = a\b ({b} = [a]{x} --> {x} = inv([a]) * {b}
template<typename T>
kmMat1<T> DivideLeft(const kmMat2<T>& a, const kmMat1<T>& b)
{
	// check the size of matrix	
	ASSERTFA(a.N1() == a.N2(), "DivideLeft in 1406");
	ASSERTFA(a.N1() == b.N1(), "DivideLeft in 1407");

	// create output
	kmMat1<T> x(a.N1());

	// init parameters
	const int64 n = a.N1();

	kmMat1<T> y(n); y.SetZero();
	
	// decompose a into LU
	kmMat2<T> lu = DecomposeLU(a);
	
	// solve Ly = b
	for(int64 i = 0; i < n; ++i)
	{
		T sum = 0;
		for(int k = 0; k < i; ++k)
		{
			sum += y(k)*lu(i,k);
		}
		y(i) = b(i) - sum;
	}

	// solve Ux = y
	for(int64 i = n - 1; i > -1; --i)
	{
		T sum = 0;
		for(int64 k = n - 1; k > i; --k)
		{
			sum += x(k)*lu(i,k);
		}
		x(i) = ((y(i) - sum)/lu(i,i));
	}
	return x;
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// signal processing functions for kmMat

// get Toeplitz matrix (or diagonal-constant matrix)
template<typename T>
kmMat2<T> Toeplitz(const kmMat1<T> b)
{
	kmMat2<T> a(b.N1(), b.N1());

	for(int64 i = 0; i < b.N1(); ++i)
	for(int64 j = 0; j <= i    ; ++j)
	{
		a(i,j) = a(j,i) = b(i-j);
	}		
	return a;
}

// get Hankel matrix (or catalecticant matrix)
template<typename T>
kmMat2<T> Hankel(const kmMat1<T>& b, const kmMat1<T>& c)
{
	// check
	ASSERTFA(b.N1() == c.N1(), "kmHankel in 4102");

	kmMat2<T> a(b.N1(), b.N1());

	for(int64 j = 0; j < a.N2(); ++j)
	for(int64 i = 0; i < a.N1(); ++i)
	{
		if(i + j < a.N1()) a(i,j) = b(i + j);
		else               a(i,j) = c(i + j - a.N1() + 1);
	}
	return a;
}

// get kernel of bpf
// w   : weight of stop bands (the weight of pass band is 1)
// wls : lower stop normalized freq.
// wlp : lower pass normalized freq.
// wup : upper pass normalized freq.
// wus : upper stop normalized freq.
template<typename T = float> 
kmMat1<T> FirlsBpf(int n_tap, float wls, float wlp, float wup, float wus, float w = 1.f)
{
	// check
	ASSERTFA( n_tap > 0 , "kmFirlsBpf in 4119");
	ASSERTFA( wls >= 0  , "kmFirlsBpf in 4120");
	ASSERTFA( wlp >= wls, "kmFirlsBpf in 4121");
	ASSERTFA( wup >  wlp, "kmFirlsBpf in 4122");
	ASSERTFA( wus >= wup, "kmFirlsBpf in 4123");
	ASSERTFA( wup <= 1.f, "kmFirlsBpf in 4124");

	// create output
	kmMat1<T> a(n_tap);

	// determine type
	int type = 2 - n_tap%2; // type 1: odd taps, type 2: even taps

	// get filter of type I
	if(type == 1)
	{
		// init parameters
		int64 m = (n_tap - 1)/2;

		// calc qn(2m+1)
		kmMat1<T> qn(2*m + 1);

		for(int64 n = 0; n < 2*m + 1; ++n)
		{
			qn(n) = wup*SINC(wup*n) - wlp*SINC(wlp*n) 
			        + w*( wls*SINC(wls*n) + SINC(n) - wus*SINC(wus*n) );
		}

		// calc bk(m + 1)
		kmMat1<T> bk(m + 1);

		for(int k = 0; k < m + 1; ++k)
		{
			bk(k) = wup*SINC(wup*k) - wlp*SINC(wlp*k);
		}

		// set Q1(m + 1, m + 1) and Q2(m + 1, m + 1)
		kmMat1<T> qn1(&qn(0), m + 1);
		kmMat1<T> qn2(&qn(m), m + 1);
		
		kmMat2<T> Q1 = Toeplitz(qn1);
		kmMat2<T> Q2 = Hankel  (qn1, qn2);

		// calc b(m + 1)
		kmMat1<T> b = DivideLeft(Q1 + Q2, bk);

		// set kernel this(2m+1, 1)
		a(m) = b(0)*2.f;

		for(int64 i = 1; i < m + 1; ++i)
		{
			a(m - i) = a(m + i) = b(i);
		}
	}
	// get filter of type II
	else if(type == 2)
	{
		// init parameters
		int64 m = n_tap/2;

		// calc qn(2m,1)
		kmMat1<T> qn(2*m);

		for(int64 n = 0; n < 2*m; ++n)
		{
			qn(n) = wup*SINC(wup*n) - wlp*SINC(wlp*n) 
			        + w*(wls*SINC(wls*n) + SINC(n) - wus*SINC(wus*n));
		}

		// calc bk(m, 1)
		kmMat1<T> bk(m);

		for(int64 k = 0; k < m; ++k)
		{
			bk(k) = wup*SINC(wup*(k + 0.5f)) - wlp*SINC(wlp*(k + 0.5f));
		}

		// set Q1(m, m) and Q2(m, m)
		kmMat1<T> qn1(&qn(0), m);
		kmMat1<T> qn2(&qn(1), m);
		kmMat1<T> qn3(&qn(m), m);

		kmMat2<T> Q1 = Toeplitz(qn1);
		kmMat2<T> Q2 = Hankel  (qn2, qn3);

		// calc a(m)
		kmMat1<T> b = DivideLeft(Q1 + Q2, bk);

		// set kernel this(2m, 1)
		for(int64 i = 0; i < m; ++i)
		{
			a(m - 1 - i) = a(m + i) = b(i);
		}
	}
	return a;
}

// get FIR filter using firls... low pass filter
// w   : weight of stop bands (the weight of pass band is 1)
// wup : upper pass normalized freq.
// wus : upper stop normalized freq.
template<typename T = float> 
kmMat1<T> FirlsLpf(int n_tap, float wup, float wus, float w = 1.f)
{
	return FirlsBpf<T>(n_tap, 0, 0, wup, wus, w);
}

// get FIR filter using firls... high pass filter
// w   : weight of stop bands (the weight of pass band is 1)
// wls : lower stop normalized freq.
// wlp : lower pass normalized freq.
template<typename T = float> 
kmMat1<T> FirlsHpf(int n_tap, float wls, float wlp, float w = 1.f)
{
	return FirlsBpf<T>(n_tap, wls, wlp, 1.f, 1.f, w);
}

// get gauss filter
template<typename T = float>
kmMat1<T> GaussKernel(int64 n_tap, float sigma = 0)
{
	ASSERTFA(n_tap > 0, "GaussKernel in 4341");

	// create output
	kmMat1<T> a(n_tap);

	// init sigma
	if(sigma == 0) sigma = float(n_tap)/(4.f*sqrt(2.f*log(2.f)));

	// init parameters
	const float size_h = float(n_tap - 1)/2.f;
	const float coeff  = 1.f/ sigma;
	
	// set Gaussian kernel.. gk(i) = exp(-0.5*((i - (N-1)/2)/sigma)^2)	
	T sum = 0;

	for(int64 i = 0; i < n_tap; ++i)
	{		
		const T exp_in = (i - size_h)*coeff;
		const T val    = exp(T(-0.5)*exp_in*exp_in);

		a(i) = val;	sum += val;
	}

	// normalize kernel... sum of kernel is one.
	return a/=sum;
}

// Gaussian window like Matlab :w = gausswin(L, alpha)
// alpha: alpha is defined as the reciprocal of the standard deviation
template<typename T = float> 
kmMat1<T> GaussWin(int64 n, float alpha = 2.5f)
{
	ASSERTFA(n > 1, "GaussWin in 4392");

	// create output
	kmMat1<T> a(n);

	// init parameters
	const float size_h = (n - 1)/2.f;

	// set Gaussian kernel.. gk(i) = exp(-0.5*((i - (N-1)/2)/sigma)^2)
	for(int64 i = 0; i < n; ++i)
	{		
		const T exp_in = alpha*(i - size_h)/size_h;

		a(i) = exp(T(-0.5)*exp_in*exp_in);
	}
	return a;
}

// get hilbert kernel
template<typename T = float>
kmMat1<T> HtKernel(int n_taph)
{
	ASSERTFA(n_taph > 3, "HtKernel in 4372");

	// create output
	kmMat1<T> a(n_taph*2 + 1); a.SetZero();

	// init parameters
	const int n       = n_taph*2;
	const int n_2     = n + 2;
	const T   inv_n_2 = T(1.)/T(n_2);
	const T   coeff   = T(2.)*inv_n_2;	
	      T*  ia      = a.P() + n_taph;
	
	// set Hilbert kernel
	// :  hh = 2/(N+2) * cot(pi*(1:Nh)'/(N+2))
	for(int64 i = 1; i <= n_taph; i+=2 )
	{
		const T h = coeff/tan(PIf*i*inv_n_2);

		*(ia - i) = -h;
		*(ia + i) =  h;
	}
	return a;
}

// get compressed Hilbert transform kernel
template<typename T = float> 
kmMat1<T> HtKernelCp(int n_taph)
{
	ASSERTFA(n_taph > 3, "HtKernel in 4401");

	// create output
	kmMat1<T> a(n_taph/2);

	// init parameters
	const int n       = n_taph*2;
	const int n_2     = n + 2;
	const T   inv_n_2 = T(1.)/T(n_2);
	const T   coeff   = T(2.)*inv_n_2;
	      T*  ia      = a.P();
	
	// set Hilbert kernel
	// :  hh = 2/(N+2) * cot(pi*(1:Nh)'/(N+2))
	for(int i = 1; i <= n_taph; i+=2)
	{
		const T h = coeff / tan(PIf*i*inv_n_2);
				
		*(ia++) =  h;
	}
	return a;
}

//////////////////////////////////////////////////////
// hash class

// chaining type
//   T : datat type, K : key type
template<typename T, typename K, int hash_n, int node_n_max> 
class kmHashC
{
	class kmNode
	{
	public:
		T*      data = nullptr;
		kmNode* prv  = nullptr;
		kmNode* nxt  = nullptr;
	};
	kmNode _table[hash_n];
	kmNode _node [node_n_max];
	int    _node_n = 0;

public:
	static const int _hash_n     = hash_n;
	static const int _node_n_max = node_n_max;
	
	// add data's address to hash table
	void Add(T* data)
	{
		const K key  = GetKey(data);		
		kmNode* node = &_table[GetHashCode(key)];

		if(node->data == nullptr) node->data = data;
		else
		{
			while(node->nxt != nullptr) node = node->nxt;

			node->nxt = &_node[_node_n++];
			node->nxt->prv  = node;
			node->nxt->data = data;
		}
	};	

	T* Find(K key) // if not found. it will return nullptr
	{
		kmNode* node = &_table[GetHashCode(key)];

		if(node->data == nullptr) return nullptr;

		while(!IsOpen(node->data, key))
		{
			if(node->nxt == nullptr) return nullptr; // cannot find
			node = node->nxt;
		}
		return node->data;
	};

	int Erase(K key)
	{
		kmNode* node = &_table[GetHashCode(key)];

		while(!IsOpen(node->data, key))
		{
			if(node->nxt == nullptr) return -1; // cannot find
			node = node->nxt;
		}		
		if(node->nxt == nullptr) node->prv->nxt = nullptr;
		else 
		{
			node->prv->nxt = node->nxt;
			node->nxt->prv = node->prv;
		}
		return int(node - _node)/sizeof(kmNode); // index of erase node
	};

	////////////////////////////////////////
	// pure virtual functions
	virtual K    GetKey     (T* data       ) = 0;
	virtual int  GetHashCode(         K key) = 0;
	virtual bool IsOpen     (T* data, K key) = 0;
};

// indexing type
//   T : datat type, K : key type
template<typename T, typename K, int hash_n, int node_n_max>
class kmHashI
{
	int _table[hash_n][node_n_max] = {0,}; // it will save idx as idx + 1
	
public:
	static const int _hash_n     = hash_n;
	static const int _node_n_max = node_n_max;
	
	// init or clear talbe
	void Init()
	{
		const int n   = _hash_n*_node_n_max;
		int*      ptr = (int*)_table;
		for(int i = 0; i < n; ++i) *(ptr + i) = 0;
	};

	// add data's index to hash table
	void Add(T* data, int idx)
	{
		const K key = GetKey(data);

		int* node = (int*)_table[GetHashCode(key)];

		if(*node == 0) *node = idx + 1;
		else
		{
			while(*node != 0) ++node;
			*node = idx + 1;
		}
	};	

	// find data from key... if not found, it'll return nullptr
	T* Find(T* data0, K key)
	{
		int* node = (int*)_table[GetHashCode(key)];

		if(*node == 0) return nullptr;

		while(!IsOpen(data0 + *node - 1, key))
		{
			++node;	if(*node == 0) return nullptr; // cannot find
		}
		return data0 + *node - 1;
	};

	// erase data of the key from hash table.. if not found, it'll return -1
	int Erase(T* data0, K key)
	{
		int* node = (int*)_table[GetHashCode(key)];

		if(*node == 0) return -1;
		
		// find data
		while(!IsOpen(data0 + *node - 1, key))
		{
			++node; if(*node == 0) return -1;
		}
		const int erased_idx = *node - 1;

		// erase 
		while(*(node + 1) != 0)
		{
			*node = *(node + 1); ++node;
		}
		*node = 0;

		return erased_idx;
	};

	////////////////////////////////////////
	// pure virtual functions
	virtual K    GetKey     (T* data       ) = 0;
	virtual int  GetHashCode(         K key) = 0;
	virtual bool IsOpen     (T* data, K key) = 0;
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// media file info for kmMat

// media file type... jpg, png, bmp, gif, mp4, mpg, avi, unknown
enum class kmMdfType { jpg, png, bmp, gif, mp4, mpg, avi, unknown };

// orientation... 1 : upper left, 3: lower right, 6: upper right, 8: lower left
enum class kmMdfOrnt : ushort { none = 0, upperleft = 1, lowerright = 3, upperright = 6, lowerleft = 8};

// tiff tag (image file directory) structure... 12byte
//   fmt 1: uchar, 2: ascii, 3: ushort, 4: uint,  5: urational = uint0/uint1,
//       6: char,            8:  short, 9:  int, 10:  rational =  int0/ int1
//       11: float, 12: double
class kmTiffTag
{
private: 
	static int _endian; // 0 : little endian, 1 : big endian

public:
	ushort id, fmt; uint cnt; int ofs;

	// set endian
	static void SetEndian(int endian) { _endian = endian; };
	static void SetLitteEndian()      { SetEndian(0);     };
	static void SetBigEndian  ()      { SetEndian(1);     };

	// get endian... 0 : little endian, 1 : big endian
	static int  GetEndian()           { return _endian;   }; 

	// read tag depending _endian
	void Read(kmFile& fobj, int64 pos0)
	{
		if(_endian == 0) ReadJust(fobj, pos0);
		else             ReadSwap(fobj, pos0);
	};

	// read tag
	void ReadJust(kmFile& fobj, int64 pos0)
	{
		fobj >> id >> fmt >> cnt;
		
		const int byte = GetTypeByte();

		if(byte*cnt > 4) (fobj >> ofs).Seek(pos0 + ofs);
		else             ofs = 0;
	};

	// read tag with swapping endian
	void ReadSwap(kmFile& fobj, int64 pos0)
	{
		fobj.ReadSwap(&id);
		fobj.ReadSwap(&fmt);
		fobj.ReadSwap(&cnt);

		const int byte = GetTypeByte();

		if(byte*cnt > 4) { fobj.ReadSwap(&ofs); fobj.Seek(pos0 + ofs); }
		else             ofs = 0;
	};

	// read data depending _endian
	void ReadData(kmFile& fobj, void* buf)
	{
		if(_endian == 0) ReadDataJust(fobj, buf);
		else             ReadDataSwap(fobj, buf);
	};

	// read data without swapping endian
	//    buf size must be equal to obj_n x sizeof(fmt's type)
	void ReadDataJust(kmFile& fobj, void* buf)
	{
		switch(fmt)
		{
		case  1: case  6: case  2: fobj.Read(( uchar*) buf, cnt  ); break;
		case  3: case  8:          fobj.Read((ushort*) buf, cnt  ); break;
		case  4: case  9:          fobj.Read((  uint*) buf, cnt  ); break;
		case  5: case 10:          fobj.Read((  uint*) buf, cnt*2); break;
		case 11:                   fobj.Read(( float*) buf, cnt  ); break;
		case 12:                   fobj.Read((double*) buf, cnt  ); break;
		}
	};

	// read data with swapping endian
	//    buf size must be equal to obj_n x sizeof(fmt's type)
	void ReadDataSwap(kmFile& fobj, void* buf)
	{
		switch(fmt)
		{
		case  1: case  6: case  2: fobj.ReadSwap(( uchar*) buf, cnt  ); break;
		case  3: case  8:          fobj.ReadSwap((ushort*) buf, cnt  ); break;
		case  4: case  9:          fobj.ReadSwap((  uint*) buf, cnt  ); break;
		case  5: case 10:          fobj.ReadSwap((  uint*) buf, cnt*2); break;
		case 11:                   fobj.ReadSwap(( float*) buf, cnt  ); break;
		case 12:                   fobj.ReadSwap((double*) buf, cnt  ); break;
		}
	};

	int GetTypeByte()
	{
		switch(fmt)
		{
		case  1: case  6: case  2: return 1;
		case  3: case  8:          return 2;
		case  4: case  9: case 11: return 4;
		case  5: case 10: case 12: return 8;
		}
		return 0;
	};
};
//int kmTiffTag::_endian = 0;

// media file class
class kmMdf
{
protected:
	static ULONG_PTR           _gdiplus_token;
	static GdiplusStartupInput _gdiplus_startupinput;

public:
	kmMdfType _type{};
	kmDate    _date{};
	kmGps     _gps{};
	kmMdfOrnt _ornt{};

	// constructor
	kmMdf() {};	
	kmMdf(kmFile& fobj) { Init(fobj); };
	kmMdf(const wchar* fname) { kmFile fobj(fname); Init(fobj); }; 
	//kmMdf(const  char* fname) { kmFile fobj(fname); Init(fobj); }; 
	
	// init from file
	int Init(kmFile& fobj)
	{
		// get type
		_type = GetType(fobj); 

		// read header info
		switch(_type)
		{
		case kmMdfType::jpg : ReadJpg(fobj); return 1;
		case kmMdfType::mp4 : ReadMp4(fobj); return 1;
		}
		return 0;
	};

	// print info
	void Print()
	{
		print("* mdf type :");
		switch(_type)
		{
		case kmMdfType::jpg : print("jpg"); break;
		case kmMdfType::png : print("png"); break;
		case kmMdfType::bmp : print("bmp"); break;
		case kmMdfType::gif : print("gif"); break;
		case kmMdfType::mp4 : print("mp4"); break;
		case kmMdfType::mpg : print("mpg"); break;
		case kmMdfType::avi : print("avi"); break;
		case kmMdfType::unknown : print("unknown");
		}
		print("\n");

		printw(L"* date : %s\n", _date.GetStrwPt().P());

		print("* orientation :");
		switch(_ornt)
		{
		case kmMdfOrnt::none       : print("none"       ); break;
		case kmMdfOrnt::upperleft  : print("upper left" ); break;
		case kmMdfOrnt::lowerright : print("lower right"); break;		
		case kmMdfOrnt::upperright : print("upper right"); break;
		case kmMdfOrnt::lowerleft  : print("lower left" ); break;
		}
		print("\n");

		_gps.Print();
	};

	//////////////////////////////////////////////////
	// static functions

	// get mdf type from file
	static kmMdfType GetType(kmFile& fobj)
	{
		// read firt 8 bytes
		uchar buf[8];

		fobj.Seek().Read(buf, 8);

		if     (IsJpg(buf)) return kmMdfType::jpg;
		else if(IsBmp(buf)) return kmMdfType::bmp;
		else if(IsPng(buf)) return kmMdfType::png;
		else if(IsMp4(buf)) return kmMdfType::mp4;

		return kmMdfType::unknown;
	};

	static bool IsJpg(uchar* hd) { return hd[0] == 0xFF && hd[1] == 0xD8; };
	static bool IsBmp(uchar* hd) { return hd[0] == 0x42 && hd[1] == 0x4D; };
	static bool IsPng(uchar* hd)
	{
		const uchar hd0[] = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };

		for(int i = 0; i < sizeof(hd0); ++i) if(hd[i] != hd0[i]) return false;

		return true;
	};
	static bool IsMp4(uchar* hd)
	{
		const uchar hd0[] = "ftyp"; //{ 0x66, 0x74, 0x79, 0x70 };
		
		for(int i = 0; i < 4; ++i) if(hd[i + 4] != hd0[i]) return false;

		return true;
	};

	// start gdiplus
	static void Startup() { ::GdiplusStartup(&_gdiplus_token, &_gdiplus_startupinput, NULL); };

	// shutdown gdiplus
	static void Shutdown() { ::GdiplusShutdown(_gdiplus_token); };

	// get clsid of gdiplus
	static CLSID GetClsid(const wchar* fmt)
	{
		uint num = 0, size = 0;

		GetImageEncodersSize(&num, &size);

		kmMat1u8 buf(size);

		ImageCodecInfo* pici = (ImageCodecInfo*)buf.P();

		GetImageEncoders(num, size, pici);

		for(uint i = 0 ; i < num; ++i)
		{
			if(wcscmp(pici[i].MimeType, fmt) == 0) return pici[i].Clsid;
		}
		return CLSID();
	};

	//////////////////////////////////////////////////
	// gdiplus functions 
	
	// get thumbnail image from file
	// * Note that you must delete the returned image pointer after using it.
	static Image* GetThumbImg(const kmStrw& path, int w_pix = 256, int h_pix = 256)
	{
		// check if file exists
		if(!kmFile::Exist(path.P()))
		{
			print("* [kmMdf::GetThumbImg] %s does not exist!\n", path.cu().P());
			return nullptr;
		}
		// read bitmap image
		Bitmap* pimg0 = (Bitmap*)Image::FromFile(path.P());

		// get orientation property.. itm, ornt
		PropertyItem itm{}; ushort ornt = 0;
		{
			uint byte = 0, n = 0; pimg0->GetPropertySize(&byte, &n);

			PropertyItem* pbuf = (PropertyItem*) malloc(byte);

			pimg0->GetAllPropertyItems(byte, n, pbuf);

			for(uint i = 0; i < n; ++i) if(pbuf[i].id == 0x0112)
			{
				ornt      = *(ushort*)pbuf[i].value;
				itm       =           pbuf[i]; 
				itm.value = (void*)&ornt;
				break;
			}
			free(pbuf);
		}

		// make thumbnail image
		int w0 = pimg0->GetWidth (), w1 = w0;
		int h0 = pimg0->GetHeight(), h1 = h0;
		
		if(w0*h_pix < h0*w_pix) h1 = w0*h_pix/w_pix;
		else                    w1 = h0*w_pix/h_pix;

		Image* pimg1 = pimg0->Clone((w0-w1)/2,(h0-h1)/2,w1,h1,pimg0->GetPixelFormat());
		Image* thumb = pimg1->GetThumbnailImage(w_pix,h_pix);

		// set property
		if(itm.id == 0x0112) thumb->SetPropertyItem(&itm);
				
		delete pimg0; delete pimg1;

		return thumb;
	};

	// save image as jpg
	static void SaveImgJpg(Image* img, const kmStrw& path)
	{
		CLSID clsid = GetClsid(L"image/jpeg"); img->Save(path.P(), &clsid);
	};

	// save image as Png
	static void SaveImgPng(Image* img, const kmStrw& path)
	{
		CLSID clsid = GetClsid(L"image/png"); img->Save(path.P(), &clsid);
	};

	// make thumbnail image from jpg or png
	static void MakeThumbImg(const kmStrw& src_path,
		                     const kmStrw& des_path, int w_pix = 256, int h_pix = 256)
	{
		// get thumbnail image
		Image* img = GetThumbImg(src_path, w_pix, h_pix);

		if(img == nullptr) return;

		// save with rotation
		kmStrw ext = des_path.GetEnd(3);

		if(ext == L"png") SaveImgPng(img, des_path);
		else              SaveImgJpg(img, des_path);

		// delete object
		delete img;
	};

	//////////////////////////////////////////////////
	// jpeg functions 

	// read jpg
	void ReadJpg(kmFile& fobj)
	{
		fobj.Seek(2); // pos after header (SOI)
		
		for(uchar mk[2] = {}; 1; mk[0] = 0)
		{
			fobj.Read(mk, 2); // read marker

			int64 eop = ReadEop(fobj); // end of position

			if(mk[0] == 0xff) switch(mk[1])
			{
			case 0xE0 : ReadJpgApp0(fobj); break; // app0
			case 0xE1 : ReadJpgApp1(fobj); break; // app1
			case 0xDB : ReadJpgDqt (fobj); break; // define quantization table
			case 0xC0 : ReadJpgSof0(fobj); break; // start of frame (baseline DCT)
			case 0xC2 : ReadJpgSof2(fobj); break; // start of frame (progressive DCT)
			case 0xC4 : ReadJpgDht (fobj); break; // define huffman table
			case 0xDA : ReadJpgSos (fobj); break; // start of scan
			case 0xFE : ReadJpgCmt (fobj); break; // comment
			default   : break; //print("*** %x\n", mk[1]);
			}
			else break;

			fobj.Seek(eop);
		}
	};

	// read end of position
	int64 ReadEop(kmFile& fobj)
	{
		const int64 pos0 = fobj.GetPos();

		uchar buf[2] = {}; fobj.Read(buf, 2);
		
		const ushort len = ((ushort)buf[0]<<8) | buf[1]; 

		return len + pos0;
	};
	
	// start of scan
	void ReadJpgSos(kmFile& fobj) {};

	// define quantization table
	void ReadJpgDqt(kmFile& fobj) {};

	// define huffman table
	void ReadJpgDht(kmFile& fobj) {};

	// start of frame (baseline DCT)
	void ReadJpgSof0(kmFile& fobj)
	{
		fobj.SeekCur(1);

		ushort h_pix, w_pix;

		fobj.Read(&h_pix); fobj.Read(&w_pix);

		h_pix = ((0x00ff&h_pix)<<8) | ((0xff00&h_pix)>>8);
		w_pix = ((0x00ff&w_pix)<<8) | ((0xff00&w_pix)>>8);

		//print("*** image size (%d, %d)\n", h_pix, w_pix);
	};

	// start of frame (progressive DCT)
	void ReadJpgSof2(kmFile& fobj) {};

	// comment
	void ReadJpgCmt(kmFile& fobj)  {};

	// app0 Jfif
	void ReadJpgApp0(kmFile& fobj) {};

	// app1 exif
	void ReadJpgApp1(kmFile& fobj) { ReadExif(fobj); };

	// read exif
	void ReadExif(kmFile& fobj)
	{
		uchar buf[6] = {}; fobj.Read(buf, 6);

		if(strcmp((char*)&buf[0], "Exif") == 0) // exif
		{
			// read tiff
			ReadTiff(fobj);
		}
	};

	// read tiff
	void ReadTiff(kmFile& fobj)
	{
		int64 pos0 = fobj.GetPos();

		uchar buf[2] = {}; fobj.Read(buf, 2);

		if(buf[0] == 0x49 && buf[1] == 0x49) // little endian
		{
			ushort mark; uint ofs;

			fobj.Read(&mark); // tag mark   ...0x2A00
			fobj.Read(&ofs);  // IFD offset ...0x08000000
			fobj.Seek(pos0 + ofs);

			kmTiffTag::SetLitteEndian(); print("**** little endian\n");

			ReadIfd(fobj, pos0);
		}
		else if(buf[0] == 0x4d && buf[1] == 0x4d) // big endian
		{
			ushort mark; uint ofs;

			fobj.ReadSwap(&mark); // tag mark   ...0x002A  
			fobj.ReadSwap(&ofs);  // IFD offset ...0x00000008
			fobj.Seek(pos0 + ofs);

			kmTiffTag::SetBigEndian(); print("**** big endian\n");

			ReadIfd(fobj, pos0);
		}
		else print("[kmTiffInfo] it is not tiff\n");

		// end of tiff
		fobj.Read(buf,2);

		ushort len = ((ushort)buf[0]<<8) | buf[1];

		fobj.Seek(pos0 + len).Read(buf,2);
	};

	// read ifd 
	void ReadIfd(kmFile& fobj, int64 pos0)
	{	
		ushort entry_n;
		
		if(kmTiffTag::GetEndian() == 0) fobj.Read    (&entry_n);
		else                            fobj.ReadSwap(&entry_n);

		for(int i = 0; i < entry_n; ++i)
		{
			int64 pos_old = fobj.GetPos();

			kmTiffTag tag; tag.Read(fobj, pos0);

			switch(tag.id)
			{
			case 0x0112 : ReadIfdOrnt(fobj, tag);       break;
			case 0x0132 : ReadIfdDate(fobj, tag);       break;
			case 0x8825 : ReadIfdGps (fobj, tag, pos0); break;
			}
			fobj.Seek(pos_old + 12);
		}
	};	

	void ReadIfdOrnt(kmFile& fobj, kmTiffTag& tag) // orientation.. fmt 3 (short)
	{
		// 1 : upper left, 3: lower right, 6: upper right, 8: lower left
		tag.ReadData(fobj, (void*)&_ornt);		
	};	
	void ReadIfdDate(kmFile& fobj, kmTiffTag& tag)
	{
		kmStra date(tag.cnt); tag.ReadData(fobj, date.P()); 

		_date.Set(date);
	};
	// read ifd gps for little endian
	void ReadIfdGps(kmFile& fobj,  kmTiffTag& tag0, int64 pos0)
	{
		uint ofs; tag0.ReadData(fobj, &ofs); fobj.Seek(pos0 + ofs);

		ushort entry_n;

		if(kmTiffTag::GetEndian() == 0) fobj.Read    (&entry_n);
		else                            fobj.ReadSwap(&entry_n);

		for(int k = 0; k < entry_n; ++k)
		{
			int64 pos_old = fobj.GetPos();

			kmTiffTag tag; tag.Read(fobj, pos0);

			switch(tag.id)
			{			
			case 0x0001: ReadIfdGpsLatRef(fobj, tag); break;
			case 0x0002: ReadIfdGpsLat   (fobj, tag); break;
			case 0x0003: ReadIfdGpsLngRef(fobj, tag); break;
			case 0x0004: ReadIfdGpsLng   (fobj, tag); break;
			case 0x0005: ReadIfdGpsAltRef(fobj, tag); break;
			case 0x0006: ReadIfdGpsAlt   (fobj, tag); break;
			case 0x0007: ReadIfdGpsTime  (fobj, tag); break;
			case 0x001d: ReadIfdGpsDate  (fobj, tag); break;
			}
			fobj.Seek(pos_old + 12);
		}
	};
	
	void ReadIfdGpsLatRef(kmFile& fobj, kmTiffTag& tag)
	{
		tag.ReadData(fobj, (void*)&_gps._lat_ref);

		if(_gps._lat_ref != 'S') _gps._lat_ref = 'N';
	};

	void ReadIfdGpsLat(kmFile& fobj, kmTiffTag& tag)
	{
		uint rat[6]; tag.ReadData(fobj, rat);

		double deg = (rat[1] == 0) ? 0 : rat[0]/double(rat[1]);
		double min = (rat[3] == 0) ? 0 : rat[2]/double(rat[3]);
		double sec = (rat[5] == 0) ? 0 : rat[4]/double(rat[5]);

		_gps._lat_deg = deg + min/60. + sec/3600.;
	};
	
	void ReadIfdGpsLngRef(kmFile& fobj, kmTiffTag& tag)
	{
		tag.ReadData(fobj, (void*)&_gps._lng_ref);

		if(_gps._lng_ref != 'W') _gps._lng_ref = 'E';
	};

	void ReadIfdGpsLng(kmFile& fobj, kmTiffTag& tag)
	{
		uint rat[6]; tag.ReadData(fobj, rat);

		double deg = (rat[1] == 0) ? 0 : rat[0]/double(rat[1]);
		double min = (rat[3] == 0) ? 0 : rat[2]/double(rat[3]);
		double sec = (rat[5] == 0) ? 0 : rat[4]/double(rat[5]);

		_gps._lng_deg = deg + min/60. + sec/3600.;
	};
	
	void ReadIfdGpsAltRef(kmFile& fobj, kmTiffTag& ifd) {};
	
	void ReadIfdGpsAlt(kmFile& fobj, kmTiffTag& tag)
	{
		uint rat[2]; tag.ReadData(fobj, rat); 

		_gps._alt_m = (rat[1] == 0) ? 0 : float(rat[0]/double(rat[1]));
	};
	
	void ReadIfdGpsDate(kmFile& fobj, kmTiffTag& tag) {};
	void ReadIfdGpsTime(kmFile& fobj, kmTiffTag& tag) {};

	//////////////////////////////////////////////////
	// mp4 functions 

	// read mp4
	void ReadMp4(kmFile& fobj)
	{	
		fobj.Seek(0); int byte;
		
		while(fobj.ReadSwap(&byte) > 0)
		{
			if(byte == 1) byte = 8;

			// read type
			char type[5] = {}; fobj.Read(type, 4);

			print("** [%s] %d bytes\n", type, byte);

			int64 endpos = fobj.GetPos() + byte - 8;

			if     (strcmp(type, "ftyp") == 0) ReadMp4Ftyp(fobj, endpos);
			else if(strcmp(type, "moov") == 0) ReadMp4Moov(fobj, endpos);
			else if(strcmp(type, "mdat") == 0) ReadMp4Mdat(fobj, endpos);

			// move to end point
			fobj.Seek(endpos);
		}
	};

	// read ftyp
	void ReadMp4Ftyp(kmFile& fobj, int64 endpos) {};

	// read moov
	void ReadMp4Moov(kmFile& fobj, int64 endpos0)
	{
		int byte;

		while(fobj.ReadSwap(&byte) > 0 && fobj.GetPos() < endpos0)
		{	
			// read type
			char type[5] = {}; fobj.Read(type, 4);

			print("**** moov [%s] %d bytes\n", type, byte);

			int64 endpos = fobj.GetPos() + byte - 8;

			if     (strcmp(type, "mvhd") == 0) {}//ReadMp4MoovUdta(fobj, endpos);
			else if(strcmp(type, "trak") == 0) {}//ReadMp4MoovUdta(fobj, endpos);
			else if(strcmp(type, "udta") == 0) ReadMp4MoovUdta(fobj, endpos);

			// move to end point
			fobj.Seek(endpos);
		}
	};

	// read moov-udta
	void ReadMp4MoovUdta(kmFile& fobj, int64 endpos0)
	{
		int byte;

		while(fobj.ReadSwap(&byte) > 0 && fobj.GetPos() < endpos0)
		{	
			// read type
			char type[5] = {}; fobj.Read(type, 4);

			print("**** moov-udta [%s] %d bytes\n", type, byte);

			int64 endpos = fobj.GetPos() + byte - 8;

			if(strcmp(type, "meta") == 0) ReadMp4MoovUdtaMeta(fobj, endpos);

			// move to end point
			fobj.Seek(endpos);
		}
	};

	// read moov-udta-meta
	void ReadMp4MoovUdtaMeta(kmFile& fobj, int64 endpos0)
	{
		int byte; fobj.SeekCur(4);

		if(fobj.ReadSwap(&byte) > 0 && fobj.GetPos() < endpos0)
		{	
			// read type
			char type[5] = {}; fobj.Read(type, 4);

			print("**** moov-udta-meta [%s] %d bytes\n", type, byte);

			int64 endpos = fobj.GetPos() + byte - 8;

			// move to end point
			fobj.Seek(endpos);			
		}
	};

	// read mdat
	void ReadMp4Mdat(kmFile& fobj, int64& endpos)
	{
		int byte; 
		
		fobj.SeekCur(4).ReadSwap(&byte);

		endpos = fobj.GetPos() + byte - 8;
	};	
};
//ULONG_PTR           kmMdf::_gdiplus_token        = 0;
//GdiplusStartupInput kmMdf::_gdiplus_startupinput = 0;

#endif /* __km7Mat_H_INCLUDED_2018_11_12__ */
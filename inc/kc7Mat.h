#ifndef __kc7Mat_H_INCLUDED_2019_09_10__
#define __kc7Mat_H_INCLUDED_2019_09_10__

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

// base header
#include "km7Mat.h"

// cuda header
#include "cuda_runtime.h"
#include "cufft.h"
#include "cuda_fp16.h"
#include "nvml.h"

#define KC7MAT

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// enum for kcMat

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// definition for kcMat

#define KC_CHECK_TIME_START {cudaEvent_t start, stop;\
							 cudaEventCreate(&start);\
							 cudaEventCreate(&stop);\
							 cudaEventRecord(start);

#define KC_CHECK_TIME_END(A) cudaEventRecord(stop); \
							 cudaEventSynchronize(stop); \
							 float cuda_time_msec=0;\
							 cudaEventElapsedTime(&cuda_time_msec, start, stop); \
							 PRINTFA("* CUDA TIME CHECK (%s): %.3fmsec\n", \
							 A, cuda_time_msec);}

#define KC_CHECK_ERROR(A)	{cudaDeviceSynchronize(); \
							 cudaError_t cuda_err = cudaGetLastError(); \
							 if(cuda_err != cudaSuccess) \
							 {PRINTFA("* CUDA ERROR CHECK (%s): %d (%s)\n", \
							 A, cuda_err, cudaGetErrorString(cuda_err)); \
							 throw KE_CUDA_ERROR;}}

#define KC_CHECK_ERROR_FFT(A) {if(_res != CUFFT_SUCCESS) \
							   {PRINTFA("* CUFFT ERROR CHECK (%s): %d\n",A, _res); \
								throw KE_CUFFT_ERROR;}}

#define KC_CHECK_ERROR_NVML(A) {if(_ret != NVML_SUCCESS) \
							    {PRINTFA("* NVML ERROR CHECK (%s) : %s\n", \
							     A, nvmlErrorString(_ret));\
								 throw KE_NVML_ERROR;}}

#define KC_PRINTF_MEM(A) {size_t a_byte, b_byte; cudaMemGetInfo(&a_byte, &b_byte); \
	                      PRINTFA("* %s:(gpu used/total) %lld/%lld MB\n",A, (b_byte - a_byte)>>20, b_byte>>20);}

#define PRINTF_DIM(BK,GD) PRINTFA("bk(%d,%d,%d) / gd(%d,%d,%d)\n",BK.x,BK.y,BK.z,GD.x,GD.y,GD.z)

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class for kcMat

///////////////////////////////////////////////////////
// cuda matrix base class
template<typename T> class kcArr
{
protected:
	T*      _p     = 0;   // pointer of data	
	kmstate _state = 0;   // state of memory allocation
	int64   _size  = 0;   // size of allocated memory (number of data)

	/////////////////////////////////////////////////
	// basic member functions	
public:	
	// * Note that _state sholud be initialized seperately.
	virtual void Init() { _p = 0;  _size = 0; };

	// construtor
	kcArr() {};
	kcArr(      int64 size) { Create(size);};
	kcArr(T* p, int64 size) { Set(p, size);};

	// destructor
	virtual ~kcArr()	{ Release(); };

	// copy constructor
	kcArr(const kcArr&    a) { Create(a.Size()); Copy(a); };
	kcArr(const kmArr<T>& a) { Create(a.Size()); Copy(a); };

	template<typename Y>
	kcArr(const kcArr<Y>& b) { Create(b.Size()); TCast(*this, b); }

	// move constructor
	kcArr(kcArr&& b) { Move(b); };

	// assignment operator
	kcArr& operator=(const kcArr&    b) { RecreateIf(b); Copy(b._p); return *this; };
	kcArr& operator=(const kmArr<T>& b) { RecreateIf(b); Copy(b);    return *this; };

	template<typename Y>
	kcArr& operator=(const kcArr<Y>& b) { RecreateIf(b); return TCast(*this, b); }

	// move assignment operator
	kcArr& operator=(kcArr&& b) { Move(b); return *this; };	

	// allocate memory... core
	void Create(int64 size)
	{
		ASSERTA(!IsCreated(), "[kcArr::Create in 149] memory has already been created");
		ASSERTA(!IsPinned (), "[kcArr::Create in 150] memory is pinned");

		if(size == 0) { _size = 0; _state = 0; return; }
		
		cudaMalloc((void**)&_p, size*sizeof(T));

		ASSERTA(_p != 0, "[kcArr::Create in 140] %lld byte",size*sizeof(T));

		_size  = size;
		_state = 1;
	};

	// expand memory... core
	// * Note that expand must be call in case that kcArr was created not set.
	void Expand(int64 size)
	{	
		ASSERTA(IsCreated(), "[kcArr::Expand in 162] memory is not created");
		ASSERTA(!IsPinned(), "[kcArr::Expand in 163] memory is pinned");

		const int64 size_new = size + _size;

		T* p; cudaMalloc((void**)&p, size_new*sizeof(T));

		ASSERTA(_p != 0, "[kcArr::Expand in 156]");

		CopyTo(p);
				
		cudaFree(_p);
		
		_size  = size_new;
		_p     = p;
		_state = 1;
	};

	// release memory... core
	void Release()
	{		
		if(IsCreated() && _p) cudaFree(_p);
		_state.is_created = 0;
		Init();
	};

	// set array... core
	void Set(T* p, int64 size)
	{
		ASSERTA(!IsCreated(), "[kcArr::Set in 193] memory has already been created");

		_size = size;
		_p    = p;
	};

	// * Note that if we know the size of the target,
	// * the size of the transfered data is always the size of the target.
	//
	// copy from (a = b), GPU to GPU... core
	void Copy(const T* b, cudaStream_t stream = 0)
	{
		ASSERTA(_p != 0, "[kcArr::Copy in 186]");

		cudaMemcpyAsync((void*) _p, (void*) b, Byte(), cudaMemcpyDeviceToDevice, stream);
	};

	// copy from (a = b), GPU to GPU... core
	void Copy(const kcArr& b, cudaStream_t stream = 0)
	{
		ASSERTA(b.Size() >= Size(), "[kcArr::Copy in 194]");

		cudaMemcpyAsync((void*) _p, (void*) b.P(), Byte(), cudaMemcpyDeviceToDevice, stream);
	};

	// copy to (b = a), GPU to GPU...  core
	void CopyTo(T* b, cudaStream_t stream = 0) const
	{
		ASSERTA(_p != 0, "[kcArr::CopyTo in 195]");

		cudaMemcpyAsync((void*) b, (void*) _p, Byte(), cudaMemcpyDeviceToDevice, stream);
	};

	// copy to (b = a), GPU to GPU... core
	void CopyTo(kcArr& b, cudaStream_t stream = 0) const
	{
		ASSERTA(Size() >= b.Size(), "[kcArr::CopyTo in 210]");

		cudaMemcpyAsync((void*) b.P(), (void*) _p, b.Byte(), cudaMemcpyDeviceToDevice, stream);
	};

	// copy from host, CPU to GPU... core
	void CopyFromHost(T* phost, cudaStream_t stream = 0)
	{
		ASSERTA(_p != 0, "[kcArr::CopyFromHost in 218]");

		cudaMemcpyAsync((void*) _p, (void*) phost, Byte(), cudaMemcpyHostToDevice, stream);
	};

	// copy from host, CPU to GPU... core
	void Copy(const kmArr<T>& b, cudaStream_t stream = 0)
	{
		ASSERTA(b.Size() >= Size(), "[kcArr::CopyFrom in 226]");

		cudaMemcpyAsync((void*) _p, (void*) b.P(), Byte(), cudaMemcpyHostToDevice, stream);
	};	

	// copy from host, CPU to GPU
	template<typename Y>
	void Copy(const kmArr<Y>& b, cudaStream_t stream = 0)
	{
		TCast(*this, kcArr<Y>(b), stream);
	}

	// copy to host, GPU to CPU...core
	void CopyToHost(T* phost, cudaStream_t stream = 0) const
	{
		ASSERTA(_p != 0, "[kcArr::CopyToHost in 226]");

		cudaMemcpyAsync((void*) phost, (void*) _p, Byte(), cudaMemcpyDeviceToHost, stream);
	};

	// copy to host, GPU to CPU... core
	void CopyTo(kmArr<T>& b, cudaStream_t stream = 0) const
	{
		ASSERTA(Size() >= b.Size(), "[kcArr::CopyTo in 234]");

		cudaMemcpyAsync((void*) b.P(), (void*) _p, b.Byte(), cudaMemcpyDeviceToHost, stream);
	};

	// copy to host, GPU to CPU as other type
	template<typename Y>
	void CopyTo(kmArr<Y>& b, cudaStream_t stream = 0) const
	{
		kcArr<Y>(*this).CopyTo(b, stream);
	}

	// release and create
	void Recreate(int64 size) { Release();	Create(size); };

	template<typename Y> void Recreate(const kcArr<Y>& b) { Recreate(b.N()); }
	template<typename Y> void Recreate(const kmArr<Y>& b) { Recreate(b.N()); }

	// recreate if
	int RecreateIf(int64 size) { if(size != _size) { Recreate(size); return 1; } return 0; };

	template<typename Y> int RecreateIf(const kcArr<Y>& b) { return RecreateIf(b.Size()); }
	template<typename Y> int RecreateIf(const kmArr<Y>& b) { return RecreateIf(b.Size()); }
		
	// move (b --> a)... core
	void Move(kcArr& b)
	{
		ASSERTA(!IsPinned(), "[kcArr::Move in 280] memory is pinned");

		if(IsCreated()) Release();

		_p     = b._p;
		_size  = b._size;
		_state = b._state; b._state = 0;
	};

	// set array
	void Set(const kcArr& a)   { Set(a._p, a._size); };

	// operator to get data
	T* P(int64 i1 = 0) const
	{
		ASSERTA(i1 < _size, "[kcArr::P in 216] %lld < %lld", i1, _size);

		return (_p + i1);
	};

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
	kcArr& Restore() { RestoreVfptr<kcArr<T>>(); _state = 0; Init(); return *this; };

	/////////////////////////////////////////////////
	// operator functions

	// conversion operator... kmArr b = (kmArr) a
	operator kmArr<T>() const
	{	
		kmArr<T> b(_size); CopyTo(b);
		return b;
	};

	/////////////////////////////////////////////////
	// general member functions

	// pin of unpin memory
	void PinMemory  () { _state.is_pinned = 1; };
	void UnpinMemory() { _state.is_pinned = 0; };

	// get info	
	int64 Size()  const { return _size;           };
	int64 Byte()  const { return _size*sizeof(T); };
	int64 State() const { return _state;          };

	// get dimension
	static int64 GetDim() { return 0; };

	const type_info& GetType() const { return typeid(T); };

	// get state
	bool IsCreated() const { return _state.is_created == 1; };
	bool IsPinned () const { return _state.is_pinned  == 1; };

	virtual bool IsNoBlank() const { return true; };

	// get the number of real elements
	virtual int64 N() const { return _size; };

	// compare
	template<typename Y> bool IsEqualSize   (const kcArr<Y>& b) const { return _size == b.Size(); }
	template<typename Y> bool IsEqualSize   (const kmArr<Y>& b) const { return _size == b.Size(); }
	template<typename Y> bool IsEqualSizeDim(const kcArr<Y>& b) const { return _size == b.Size(); }
	template<typename Y> bool IsEqualSizeDim(const kmArr<Y>& b) const { return _size == b.Size(); }
	template<typename Y> bool IsEqualN      (const kcArr<Y>& b) const { return N() == b.N(); }
	template<typename Y> bool IsEqualN      (const kmArr<Y>& b) const { return N() == b.N(); }

	// set value
	void SetZero(cudaStream_t stream = 0)
	{
		cudaMemsetAsync(_p, 0, Byte(), stream);
	};

	// display member info
	virtual void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		PRINTFA("  _p     : %p\n"  ,        _p);
		PRINTFA("  _state : %lld\n", (int64)_state);
		PRINTFA("  _size  : %lld\n",        _size);
	};

	// display dimension
	virtual void PrintDim(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		PRINTFA("  dim : %lld\n", _size);
	};

	// display member value
	virtual void  PrintVal(int64 s_idx, int64 e_idx) const
	{
		kmArr<T>(*this).PrintVal(s_idx, e_idx);
	};

	void PrintVal() const {	PrintVal(0, N()-1); };

	///////////////////////////////////////////////
	// functions for kernel

	void GetBkGd(dim3& bk, dim3& gd, const uint nx1 = 256) const 
	{
		bk = {nx1, 1, 1};
		gd = {((int)N()-1)/nx1 + 1, 1, 1};

		CheckBkGd(bk, gd);
	};

	///////////////////////////////////////////////
	// static functions

	static void GetCudaProp(cudaDeviceProp& prop, int id = 0)
	{		
		cudaGetDeviceProperties(&prop, id);		
	};

	static void CheckBkGd(dim3& bk, dim3& gd)
	{		
		ASSERTFA(bk.x <= 1024, "bk.x(%d) is over the limit(%d)", bk.x, 1024);
		ASSERTFA(bk.y <= 1024, "bk.y(%d) is over the limit(%d)", bk.y, 1024);
		ASSERTFA(bk.z <=   64, "bk.z(%d) is over the limit(%d)", bk.z,   64);
		
		ASSERTFA(gd.x <= 2147483647, "gd.x(%d) is over the limit(%d)", gd.x, 2147483647);
		ASSERTFA(gd.y <=      65535, "gd.y(%d) is over the limit(%d)", gd.y,      65535);
		ASSERTFA(gd.z <=      65535, "gd.z(%d) is over the limit(%d)", gd.z,      65535);
	};
};

//////////////////////////////////////////////////////////
// 1D matrix class
template<typename T> class kcMat1 : public kcArr<T>
{
protected:
	// using for members of parents class
	using kcArr<T>::_p, kcArr<T>::_state, kcArr<T>::_size;

	// member variables
	int64 _n1 = 0;    // n of dim1

public:
	// using for functions of parents class
	using kcArr<T>::GetKmClass;
	using kcArr<T>::Release;	
	using kcArr<T>::P;
	using kcArr<T>::CheckBkGd;
	using kcArr<T>::CopyTo;
	using kcArr<T>::Copy;

	/////////////////////////////////////////////////
	// basic member functions
public:	
	virtual void Init() { kcArr<T>::Init(); _n1 = 0; };

	// constructor
	kcMat1() {};
	kcMat1(      int64 n1)             { Create(n1);       };	
	kcMat1(T* p, int64 n1)             { Set(p, n1);       };	
	kcMat1(      int64 n1, int64 size) { Create(n1, size); };
	kcMat1(T* p, int64 n1, int64 size) { Set(p, n1, size); };

	// destructor
	virtual ~kcMat1() {};
		
	// copy constructor	
	kcMat1(const kcMat1<T>& b) { Create(b.N1()); Copy(b.Begin()); };
	kcMat1(const kmMat1<T>& a) { Create(a.N1()); Copy(a); };

	template<typename Y>
	kcMat1(const kcMat1<Y>& b) { Create(b.N1()); TCast(*this, b); }

	// move constructor
	kcMat1(kcMat1&& b) { Move(b); };
		
	// assignment operator
	kcMat1& operator=(const kcMat1&    b) { RecreateIf(b); Copy(b._p); return *this; };
	kcMat1& operator=(const kmMat1<T>& b) { RecreateIf(b); Copy(b);    return *this; };

	template<typename Y>
	kcMat1& operator=(const kcMat1<Y>& b) { RecreateIf(b); return TCast(*this, b);}

	// move assignment operator
	kcMat1& operator=(kcMat1&& b) { Move(b); return *this; };	
		
	// allocate memory
	void Create(int64 n1)	            { Create(n1, n1); };
	void Create(int64 n1, int64 size)
	{
		ASSERTA( n1 <= size, "[kcMat1::Create in 444] %lld <= %lld", n1, size);
		
		_n1 = n1; kcArr<T>::Create(size);
	};

	// release and create
	void Recreate(int64 n1)             { Release(); Create(n1); };
	void Recreate(int64 n1, int64 size) { Release(); Create(n1, size); };

	template<typename Y> void Recreate(const kcMat1<Y>& b) { Recreate(b.N1()); }
	template<typename Y> void Recreate(const kmMat1<Y>& b) { Recreate(b.N1()); }

	// recreate if
	int RecreateIf(int64 n1) { if(n1 != _n1) { Recreate(n1); return 1; } return 0; };

	template<typename Y> int RecreateIf(const kcMat1<Y>& b) { return RecreateIf(b.N1()); }
	template<typename Y> int RecreateIf(const kmMat1<Y>& b) { return RecreateIf(b.N1()); }
	
	// move 
	void Move(kcMat1& b) { kcArr<T>::Move(b); _n1 = b._n1;};

	// set array
	void Set(T* p, int64 n1)            { Set(p, n1, n1); };
	void Set(T* p, int64 n1, int64 size)
	{
		ASSERTA( n1 <= size, "[kcMat1::Set in 195]");

		_n1 = n1; kcArr<T>::Set(p, size);
	};

	// set array... a.Set(b)
	void Set(const kcMat1<T>& b) { Set(b.P(), b.N1(), b.Size()); };
	void Set(const kcArr <T>& b) { Set(b.P(), b.Size());         };

	// operator to get data
	T* End() const { return P(_n1 - 1); };

	// restore the class including vfptr
	// * Note that this will clear and reset _p, _state, _size without release,
	// * which can cause a memory leak. So, you should use this very carefully.
	kcMat1& Restore() { RestoreVfptr(*this); _state = 0; Init(); return *this; };

	/////////////////////////////////////////////////
	// operator functions

	// conversion operator... kmMat1 b = (kmMat1) a
	operator kmMat1<T>() const
	{	
		kmMat1<T> b(_n1); CopyTo(b);
		return b;
	};

	/////////////////////////////////////////////////
	// general member functions

	// get mat1
	kcMat1<T> Mat1(kmI i1) const
	{
		i1.e = MIN(i1.e, _n1-1);

		return kcMat1<T>(P(i1.s), i1.Len());
	};

	// get info
	int64 N1() const { return _n1; };

	// get the number of real elements
	virtual int64 N() const { return _n1; };

	// get dimension
	static int64 GetDim() { return 1; };

	// display member info
	void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		kcArr<T>::PrintInfo();
		PRINTFA("  _n1    : %lld\n", _n1);
	};

	// display dimension
	void PrintDim(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		PRINTFA("  dim : %lld (%lld) \n", _n1, _size);
	};

	// display member value
	void  PrintVal(int64 s_idx, int64 e_idx) const
	{
		kmMat1<T>(*this).PrintVal(s_idx, e_idx);
	};

	// display matrix
	void PrintMat(LPCSTR str = nullptr) const
	{
		kmMat1<T>(*this).PrintMat(str);
	};

	// compare
	template<typename Y> bool IsEqualSizeDim(const kcMat1<Y>& b) const { return _n1 == b.N1(); }
	template<typename Y> bool IsEqualSizeDim(const kmMat1<Y>& b) const { return _n1 == b.N1(); }

	///////////////////////////////////////////////
	// functions for kernel

	void GetBkGd(dim3& bk, dim3& gd, uint nx1 = 128) const	
	{
		bk = {nx1, 1, 1};
		gd = {((int)N()-1)/nx1 + 1, 1, 1};

		CheckBkGd(bk, gd);
	};

	// get bk and gd for reduction 
	void GetBkGdRdc(dim3& bk, dim3& gd, int& sm) const
	{
		bk = MIN(MAX(32, NextPow2((uint)N1())),1024);
		gd = 1; 
		sm = bk.x*sizeof(T);

		CheckBkGd(bk, gd);
	};
};

//////////////////////////////////////////////////////////
// 2D matrix class
template<typename T> class kcMat2 : public kcMat1<T>
{
protected:
	// using for members of parents class
	using kcArr <T>::_p, kcArr<T>::_state, kcArr<T>::_size;
	using kcMat1<T>::_n1;

	// member variables
	int64 _p1 = 0;  // pitch of dim1
	int64 _n2 = 1;  // number of dim2	

public:
	// using for functions of parents class
	using kcArr <T>::GetKmClass;
	using kcArr <T>::Release;
	using kcArr <T>::CheckBkGd;
	using kcArr <T>::Copy;
	using kcMat1<T>::N1;

	/////////////////////////////////////////////////
	// basic member functions
public:
	virtual void Init() { kcMat1<T>::Init(); _p1 = 0; _n2 = 1; };

	// constructor
	kcMat2() {};
	kcMat2(int64 n) { Create(n, 1);};

	kcMat2(      int64 n1, int64 n2) { Create(n1, n2);};
	kcMat2(T* p, int64 n1, int64 n2) { Set(p, n1, n2);};

	kcMat2(      int64 n1, int64 n2, int64 p1) { CreateP(n1, n2, p1);};
	kcMat2(T* p, int64 n1, int64 n2, int64 p1) { SetP(p, n1, n2, p1);};

	kcMat2(      int64 n1, int64 n2, int64 p1, int64 size) { CreateP(n1, n2, p1, size);};
	kcMat2(T* p, int64 n1, int64 n2, int64 p1, int64 size) { SetP(p, n1, n2, p1, size);};

	kcMat2(const kcMat1<T>& a) { Set(a); };

	// destructor
	virtual ~kcMat2() {};
	
	// copy constructor	
	kcMat2(const kcMat2&    b) { Create(b.N1(), b.N2()); CopyFrom(b); };
	kcMat2(const kmMat2<T>& b) { Create(b.N1(), b.N2()); CopyFrom(b); };

	template<typename Y>
	kcMat2(const kcMat2<Y>& b) { Create(b.N1(), b.N2()); TCast(*this, b); }

	// move constructor
	kcMat2(kcMat2&& b) { Move(b); };

	// assignment operator
	kcMat2& operator=(const kcMat2&    b) { RecreateIf(b); return CopyFrom(b); };
	kcMat2& operator=(const kmMat2<T>& b) { RecreateIf(b); return CopyFrom(b); };
		
	template<typename Y>
	kcMat2& operator=(const kcMat2<Y>& b) { RecreateIf(b); return TCast(*this, b); }

	// move assignment operator
	kcMat2& operator=(kcMat2&& b) { Move(b); return *this; };

	// allocate memory
	void Create (int64 n1, int64 n2 = 1)         { CreateP(n1, n2, n1, n1*n2); };
	void Create (int64 n1, int64 n2, int64 size) { CreateP(n1, n2, n1, size ); };
	void CreateP(int64 n1, int64 n2, int64 p1)   { CreateP(n1, n2, p1, p1*n2); };
	void CreateP(int64 n1, int64 n2, int64 p1, int64 size)
	{	
		ASSERTA(p1*n2 <= size, "[kcMat2::CreateP in 620]");

		_n1 = n1; _n2 = n2; _p1 = p1; kcArr<T>::Create(size);
	};

	// release and create
	void Recreate (int64 n1, int64 n2 = 1)         { Release(); Create (n1, n2      ); };
	void Recreate (int64 n1, int64 n2, int64 size) { Release(); Create (n1, n2, size); };
	void RecreateP(int64 n1, int64 n2, int64 p1)   { Release(); CreateP(n1, n2, p1  ); };
	void RecreateP(int64 n1, int64 n2, int64 p1, int64 size)
	{
		Release(); CreateP(n1, n2, p1, size);
	};
	template<typename Y> void Recreate(const kcMat2<Y>& b) { Recreate(b.N1(), b.N2()); }
	template<typename Y> void Recreate(const kmMat2<Y>& b) { Recreate(b.N1(), b.N2()); }

	// recreate if
	int RecreateIf(int64 n1, int64 n2)
	{
		if(n1 != _n1 || n2 != _n2) { Recreate(n1, n2); return 1; } return 0;
	};
	template<typename Y> int RecreateIf(const kcMat2<Y>& b) { return RecreateIf(b.N1(), b.N2()); }
	template<typename Y> int RecreateIf(const kmMat2<Y>& b) { return RecreateIf(b.N1(), b.N2()); }

	// move 
	void Move(kcMat2& b) { kcMat1<T>::Move(b); _n2 = b._n2; _p1 = b._p1; };

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

		ASSERTA(min_size <= size, "[kcMat2::SetP in 645]");

		_n1 = n1; _n2 = n2; _p1 = p1; kcArr<T>::Set(p, size);
	};

	// calculate the minimum size.
	// * Note that (p - n) of the last element is actually not used.
	static int64 CalcMinSize(int64 n1, int64 n2, int64 p1)
	{
		return p1*n2 - (p1 - n1);
	};

	// set array... a.Set(b)
	void Set(const kcMat2<T>& a) { SetP(a.P(), a.N1(), a.N2(), a.P1(), a.Size()); };
	void Set(const kcMat1<T>& a) { Set (a.P(), a.N1(),      1, a.Size()); };
	void Set(const kcArr <T>& a) { Set (a.P(), a.Size(),    1, a.Size()); };
	
	// operator to get data
	T* P(int64 i1, int64 i2) const
	{
		const int64 idx = i1 + _p1*i2;

		ASSERTA(idx <= _size,"[kcMat2::P in 646] idx: %lld (%lld, %lld), _size: %lld", idx, i1, i2, _size);

		return _p + idx;
	};

	T* P(int64 i) const
	{		
		const int64 i2 = i/_n1; i-= i2*_n1;		

		return P(i, i2);
	};

	T* P() const { return _p; };
	
	T* End() const { return P(_n1-1, _n2-1); };

	// restore the class including vfptr
	// * Note that this will clear and reset _p, _state, _size without release,
	// * which can cause a memory leak. So, you should use this very carefully.
	kcMat2& Restore() { RestoreVfptr(*this); _state = 0; Init(); return *this; };

	// reshape
	kcMat2& Reshape(int64 n1, int64 n2, int64 p1 = 0)
	{
		// init arguement
		if(p1 == 0) p1 = n1;

		// check size
		ASSERTA(p1*n2 <= _size, "[kcMat2::Reshape in 669]");

		// reset members
		_n1 = n1; _n2 = n2;	_p1 = p1;

		return *this;
	};

	// copy to device for kcMat2
	kcMat2& CopyTo(kcMat2<T>& b, cudaStream_t stream = 0) const
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat2::CopyTo in 781]");

		cudaMemcpy2DAsync((void*) b.P(), b.P1()*sizeof(T), 
			              (void*)   P(),   P1()*sizeof(T),
				          _n1*sizeof(T), _n2, cudaMemcpyDeviceToDevice, stream);
		return const_cast<kcMat2&>(*this);
	};

	// copy to host for kmMat2
	kcMat2& CopyTo(kmMat2<T>& b, cudaStream_t stream = 0) const
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat2::CopyTo in 791]");
	
		cudaMemcpy2DAsync((void*) b.P(), b.P1()*sizeof(T), 
			              (void*)   P(),   P1()*sizeof(T),
			              _n1*sizeof(T), _n2, cudaMemcpyDeviceToHost, stream);
		return const_cast<kcMat2&>(*this);
	};

	// copy from host for kcMat2
	kcMat2& CopyFrom(const kcMat2<T>& b, cudaStream_t stream = 0)
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat2::CopyFrom in 801]");
	
		cudaMemcpy2DAsync((void*)   P(),   P1()*sizeof(T), 
			              (void*) b.P(), b.P1()*sizeof(T),
			              _n1*sizeof(T), _n2, cudaMemcpyDeviceToDevice, stream);
		return *this;
	};

	// copy from host for kmMat2
	kcMat2& CopyFrom(const kmMat2<T>& b, cudaStream_t stream = 0)
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat2::CopyFrom in 811]");
	
		cudaMemcpy2DAsync((void*)   P(),   P1()*sizeof(T), 
			              (void*) b.P(), b.P1()*sizeof(T),
			              _n1*sizeof(T), _n2, cudaMemcpyHostToDevice, stream);
		return *this;
	};
		
	/////////////////////////////////////////////////
	// operator functions
	
	// conversion operator... kmMat2 b = (kmMat2) a
	operator kmMat2<T>() const
	{	
		kmMat2<T> b(_n1, _n2, _p1); CopyTo(b);
		return b;
	};

	/////////////////////////////////////////////////
	// general member functions

	// get mat1
	kcMat1<T> Mat1(int64 i2) const { return kcMat1<T>(P(0,i2), _n1);};
		
	// get mat2
	kcMat2<T> Mat2(kmI i1, kmI i2) const
	{
		i1.e = MIN(i1.e, _n1-1);
		i2.e = MIN(i2.e, _n2-1);		

		return kcMat2<T>(P(i1.s, i2.s), i1.Len(), i2.Len(), _p1);
	};

	// get flat matrix
	kcMat1<T> Flat() const
	{	
		ASSERTA(IsNoBlank(), "[kcMat2::Flat in 1423]");
	
		return kcMat1<T>(P(), N());
	};

	// get info
	int64 N2() const { return _n2; };
	int64 P1() const { return _p1; };

	virtual bool IsNoBlank() const { return _p1 == _n1; };
	
	// get the number of real elements
	virtual int64 N() const { return _n1*_n2;};

	// get dimension
	static int64 GetDim() { return 2; };
	
	// display member info
	void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		kcMat1<T>::PrintInfo();
		PRINTFA("  _p1    : %lld\n", _p1);
		PRINTFA("  _n2    : %lld\n", _n2);
	};

	// display dimension
	void PrintDim(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]", str);

		PRINTFA(" dim : %lld, %lld (%lld, %p) \n", _n1, _n2, _size, _p);
	};

	// display member value
	void  PrintVal(int64 s_idx, int64 e_idx) const
	{
		kmMat2<T>(*this).PrintVal(s_idx, e_idx);
	};

	void PrintMat(LPCSTR str = nullptr) const
	{
		kmMat2<T>(*this).PrintMat(str);
	};

	// compare
	template<typename Y> bool IsEqualSizeDim (const kcMat2<Y>& b) const 
	{
		return _n1 == b.N1() && _n2 == b.N2();
	}

	template<typename Y> bool IsEqualSizeDim (const kmMat2<Y>& b) const 
	{
		return _n1 == b.N1() && _n2 == b.N2();
	}

	template<typename Y> bool IsEqualSizeDimP(const kcMat2<Y>& b) const
	{
		return (_p1 == b.P1()) && IsEqualSizeDim(b);
	}

	template<typename Y> bool IsEqualSizeDimP(const kmMat2<Y>& b) const
	{
		return (_p1 == b.P1()) && IsEqualSizeDim(b);
	}

	template<typename Y> bool IsEqualSizeAll (const kcMat2<Y>& b) const 
	{
		return IsEqualSizeDimP(b) && IsEqualSize(b);
	}

	template<typename Y> bool IsEqualSizeAll (const kmMat2<Y>& b) const 
	{
		return IsEqualSizeDimP(b) && IsEqualSize(b);
	}
	
	///////////////////////////////////////////////
	// functions for kernel

	void GetBkGd(dim3& bk, dim3& gd, uint nx1 = 32, uint ny1 = 32) const
	{
		bk = {nx1, ny1, 1};
		gd = {((int)N1()-1)/nx1 + 1, ((int)N2()-1)/ny1 + 1, 1};

		CheckBkGd(bk, gd);
	};

	// get bk and gd for reduction 
	// exdm : excluded dimension
	void GetBkGdRdc(dim3& bk, dim3& gd, int& sm, const int exdm = 0) const
	{
		bk = MIN(MAX(32, NextPow2((uint)N1())),1024);
		sm = bk.x*sizeof(T);

		if     (exdm == 0) gd = 1;
		else if(exdm == 2) gd = (uint)N2();
		else
		{
			PRINTFA("exdm(%d) is outof range in kcMat2\n", exdm);
			throw KE_OUTOF_RANGE;
		}

		CheckBkGd(bk, gd);
	};
};

//////////////////////////////////////////////////////////
// 3D matrix class
template<typename T> class kcMat3 : public kcMat2<T>
{
protected:
	// using for members of parents class
	using kcArr <T>::_p, kcArr<T>::_state, kcArr<T>::_size;
	using kcMat1<T>::_n1;
	using kcMat2<T>::_n2, kcMat2<T>::_p1;

	// member variables
	int64 _p2 = 0;  // pitch of dim2
	int64 _n3 = 1;  // number of dim3

public:
	// using for functions of parents class
	using kcArr <T>::GetKmClass;
	using kcArr <T>::Release;
	using kcArr <T>::CheckBkGd;
	using kcArr <T>::RestoreVfptr;
	using kcMat1<T>::N1;
	using kcMat2<T>::N2;
	using kcMat2<T>::P1;

	/////////////////////////////////////////////////
	// basic member functions
public:
	virtual void Init() { kcMat2<T>::Init(); _p2 = 0; _n3 = 1; };

	// constructor
	kcMat3() {};
	kcMat3(int64 n) { Create(n, 1, 1);};

	kcMat3(      int64 n1, int64 n2, int64 n3) { Create(n1, n2, n3);};
	kcMat3(T* p, int64 n1, int64 n2, int64 n3) { Set(p, n1, n2, n3);};

	kcMat3(      int64 n1, int64 n2, int64 n3, int64 p1) { CreateP(n1, n2, n3, p1);};
	kcMat3(T* p, int64 n1, int64 n2, int64 n3, int64 p1) { SetP(p, n1, n2, n3, p1);};

	kcMat3(      int64 n1, int64 n2, int64 n3, int64 p1, int64 p2) { CreateP(n1, n2, n3, p1, p2);};
	kcMat3(T* p, int64 n1, int64 n2, int64 n3, int64 p1, int64 p2) { SetP(p, n1, n2, n3, p1, p2);};

	kcMat3(      int64 n1, int64 n2, int64 n3, int64 p1, int64 p2, int64 size) { CreateP(n1, n2, n3, p1, p2, size);};
	kcMat3(T* p, int64 n1, int64 n2, int64 n3, int64 p1, int64 p2, int64 size) { SetP(p, n1, n2, n3, p1, p2, size);};
	
	kcMat3(const kcMat2<T>& a) { Set(a); };
	kcMat3(const kcMat1<T>& a) { Set(a); };

	// destructor
	virtual ~kcMat3() {};

	// copy constructor	
	kcMat3(const kcMat3&    b) { Create(b.N1(), b.N2(), b.N3()); CopyFrom(b); };
	kcMat3(const kmMat3<T>& b) { Create(b.N1(), b.N2(), b.N3()); CopyFrom(b); };

	template<typename Y>
	kcMat3(const kcMat3<Y>& b) { Create(b.N1(), b.N2(), b.N3()); TCast(*this, b); }

	// move constructor
	kcMat3(kcMat3&& b) { Move(b); };

	// assignment operator
	kcMat3& operator=(const kcMat3&    b) { RecreateIf(b); return CopyFrom(b); };
	kcMat3& operator=(const kmMat3<T>& b) { RecreateIf(b); return CopyFrom(b); };

	template<typename Y>
	kcMat3& operator=(const kcMat3<Y>& b) { RecreateIf(b); return TCast(*this, b); }
	
	// move assignment operator
	kcMat3& operator=(kcMat3&& b) { Move(b); return *this; };
		
	// allocate memory
	void Create (int64 n1, int64 n2, int64 n3 = 1)                 { CreateP(n1, n2, n3, n1, n2, n1*n2*n3); };
	void Create (int64 n1, int64 n2, int64 n3, int64 size)         { CreateP(n1, n2, n3, n1, n2, size    ); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 p1  )         { CreateP(n1, n2, n3, p1, n2, p1*n2*n3); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 p1, int64 p2) { CreateP(n1, n2, n3, p1, p2, p1*p2*n3); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 p1, int64 p2, int64 size)
	{
		ASSERTA(p1*p2*n3 <= size,"[kcMat3::CreateP in 874]");

		_n1 = n1; _n2 = n2; _n3 = n3; _p1 = p1; _p2 = p2; kcArr<T>::Create(size);
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
	template<typename Y> void Recreate(const kcMat3<Y>& b) { Recreate(b.N1(), b.N2(), b.N3()); }
	template<typename Y> void Recreate(const kmMat3<Y>& b) { Recreate(b.N1(), b.N2(), b.N3()); }

	// recreate if
	int RecreateIf(int64 n1, int64 n2, int64 n3)
	{
		if(n1 != _n1 || n2 != _n2 || n3 != _n3) { Recreate(n1, n2, n3); return 1; }	return 0;
	};
	template<typename Y> int RecreateIf(const kcMat3<Y>& b) { return RecreateIf(b.N1(), b.N2(), b.N3()); }
	template<typename Y> int RecreateIf(const kmMat3<Y>& b) { return RecreateIf(b.N1(), b.N2(), b.N3()); }

	// move 
	void Move(kcMat3& b) { kcMat2<T>::Move(b); _n3 = b._n3; _p2 = b._p2; };

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

		ASSERTA(min_size <= size, "[kcMat3::SetP in 904] %lld <= %lld", min_size, size);

		_n1 = n1; _n2 = n2; _n3 = n3; _p1 = p1; _p2 = p2; kcArr<T>::Set(p, size);
	};

	// calculate the minimum size.
	// * Note that (p - n) of the last element is actually not used.
	static int64 CalcMinSize(int64 n1, int64 n2, int64 n3, int64 p1, int64 p2)
	{
		return p1*p2*n3 - p1*(p2 - n2) - (p1 - n1);
	};

	// set array... a.Set(b)
	void Set(const kcMat3<T>& b) { SetP(b.P(), b.N1(), b.N2(), b.N3(), b.P1(), b.P2(), b.Size()); };
	void Set(const kcMat2<T>& b) { SetP(b.P(), b.N1(), b.N2(),      1, b.P1(), b.N2(), b.Size()); };
	void Set(const kcMat1<T>& b) { Set (b.P(), b.N1(),      1,      1, b.Size()); };
	void Set(const kcArr <T>& b) { Set (b.P(), b.Size(),    1,      1, b.Size()); };
	
	// operator to get data
	T* P(int64 i1, int64 i2 , int64 i3) const
	{
		const int64 idx = i1 + _p1*(i2 + _p2*i3);

		ASSERTA(idx < _size, "[kcMat3::P in 920] %lld < %lld", idx, _size);

		return _p + idx;
	};

	T* P(int64 i1, int64 i) const
	{
		const int64 i3 = i/_n2; i-= i3*_n2;		

		return P(i1, i, i3);
	};

	T* P(int64 i) const
	{
		const int64 n12 = _n1*_n2;
		const int64 n1  = _n1;

		const int64 i3  = i/n12; i-= i3*n12;
		const int64 i2  = i/n1;  i-= i2*n1;
		
		return P(i, i2, i3);
	};

	T* P() const { return _p; };

	T* End() const { return P(_n1-1, _n2-1, _n3-1); };

	// restore the class including vfptr
	// * Note that this will clear and reset _p, _state, _size without release,
	// * which can cause a memory leak. So, you should use this very carefully.
	kcMat3& Restore() { RestoreVfptr(*this); _state = 0; Init(); return *this; };

	// reshape
	kcMat3& Reshape(int64 n1, int64 n2, int64 n3, int64 p1 = 0, int64 p2 = 0)
	{
		// init arguement
		if(p1 == 0) p1 = n1;
		if(p2 == 0) p2 = n2;

		// check size
		ASSERTA(p1*p2*n3 <= _size, "[kcMat3::Reshape in 955]");

		// reset members
		_n1 = n1; _n2 = n2;	_n3 = n3; _p1 = p1; _p2 = p2;

		return *this;
	};

	// copy to device for kcMat3
	kcMat3& CopyTo(const kcMat3<T>& b, cudaStream_t stream = 0) const
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat3::CopyTo in 1123]");
	
		for(int64 i3 = 0; i3 < _n3; ++i3)
		{
			cudaMemcpy2DAsync((void*) b.P(0,0,i3), b.P1()*sizeof(T), 
				              (void*)   P(0,0,i3),   P1()*sizeof(T),
				              _n1*sizeof(T), _n2, cudaMemcpyDeviceToDevice, stream);
		}
		return const_cast<kcMat3&>(*this);
	};

	// copy to host for kmMat3
	kcMat3& CopyTo(kmMat3<T>& b, cudaStream_t stream = 0) const
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat3::CopyTo in 1097]");
	
		for(int64 i3 = 0; i3 < _n3; ++i3)
		{
			cudaMemcpy2DAsync((void*) b.P(0,0,i3), b.P1()*sizeof(T), 
				              (void*)   P(0,0,i3),   P1()*sizeof(T),
				              _n1*sizeof(T), _n2, cudaMemcpyDeviceToHost, stream);
		}
		return const_cast<kcMat3&>(*this);
	};

	// copy from host for kcMat3
	kcMat3& CopyFrom(const kcMat3<T>& b, cudaStream_t stream = 0)
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat3::CopyFrom in 1188]");
	
		for(int64 i3 = 0; i3 < _n3; ++i3)
		{
			cudaMemcpy2DAsync((void*)   P(0,0,i3),   P1()*sizeof(T), 
				              (void*) b.P(0,0,i3), b.P1()*sizeof(T),
				              _n1*sizeof(T), _n2, cudaMemcpyDeviceToDevice, stream);
		}
		return *this;
	};

	// copy from host for kmMat3
	kcMat3& CopyFrom(const kmMat3<T>& b, cudaStream_t stream = 0)
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat3::CopyFrom in 1123]");
	
		for(int64 i3 = 0; i3 < _n3; ++i3)
		{
			cudaMemcpy2DAsync((void*)   P(0,0,i3),   P1()*sizeof(T), 
				              (void*) b.P(0,0,i3), b.P1()*sizeof(T),
				              _n1*sizeof(T), _n2, cudaMemcpyHostToDevice, stream);
		}
		return *this;
	};

	/////////////////////////////////////////////////
	// operator functions

	// conversion operator... kmMat3 b = (kmMat3) a
	operator kmMat3<T>() const
	{	
		kmMat3<T> b(_n1, _n2, _n3, _p1, _p2); CopyTo(b);
		return b;
	};
	
	/////////////////////////////////////////////////
	// general member functions

	// get mat1
	kcMat1<T> Mat1(int64 i23)          const { return kcMat1<T>(P(0,i23)  , _n1);};
	kcMat1<T> Mat1(int64 i2, int64 i3) const { return kcMat1<T>(P(0,i2,i3), _n1);};

	// get mat2
	kcMat2<T> Mat2(int64 i3) const { return kcMat2<T>(P(0,0,i3), _n1, _n2, _p1);};

	// get mat3
	kcMat3<T> Mat3(kmI i1, kmI i2, kmI i3) const
	{
		i1.e = MIN(i1.e, _n1-1);
		i2.e = MIN(i2.e, _n2-1);
		i3.e = MIN(i3.e, _n3-1);

		return kcMat3<T>(P(i1.s, i2.s, i3.s), i1.Len(), i2.Len(), i3.Len(), _p1, _p2);
	};

	// get info
	int64 N3() const { return _n3; };
	int64 P2() const { return _p2; };

	virtual bool IsNoBlank() const { return (_p1 == _n1) && (_p2 == _n2); };
	
	// get the number of real elements
	virtual int64 N() const { return _n1*_n2*_n3;};

	// get dimension
	static int64 GetDim() { return 3; };

	// display member info
	void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		kcMat2<T>::PrintInfo();
		PRINTFA("  _p2    : %lld\n", _p2);
		PRINTFA("  _n3    : %lld\n", _n3);
	};

	// display dimension
	void PrintDim(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		PRINTFA("  dim : %lld, %lld, %lld (%lld) \n", _n1, _n2, _n3, _size);
	};

	// display member value
	void  PrintVal(int64 s_idx, int64 e_idx) const
	{
		kmMat3<T>(*this).PrintVal(s_idx, e_idx);
	};

	void PrintMat(LPCSTR str = nullptr) const
	{
		kmMat3<T>(*this).PrintMat(str);
	};

	// compare
	template<typename Y> bool IsEqualSizeDim(const kcMat3<Y>& b) const
	{
		return _n1 == b.N1() && _n2 == b.N2() && _n3 == b.N3();
	}

	template<typename Y> bool IsEqualSizeDim(const kmMat3<Y>& b) const
	{
		return _n1 == b.N1() && _n2 == b.N2() && _n3 == b.N3();
	}

	template<typename Y> bool IsEqualSizeDimP(const kcMat3<Y>& b) const
	{
		return (_p1 == b.P1()) && (_p2 == b.P2()) && IsEqualSizeDim(b);
	}

	template<typename Y> bool IsEqualSizeDimP(const kmMat3<Y>& b) const
	{
		return (_p1 == b.P1()) && (_p2 == b.P2()) && IsEqualSizeDim(b);
	}

	template<typename Y> bool IsEqualSizeAll(const kcMat3<Y>& b) const
	{
		return IsEqualSizeDimP(b) && IsEqualSize(b);
	}

	template<typename Y> bool IsEqualSizeAll(const kmMat3<Y>& b) const
	{
		return IsEqualSizeDimP(b) && IsEqualSize(b);
	}

	///////////////////////////////////////////////
	// functions for kernel

	void GetBkGd(dim3& bk, dim3& gd, uint nx1, uint ny1) const
	{	
		bk = {nx1, ny1, 1};
		gd = {((int)N1()-1)/nx1 + 1, ((int)N2()-1)/ny1 + 1, (uint)N3()};

		CheckBkGd(bk, gd);
	};

	void GetBkGd(dim3& bk, dim3& gd) const
	{
		uint nx1 = MIN((uint)N1(), 1024), ny1 = 1;
		
		GetBkGd(bk, gd, nx1, ny1);

		//PRINTFA("kcMat3::GetBkGd(n1, n2, n3) (nx1, ny1) : (%d, %d, %d, %d) - (%d, %d)\n", N1(), N2(), N3(), nx1, ny1);		
	};

	// get bk and gd for reduction 
	// exdm : excluded dimension
	void GetBkGdRdc(dim3& bk, dim3& gd, int& sm, const int exdm1 = 0, const int exdm2 = 0) const
	{
		bk = MIN(MAX(32, NextPow2((uint)N1())),1024);
		sm = bk.x*sizeof(T);

		if     (exdm1 == 0 && exdm2 == 0) gd = 1;
		else if(exdm1 == 2 && exdm2 == 0) gd =  (uint)N2();
		else if(exdm1 == 3 && exdm2 == 0) gd =  (uint)N3();
		else if(exdm1 == 2 && exdm2 == 3) gd = {(uint)N2(), (uint)N3(), 1};
		else
		{
			PRINTFA("exdm(%d, %d) is outof range in kcMat3\n", exdm1, exdm2);
			throw KE_OUTOF_RANGE;
		};
		CheckBkGd(bk, gd);
	};
};

//////////////////////////////////////////////////////////
// 4D matrix class
template<typename T> class kcMat4 : public kcMat3<T>
{
protected:
	// using for members of parents class
	using kcArr <T>::_p, kcArr<T>::_state, kcArr<T>::_size;
	using kcMat1<T>::_n1;
	using kcMat2<T>::_n2, kcMat2<T>::_p1;
	using kcMat3<T>::_n3, kcMat3<T>::_p2;

	// member variables
	int64 _p3 = 0;   // pitch of dim3
	int64 _n4 = 1;   // number of dim4

public:
	// using for functions of parents class
	using kcArr <T>::GetKmClass;
	using kcArr <T>::Release;
	using kcArr <T>::CheckBkGd;
	using kcMat1<T>::N1;
	using kcMat2<T>::N2;
	using kcMat2<T>::P1;
	using kcMat3<T>::N3;
	using kcMat3<T>::P2;

	/////////////////////////////////////////////////
	// basic member functions
public:
	virtual void Init() { kcMat3<T>::Init(); _p3 = 0; _n4 = 1; };

	// constructor
	kcMat4() {};
	kcMat4(int64 n) { Create(n, 1, 1, 1);};

	kcMat4(      int64 n1, int64 n2, int64 n3, int64 n4) { Create(n1, n2, n3, n4);};
	kcMat4(T* p, int64 n1, int64 n2, int64 n3, int64 n4) { Set(p, n1, n2, n3, n4);};

	kcMat4(      int64 n1, int64 n2, int64 n3, int64 n4, int64 p1) { CreateP(n1, n2, n3, n4, p1);};
	kcMat4(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1) { SetP(p, n1, n2, n3, n4, p1);};

	kcMat4(      int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2) { CreateP(n1, n2, n3, n4, p1, p2);};
	kcMat4(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2) { SetP(p, n1, n2, n3, n4, p1, p2);};

	kcMat4(      int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3) { CreateP(n1, n2, n3, n4, p1, p2, p3);};
	kcMat4(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3) { SetP(p, n1, n2, n3, n4, p1, p2, p3);};
												  
	kcMat4(      int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3, int64 size) { CreateP(n1, n2, n3, n4, p1, p2, p3, size);};
	kcMat4(T* p, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3, int64 size) { SetP(p, n1, n2, n3, n4, p1, p2, p3, size);};
	
	kcMat4(const kcMat3<T>& a) { Set(a); };
	kcMat4(const kcMat2<T>& a) { Set(a); };
	kcMat4(const kcMat1<T>& a) { Set(a); };

	// destructor
	virtual ~kcMat4() {};

	// copy constructor	
	kcMat4(const kcMat4&    b) { Create(b.N1(), b.N2(), b.N3(), b.N4()); CopyFrom(b); };
	kcMat4(const kmMat4<T>& b) { Create(b.N1(), b.N2(), b.N3(), b.N4()); CopyFrom(b); };

	template<typename Y>
	kcMat4(const kcMat4<Y>& b) { Create(b.N1(), b.N2(), b.N3(), b.N4()); TCast(*this, b); }

	// move constructor
	kcMat4(kcMat4&& b) { Move(b); };
	
	// assignment operator
	kcMat4& operator=(const kcMat4&    b) { RecreateIf(b); return CopyFrom(b);};
	kcMat4& operator=(const kmMat4<T>& b) {	RecreateIf(b); return CopyFrom(b);};

	template<typename Y>
	kcMat4& operator=(const kcMat4<Y>& b) { RecreateIf(b); return TCast(*this, b);}

	// move assignment operator
	kcMat4& operator=(kcMat4&& b) { Move(b); return *this; };

	// allocate memory
	void Create (int64 n1, int64 n2, int64 n3, int64 n4 = 1)                           { CreateP(n1, n2, n3, n4, n1, n2, n3, n1*n2*n3*n4); };
	void Create (int64 n1, int64 n2, int64 n3, int64 n4, int64 size)                   { CreateP(n1, n2, n3, n4, n1, n2, n3, size       ); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1  )                   { CreateP(n1, n2, n3, n4, p1, n2, n3, p1*n2*n3*n4); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2)           { CreateP(n1, n2, n3, n4, p1, p2, n3, p1*p2*n3*n4); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3) { CreateP(n1, n2, n3, n4, p1, p2, p3, p1*p2*p3*n4); };
	void CreateP(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3, int64 size)
	{
		ASSERTA(p1*p2*p3*n4 <= size,"[kcMat4::CreateP in 1141]");

		_n1 = n1; _n2 = n2; _n3 = n3; _n4 = n4; _p1 = p1; _p2 = p2; _p3 = p3; kcArr<T>::Create(size);
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
	template<typename Y> void Recreate(const kcMat4<Y>& b) { Recreate(b.N1(), b.N2(), b.N3(), b.N4()); }
	template<typename Y> void Recreate(const kmMat4<Y>& b) { Recreate(b.N1(), b.N2(), b.N3(), b.N4()); }

	// recreate if
	int RecreateIf(int64 n1, int64 n2, int64 n3, int64 n4)
	{
		if(n1 != _n1 || n2 != _n2 || n3 != _n3 || n4 != _n4) { Recreate(n1,n2,n3,n4); return 1; }	return 0;
	};
	template<typename Y> int RecreateIf(const kcMat4<Y>& b) { return RecreateIf(b.N1(), b.N2(), b.N3(), b.N4()); }
	template<typename Y> int RecreateIf(const kmMat4<Y>& b) { return RecreateIf(b.N1(), b.N2(), b.N3(), b.N4()); }

	// move
	void Move(kcMat4& b) { kcMat3<T>::Move(b); _n4 = b._n4; _p3 = b._p3; };

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

		ASSERTA(min_size <= size, "[kcMat4::SetP in 1285] %lld <= %lld", min_size, size);		

		_n1 = n1; _n2 = n2; _n3 = n3; _n4 = n4; _p1 = p1; _p2 = p2; _p3 = p3; kcArr<T>::Set(p, size);
	};

	// calculate the minimum size.
	// * Note that (p - n) of the last element is actually not used.
	static int64 CalcMinSize(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1, int64 p2, int64 p3)
	{
		return p1*p2*p3*n4 - p1*p2*(p3 - n3) - p1*(p2 - n2) - (p1 - n1);
	};

	// set array... a.Set(b)
	void Set(const kcMat4<T>& b) { SetP(b.P(), b.N1(), b.N2(), b.N3(), b.N4(), b.P1(), b.P2(), b.P3(), b.Size()); };
	void Set(const kcMat3<T>& b) { SetP(b.P(), b.N1(), b.N2(), b.N3(),      1, b.P1(), b.P2(), b.N3(), b.Size()); };
	void Set(const kcMat2<T>& b) { SetP(b.P(), b.N1(), b.N2(),      1,      1, b.P1(), b.N2(),     1 , b.Size()); };
	void Set(const kcMat1<T>& b) { Set (b.P(), b.N1(),     1,       1,      1, b.Size()); };
	void Set(const kcArr <T>& b) { Set (b.P(), b.Size(),   1,       1,      1, b.Size()); };
	
	// operator to get data
	T* P(int64 i1, int64 i2 , int64 i3, int64 i4) const
	{
		const int64 idx = i1 + _p1*(i2 + _p2*(i3 + _p3*i4));

		ASSERTA(idx < _size, "[kcMat4::P in 1200] %lld < %lld", idx, _size);

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
		const int64 n123 = _n1*_n2*_n3;
		const int64 n12  = _n1*_n2;
		const int64 n1   = _n1;

		const int64 i4 = i/n123; i-= i4*n123;
		const int64 i3 = i/n12;  i-= i3*n12;
		const int64 i2 = i/n1;   i-= i2*n1;

		return P(i, i2, i3, i4);
	};

	T* P() const { return _p; };

	T* End() const { return P(_n1-1, _n2-1, _n3-1, _n4-1); };

	// restore the class including vfptr
	// * Note that this will clear and reset _p, _state, _size without release,
	// * which can cause a memory leak. So, you should use this very carefully.
	kcMat4& Restore() { RestoreVfptr(*this); _state = 0; Init(); return *this; };

	// reshape
	kcMat4& Reshape(int64 n1, int64 n2, int64 n3, int64 n4, int64 p1 = 0, int64 p2 = 0, int64 p3 = 0)
	{
		// init arguement
		if(p1 == 0) p1 = n1;
		if(p2 == 0) p2 = n2;
		if(p3 == 0) p3 = n3;

		// check size
		ASSERTA(p1*p2*p3*n4 <= _size, "[kcMat4::Reshape in 1249]");

		// reset members
		_n1 = n1; _n2 = n2;	_n3 = n3; _n4 = n4; _p1 = p1; _p2 = p2; _p3 = p3;

		return *this;
	};

	// copy to device for kcMat3
	kcMat4& CopyTo(const kcMat4<T>& b, cudaStream_t stream = 0) const
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat4::CopyTo in 1614]");

		for(int64 i4 = 0; i4 < _n4; ++i4)
		for(int64 i3 = 0; i3 < _n3; ++i3)
		{
			cudaMemcpy2DAsync((void*) b.P(0,0,i3,i4), b.P1()*sizeof(T), 
			                  (void*)   P(0,0,i3,i4),   P1()*sizeof(T),
				               _n1*sizeof(T), _n2, cudaMemcpyDeviceToDevice, stream);
		}
		return const_cast<kcMat4&>(*this);
	};

	// copy to host for kmMat4
	kcMat4& CopyTo(kmMat4<T>& b, cudaStream_t stream = 0) const
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat4::CopyTo in 1627]");
	
		for(int64 i4 = 0; i4 < _n4; ++i4)
		for(int64 i3 = 0; i3 < _n3; ++i3)
		{
			cudaMemcpy2DAsync((void*) b.P(0,0,i3,i4), b.P1()*sizeof(T), 
				              (void*)   P(0,0,i3,i4),   P1()*sizeof(T),
				              _n1*sizeof(T), _n2, cudaMemcpyDeviceToHost, stream);
		}
		return const_cast<kcMat4&>(*this);
	};

	// copy from host for kcMat4
	kcMat4& CopyFrom(const kcMat4<T>& b, cudaStream_t stream = 0)
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat4::CopyFrom in 1642]");

		for(int64 i4 = 0; i4 < _n4; ++i4)
		for(int64 i3 = 0; i3 < _n3; ++i3)
		{
			cudaMemcpy2DAsync((void*)   P(0,0,i3,i4),   P1()*sizeof(T), 
				              (void*) b.P(0,0,i3,i4), b.P1()*sizeof(T),
				              _n1*sizeof(T), _n2, cudaMemcpyDeviceToDevice, stream);
		}
		return *this;
	};

	// copy from host for kmMat4
	kcMat4& CopyFrom(const kmMat4<T>& b, cudaStream_t stream = 0)
	{
		ASSERTA(IsEqualSizeDim(b), "[kcMat4::CopyFrom in 1655]");

		for(int64 i4 = 0; i4 < _n4; ++i4)
		for(int64 i3 = 0; i3 < _n3; ++i3)
		{
			cudaMemcpy2DAsync((void*)   P(0,0,i3,i4),   P1()*sizeof(T), 
				              (void*) b.P(0,0,i3,i4), b.P1()*sizeof(T),
				              _n1*sizeof(T), _n2, cudaMemcpyHostToDevice, stream);
		}
		return *this;
	};

	/////////////////////////////////////////////////
	// operator functions

	// conversion operator... kmMat4 b = (kmMat4) a
	operator kmMat4<T>() const
	{	
		kmMat4<T> b(_n1, _n2, _n3, _n4, _p1, _p2, _p3); CopyTo(b);
		return b;
	};
	
	/////////////////////////////////////////////////
	// general member functions

	// get mat1
	kcMat1<T> Mat1(int64 idx)                    const { return kcMat1<T>(P(0,idx)     , _n1);};
	kcMat1<T> Mat1(int64 i2, int64 i3, int64 i4) const { return kcMat1<T>(P(0,i2,i3,i4), _n1);};

	// get mat2
	kcMat2<T> Mat2(int64 idx)          const { return kcMat2<T>(P(0,0,idx)  , _n1, _n2, _p1);};
	kcMat2<T> Mat2(int64 i3, int64 i4) const { return kcMat2<T>(P(0,0,i3,i4), _n1, _n2, _p1);};

	// get mat3
	kcMat3<T> Mat3(int64 idx) const
	{
		return kcMat3<T>(P(0,0,0,idx), _n1, _n2, _n3, _p1, _p2);
	};

	// get mat4
	kcMat4 Mat4(kmI i1, kmI i2, kmI i3, kmI i4) const
	{
		i1.e = MIN(i1.e, _n1-1);
		i2.e = MIN(i2.e, _n2-1);
		i3.e = MIN(i3.e, _n3-1);
		i4.e = MIN(i4.e, _n4-1);

		return kcMat4<T>(P(i1.s, i2.s, i3.s, i4.s), i1.Len(), i2.Len(), i3.Len(), i4.Len(), _p1, _p2, _p3);
	};

	// get info
	int64 N4() const { return _n4; };
	int64 P3() const { return _p3; };

	virtual bool IsNoBlank() const { return (_p1 == _n1) && (_p2 == _n2) && (_p3 == _n3); };
	
	// get the number of real elements
	virtual int64 N() const { return _n1*_n2*_n3*_n4;};

	// get dimension
	static int64 GetDim() { return 0; };
	
	// display member info
	void PrintInfo(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]\n", str);

		kcMat3<T>::PrintInfo();
		PRINTFA("  _p3    : %lld\n", _p3);
		PRINTFA("  _n4    : %lld\n", _n4);
	};

	// display dimension
	void PrintDim(LPCSTR str = nullptr) const
	{
		if(str != nullptr) PRINTFA("[%s]", str);

		PRINTFA(" dim : %lld, %lld, %lld, %lld (%lld, %p) \n", _n1, _n2, _n3, _n4, _size, _p);
	};

	// display member value
	void  PrintVal(int64 s_idx, int64 e_idx) const
	{
		kmMat4<T>(*this).PrintVal(s_idx, e_idx);
	};

	// compare
	template<typename Y> bool IsEqualSizeDim(const kcMat4<Y>& b) const
	{
		return _n1 == b.N1() && _n2 == b.N2() && _n3 == b.N3() && _n4 == b.N4();
	}

	template<typename Y> bool IsEqualSizeDim(const kmMat4<Y>& b) const
	{
		return _n1 == b.N1() && _n2 == b.N2() && _n3 == b.N3() && _n4 == b.N4();
	}

	template<typename Y> bool IsEqualSizeDimP(const kcMat4<Y>& b) const
	{
		return _p1 == b.P1() && _p2 == b.P2() && _p3 == b.P3() && IsEqualSizeDim(b);
	}

	template<typename Y> bool IsEqualSizeDimP(const kmMat4<Y>& b) const
	{
		return _p1 == b.P1() && _p2 == b.P2() && _p3 == b.P3() && IsEqualSizeDim(b);
	}

	template<typename Y> bool IsEqualSizeAll(const kcMat4<Y>& b) const
	{
		return IsEqualSizeDimP(b) && IsEqualSize(b);
	}

	template<typename Y> bool IsEqualSizeAll(const kmMat4<Y>& b) const
	{
		return IsEqualSizeDimP(b) && IsEqualSize(b);
	}

	///////////////////////////////////////////////
	// functions for kernel

	void GetBkGd(dim3& bk, dim3& gd, uint nx1, uint ny1) const
	{
		bk = {nx1, ny1, 1};
		gd = {(uint)(N3()*N4()), ((int)N2()-1)/ny1 + 1, ((int)N1()-1)/nx1 + 1};

		CheckBkGd(bk, gd);
	}

	void GetBkGd(dim3& bk, dim3& gd) const
	{	
		uint nx1 = MIN((uint)N1(), 1024), ny1 = 1;

		GetBkGd(bk, gd, nx1, ny1);

		//PRINTFA("kcMat4::GetBkGd(n1, n2, n3, n4) (nx1, ny1) : (%d, %d, %d, %d) - (%d, %d)\n", N1(), N2(), N3(), N4(), nx1, ny1);
	};

	// get bk and gd for reduction 
	// exdm : excluded dimension
	void GetBkGdRdc(dim3& bk, dim3& gd, int& sm, const int exdm1 = 0, const int exdm2 = 0, const int exdm3 = 0) const
	{	
		bk = MIN(MAX(32, NextPow2((uint)N1())),1024);		
		sm = bk.x*sizeof(T);

		if     (exdm1 == 0 && exdm2 == 0 && exdm3 == 0) gd = 1;
		else if(exdm1 == 2 && exdm2 == 0 && exdm3 == 0) gd =  (uint)N2();
		else if(exdm1 == 3 && exdm2 == 0 && exdm3 == 0) gd =  (uint)N3();
		else if(exdm1 == 4 && exdm2 == 0 && exdm3 == 0) gd =  (uint)N4();
		else if(exdm1 == 2 && exdm2 == 3 && exdm3 == 0) gd = {(uint)N2(), (uint)N3(), 1};
		else if(exdm1 == 2 && exdm2 == 4 && exdm3 == 0) gd = {(uint)N2(), (uint)N4(), 1};
		else if(exdm1 == 3 && exdm2 == 4 && exdm3 == 0) gd = {(uint)N3(), (uint)N4(), 1};
		else if(exdm1 == 2 && exdm2 == 3 && exdm3 == 4) gd = {(uint)N2(), (uint)N3(), (uint)N4()};
		else
		{
			PRINTFA("exdm(%d, %d, %d) is outof range in kcMat4\n", exdm1, exdm2, exdm3);
			throw KE_OUTOF_RANGE;
		};
		CheckBkGd(bk, gd);
	};
};

// define type for kcMat
typedef kcArr <char>			kcArr0i8;
typedef kcMat1<char>			kcMat1i8;
typedef kcMat2<char>			kcMat2i8;
typedef kcMat3<char>			kcMat3i8;
typedef kcMat4<char>			kcMat4i8;

typedef kcArr <uchar>			kcArr0u8;
typedef kcMat1<uchar>			kcMat1u8;
typedef kcMat2<uchar>			kcMat2u8;
typedef kcMat3<uchar>			kcMat3u8;
typedef kcMat4<uchar>			kcMat4u8;

typedef kcArr <short>			kcArr0i16;
typedef kcMat1<short>			kcMat1i16;
typedef kcMat2<short>			kcMat2i16;
typedef kcMat3<short>			kcMat3i16;
typedef kcMat4<short>			kcMat4i16;

typedef kcArr <ushort>			kcArr0u16;
typedef kcMat1<ushort>			kcMat1u16;
typedef kcMat2<ushort>			kcMat2u16;
typedef kcMat3<ushort>			kcMat3u16;
typedef kcMat4<ushort>			kcMat4u16;

typedef kcArr <int>				kcArr0i32;
typedef kcMat1<int>				kcMat1i32;
typedef kcMat2<int>				kcMat2i32;
typedef kcMat3<int>				kcMat3i32;
typedef kcMat4<int>				kcMat4i32;

typedef kcArr <uint>			kcArr0u32;
typedef kcMat1<uint>			kcMat1u32;
typedef kcMat2<uint>			kcMat2u32;
typedef kcMat3<uint>			kcMat3u32;
typedef kcMat4<uint>			kcMat4u32;

typedef kcArr <int64>			kcArr0i64;
typedef kcMat1<int64>			kcMat1i64;
typedef kcMat2<int64>			kcMat2i64;
typedef kcMat3<int64>			kcMat3i64;
typedef kcMat4<int64>			kcMat4i64;

typedef kcArr <float>			kcArr0f32;
typedef kcMat1<float>			kcMat1f32;
typedef kcMat2<float>			kcMat2f32;
typedef kcMat3<float>			kcMat3f32;
typedef kcMat4<float>			kcMat4f32;

typedef kcArr <double>			kcArr0f64;
typedef kcMat1<double>			kcMat1f64;
typedef kcMat2<double>			kcMat2f64;
typedef kcMat3<double>			kcMat3f64;
typedef kcMat4<double>			kcMat4f64;

typedef kcArr <float>			kcArr0f32;
typedef kcMat1<float>			kcMat1f32;
typedef kcMat2<float>			kcMat2f32;
typedef kcMat3<float>			kcMat3f32;
typedef kcMat4<float>			kcMat4f32;

typedef kcArr <float2>			kcArr0c32;
typedef kcMat1<float2>			kcMat1c32;
typedef kcMat2<float2>			kcMat2c32;
typedef kcMat3<float2>			kcMat3c32;
typedef kcMat4<float2>			kcMat4c32;

typedef kcArr <short2>			kcArr0q16;
typedef kcMat1<short2>			kcMat1q16;
typedef kcMat2<short2>			kcMat2q16;
typedef kcMat3<short2>			kcMat3q16;
typedef kcMat4<short2>			kcMat4q16;

typedef kcArr <f32xy>			kcArr0f32xy;
typedef kcMat1<f32xy>			kcMat1f32xy;
typedef kcMat2<f32xy>			kcMat2f32xy;
typedef kcMat3<f32xy>			kcMat3f32xy;
typedef kcMat4<f32xy>			kcMat4f32xy;

typedef kcArr <f32yz>			kcArr0f32yz;
typedef kcMat1<f32yz>			kcMat1f32yz;
typedef kcMat2<f32yz>			kcMat2f32yz;
typedef kcMat3<f32yz>			kcMat3f32yz;
typedef kcMat4<f32yz>			kcMat4f32yz;

typedef kcArr <f32zx>			kcArr0f32zx;
typedef kcMat1<f32zx>			kcMat1f32zx;
typedef kcMat2<f32zx>			kcMat2f32zx;
typedef kcMat3<f32zx>			kcMat3f32zx;
typedef kcMat4<f32zx>			kcMat4f32zx;

typedef kcArr <f32xyz>			kcArr0f32xyz;
typedef kcMat1<f32xyz>			kcMat1f32xyz;
typedef kcMat2<f32xyz>			kcMat2f32xyz;
typedef kcMat3<f32xyz>			kcMat3f32xyz;
typedef kcMat4<f32xyz>			kcMat4f32xyz;

typedef kcArr <half>			kcArr0f16;
typedef kcMat1<half>			kcMat1f16;
typedef kcMat2<half>			kcMat2f16;
typedef kcMat3<half>			kcMat3f16;
typedef kcMat4<half>			kcMat4f16;

///////////////////////////////////////////////////////////////
// memory block class for manual memory allocation
template<typename T> class kcMem : public kcMat1<T>
{
protected:
	// using for members for parents class
	using kcArr <T>::_p, kcArr<T>::_state, kcArr<T>::_size;
	using kcMat1<T>::_n1;
	
public:
	// using for functions for parents class
	using kcArr<T>::GetKmClass;
	using kcArr<T>::Byte;

	/////////////////////////////////////////////////
	// basic member functions
public:	
	virtual void Init() { kcMat1<T>::Init(); };

	// constructor
	kcMem() {};
	kcMem(      int64 byte) { Create(byte);};
	kcMem(T* p, int64 byte) { Set(p, byte);};

	// destructor
	virtual ~kcMem() {};
		
	// allocate memory
	void Create(int64 byte) { kcMat1<T>::Create(0, Byte2Size(byte)); };
	
	// release and create
	void Recreate(int64 byte) { kcMat1<T>::Release(); Create(byte); };

	// set array
	void Set(T* p, int64 byte) { kcMat1<T>::Set(p, 0, Byte2Size(byte)); };

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
		else               PRINTFA("* kcMem : ");
		PRINTFA("%.1f / %lld MB\n", GetMbyteUsed(), Byte()>>20);
	};

	// get memory block
	void* GetMem(int64 byte)
	{
		// calc size
		int64 size = Byte2Size(byte); 

		// check size
		ASSERTA(_n1 + size <= _size, "[kcMem::GetMem in 1883] over memory");

		// update index
		int64 idx = _n1; _n1 += size;
				
		return (void*) (_p + idx);
	};

	// reset allocated memory block
	void Reset(const int64 idx = 0) { _n1 = idx; };

	// allocate memory to kmArr and kmMat
	template<typename Y> void Give(kcArr <Y>& a, int64 n ) { a.Set((Y*)this->GetMem(n *sizeof(Y)), n );}
	template<typename Y> void Give(kcMat1<Y>& a, int64 n1) { a.Set((Y*)this->GetMem(n1*sizeof(Y)), n1);}

	template<typename Y> 
	void Give(kcMat1<Y>& a, const kmMat1<Y>& b)
	{
		Give(a, b.N1()); a.Copy(b);
	}

	// allocate memory to kcMat2
	template<typename Y> 
	void Give(kcMat2<Y>& a, int64 n1, int64 n2, int64 p1 = 0)
	{
		if(p1 == 0) p1 = n1;

		a.SetP((Y*)this->GetMem(p1*n2*sizeof(Y)), n1, n2, p1);
	}

	template<typename Y> 
	void Give(kcMat2<Y>& a, const kmMat2<Y>& b)
	{
		Give(a, b.N1(), b.N2(), b.P1()); a.CopyFrom(b);
	}

	// allocate memory to kcMat3
	template<typename Y> 
	void Give(kcMat3<Y>& a, int64 n1, int64 n2, int64 n3, int64 p1 = 0, int64 p2 = 0)
	{
		if(p1 == 0) p1 = n1; if(p2 == 0) p2 = n2;

		a.SetP((Y*)this->GetMem(p1*p2*n3*sizeof(Y)), n1, n2, n3, p1, p2);
	}

	// allocate memory to kcMat4
	template<typename Y> 
	void Give(kcMat4<Y>& a, int64 n1, int64 n2, int64 n3, int64 n4, int64 p1 = 0, int64 p2 = 0, int64 p3 = 0)
	{
		if(p1 == 0) p1 = n1; if(p2 == 0) p2 = n2; if(p3 == 0) p3 = n3;

		a.SetP((Y*)this->GetMem(p1*p2*p3*n4*sizeof(Y)), n1, n2, n3, n4, p1, p2, p3);
	}
};

typedef kcMem<char>    kcMem8;
typedef kcMem<short>   kcMem16;
typedef kcMem<int>     kcMem32;
typedef kcMem<int64>   kcMem64;

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// class for kernel arguments

//////////////////////////////////////////////////
// define kckArr for constant memory and kckArr
template<typename T> class kccArr
{
public:
	T* p; int size;

	__device__ int Idx       (int i1) const { return       i1;  };
	__device__  T& operator()(int i1) const { return *(p + i1); };
};

template<typename T> class kckArr : public kccArr<T>
{
public:	
	kckArr(const kcArr<T>& b)
	{
		this->p    = b.P();
		this->size = (int)b.Size();
	};
};

//////////////////////////////////////////////////
// define kccMat2 for constant memory and kckMat1
template<typename T> class kccMat1
{
public:
	T* p; int n1;

	__device__ int Idx       (int i1) const { return       i1; };
	__device__  T& operator()(int i1) const { return *(p + i1);};

	__device__  T cg(int i1) const { return __ldcg(p + i1); };
	__device__  T ca(int i1) const { return __ldca(p + i1); };
	__device__  T cs(int i1) const { return __ldcs(p + i1); };
	__device__  T lu(int i1) const { return __ldlu(p + i1); };
	//__device__  T cv(int i1) const { return __ldcv(p + i1); };
};

template<typename T> class kckMat1 : public kccMat1<T>
{
public:
	using kccMat1<T>::p, kccMat1<T>::n1;

	kckMat1(const kcMat1<T>& b)
	{
		p  = b.P();
		n1 = (int)b.N1();
	};
};

//////////////////////////////////////////////////
// define kccMat2 for constant memory and kckMat2
template<typename T> class kccMat2
{
public:
	T* p; int n1, n2, p1;

	__device__ int Idx      (int i1, int i2) const { return       i1 + p1*i2; };
	__device__ T& operator()(int i1, int i2) const { return *(p + i1 + p1*i2);};
	__device__ T* P         (int i1, int i2) const { return  (p + i1 + p1*i2);};
	__device__ T* End()                      const { return P(n1-1, n2-1);    };

	__device__ T cg(int i1, int i2) const { return __ldcg(p + i1 + p1*i2); };
	__device__ T ca(int i1, int i2) const { return __ldca(p + i1 + p1*i2); };
	__device__ T cs(int i1, int i2) const { return __ldcs(p + i1 + p1*i2); };
	__device__ T lu(int i1, int i2) const { return __ldlu(p + i1 + p1*i2); };
	__device__ T cv(int i1, int i2) const { return __ldcv(p + i1 + p1*i2); };
};

template<typename T> class kckMat2 : public kccMat2<T>
{
public:	
	using kccMat2<T>::p, kccMat2<T>::n1, kccMat2<T>::n2, kccMat2<T>::p1;

	kckMat2(const kcMat2<T>& b)
	{
		p  = b.P();
		n1 = (int)b.N1(); n2 = (int)b.N2();
		p1 = (int)b.P1();
	};

	__device__ kckMat2(T* __p, int __n1, int __n2)
	{
		p = __p, n1 = __n1, n2 = __n2, p1 = __n1;
	};

	__device__ kckMat2(T* __p, int __n1, int __n2, int __p1)
	{
		p = __p, n1 = __n1, n2 = __n2, p1 = __p1;
	};
};

//////////////////////////////////////////////////
// define kccMat3 for constant memory and kckMat3
template<typename T> class kccMat3
{
public:
	T* p; int n1, n2, n3, p1, p2;

	__device__ int Idx      (int i1, int i2, int i3) const { return       i1 + p1*(i2 + p2*i3);	};
	__device__ T& operator()(int i1, int i2, int i3) const { return *(p + i1 + p1*(i2 + p2*i3));};
	__device__ T* P         (int i1, int i2, int i3) const { return  (p + i1 + p1*(i2 + p2*i3));};
	__device__ T* End()                              const { return P(n1-1, n2-1, n3-1);        };	
};

template<typename T> class kckMat3 : public kccMat3<T>
{
public:
	using kccMat3<T>::p , kccMat3<T>::n1, kccMat3<T>::n2, kccMat3<T>::n3;
	using kccMat3<T>::p1, kccMat3<T>::p2;

	kckMat3(const kcMat3<T>& b)
	{
		p  = b.P();
		n1 = (int)b.N1(); n2 = (int)b.N2(); n3 = (int)b.N3();
		p1 = (int)b.P1(); p2 = (int)b.P2();
	};
};

//////////////////////////////////////////////////
// define kccMat4 for constant memory and kckMat4
template<typename T> class kccMat4
{
public:
	T* p; int n1, n2, n3, n4, p1, p2, p3;

	__device__ int Idx      (int i1, int i2, int i3, int i4) const { return       i1 + p1*(i2 + p2*(i3 + p3*i4)); };
	__device__ T& operator()(int i1, int i2, int i3, int i4) const { return *(p + i1 + p1*(i2 + p2*(i3 + p3*i4)));};
};

template<typename T> class kckMat4 : public kccMat4<T>
{
public:
	using kccMat4<T>::p , kccMat4<T>::n1, kccMat4<T>::n2, kccMat4<T>::n3, kccMat4<T>::n4;
	using kccMat4<T>::p1, kccMat4<T>::p2, kccMat4<T>::p3;

	kckMat4(const kcMat4<T>& b)
	{
		p  = b.P();
		n1 = (int)b.N1(); n2 = (int)b.N2(); n3 = (int)b.N3(); n4 = (int)b.N4();
		p1 = (int)b.P1(); p2 = (int)b.P2(); p3 = (int)b.P3();
	};
};

// conversion functions from kcMat to kckMat
template<typename T> kckArr <T> kckMat(const kcArr <T>& a) { return kckArr <T>(a); };
template<typename T> kckMat1<T> kckMat(const kcMat1<T>& a) { return kckMat1<T>(a); };
template<typename T> kckMat2<T> kckMat(const kcMat2<T>& a) { return kckMat2<T>(a); };
template<typename T> kckMat3<T> kckMat(const kcMat3<T>& a) { return kckMat3<T>(a); };
template<typename T> kckMat4<T> kckMat(const kcMat4<T>& a) { return kckMat4<T>(a); };

// define type for kcMat
typedef kckArr <char>			kckArr0i8;
typedef kckMat1<char>			kckMat1i8;
typedef kckMat2<char>			kckMat2i8;
typedef kckMat3<char>			kckMat3i8;
typedef kckMat4<char>			kckMat4i8;
typedef kckArr <uchar>			kckArr0u8;
typedef kckMat1<uchar>			kckMat1u8;
typedef kckMat2<uchar>			kckMat2u8;
typedef kckMat3<uchar>			kckMat3u8;
typedef kckMat4<uchar>			kckMat4u8;
typedef kckArr <short>			kckArr0i16;
typedef kckMat1<short>			kckMat1i16;
typedef kckMat2<short>			kckMat2i16;
typedef kckMat3<short>			kckMat3i16;
typedef kckMat4<short>			kckMat4i16;
typedef kckArr <ushort>			kckArr0u16;
typedef kckMat1<ushort>			kckMat1u16;
typedef kckMat2<ushort>			kckMat2u16;
typedef kckMat3<ushort>			kckMat3u16;
typedef kckMat4<ushort>			kckMat4u16;
typedef kckArr <int>			kckArr0i32;
typedef kckMat1<int>			kckMat1i32;
typedef kckMat2<int>			kckMat2i32;
typedef kckMat3<int>			kckMat3i32;
typedef kckMat4<int>			kckMat4i32;
typedef kckArr <uint>			kckArr0u32;
typedef kckMat1<uint>			kckMat1u32;
typedef kckMat2<uint>			kckMat2u32;
typedef kckMat3<uint>			kckMat3u32;
typedef kckMat4<uint>			kckMat4u32;
typedef kckArr <int64>			kckArr0i64;
typedef kckMat1<int64>			kckMat1i64;
typedef kckMat2<int64>			kckMat2i64;
typedef kckMat3<int64>			kckMat3i64;
typedef kckMat4<int64>			kckMat4i64;
typedef kckArr <float>			kckArr0f32;
typedef kckMat1<float>			kckMat1f32;
typedef kckMat2<float>			kckMat2f32;
typedef kckMat3<float>			kckMat3f32;
typedef kckMat4<float>			kckMat4f32;
typedef kckArr <double>			kckArr0f64;
typedef kckMat1<double>			kckMat1f64;
typedef kckMat2<double>			kckMat2f64;
typedef kckMat3<double>			kckMat3f64;
typedef kckMat4<double>			kckMat4f64;
typedef kckArr <float>			kckArr0f32;
typedef kckMat1<float>			kckMat1f32;
typedef kckMat2<float>			kckMat2f32;
typedef kckMat3<float>			kckMat3f32;
typedef kckMat4<float>			kckMat4f32;
typedef kckArr <float2>			kckArr0c32;
typedef kckMat1<float2>			kckMat1c32;
typedef kckMat2<float2>			kckMat2c32;
typedef kckMat3<float2>			kckMat3c32;
typedef kckMat4<float2>			kckMat4c32;
typedef kckArr <f32xy>			kckArr0f32xy;
typedef kckMat1<f32xy>			kckMat1f32xy;
typedef kckMat2<f32xy>			kckMat2f32xy;
typedef kckMat3<f32xy>			kckMat3f32xy;
typedef kckMat4<f32xy>			kckMat4f32xy;
typedef kckArr <f32yz>			kckArr0f32yz;
typedef kckMat1<f32yz>			kckMat1f32yz;
typedef kckMat2<f32yz>			kckMat2f32yz;
typedef kckMat3<f32yz>			kckMat3f32yz;
typedef kckMat4<f32yz>			kckMat4f32yz;
typedef kckArr <f32zx>			kckArr0f32zx;
typedef kckMat1<f32zx>			kckMat1f32zx;
typedef kckMat2<f32zx>			kckMat2f32zx;
typedef kckMat3<f32zx>			kckMat3f32zx;
typedef kckMat4<f32zx>			kckMat4f32zx;
typedef kckArr <f32xyz>			kckArr0f32xyz;
typedef kckMat1<f32xyz>			kckMat1f32xyz;
typedef kckMat2<f32xyz>			kckMat2f32xyz;
typedef kckMat3<f32xyz>			kckMat3f32xyz;
typedef kckMat4<f32xyz>			kckMat4f32xyz;
typedef kckArr <half>			kckArr0f16;
typedef kckMat1<half>			kckMat1f16;
typedef kckMat2<half>			kckMat2f16;
typedef kckMat3<half>			kckMat3f16;
typedef kckMat4<half>			kckMat4f16;

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// additional class for kcMat

////////////////////////////////////////////////
// class cuda compatible stream
class kcStream : public kmArr<cudaStream_t>
{
public:
	int64 idx = 0;

	/////////////////////////////////////////////////
	// basic member functions
		
	// create stream... core
	void Create(int64 size)
	{
		kmArr::Create(size);
		
		// create stream
		for(int64 i = 0; i < _size; ++i)
		{
			cudaStreamCreate(&a(i));
		}
		KC_CHECK_ERROR("kcStream::Create");
	};

	// release stream... core
	virtual void Release()
	{	
		if(IsCreated() && _p) 
		{
			for(int64 i = 0; i < _size; ++i) cudaStreamDestroy(a(i));
			kmArr::Release();
			KC_CHECK_ERROR("kcStream::Release");
		}
	};

	// recreate
	void Recreate(int64 size) { Release(); Create(size); };

	// recreate if
	int RecreateIf(int64 size)
	{
		if(size != _size) { Recreate(size); return 1; } return 0;
	};

	// get next stream
	cudaStream_t GetNext() { if(idx == N()-1) idx = 0; else ++idx; return a(idx); };

	// get previous stream
	cudaStream_t GetPrev() { if(idx == 0) idx = N()-1; else --idx; return a(idx); };

	// get current stream
	int64        GetCurrIdx() { return idx; };
	cudaStream_t GetCurr   () { return a(idx); };

	// wait stream
	void WaitCurrStream()
	{
		if(N() > 0) cudaStreamSynchronize(GetCurr()); 
	};
	void WaitAllStreams()
	{
		for(int64 i = 0; i < N(); ++i) cudaStreamSynchronize(a(i));
	};
};

////////////////////////////////////////////////
// class cuda compatible texture object

// * Note that you must set and create kuTexobj to use it properly.
class kcTexObj
{
protected:
	cudaTextureObject_t	_tex         = 0;	
	int                 _is_created  = 0;
	cudaResourceDesc    _resDesc;
	cudaTextureDesc     _texDesc;
	
	void Init() 
	{	
		_tex			= 0;
		_is_created		= 0;
		
		// set texDesc and resDesc
		memset(&_resDesc, 0, sizeof(_resDesc));
		memset(&_texDesc, 0, sizeof(_texDesc));
	};

public:
	kcTexObj()	{ Init();		};
	~kcTexObj()	{ Release();	};

	///////////////////////////////////////////////
	// create and release
		
	// create
	void Create()
	{
		ASSERTFA(_is_created    == 0, "kcTexObj::Create in 3826");
		ASSERTFA(IsSetResDesc() == 1, "kcTexObj::Create in 3827");

		// create texture object
		cudaCreateTextureObject(&_tex, &_resDesc, &_texDesc, NULL);

		//__CUDA_ERROR_CHECK_D("kcTexObj::Create");

		// set created flag
		_is_created = 1;
	};

	// recreate
	void Recreate()
	{
		// release
		if(_is_created == 1) Release();
		
		// create
		Create();
	};

	// release
	void Release()
	{		
		if(_is_created == 1)
		{
			// destroy texture object
			cudaDestroyTextureObject(_tex);

			// set created flag
			_is_created = 0;
		}
	};

	///////////////////////////////////////////////
	// get function

	cudaTextureObject_t Get() const
	{
		ASSERTFA(_is_created == 1, "kcTexObj::Get");

		return _tex;
	};

	///////////////////////////////////////////////
	// set function
	// * Note that you can set one which has been already created,
	// * but you have to re-create it to apply changes.

	void Set(const kcMat3f32& mat)
	{
		// set rescource description
		SetResDesc2D(mat);
		//SetResDescLinear(mat);

		// set texture description
		// address_mode -  0: wrap, 1: clamp, 2: mirror, 3: border
		// filter_mode  -  0: point, 1: linear
		// read_mode    -  0: element, 1: normalized float
		// normal_mode  -  0: no normalized, 1: normalized
		SetTexDesc(1, 1, 0, 0);
	};

	///////////////////////////////////////////////
	// set resource description

	void SetResDescArray(cudaArray_t array)
	{
		_resDesc.resType			= cudaResourceTypeArray;	
		_resDesc.res.array.array	= array;
	};

	void SetResDescMipmap(cudaMipmappedArray_t mipmap)
	{
		_resDesc.resType			= cudaResourceTypeMipmappedArray;	
		_resDesc.res.mipmap.mipmap	= mipmap;
	};

	void SetResDescLinear(const kcMat3f32& mat)
	{
		_resDesc.resType		   = cudaResourceTypeLinear;
		_resDesc.res.linear.devPtr = mat.Begin();
		_resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
		_resDesc.res.linear.desc.x = 32; // bits per channel

		_resDesc.res.linear.sizeInBytes = mat.P1()*sizeof(float);
	};

	void SetResDescLinear( void *devPtr, 
							cudaChannelFormatKind f, 
							size_t sizeInBytes)
	{
		_resDesc.resType				= cudaResourceTypeLinear;
		_resDesc.res.linear.devPtr		= devPtr;
		_resDesc.res.linear.desc.f		= f;
		_resDesc.res.linear.desc.x		= 32; // bits per channel
		_resDesc.res.linear.sizeInBytes	= sizeInBytes;
	};

	void SetResDesc2D(void *devPtr, 
						cudaChannelFormatKind f, 
						size_t width, 
						size_t height, 
						size_t pitchInBytes)
	{		
		_resDesc.resType			= cudaResourceTypePitch2D;
		_resDesc.res.pitch2D.devPtr = devPtr;
		_resDesc.res.pitch2D.desc.f = f;
		_resDesc.res.pitch2D.width	= width;
		_resDesc.res.pitch2D.height = height;
		_resDesc.res.pitch2D.desc.x	= 32; // bits per channel

		_resDesc.res.pitch2D.pitchInBytes = pitchInBytes;
	};

	void SetResDesc2D(const kcMat3f32& mat)
	{
		_resDesc.resType			= cudaResourceTypePitch2D;
		_resDesc.res.pitch2D.devPtr = mat.Begin();
		_resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
		_resDesc.res.pitch2D.width	= mat.N1();
		_resDesc.res.pitch2D.height = mat.N2()*mat.N3();
		_resDesc.res.pitch2D.desc.x	= 32; // bits per channel

		_resDesc.res.pitch2D.pitchInBytes = mat.P1()*sizeof(float);
	};

	int IsSetResDesc()
	{
		switch(_resDesc.resType)
		{
		case cudaResourceTypeArray:			 if(_resDesc.res.array.array    == NULL) return 0; break;
		case cudaResourceTypeMipmappedArray: if(_resDesc.res.mipmap.mipmap  == NULL) return 0; break;
		case cudaResourceTypeLinear:		 if(_resDesc.res.linear.devPtr  == NULL) return 0; break;
		case cudaResourceTypePitch2D:		 if(_resDesc.res.pitch2D.devPtr == NULL) return 0; break;
		}
		return 1;
	};

	////////////////////////////////////////////////////////
	// set texture description

	void SetTexDesc(cudaTextureAddressMode addressMode,
					cudaTextureFilterMode  filterMode,   
					cudaTextureReadMode    readMode, 
					int normalizedCoords , int dim)
	{
		for(int i = 0; i < dim; ++i) _texDesc.addressMode[i]	= addressMode;		
		_texDesc.filterMode       = filterMode ;
		_texDesc.readMode         = readMode;
		_texDesc.normalizedCoords = normalizedCoords;
	};	

	void SetTexDesc(const int address_mode, const int filter_mode, 
		            const int read_mode   , const int normal_mode)
	{
		SetAddressMode(address_mode);
		SetFilterMode (filter_mode );
		SetReadMode   (read_mode   );
		SetNormalMode (normal_mode );
	};

	// set address mode
	// 0, cudaAddressModeWrap   : Wrapping address mode
	// 1, cudaAddressModeClamp  : Clamp to edge address mode
	// 2, cudaAddressModeMirror : Mirror address mode
	// 3, cudaAddressModeBorder : Border address mode 
	void SetAddressMode(const int address_mode)
	{
		_texDesc.addressMode[0] = (cudaTextureAddressMode) address_mode;
		_texDesc.addressMode[1] = (cudaTextureAddressMode) address_mode;
		_texDesc.addressMode[2] = (cudaTextureAddressMode) address_mode;
	};

	// set filter mode
	// 0, cudaFilterModePoint  : Point filter mode
	// 1, cudaFilterModeLinear : Linear filter mode
	void SetFilterMode(const int filter_mode)
	{
		_texDesc.filterMode = (cudaTextureFilterMode) filter_mode;
	};

	// set read mode
	// 0, cudaReadModeElementTypeRead : texture as specified element type
	// 1, cudaReadModeNormalizedFloat : Read texture as normalized float
	void SetReadMode(const int read_mode)
	{
		_texDesc.readMode = (cudaTextureReadMode) read_mode;
	};	

	// set normalized mode
	// 0 :  no normalization
	// 1 :  normalizization
	void SetNormalMode(const int normal_mode)
	{
		_texDesc.normalizedCoords = normal_mode;
	};
};

////////////////////////////////////////////////
// class GPU info with NVML (NVIDIA Management Library)
class kcNvml
{
protected:
	nvmlDevice_t _dev;
	nvmlReturn_t _ret;
	int          _is_created; // 1: init, 2: get handle

	void Init()
	{
		_dev = 0; _is_created = 0; _ret = NVML_SUCCESS;
	};

public:
	kcNvml() { Init(); };
	~kcNvml() { Release(); };

	////////////////////////////////
	// create and release
	void Create(int dev_idx = 0)
	{
		ASSERTFA(_is_created == 0, "kcNvml::Create in 4183");

		try
		{
			// init nvml
			_ret = nvmlInit(); KC_CHECK_ERROR_NVML("init");

			_is_created = 1;

			// get device handle
			_ret = nvmlDeviceGetHandleByIndex(dev_idx, &_dev);

			KC_CHECK_ERROR_NVML("get device handle");

			_is_created = 2;
		}
		catch (kmException e)
		{
			_is_created = 0;
			PRINTFA("* kcNvml::Created() catched the exception\n");
			kmPrintException(e);
		}
	};

	void Release()
	{
		if (_is_created > 0)
		{
			_ret = nvmlShutdown();	KC_CHECK_ERROR_NVML("shutdown");
			Init();
		}
	};

	////////////////////////////////////
	// set fucntion
	//
	// * Note that you must run exe with administrator mode to ues set-functions

	// set driver model (0 : WDDM, 1: WDM (TCC))
	void SetDriverModel(int driver_model)
	{
		if(_is_created != 2) return;
		
		switch (driver_model)
		{
		case 0 : _ret = nvmlDeviceSetDriverModel(_dev, NVML_DRIVER_WDDM, 0x00); break;
		case 1 : _ret = nvmlDeviceSetDriverModel(_dev, NVML_DRIVER_WDM , 0x00); break;
		default:
			PRINTFA("* nvml driver model (%d) is not supported\n", driver_model);
		}
		KC_CHECK_ERROR_NVML("SetDriverModel");
	};

	// set memory and gpu clocks
	void SetClocks(uint mem_clock_MHz, uint gpu_clock_MHz)
	{
		if(_is_created != 2) return;
		
		_ret = nvmlDeviceSetApplicationsClocks(_dev, mem_clock_MHz, gpu_clock_MHz);

		KC_CHECK_ERROR_NVML("SetClocks");
	};

	// set max memory and desired gpu clocks
	void SetClocksMaxMem(uint desired_gpu_clk_MHz)
	{
		if(_is_created != 2) return;

		kmMat1u32 mem_clk_MHz, gpu_clk_MHz;

		GetSupportedMemClock(mem_clk_MHz);
		GetSupportedGpuClock(gpu_clk_MHz, mem_clk_MHz(0));

		int i = 0; for(; i < gpu_clk_MHz.N(); ++i) if(desired_gpu_clk_MHz >= gpu_clk_MHz(i)) break;

		SetClocks(mem_clk_MHz(0), gpu_clk_MHz(i));

		PRINTFA("* SetClocksMaxMem : mem %d MHz, gpu %d MHz\n",
			    mem_clk_MHz(0), gpu_clk_MHz(i));
	};

	// set max memory and gpu clocks
	void SetClocksMax()
	{
		if(_is_created != 2) return;

		kmMat1u32 mem_clk_MHz, gpu_clk_MHz;

		GetSupportedMemClock(mem_clk_MHz);
		GetSupportedGpuClock(gpu_clk_MHz, mem_clk_MHz(0));

		SetClocks(mem_clk_MHz(0), gpu_clk_MHz(0));

		PRINTFA("* SetClocksMax : mem %d MHz, gpu %d MHz\n",
			    mem_clk_MHz(0), gpu_clk_MHz(0));
	};

	// reset clocks as default state
	void ResetClocks()
	{
		if(_is_created != 2) return;

		_ret = nvmlDeviceResetApplicationsClocks(_dev);

		KC_CHECK_ERROR_NVML("ResetClocks");
		PRINTFA("* ResetClocks : application clocks was reset\n");
	};

	////////////////////////////////////
	// get gpu info

	// get temperature of gpu in degree
	uint GetTemp()
	{
		if(_is_created != 2) return 0;

		uint temp = 0;
		_ret = nvmlDeviceGetTemperature(_dev, NVML_TEMPERATURE_GPU, &temp);

		KC_CHECK_ERROR_NVML("GetTemp");
		return temp;
	};

	// get fan speed in percentage
	uint GetFanSpeed()
	{
		if(_is_created != 2) return 0;

		uint speed = 0;
		_ret = nvmlDeviceGetFanSpeed(_dev, &speed);

		KC_CHECK_ERROR_NVML("GetFanSpeed");
		return speed;
	};

	// get temperature threshold of gpu in degree
	uint GetTempThres_ShutDown()
	{
		if(_is_created != 2) return 0;

		uint temp = 0;
		_ret = nvmlDeviceGetTemperatureThreshold(_dev, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, &temp);

		KC_CHECK_ERROR_NVML("GetTempThreshold");
		return temp;
	};

	// get temperature threshold of gpu in degree
	uint GetTempThres_SlowDown()
	{
		if(_is_created != 2) return 0;

		uint temp = 0;
		_ret = nvmlDeviceGetTemperatureThreshold(_dev, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &temp);

		KC_CHECK_ERROR_NVML("GetTempThreshold");
		return temp;
	};

	// get memory info in bytes
	// * Note that this is a time-consuming operation.
	void GetMemInfo(uint64* used, uint64* free, uint64* total)
	{
		if(_is_created != 2) return;

		nvmlMemory_t mem = {0};
		_ret = nvmlDeviceGetMemoryInfo(_dev, &mem);

		KC_CHECK_ERROR_NVML("GetMemInfo");

		*total = mem.total; *free = mem.free; *used = mem.used;
	};

	// get used memory of gpu in bytes
	// * Note that this is a time-consuming operation.
	uint64 GetMemUsed()
	{
		if(_is_created != 2) return 0;

		nvmlMemory_t mem = {0, };
		_ret = nvmlDeviceGetMemoryInfo(_dev, &mem);

		KC_CHECK_ERROR_NVML("GetMemUsed");
		return mem.used;
	};

	// get free memory of gpu in bytes
	// * Note that this is a time-consuming operation.
	uint64 GetMemFree()
	{
		if(_is_created != 2) return 0;

		nvmlMemory_t mem = { 0, };
		_ret = nvmlDeviceGetMemoryInfo(_dev, &mem);

		KC_CHECK_ERROR_NVML("GetMemFree");
		return mem.free;
	};

	// get total memory of gpu in bytes
	// * Note that this is a time-consuming operation.
	uint64 GetMemTotal()
	{
		if(_is_created != 2) return 0;

		nvmlMemory_t mem = { 0, };
		_ret = nvmlDeviceGetMemoryInfo(_dev, &mem);

		KC_CHECK_ERROR_NVML("GetMemTotal");
		return mem.total;
	};

	// get power of gpu in mW
	uint GetPower()
	{
		if(_is_created != 2) return 0;

		uint power_mW = 0;
		_ret = nvmlDeviceGetPowerUsage(_dev, &power_mW);

		KC_CHECK_ERROR_NVML("GetPower");
		return power_mW;
	};

	float GetPower_mW() { return (float)GetPower(); };
	float GetPower_W () { return (float)GetPower()*1e-3f; };

	// get limit power of gpu in mW
	uint GetPowerLimit()
	{
		if(_is_created != 2) return 0;

		uint power_mW = 0;
		_ret = nvmlDeviceGetPowerManagementLimit(_dev, &power_mW);

		KC_CHECK_ERROR_NVML("GetPowerLimit");
		return power_mW;
	};

	float GetPowerLimit_mW() { return (float)GetPowerLimit(); }
	float GetPowerLimit_W () { return (float)GetPowerLimit()*1e-3f; };

	// get max limit power of gpu in mW
	uint GetPowerMax()
	{
		if(_is_created != 2) return 0;

		uint min_mW = 0, max_mW = 0;
		_ret = nvmlDeviceGetPowerManagementLimitConstraints(_dev, &min_mW, &max_mW);

		KC_CHECK_ERROR_NVML("GetPowerMax");
		return max_mW;
	};

	float GetPowerMax_mW() { return (float)GetPowerMax(); }
	float GetPowerMax_W () { return (float)GetPowerMax()*1e-3f; };

	// get min limit power of gpu in mW
	uint GetPowerMin()
	{
		if(_is_created != 2) return 0;

		uint min_mW = 0, max_mW = 0;
		_ret = nvmlDeviceGetPowerManagementLimitConstraints(_dev, &min_mW, &max_mW);

		KC_CHECK_ERROR_NVML("GetPowerMin");
		return min_mW;
	};

	float GetPowerMin_mW() { return (float)GetPowerMin(); }
	float GetPowerMin_W () { return (float)GetPowerMin()*1e-3f; };

	// get default limit power of gpu in mW
	uint GetPowerDefault()
	{
		if(_is_created != 2) return 0;

		uint default_mW = 0;		
		_ret = nvmlDeviceGetPowerManagementDefaultLimit(_dev, &default_mW);

		KC_CHECK_ERROR_NVML("GetPowerDefault");
		return default_mW;
	};

	float GetPowerDefault_mW() { return (float)GetPowerDefault(); }
	float GetPowerDefault_W () { return (float)GetPowerDefault()*1e-3f; };

	// get driver model... 0 : WDDM, 1 : TCC
	uint GetDriverModel()
	{
		if(_is_created != 2) return 0;

		nvmlDriverModel_t curr, pend;
		_ret = nvmlDeviceGetDriverModel(_dev, &curr, &pend);

		KC_CHECK_ERROR_NVML("GetPowerDefault");
		return (uint)curr;
	};

	// get gpu usage rate
	uint GetUsage()
	{
		if(_is_created != 2) return 0;

		nvmlUtilization_t utz;
		_ret = nvmlDeviceGetUtilizationRates(_dev, &utz);

		KC_CHECK_ERROR_NVML("GetUsage");
		return utz.gpu;
	};

	// get gpu performance state
	uint GetState()
	{
		if(_is_created != 2) return 0;

		nvmlPstates_t state = (nvmlPstates_t)0;
		_ret = nvmlDeviceGetPerformanceState(_dev, &state);

		KC_CHECK_ERROR_NVML("GetState");
		return (uint) state;
	};

	// get gpu clock (MHz)
	uint GetGpuClock()
	{
		if(_is_created != 2) return 0;

		uint clock = 0;
		_ret = nvmlDeviceGetClock(_dev, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &clock);

		KC_CHECK_ERROR_NVML("GetGpuClock");
		return (uint) clock;
	};

	// get mem clock (MHz)
	uint GetMemClock()
	{
		if(_is_created != 2) return 0;

		uint clock = 0;
		_ret = nvmlDeviceGetClock(_dev, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &clock);

		KC_CHECK_ERROR_NVML("GetMemClock");
		return (uint) clock;
	};

	// get supported mem clock (MHz)
	void GetSupportedMemClock(kmMat1u32& clock_MHz)
	{
		if(_is_created != 2) return;

		uint n = 256, clock[256];
		_ret = nvmlDeviceGetSupportedMemoryClocks(_dev, &n, clock);

		if(_ret == NVML_SUCCESS)
		{
			clock_MHz.Recreate(n); clock_MHz.Copy(clock);
		}
		KC_CHECK_ERROR_NVML("GetSupportedMemClock");
	};

	// get supported gpu clock (MHz)
	void GetSupportedGpuClock(kmMat1u32& clock_MHz, uint mem_clock_MHz)
	{
		if(_is_created != 2) return;
		
		uint n = 256, clock[256];
		_ret = nvmlDeviceGetSupportedGraphicsClocks(_dev, mem_clock_MHz, &n, clock);

		if(_ret == NVML_SUCCESS)
		{
			clock_MHz.Recreate(n); clock_MHz.Copy(clock);
		}
		KC_CHECK_ERROR_NVML("GetSupportedGpuClock");
	};

	// is gpu clock max ?
	bool IsGpuClockMax()
	{
		kmMat1u32 mem_clk_MHz, gpu_clk_MHz; 
		
		GetSupportedMemClock(mem_clk_MHz);
		GetSupportedGpuClock(gpu_clk_MHz, mem_clk_MHz(0));

		return gpu_clk_MHz(0) == GetGpuClock();
	};

	//////////////////////////////////////////////////////////////////////////
	// display function
	void DisplayProcessInfo()
	{
		if(_is_created != 2) return;

		uint              n = 8;
		nvmlProcessInfo_t info[8] = { 0, };

		_ret = nvmlDeviceGetComputeRunningProcesses(_dev, &n, info);
		//__NVML_ERROR_CHECK_D("DisplayProcessInfo");

		if(_ret == NVML_SUCCESS)
		for(uint i = 0; i < n; ++i)
		{
			char name[64];
			nvmlSystemGetProcessName(info[i].pid, name, 63);

			PRINTFA("* %lu (%s): %llu MB\n",
				info[i].pid, name, info[i].usedGpuMemory >> 20);
		}
	};
};

// kcNvml's dummy class
class kcNvml_
{
protected:
	nvmlDevice_t _dev;
	nvmlReturn_t _ret;
	int          _is_created; // 1: init, 2: get handle

	void Init()
	{
		_dev = 0; _is_created = 0; _ret = NVML_SUCCESS;
	};

public:
	kcNvml_()  { Init(); };
	~kcNvml_() { Release(); };

	////////////////////////////////
	// create and release
	void Create(int dev_idx = 0) {};
	void Release() {};

	////////////////////////////////////
	// set fucntion
	//
	// * Note that you must run exe with administrator mode to ues set-functions
		
	void SetDriverModel (int driver_model) {};
	void SetClocks      (uint mem_clock_MHz, uint gpu_clock_MHz) {};
	void SetClocksMaxMem(uint desired_gpu_clk_MHz) {};
	void SetClocksMax   () {};
	void ResetClocks    () {};

	////////////////////////////////////
	// get gpu info
		
	uint GetTemp              () { return 0; };	
	uint GetFanSpeed          () { return 0; };
	uint GetTempThres_ShutDown() { return 0; };	
	uint GetTempThres_SlowDown() { return 0; };

	// get memory info in bytes	
	void GetMemInfo(uint64* used, uint64* free, uint64* total)
	{
		*used = (uint64)1e9; *free = (uint64)7e9; *total = (uint64)8e9;
	};
	uint64 GetMemUsed () { return (uint64)1e9; };
	uint64 GetMemFree () { return (uint64)7e9; };
	uint64 GetMemTotal() { return (uint64)8e9; };

	// get power of gpu in mW
	uint  GetPower()    { return 0; };
	float GetPower_mW() { return (float)GetPower(); };
	float GetPower_W () { return (float)GetPower()*1e-3f; };

	// get limit power of gpu in mW
	uint  GetPowerLimit()    { return 0; };
	float GetPowerLimit_mW() { return (float)GetPowerLimit(); }
	float GetPowerLimit_W () { return (float)GetPowerLimit()*1e-3f; };

	// get max limit power of gpu in mW
	uint  GetPowerMax()    { return 0; };
	float GetPowerMax_mW() { return (float)GetPowerMax(); }
	float GetPowerMax_W () { return (float)GetPowerMax()*1e-3f; };

	// get min limit power of gpu in mW
	uint  GetPowerMin()    { return 0; };
	float GetPowerMin_mW() { return (float)GetPowerMin(); }
	float GetPowerMin_W () { return (float)GetPowerMin()*1e-3f; };

	// get default limit power of gpu in mW
	uint  GetPowerDefault()    { return 0; };
	float GetPowerDefault_mW() { return (float)GetPowerDefault(); }
	float GetPowerDefault_W () { return (float)GetPowerDefault()*1e-3f; };

	// get driver model... 0 : WDDM, 1 : TCC
	uint GetDriverModel() { return 0; };
	uint GetUsage      () { return 0; };
	uint GetState      () { return 1; };

	// get gpu clock (MHz)
	uint GetGpuClock() { return 1000; };
	uint GetMemClock() { return 1000; };

	// get supported mem clock (MHz)
	void GetSupportedMemClock(kmMat1u32& clock_MHz)
	{
		clock_MHz.Recreate(1); clock_MHz(0) = 1000;
	};
	void GetSupportedGpuClock(kmMat1u32& clock_MHz, uint mem_clock_MHz)
	{
		GetSupportedMemClock(clock_MHz);
	};

	// is gpu clock max ?
	bool IsGpuClockMax() { return true; };

	//////////////////////////////////////////////////////////////////////////
	// display function
	void DisplayProcessInfo() { print("*** kcNvml is set as a dummy"); };
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// cuda device math functions

#define __indevice__ __inline__ __device__

__indevice__ float2& operator*=(float2& a, const float&  b) { a.x *= b;   a.y *= b;   return a; };
__indevice__ float2& operator/=(float2& a, const float&  b) { a.x /= b;   a.y /= b;   return a; };
__indevice__ float2& operator+=(float2& a, const float&  b) { a.x += b;   a.y += b;   return a; };
__indevice__ float2& operator-=(float2& a, const float&  b) { a.x -= b;   a.y -= b;   return a; };
__indevice__ float2& operator+=(float2& a, const float2& b) { a.x += b.x; a.y += b.y; return a; };
__indevice__ float2& operator-=(float2& a, const float2& b) { a.x -= b.x; a.y -= b.y; return a; };

__indevice__ float  sumsq(const float2  a) { return a.x*a.x + a.y*a.y; };
__indevice__ double sumsq(const double2 a) { return a.x*a.x + a.y*a.y; };

__indevice__ float  abs(const float2  a) { return sqrt(a.x*a.x + a.y*a.y); };
__indevice__ double abs(const double2 a) { return sqrt(a.x*a.x + a.y*a.y); };

__indevice__ float  ang(const float2  a) { return atan2(a.y, a.x); };
__indevice__ double ang(const double2 a) { return atan2(a.y, a.x); };

__indevice__ float hypot(const f32zx& a) { return hypot(a.z, a.x); };

__indevice__ float2 operator*(const float2& a, const float b) { float2 c; c.x = a.x * b; c.y = a.y * b; return c; };
__indevice__ float2 operator+(const float2& a, const float b) { float2 c; c.x = a.x + b; c.y = a.y + b; return c; };

__indevice__ f32zx operator+(const f32zx& a, const f32zx& b) { f32zx c; c.z = a.z + b.z; c.x = a.x + b.x; return c; };
__indevice__ f32zx operator-(const f32zx& a, const f32zx& b) { f32zx c; c.z = a.z - b.z; c.x = a.x - b.x; return c; };

__indevice__ f32zx operator*(const f32zx& a, const float b)  { f32zx c; c.z = a.z*b; c.x = a.x*b; return c; };
__indevice__ f32zx operator*(const float b, const f32zx& a)  { f32zx c; c.z = a.z*b; c.x = a.x*b; return c; };

// dot product
__indevice__ float operator*(const f32zx& a, const f32zx& b) { return a.z*b.z + a.x*b.x; };

template<typename T>
__indevice__ float subval(const T* a, const float idx)
{
	const float idx0 = floor(idx);
	const float rat1 = idx - idx0;

	const T val0 = *(a += (int)idx0);
	const T val1 = *(++a);

	return (float)(val0*( (1.0f) - rat1) + val1*rat1);
};

// interpolation function (ratio = 0.5)
__indevice__ float2 interp(const float2 a, const float2 b)
{
	const float abs_a = abs(a);
	const float abs_b = abs(b);
	const float abs_c = (abs_a + abs_b)*0.5f;

	const float ang_a  = ang(a);
	      float ang_b  = ang(b);
	const float ang_ab = ang_a - ang_b;

	if     (ang_ab >  PIHf) ang_b += PIf;
	else if(ang_ab < -PIHf) ang_b -= PIf;
		
	const float ang_c = (ang_a + ang_b)*0.5f;

	return float2({cos(ang_c)*abs_c, sin(ang_c)*abs_c});
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// standard xyz index of thread & block... new version

__indevice__ int GetStdIdx1(dim3 threadIdx, dim3 blockIdx, dim3 blockDim) { return threadIdx.x + blockDim.x*blockIdx.x; };
__indevice__ int GetStdIdx2(dim3 threadIdx, dim3 blockIdx, dim3 blockDim) { return threadIdx.y + blockDim.y*blockIdx.y; };
__indevice__ int GetStdIdx3(dim3 blockIdx)                                { return blockIdx.z; };
__indevice__ int GetStdIdx4(int& i3, int n3)
{
	int i4 = i3/n3; i3 -= i4*n3; return i4;
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// standard xyz index of thread & block... old version

// reduction define for no exdm... ix1, nx1, nx2
#define KC_RDC_INDEX_EXDM0(A)           \
	const int ix1 = threadIdx.x;        \
	const int nx1 = blockDim .x;        \
	const int nx2 = (A.n1 - 1)/nx1 + 1

// reduction define for one exdm... ix1, nx1, nx2, ie1
#define KC_RDC_INDEX_EXDM1(A)           \
	const int ix1 = threadIdx.x;        \
	const int nx1 = blockDim .x;        \
	const int nx2 = (A.n1 - 1)/nx1 + 1; \
	const int ie1 = blockIdx .x

// reduction define for two exdm... ix1, nx1, nx2, ie1, ie2
#define KC_RDC_INDEX_EXDM2(A)           \
	const int ix1 = threadIdx.x;        \
	const int nx1 = blockDim .x;        \
	const int nx2 = (A.n1 - 1)/nx1 + 1; \
	const int ie1 = blockIdx .x;        \
	const int ie2 = blockIdx .y

// reduction define for three exdm... ix1, nx1, nx2, ie1, ie2, ie3
#define KC_RDC_INDEX_EXDM3(A)           \
	const int ix1 = threadIdx.x;        \
	const int nx1 = blockDim .x;        \
	const int nx2 = (A.n1 - 1)/nx1 + 1; \
	const int ie1 = blockIdx .x;        \
	const int ie2 = blockIdx .y;        \
    const int ie3 = blockIdx .z

// reduction define for reduction
#define KC_RDC_REDUCTION(A) \
	sm[ix1] = A; \
	for(int size_s = nx1>>1; size_s > 0; size_s >>=1) \
	{ __syncthreads(); \
	if(ix1 < size_s) sm[ix1] += sm[ix1 + size_s]; }

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// math functions for kcMat

// casting function ... a = (T)b
kcArr0f32& TCast(kcArr0f32& a, const kcArr0i8 & b, cudaStream_t s = 0);
kcArr0f32& TCast(kcArr0f32& a, const kcArr0i16& b, cudaStream_t s = 0);
kcArr0f32& TCast(kcArr0f32& a, const kcArr0i32& b, cudaStream_t s = 0);
kcArr0f32& TCast(kcArr0f32& a, const kcArr0i64& b, cudaStream_t s = 0);
kcArr0f32& TCast(kcArr0f32& a, const kcArr0f16& b, cudaStream_t s = 0);
kcArr0f32& TCast(kcArr0f32& a, const kcArr0f64& b, cudaStream_t s = 0);
kcArr0f16& TCast(kcArr0f16& a, const kcArr0i8 & b, cudaStream_t s = 0);
kcArr0f16& TCast(kcArr0f16& a, const kcArr0i16& b, cudaStream_t s = 0);
kcArr0f16& TCast(kcArr0f16& a, const kcArr0i32& b, cudaStream_t s = 0);
kcArr0f16& TCast(kcArr0f16& a, const kcArr0i64& b, cudaStream_t s = 0);
kcArr0f16& TCast(kcArr0f16& a, const kcArr0f32& b, cudaStream_t s = 0);
kcArr0f16& TCast(kcArr0f16& a, const kcArr0f64& b, cudaStream_t s = 0);

kcMat1f32& TCast(kcMat1f32& a, const kcMat1i8 & b, cudaStream_t s = 0);
kcMat1f32& TCast(kcMat1f32& a, const kcMat1i16& b, cudaStream_t s = 0);
kcMat1f32& TCast(kcMat1f32& a, const kcMat1i32& b, cudaStream_t s = 0);
kcMat1f32& TCast(kcMat1f32& a, const kcMat1i64& b, cudaStream_t s = 0);
kcMat1f32& TCast(kcMat1f32& a, const kcMat1f16& b, cudaStream_t s = 0);
kcMat1f32& TCast(kcMat1f32& a, const kcMat1f64& b, cudaStream_t s = 0);
kcMat1f16& TCast(kcMat1f16& a, const kcMat1i8 & b, cudaStream_t s = 0);
kcMat1f16& TCast(kcMat1f16& a, const kcMat1i16& b, cudaStream_t s = 0);
kcMat1f16& TCast(kcMat1f16& a, const kcMat1i32& b, cudaStream_t s = 0);
kcMat1f16& TCast(kcMat1f16& a, const kcMat1i64& b, cudaStream_t s = 0);
kcMat1f16& TCast(kcMat1f16& a, const kcMat1f32& b, cudaStream_t s = 0);
kcMat1f16& TCast(kcMat1f16& a, const kcMat1f64& b, cudaStream_t s = 0);

kcMat2f32& TCast(kcMat2f32& a, const kcMat2i8 & b, cudaStream_t s = 0);
kcMat2f32& TCast(kcMat2f32& a, const kcMat2i16& b, cudaStream_t s = 0);
kcMat2f32& TCast(kcMat2f32& a, const kcMat2i32& b, cudaStream_t s = 0);
kcMat2f32& TCast(kcMat2f32& a, const kcMat2i64& b, cudaStream_t s = 0);
kcMat2f32& TCast(kcMat2f32& a, const kcMat2f16& b, cudaStream_t s = 0);
kcMat2f32& TCast(kcMat2f32& a, const kcMat2f64& b, cudaStream_t s = 0);
kcMat2f16& TCast(kcMat2f16& a, const kcMat2i8 & b, cudaStream_t s = 0);
kcMat2f16& TCast(kcMat2f16& a, const kcMat2i16& b, cudaStream_t s = 0);
kcMat2f16& TCast(kcMat2f16& a, const kcMat2i32& b, cudaStream_t s = 0);
kcMat2f16& TCast(kcMat2f16& a, const kcMat2i64& b, cudaStream_t s = 0);
kcMat2f16& TCast(kcMat2f16& a, const kcMat2f32& b, cudaStream_t s = 0);
kcMat2f16& TCast(kcMat2f16& a, const kcMat2f64& b, cudaStream_t s = 0);

kcMat3f32& TCast(kcMat3f32& a, const kcMat3i8 & b, cudaStream_t s = 0);
kcMat3f32& TCast(kcMat3f32& a, const kcMat3i16& b, cudaStream_t s = 0);
kcMat3f32& TCast(kcMat3f32& a, const kcMat3i32& b, cudaStream_t s = 0);
kcMat3f32& TCast(kcMat3f32& a, const kcMat3i64& b, cudaStream_t s = 0);
kcMat3f32& TCast(kcMat3f32& a, const kcMat3f16& b, cudaStream_t s = 0);
kcMat3f32& TCast(kcMat3f32& a, const kcMat3f64& b, cudaStream_t s = 0);
kcMat3f16& TCast(kcMat3f16& a, const kcMat3i8 & b, cudaStream_t s = 0);
kcMat3f16& TCast(kcMat3f16& a, const kcMat3i16& b, cudaStream_t s = 0);
kcMat3f16& TCast(kcMat3f16& a, const kcMat3i32& b, cudaStream_t s = 0);
kcMat3f16& TCast(kcMat3f16& a, const kcMat3i64& b, cudaStream_t s = 0);
kcMat3f16& TCast(kcMat3f16& a, const kcMat3f32& b, cudaStream_t s = 0);
kcMat3f16& TCast(kcMat3f16& a, const kcMat3f64& b, cudaStream_t s = 0);

kcMat4f32& TCast(kcMat4f32& a, const kcMat4i8 & b, cudaStream_t s = 0);
kcMat4f32& TCast(kcMat4f32& a, const kcMat4i16& b, cudaStream_t s = 0);
kcMat4f32& TCast(kcMat4f32& a, const kcMat4i32& b, cudaStream_t s = 0);
kcMat4f32& TCast(kcMat4f32& a, const kcMat4i64& b, cudaStream_t s = 0);
kcMat4f32& TCast(kcMat4f32& a, const kcMat4f16& b, cudaStream_t s = 0);
kcMat4f32& TCast(kcMat4f32& a, const kcMat4f64& b, cudaStream_t s = 0);
kcMat4f16& TCast(kcMat4f16& a, const kcMat4i8 & b, cudaStream_t s = 0);
kcMat4f16& TCast(kcMat4f16& a, const kcMat4i16& b, cudaStream_t s = 0);
kcMat4f16& TCast(kcMat4f16& a, const kcMat4i32& b, cudaStream_t s = 0);
kcMat4f16& TCast(kcMat4f16& a, const kcMat4i64& b, cudaStream_t s = 0);
kcMat4f16& TCast(kcMat4f16& a, const kcMat4f32& b, cudaStream_t s = 0);
kcMat4f16& TCast(kcMat4f16& a, const kcMat4f64& b, cudaStream_t s = 0);

// sigmoid function... a = sigmoid(b)
kcMat1f32& Sigmoid(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s = 0);
kcMat2f32& Sigmoid(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s = 0);
kcMat3f32& Sigmoid(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s = 0);
kcMat4f32& Sigmoid(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s = 0);
kcMat1f32  Sigmoid(              const kcMat1f32& b, cudaStream_t s = 0);
kcMat2f32  Sigmoid(              const kcMat2f32& b, cudaStream_t s = 0);
kcMat3f32  Sigmoid(              const kcMat3f32& b, cudaStream_t s = 0);
kcMat4f32  Sigmoid(              const kcMat4f32& b, cudaStream_t s = 0);

kcMat1f16& Sigmoid(kcMat1f16& a, const kcMat1f16& b, cudaStream_t s = 0);
kcMat2f16& Sigmoid(kcMat2f16& a, const kcMat2f16& b, cudaStream_t s = 0);
kcMat3f16& Sigmoid(kcMat3f16& a, const kcMat3f16& b, cudaStream_t s = 0);
kcMat4f16& Sigmoid(kcMat4f16& a, const kcMat4f16& b, cudaStream_t s = 0);
kcMat1f16  Sigmoid(              const kcMat1f16& b, cudaStream_t s = 0);
kcMat2f16  Sigmoid(              const kcMat2f16& b, cudaStream_t s = 0);
kcMat3f16  Sigmoid(              const kcMat3f16& b, cudaStream_t s = 0);
kcMat4f16  Sigmoid(              const kcMat4f16& b, cudaStream_t s = 0);

// softmax function... a = softmax(b)
kcMat1f32& Softmax(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s = 0);
kcMat2f32& Softmax(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s = 0);
kcMat3f32& Softmax(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s = 0);
kcMat4f32& Softmax(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s = 0);
kcMat1f32  Softmax(              const kcMat1f32& b, cudaStream_t s = 0);
kcMat2f32  Softmax(              const kcMat2f32& b, cudaStream_t s = 0);
kcMat3f32  Softmax(              const kcMat3f32& b, cudaStream_t s = 0);
kcMat4f32  Softmax(              const kcMat4f32& b, cudaStream_t s = 0);

// ReLU function... a = ReLU(b)
kcMat1f32& ReLU(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s = 0);
kcMat2f32& ReLU(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s = 0);
kcMat3f32& ReLU(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s = 0);
kcMat4f32& ReLU(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s = 0);
kcMat1f32  ReLU(              const kcMat1f32& b, cudaStream_t s = 0);
kcMat2f32  ReLU(              const kcMat2f32& b, cudaStream_t s = 0);
kcMat3f32  ReLU(              const kcMat3f32& b, cudaStream_t s = 0);
kcMat4f32  ReLU(              const kcMat4f32& b, cudaStream_t s = 0);

kcMat1f16& ReLU(kcMat1f16& a, const kcMat1f16& b, cudaStream_t s = 0);
kcMat2f16& ReLU(kcMat2f16& a, const kcMat2f16& b, cudaStream_t s = 0);
kcMat3f16& ReLU(kcMat3f16& a, const kcMat3f16& b, cudaStream_t s = 0);
kcMat4f16& ReLU(kcMat4f16& a, const kcMat4f16& b, cudaStream_t s = 0);
kcMat1f16  ReLU(              const kcMat1f16& b, cudaStream_t s = 0);
kcMat2f16  ReLU(              const kcMat2f16& b, cudaStream_t s = 0);
kcMat3f16  ReLU(              const kcMat3f16& b, cudaStream_t s = 0);
kcMat4f16  ReLU(              const kcMat4f16& b, cudaStream_t s = 0);

// exp function... a = e^(b)
kcMat1f32& Exp(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s = 0);
kcMat2f32& Exp(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s = 0);
kcMat3f32& Exp(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s = 0);
kcMat4f32& Exp(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s = 0);
kcMat1f32  Exp(              const kcMat1f32& b, cudaStream_t s = 0);
kcMat2f32  Exp(              const kcMat2f32& b, cudaStream_t s = 0);
kcMat3f32  Exp(              const kcMat3f32& b, cudaStream_t s = 0);
kcMat4f32  Exp(              const kcMat4f32& b, cudaStream_t s = 0);

// AddEq function... a += b
kcMat1f32& AddEq(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s = 0);
kcMat2f32& AddEq(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s = 0);
kcMat3f32& AddEq(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s = 0);
kcMat4f32& AddEq(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s = 0);

kcMat1f32& operator+=(kcMat1f32& a, const kcMat1f32& b);
kcMat2f32& operator+=(kcMat2f32& a, const kcMat2f32& b);
kcMat3f32& operator+=(kcMat3f32& a, const kcMat3f32& b);
kcMat4f32& operator+=(kcMat4f32& a, const kcMat4f32& b);
kcMat1f32& operator+=(kcMat1f32& a, const float      b);
kcMat2f32& operator+=(kcMat2f32& a, const float      b);
kcMat3f32& operator+=(kcMat3f32& a, const float      b);
kcMat4f32& operator+=(kcMat4f32& a, const float      b);

kcMat1f16& operator+=(kcMat1f16& a, const kcMat1f16& b);
kcMat2f16& operator+=(kcMat2f16& a, const kcMat2f16& b);
kcMat3f16& operator+=(kcMat3f16& a, const kcMat3f16& b);
kcMat4f16& operator+=(kcMat4f16& a, const kcMat4f16& b);

// SubEq function... a -= b
kcMat1f32& SubEq(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s = 0);
kcMat2f32& SubEq(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s = 0);
kcMat3f32& SubEq(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s = 0);
kcMat4f32& SubEq(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s = 0);

kcMat1f32& operator-=(kcMat1f32& a, const kcMat1f32& b);
kcMat2f32& operator-=(kcMat2f32& a, const kcMat2f32& b);
kcMat3f32& operator-=(kcMat3f32& a, const kcMat3f32& b);
kcMat4f32& operator-=(kcMat4f32& a, const kcMat4f32& b);

// MulEq function... a *= b
kcMat1f32& MulEq(kcMat1f32& a, const kcMat1f32& b, cudaStream_t s = 0);
kcMat2f32& MulEq(kcMat2f32& a, const kcMat2f32& b, cudaStream_t s = 0);
kcMat3f32& MulEq(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s = 0);
kcMat4f32& MulEq(kcMat4f32& a, const kcMat4f32& b, cudaStream_t s = 0);
kcMat1f32& MulEq(kcMat1f32& a, const float      b, cudaStream_t s = 0);
kcMat2f32& MulEq(kcMat2f32& a, const float      b, cudaStream_t s = 0);
kcMat3f32& MulEq(kcMat3f32& a, const float      b, cudaStream_t s = 0);
kcMat4f32& MulEq(kcMat4f32& a, const float      b, cudaStream_t s = 0);

kcMat1f32& operator*=(kcMat1f32& a, const kcMat1f32& b);
kcMat2f32& operator*=(kcMat2f32& a, const kcMat2f32& b);
kcMat3f32& operator*=(kcMat3f32& a, const kcMat3f32& b);
kcMat4f32& operator*=(kcMat4f32& a, const kcMat4f32& b);
kcMat1f32& operator*=(kcMat1f32& a, const float      b);
kcMat2f32& operator*=(kcMat2f32& a, const float      b);
kcMat3f32& operator*=(kcMat3f32& a, const float      b);
kcMat4f32& operator*=(kcMat4f32& a, const float      b);

// matrix multiplication and addition function... a = b*c
kcMat2f32& Dot(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s = 0);

// matrix multiplication and addition function... a = b*c... shared memory version
kcMat2f32& DotS1(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s = 0);
kcMat2f32& DotS2(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s = 0);
#ifdef __CUDA_IMMA__

kcMat2f32& DotS3(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s = 0);

#endif

// matrix multiplication and addition function... a = b*c + d
kcMat2f32& DotAdd(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, const kcMat2f32& d, cudaStream_t s = 0);
kcMat3f32& DotAdd(kcMat3f32& a, const kcMat3f32& b, const kcMat3f32& c, const kcMat3f32& d, cudaStream_t s = 0);
kcMat4f32& DotAdd(kcMat4f32& a, const kcMat4f32& b, const kcMat4f32& c, const kcMat4f32& d, cudaStream_t s = 0);

kcMat2f32  DotAdd(              const kcMat2f32& b, const kcMat2f32& c, const kcMat2f32& d, cudaStream_t s = 0);
kcMat3f32  DotAdd(              const kcMat3f32& b, const kcMat3f32& c, const kcMat3f32& d, cudaStream_t s = 0);
kcMat4f32  DotAdd(              const kcMat4f32& b, const kcMat4f32& c, const kcMat4f32& d, cudaStream_t s = 0);

// addition function... a = b + c
kcMat1f32& Add(kcMat1f32& a, const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s = 0);
kcMat2f32& Add(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s = 0);
kcMat3f32& Add(kcMat3f32& a, const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s = 0);
kcMat4f32& Add(kcMat4f32& a, const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s = 0);
kcMat1f16& Add(kcMat1f16& a, const kcMat1f16& b, const kcMat1f16& c, cudaStream_t s = 0);
kcMat2f16& Add(kcMat2f16& a, const kcMat2f16& b, const kcMat2f16& c, cudaStream_t s = 0);
kcMat3f16& Add(kcMat3f16& a, const kcMat3f16& b, const kcMat3f16& c, cudaStream_t s = 0);
kcMat4f16& Add(kcMat4f16& a, const kcMat4f16& b, const kcMat4f16& c, cudaStream_t s = 0);

template<typename T> kcMat1<T> Add(const kcMat1<T>& b, const kcMat1<T>& c, cudaStream_t s = 0) { kcMat1<T> a; Add(a, b, c, s); return a;}
template<typename T> kcMat2<T> Add(const kcMat2<T>& b, const kcMat2<T>& c, cudaStream_t s = 0) { kcMat2<T> a; Add(a, b, c, s); return a;}
template<typename T> kcMat3<T> Add(const kcMat3<T>& b, const kcMat3<T>& c, cudaStream_t s = 0) { kcMat3<T> a; Add(a, b, c, s); return a;}
template<typename T> kcMat4<T> Add(const kcMat4<T>& b, const kcMat4<T>& c, cudaStream_t s = 0) { kcMat4<T> a; Add(a, b, c, s); return a;}

template<typename T> kcMat1<T> operator+(const kcMat1<T>& a, const kcMat1<T>& b) { return Add(a, b); }
template<typename T> kcMat2<T> operator+(const kcMat2<T>& a, const kcMat2<T>& b) { return Add(a, b); }
template<typename T> kcMat3<T> operator+(const kcMat3<T>& a, const kcMat3<T>& b) { return Add(a, b); }
template<typename T> kcMat4<T> operator+(const kcMat4<T>& a, const kcMat4<T>& b) { return Add(a, b); }

// sum... a = Sum(b)
// - kcMat1 a = Sum(kcMat2 b);
//kcMat1f32& Sum(kcMat1f32& a, const kcMat3f32& b, cudaStream_t s = 0);

// sum... a = Sum(b)
float Sum(const kcMat1f32& b, cudaStream_t s = 0);
float Sum(const kcMat2f32& b, cudaStream_t s = 0);
float Sum(const kcMat3f32& b, cudaStream_t s = 0);
float Sum(const kcMat4f32& b, cudaStream_t s = 0);

// mean... a = Mean(b)
float Mean(const kcMat1f32& b, cudaStream_t s = 0);
float Mean(const kcMat2f32& b, cudaStream_t s = 0);
float Mean(const kcMat3f32& b, cudaStream_t s = 0);
float Mean(const kcMat4f32& b, cudaStream_t s = 0);

// sum of square... a = SumSq(b)
float SumSq(const kcMat1f32& b, cudaStream_t s = 0);
float SumSq(const kcMat2f32& b, cudaStream_t s = 0);
float SumSq(const kcMat3f32& b, cudaStream_t s = 0);
float SumSq(const kcMat4f32& b, cudaStream_t s = 0);

// variance... a = Var(b)
float Var(const kcMat1f32& b, cudaStream_t s = 0);
float Var(const kcMat2f32& b, cudaStream_t s = 0);
float Var(const kcMat3f32& b, cudaStream_t s = 0);
float Var(const kcMat4f32& b, cudaStream_t s = 0);

// standard deviation... a = Std(b)
float Std(const kcMat1f32& b, cudaStream_t s = 0);
float Std(const kcMat2f32& b, cudaStream_t s = 0);
float Std(const kcMat3f32& b, cudaStream_t s = 0);
float Std(const kcMat4f32& b, cudaStream_t s = 0);

// max... a = Max(b)
float Max(const kcMat1f32& b, cudaStream_t s = 0);
float Max(const kcMat2f32& b, cudaStream_t s = 0);
float Max(const kcMat3f32& b, cudaStream_t s = 0);
float Max(const kcMat4f32& b, cudaStream_t s = 0);

// min... a = Min(b)
float Min(const kcMat1f32& b, cudaStream_t s = 0);
float Min(const kcMat2f32& b, cudaStream_t s = 0);
float Min(const kcMat3f32& b, cudaStream_t s = 0);
float Min(const kcMat4f32& b, cudaStream_t s = 0);

// subeqmul... a -= b*c
kcMat1f32& SubEqMul(kcMat1f32& a, const kcMat1f32& b, const float c, cudaStream_t s = 0);
kcMat2f32& SubEqMul(kcMat2f32& a, const kcMat2f32& b, const float c, cudaStream_t s = 0);
kcMat3f32& SubEqMul(kcMat3f32& a, const kcMat3f32& b, const float c, cudaStream_t s = 0);
kcMat4f32& SubEqMul(kcMat4f32& a, const kcMat4f32& b, const float c, cudaStream_t s = 0);

// gradient of mean square error
kcMat1f32& GradMse(kcMat1f32& a, const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s = 0);
kcMat2f32& GradMse(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s = 0);
kcMat3f32& GradMse(kcMat3f32& a, const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s = 0);
kcMat4f32& GradMse(kcMat4f32& a, const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s = 0);
kcMat1f16& GradMse(kcMat1f16& a, const kcMat1f16& b, const kcMat1f16& c, cudaStream_t s = 0);
kcMat2f16& GradMse(kcMat2f16& a, const kcMat2f16& b, const kcMat2f16& c, cudaStream_t s = 0);
kcMat3f16& GradMse(kcMat3f16& a, const kcMat3f16& b, const kcMat3f16& c, cudaStream_t s = 0);
kcMat4f16& GradMse(kcMat4f16& a, const kcMat4f16& b, const kcMat4f16& c, cudaStream_t s = 0);

// gradient of cross entroy error
kcMat1f32& GradCee(kcMat1f32& a, const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s = 0);
kcMat2f32& GradCee(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s = 0);
kcMat3f32& GradCee(kcMat3f32& a, const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s = 0);
kcMat4f32& GradCee(kcMat4f32& a, const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s = 0);
kcMat1f16& GradCee(kcMat1f16& a, const kcMat1f16& b, const kcMat1f16& c, cudaStream_t s = 0);
kcMat2f16& GradCee(kcMat2f16& a, const kcMat2f16& b, const kcMat2f16& c, cudaStream_t s = 0);
kcMat3f16& GradCee(kcMat3f16& a, const kcMat3f16& b, const kcMat3f16& c, cudaStream_t s = 0);
kcMat4f16& GradCee(kcMat4f16& a, const kcMat4f16& b, const kcMat4f16& c, cudaStream_t s = 0);

// square error
kcMat1f32& Sqe(kcMat1f32& a, const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s = 0);
kcMat2f32& Sqe(kcMat2f32& a, const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s = 0);
kcMat3f32& Sqe(kcMat3f32& a, const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s = 0);
kcMat4f32& Sqe(kcMat4f32& a, const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s = 0);

// mean square error 
float Mse(const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s = 0);
float Mse(const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s = 0);
float Mse(const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s = 0);
float Mse(const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s = 0);

// cross entropy error 
float Cee(const kcMat1f32& b, const kcMat1f32& c, cudaStream_t s = 0);
float Cee(const kcMat2f32& b, const kcMat2f32& c, cudaStream_t s = 0);
float Cee(const kcMat3f32& b, const kcMat3f32& c, cudaStream_t s = 0);
float Cee(const kcMat4f32& b, const kcMat4f32& c, cudaStream_t s = 0);

// Hilbert trasform : a is compliex of b (b is real signal)
// * n_tap_h is fixed with 32.
// * This allows over 0.7MHz signal at 45MHz sampling.
void Hilbert(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s = 0);
void Hilbert(kcMat3c32& a, const kcMat3f32& b, cudaStream_t s = 0);

// Hibert transform : a is abs of b (b is real signal)
// * n_tap_h is fixed with 32.
// * This allows over 0.7MHz signal at 45MHz sampling.
void HilbertAbs(kcMat3f32& a, const kcMat3f32& b, cudaStream_t s = 0);

// IqDemodulate
void IqDemodulate(kcMat3c32& a, const kcMat3f32& b, float fs_MHz, float fd_MHz, cudaStream_t s = 0);

// filering
void Filter1DX1_N(kcMat2f32& a, const kcMat2f32& b, const kcMat1f32& k, cudaStream_t s = 0);
void Filter1DX1_N(kcMat2c32& a, const kcMat2c32& b, const kcMat1f32& k, cudaStream_t s = 0);

// down sampling
void Downsample(kcMat3f32& a, const kcMat3f32& b, const int downrate, cudaStream_t s = 0);
void Downsample(kcMat3c32& a, const kcMat3c32& b, const int downrate, cudaStream_t s = 0);

// log-compression
void LogCompression(kcMat2f32& a, const kcMat2c32& b, cudaStream_t s = 0);
void LogCompression(kcMat3f32& a, const kcMat3c32& b, cudaStream_t s = 0);
void LogCompression(kcMat3f32& a, const kcMat3q16& b, cudaStream_t s = 0);

kcMat2f32 LogCompression(const kcMat2c32& b, cudaStream_t s = 0);
kcMat3f32 LogCompression(const kcMat3c32& b, cudaStream_t s = 0);
kcMat3f32 LogCompression(const kcMat3q16& b, cudaStream_t s = 0);

// abs for complex
void Abs(kcMat2f32& a, const kcMat2c32& b, cudaStream_t s = 0);
void Abs(kcMat3f32& a, const kcMat3c32& b, cudaStream_t s = 0);
void Abs(kcMat3f32& a, const kcMat3q16& b, cudaStream_t s = 0);

kcMat2f32 Abs(const kcMat2c32& b, cudaStream_t s = 0);
kcMat3f32 Abs(const kcMat3c32& b, cudaStream_t s = 0);
kcMat3f32 Abs(const kcMat3q16& b, cudaStream_t s = 0);

// set diagonal stripes
void SetDiagStripe(kcMat2c32& a, float vs, float ve, int cnt0 = 0, cudaStream_t s = 0);

// color coding
void ConvertBgrTp(kcMat2<kmBgr>& img, const kcMat2f32& a, float cmin, float cmax, const kcMat1<kmBgr>& cmap, cudaStream_t s = 0);

// decompose complex
// * complex (float2) --> real, imaginary, magnitude, (phase)
void DecomposeCmplx(kcMat2f32& re, kcMat2f32& im, kcMat2f32& ma, const kcMat2c32& cmplx, cudaStream_t s = 0);
void DecomposeCmplx(kcMat3f32& re, kcMat3f32& im, kcMat3f32& ma, const kcMat3c32& cmplx, cudaStream_t s = 0);

// contrast with Laplacian
float ContrastLp (const kcMat2f32& a, cudaStream_t s = 0);

// contrast with Sobel
float ContrastSobel(const kcMat2f32& a, cudaStream_t s = 0);

#endif /* __kc7Mat_H_INCLUDED_2019_09_10__ */
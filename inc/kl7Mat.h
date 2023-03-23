#ifndef __kl7Mat_H_INCLUDED_2021_09_28__
#define __kl7Mat_H_INCLUDED_2021_09_28__

/* Note ----------------------
* kmMat has been created by Choi, Kiwan
* This is version 7
* kmMat v7 is including the following
*   - km7Define.h
*   - km7Define.h -> km7Mat.h
*   - km7Define.h -> km7Mat.h -> km7Wnd.h
*   - km7Define.h -> km7Mat.h -> km7Dnn.h
*   - km7Define.h -> km7Mat.h -> kc7Mat.h
*   - km7Define.h -> km7Mat.h -> kl7Mat.h
*/

// base header
#include "km7Mat.h"

// openCL header
#include "CL/cl.h"

// link lib
#pragma comment(lib, "opencl.lib")

#define KL7MAT

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// enum for klMat

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// definition for klMat

// check kl result
void klCheckRes(cl_int ret,  LPCSTR str = nullptr)
{
	if(ret == CL_SUCCESS)
	{
		if(str != nullptr) print("* %s : success\n", str);
	}
	else
	{
#define CASE_STD_RUNTIME_ERROR(A) case A : throw runtime_error(kmStra(" %s ("#A")",str).P())
		//#define CASE_STD_RUNTIME_ERROR(A) case A : print("* %s : "#A"\n", str); break

		switch(ret)
		{
			CASE_STD_RUNTIME_ERROR( CL_SUCCESS                                  );
			CASE_STD_RUNTIME_ERROR( CL_DEVICE_NOT_FOUND                         );
			CASE_STD_RUNTIME_ERROR( CL_DEVICE_NOT_AVAILABLE                     );
			CASE_STD_RUNTIME_ERROR( CL_COMPILER_NOT_AVAILABLE                   );
			CASE_STD_RUNTIME_ERROR( CL_MEM_OBJECT_ALLOCATION_FAILURE            );
			CASE_STD_RUNTIME_ERROR( CL_OUT_OF_RESOURCES                         );
			CASE_STD_RUNTIME_ERROR( CL_OUT_OF_HOST_MEMORY                       );
			CASE_STD_RUNTIME_ERROR( CL_PROFILING_INFO_NOT_AVAILABLE             );
			CASE_STD_RUNTIME_ERROR( CL_MEM_COPY_OVERLAP                         );
			CASE_STD_RUNTIME_ERROR( CL_IMAGE_FORMAT_MISMATCH                    );
			CASE_STD_RUNTIME_ERROR( CL_IMAGE_FORMAT_NOT_SUPPORTED               );
			CASE_STD_RUNTIME_ERROR( CL_BUILD_PROGRAM_FAILURE                    );
			CASE_STD_RUNTIME_ERROR( CL_MAP_FAILURE                              );
			CASE_STD_RUNTIME_ERROR( CL_MISALIGNED_SUB_BUFFER_OFFSET             );
			CASE_STD_RUNTIME_ERROR( CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
			CASE_STD_RUNTIME_ERROR( CL_COMPILE_PROGRAM_FAILURE                  );
			CASE_STD_RUNTIME_ERROR( CL_LINKER_NOT_AVAILABLE                     );
			CASE_STD_RUNTIME_ERROR( CL_LINK_PROGRAM_FAILURE                     );
			CASE_STD_RUNTIME_ERROR( CL_DEVICE_PARTITION_FAILED                  );
			CASE_STD_RUNTIME_ERROR( CL_KERNEL_ARG_INFO_NOT_AVAILABLE            );

			CASE_STD_RUNTIME_ERROR( CL_INVALID_VALUE                  );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_DEVICE_TYPE            );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_PLATFORM               );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_DEVICE                 );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_CONTEXT                );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_QUEUE_PROPERTIES       );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_COMMAND_QUEUE          );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_HOST_PTR               );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_MEM_OBJECT             );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
			CASE_STD_RUNTIME_ERROR( CL_INVALID_IMAGE_SIZE             );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_SAMPLER                );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_BINARY                 );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_BUILD_OPTIONS          );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_PROGRAM                );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_PROGRAM_EXECUTABLE     );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_KERNEL_NAME            );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_KERNEL_DEFINITION      );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_KERNEL                 );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_ARG_INDEX              );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_ARG_VALUE              );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_ARG_SIZE               );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_KERNEL_ARGS            );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_WORK_DIMENSION         );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_WORK_GROUP_SIZE        );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_WORK_ITEM_SIZE         );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_GLOBAL_OFFSET          );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_EVENT_WAIT_LIST        );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_EVENT                  );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_OPERATION              );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_GL_OBJECT              );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_BUFFER_SIZE            );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_MIP_LEVEL              );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_GLOBAL_WORK_SIZE       );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_PROPERTY               );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_IMAGE_DESCRIPTOR       );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_COMPILER_OPTIONS       );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_LINKER_OPTIONS         );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_DEVICE_PARTITION_COUNT );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_PIPE_SIZE              );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_DEVICE_QUEUE           );
			CASE_STD_RUNTIME_ERROR( CL_INVALID_SPEC_ID                );
			CASE_STD_RUNTIME_ERROR( CL_MAX_SIZE_RESTRICTION_EXCEEDED  );

		default: throw runtime_error(kmStra(" %s (unknow)",str).P());
		}
	}
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// class for klMat

///////////////////////////////////////////////////////
// opencl matrix base class to register context and command queue
class klBase
{
public:
	static cl_context       _cnt;
	static cl_command_queue _que;
	static void Register(cl_context cnt, cl_command_queue que) { _cnt = cnt; _que = que; };
};
cl_context       klBase::_cnt = nullptr;
cl_command_queue klBase::_que = nullptr;

///////////////////////////////////////////////////////
// openCl context class
class klCnt
{
public:
	cl_context               _cnt{};  // opencl context
	cl_platform_id           _plf{};  // platform id
	cl_device_id             _dev{};  // device id
	kmMat1<cl_command_queue> _ques;   // command queue
	kmMat1<cl_program>       _prgs;   // programs
	kmMat1<cl_kernel>        _krns;   // kernels

	// constructor

	// destructor
	virtual ~klCnt() { Release(); };

	//////////////////////////////////////////////////
	// member functions

	// display availble platforms and devices
	klCnt& DispPlfsDevs()
	{
		cl_int res = 0;

		// get platforms
		kmMat1<cl_platform_id> plfs; cl_uint plf_n = 0;

		res = clGetPlatformIDs(    0,     NULL, &plf_n); plfs.Create(plf_n);
		res = clGetPlatformIDs(plf_n, plfs.P(), &plf_n);

		// get platform info
		size_t str_n = 128; kmStra str(str_n);

		for(uint i = 0; i < plf_n; ++i)
		{
			res = clGetPlatformInfo(plfs(i), CL_PLATFORM_NAME, str.Size(), str.P(), &str_n);

			print("* platform[%d] %s\n", i, str.P());
			
			// get device ids
			kmMat1<cl_device_id> devs(0); cl_uint dev_n = 0;

			res = clGetDeviceIDs(plfs(i), CL_DEVICE_TYPE_GPU,     0,     NULL, &dev_n); devs.Create(dev_n);
			res = clGetDeviceIDs(plfs(i), CL_DEVICE_TYPE_GPU, dev_n, devs.P(), &dev_n);

			// get device info
			for(uint j = 0; j < dev_n; ++j)
			{
				res = clGetDeviceInfo(devs(j), CL_DEVICE_NAME, str.Size(), str.P(), &str_n);

				print("     - device[%d] %s\n", j, str.P());
			}
		}
		return *this;
	};

	// init opencl device
	klCnt& Init(int plf_idx = 0, int dev_idx = 0, int que_n = 1)
	{
		// init parameters
		cl_int res   = 0;
		size_t str_n = 128; kmStra str(str_n);

		///////////////////////////////////////////////
		// get platform id... _plf
		
		// get platform ids
		kmMat1<cl_platform_id> plfs; cl_uint plf_n = 0;

		res = clGetPlatformIDs(    0,     NULL, &plf_n); plfs.Create(plf_n);
		res = clGetPlatformIDs(plf_n, plfs.P(), &plf_n);

		// choose platform
		_plf = plfs(plf_idx);

		///////////////////////////////////////////////
		// get device id... _dev

		// get device ids
		kmMat1<cl_device_id> devs; cl_uint dev_n = 0;

		res = clGetDeviceIDs(_plf, CL_DEVICE_TYPE_GPU,     0,     NULL, &dev_n); devs.Create(dev_n);
		res = clGetDeviceIDs(_plf, CL_DEVICE_TYPE_GPU, dev_n, devs.P(), &dev_n);

		// choose devices
		_dev = devs(dev_idx);
		
		////////////////////////////////////////////////
		// create context... _cnt

		_cnt = clCreateContext(NULL, 1, &_dev, NULL, NULL, &res);

		klCheckRes(res, "creating context");

		////////////////////////////////////////////////
		// create command queue... _que
		
		_ques.Recreate(que_n);

		for(int i = 0; i < que_n; ++i)
		{
			_ques(i) = clCreateCommandQueue(_cnt, _dev, 0, &res);

			klCheckRes(res, "creating command queue");
		}

		/////////////////////////////////////////////////
		// post processing

		// pre-create prgs and krns
		_prgs.Recreate(0,16);
		_krns.Recreate(0,16);

		// register for klArr
		klBase::Register(_cnt, _ques(0));

		return *this;
	};

	// build program
	//  : return prg_idx
	int Build(kmStra src_file_name)
	{
		// init parameters
		cl_int res = 0;

		////////////////////////////////////////
		// get soruce from file... src_str

		kmFile file(src_file_name, KF_READ_TXT);

		const int byte = file.GetByte();

		kmStra src_str(byte); src_str.SetZero();

		file.Read(src_str.P(), byte);
		file.Close();

		src_str.Printf();

		////////////////////////////////////////
		// create and build program... prg

		const char* src = src_str.P();
		
		// create program
		cl_program prg = clCreateProgramWithSource(_cnt, 1, &src, NULL, &res);

		klCheckRes(res, "creating program");
		
		// build program
		res = clBuildProgram(prg, 0, NULL, NULL, NULL, NULL);
		
		if(res == CL_SUCCESS) print("* program is built\n");
		else
		{
			print("* failed to build a program (code : %d)\n", (int)res);
			char buf[2048]; size_t len;
			clGetProgramBuildInfo(prg, _dev, CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, &len);
			print("------------------------\n%s------------------------\n", buf);
		}
		return (int)_prgs.PushBack(prg);
	};

	// create kernel
	//  : return krn_idx
	int CreateKrn(int prg_idx, LPCSTR fun_name)
	{
		// init parameters
		cl_int res = 0;
		
		// create kernel... krn
		cl_kernel krn = clCreateKernel(_prgs(prg_idx), fun_name, &res);

		klCheckRes(res, "creating kernel");

		return (int)_krns.PushBack(krn);
	};

	// set argument to kernel
	template<typename T>
	void SetArg(int krn_idx, int arg_idx, T arg)
	{
		//KM_CHECK_TIME_START;

		cl_int res = clSetKernelArg(_krns(krn_idx), arg_idx, sizeof(T), &arg);

		klCheckRes(res, "setting argument");

		//KM_CHECK_TIME_END("setting arguments");
	}

	// set arguments to kernel with variable arguments
	template<typename T, typename... Ts>
	void SetArgs(int krn_idx, T arg, Ts... args)
	{
		SetArg(krn_idx, 0, arg); _SetArgs(krn_idx, 1, args...);
	}

protected:
	// set arguments with variable arguments.. hidden functions
	template<typename T, typename... Ts>
	void _SetArgs(int krn_idx, int arg_idx, T arg, Ts... args)
	{
		SetArg(krn_idx, arg_idx, arg); _SetArgs(krn_idx, ++arg_idx, args...);
	}
	void _SetArgs(int krn_idx, int arg_idx) {};

public:
	// launch kernel
	void Launch(int que_idx, int krn_idx, uint dim, size_t* glb, size_t* loc)
	{
		//KM_CHECK_TIME_START;

		// add kernel to queue
		cl_int res = clEnqueueNDRangeKernel(_ques(que_idx), _krns(krn_idx), dim, NULL, glb, loc, 0, NULL, NULL);

		klCheckRes(res, "launching kernel");

		//KM_CHECK_TIME_END("enqueue ndrange kernel");
	};

	// launch kernel with arguments
	template<typename T, typename... Ts>
	void Launch(int que_idx, int krn_idx, uint dim, size_t* glb, size_t* loc, T arg, Ts... args)
	{
		// set arguments
		_SetArgs(krn_idx, 0, arg, args...);
	
		// launch kernel
		Launch(que_idx, krn_idx, dim, glb, loc);
	}

	// flush the command queue
	klCnt& Flush(int que_idx = 0) { clFlush(_ques(que_idx)); return *this; };

	// wait until the command queue is done.
	klCnt& Wait(int que_idx = 0) { clFinish(_ques(que_idx)); return *this; };

	// release opencl
	void Release()
	{
		for(int i = (int)_krns.N(); i--;) clReleaseKernel      (_krns(i));
		for(int i = (int)_prgs.N(); i--;) clReleaseProgram     (_prgs(i));
		for(int i = (int)_ques.N(); i--;) clReleaseCommandQueue(_ques(i));

		clReleaseContext(_cnt);

		print("* opencl is released\n");
	};
};

///////////////////////////////////////////////////////
// openCl matrix base class
template<typename T> class klArr : klBase
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
	klArr() {};
	klArr(      int64 size) { Create(size);};
	klArr(T* p, int64 size) { Set(p, size);};

	// destructor
	virtual ~klArr()	{ Release(); };

	// copy constructor
	klArr(const klArr&    a) { Create(a.Size()); Copy(a); };
	klArr(const kmArr<T>& a) { Create(a.Size()); Copy(a); };

	template<typename Y>
	klArr(const klArr<Y>& b) { Create(b.Size()); TCast(*this, b); }

	// move constructor
	klArr(klArr&& b) { Move(b); };

	// assignment operator
	klArr& operator=(const klArr&    b) { RecreateIf(b); Copy(b._p); return *this; };
	klArr& operator=(const kmArr<T>& b) { RecreateIf(b); Copy(b);    return *this; };

	template<typename Y>
	klArr& operator=(const klArr<Y>& b) { RecreateIf(b); return TCast(*this, b); }

	// move assignment operator
	klArr& operator=(klArr&& b) { Move(b); return *this; };	

	// allocate memory... core
	void Create(int64 size)
	{
		ASSERTA(!IsCreated(), "[klArr::Create in 149] memory has already been created");
		ASSERTA(!IsPinned (), "[klArr::Create in 150] memory is pinned");

		if(size == 0) { _size = 0; _state = 0; return; }
		
		//cudaMalloc((void**)&_p, size*sizeof(T));
		cl_int res = 0; _p =(T*)clCreateBuffer(_cnt, CL_MEM_READ_WRITE, size*sizeof(T), NULL, &res);

		ASSERTA(_p != 0, "[klArr::Create in 140] %lld byte",size*sizeof(T));

		_size  = size;
		_state = 1;
	};

	// expand memory... core
	// * Note that expand must be call in case that klArr was created not set.
	void Expand(int64 size)
	{	
		ASSERTA(IsCreated(), "[klArr::Expand in 162] memory is not created");
		ASSERTA(!IsPinned(), "[klArr::Expand in 163] memory is pinned");

		const int64 size_new = size + _size;

		T* p; 		
		//cudaMalloc((void**)&p, size_new*sizeof(T));
		cl_int res = 0; p = (T*)clCreateBuffer(_cnt, CL_MEM_READ_WRITE, size_new*sizeof(T), NULL, &res);

		ASSERTA(_p != 0, "[klArr::Expand in 156]");

		CopyTo(p);
		
		//cudaFree(_p);
		clReleaseMemObject((cl_mem)_p);
		
		_size  = size_new;
		_p     = p;
		_state = 1;
	};

	// release memory... core
	void Release()
	{		
		if(IsCreated() && _p) clReleaseMemObject((cl_mem)_p); //cudaFree(_p);
		_state.is_created = 0;
		Init();
	};

	// set array... core
	void Set(T* p, int64 size)
	{
		ASSERTA(!IsCreated(), "[klArr::Set in 193] memory has already been created");

		_size = size;
		_p    = p;
	};

	// * Note that if we know the size of the target,
	// * the size of the transfered data is always the size of the target.
	//
	// copy from (a = b), GPU to GPU... core
	void Copy(const T* b, cudaStream_t stream = 0)
	{
		ASSERTA(_p != 0, "[klArr::Copy in 186]");

		//cudaMemcpyAsync((void*) _p, (void*) b, Byte(), cudaMemcpyDeviceToDevice, stream);
	};

	// copy from (a = b), GPU to GPU... core
	void Copy(const klArr& b, cudaStream_t stream = 0)
	{
		ASSERTA(b.Size() >= Size(), "[klArr::Copy in 194]");

		//cudaMemcpyAsync((void*) _p, (void*) b.P(), Byte(), cudaMemcpyDeviceToDevice, stream);
	};

	// copy to (b = a), GPU to GPU...  core
	void CopyTo(T* b, cudaStream_t stream = 0) const
	{
		ASSERTA(_p != 0, "[klArr::CopyTo in 195]");

		//cudaMemcpyAsync((void*) b, (void*) _p, Byte(), cudaMemcpyDeviceToDevice, stream);
	};

	// copy to (b = a), GPU to GPU... core
	void CopyTo(klArr& b, cudaStream_t stream = 0) const
	{
		ASSERTA(Size() >= b.Size(), "[klArr::CopyTo in 210]");

		//cudaMemcpyAsync((void*) b.P(), (void*) _p, b.Byte(), cudaMemcpyDeviceToDevice, stream);
	};

	// copy from host, CPU to GPU... core
	void CopyFromHost(T* phost)
	{
		ASSERTA(_p != 0, "[klArr::CopyFromHost in 218]");

		//cudaMemcpyAsync((void*) _p, (void*) phost, Byte(), cudaMemcpyHostToDevice, stream);
		cl_int res = clEnqueueWriteBuffer(_que, (cl_mem)_p, CL_TRUE, 0, Byte(), (void*) phost, 0, NULL, NULL);

		klCheckRes(res, "copying from host");
	};

	// copy from host, CPU to GPU... core
	void Copy(const kmArr<T>& b)
	{
		ASSERTA(b.Size() >= Size(), "[klArr::CopyFrom in 226]");

		//cudaMemcpyAsync((void*) _p, (void*) b.P(), Byte(), cudaMemcpyHostToDevice, stream);
		cl_int res = clEnqueueWriteBuffer(_que, (cl_mem)_p, CL_TRUE, 0, Byte(), (void*)b.P(), 0, NULL, NULL);

		klCheckRes(res, "copying");
	};	

	// copy from host, CPU to GPU
	template<typename Y>
	void Copy(const kmArr<Y>& b, cudaStream_t stream = 0)
	{
		TCast(*this, klArr<Y>(b), stream);
	}

	// copy to host, GPU to CPU...core
	void CopyToHost(T* phost, cudaStream_t stream = 0) const
	{
		ASSERTA(_p != 0, "[klArr::CopyToHost in 226]");

		//cudaMemcpyAsync((void*) phost, (void*) _p, Byte(), cudaMemcpyDeviceToHost, stream);
		cl_int res = clEnqueueReadBuffer(_que, (cl_mem)_p, CL_TRUE, 0, Byte(), (void*)phost, 0, NULL, NULL);

		klCheckRes(res, "copying to host");
	};

	// copy to host, GPU to CPU... core
	void CopyTo(kmArr<T>& b, cudaStream_t stream = 0) const
	{
		ASSERTA(Size() >= b.Size(), "[klArr::CopyTo in 234]");

		//cudaMemcpyAsync((void*) b.P(), (void*) _p, b.Byte(), cudaMemcpyDeviceToHost, stream);
		cl_int res = clEnqueueReadBuffer(_que, (cl_mem)_p, CL_TRUE, 0, Byte(), (void*)b.P(), 0, NULL, NULL);

		klCheckRes(res, "copying to host");
	};

	// copy to host, GPU to CPU as other type
	template<typename Y>
	void CopyTo(kmArr<Y>& b, cudaStream_t stream = 0) const
	{
		klArr<Y>(*this).CopyTo(b, stream);
	}

	// release and create
	void Recreate(int64 size) { Release();	Create(size); };

	template<typename Y> void Recreate(const klArr<Y>& b) { Recreate(b.N()); }
	template<typename Y> void Recreate(const kmArr<Y>& b) { Recreate(b.N()); }

	// recreate if
	int RecreateIf(int64 size) { if(size != _size) { Recreate(size); return 1; } return 0; };

	template<typename Y> int RecreateIf(const klArr<Y>& b) { return RecreateIf(b.Size()); }
	template<typename Y> int RecreateIf(const kmArr<Y>& b) { return RecreateIf(b.Size()); }
		
	// move (b --> a)... core
	void Move(klArr& b)
	{
		ASSERTA(!IsPinned(), "[klArr::Move in 280] memory is pinned");

		if(IsCreated()) Release();

		_p     = b._p;
		_size  = b._size;
		_state = b._state; b._state = 0;
	};

	// set array
	void Set(const klArr& a)   { Set(a._p, a._size); };

	// operator to get data
	T* P(int64 i1 = 0) const
	{
		ASSERTA(i1 < _size, "[klArr::P in 216] %lld < %lld", i1, _size);

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
	klArr& Restore() { RestoreVfptr<klArr<T>>(); _state = 0; Init(); return *this; };

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
	template<typename Y> bool IsEqualSize   (const klArr<Y>& b) const { return _size == b.Size(); }
	template<typename Y> bool IsEqualSize   (const kmArr<Y>& b) const { return _size == b.Size(); }
	template<typename Y> bool IsEqualSizeDim(const klArr<Y>& b) const { return _size == b.Size(); }
	template<typename Y> bool IsEqualSizeDim(const kmArr<Y>& b) const { return _size == b.Size(); }
	template<typename Y> bool IsEqualN      (const klArr<Y>& b) const { return N() == b.N(); }
	template<typename Y> bool IsEqualN      (const kmArr<Y>& b) const { return N() == b.N(); }

	// set value
	void SetZero(cudaStream_t stream = 0)
	{
		//cudaMemsetAsync(_p, 0, Byte(), stream);
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
		//cudaGetDeviceProperties(&prop, id);		
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

#endif /* __kl7Mat_H_INCLUDED_2021_09_28__ */
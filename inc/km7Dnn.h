#ifndef __km7Dnn_H_INCLUDED_2019_03_05__
#define __km7Dnn_H_INCLUDED_2019_03_05__

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
#include "kc7Mat.h"

///////////////////////////////////////////////////////////////
// enum for dnn

// type of layer
enum kmeDnnLyType
{
	LY_BASE,       // base... it cannot be used.
	LY_FULL,       // fully connected
	LY_CONV,       // convolution
	LY_POOL,       // pooling 
	LY_ACTF,       // active function
	LY_COMB,       // combination
	LY_ADD,        // adding
	LY_BNRM,       // batch normalization
	LY_CNVF        // convolution filter
};

// type of activation function
enum kmeDnnActf
{	
	ACTF_NONE,       // none
	ACTF_SIGM,       // sigmoid
	ACTF_RELU,       // ReLU
	ACTF_SMAX        // soft max
};

// type of loss function
enum kmeDnnLoss
{
	LOSS_MSE,        // mean square error
	LOSS_CEE         // cross entropy error
};

// type of pooling
enum kmeDnnPool
{
	POOL_MAX,       // max pooling
	POOL_AVG        // average pooling
};

//////////////////////////////////////////////////////////////
// state union 
union kmDnnLyState
{
	// members
	uint64 val = 0;
	struct
	{
		uchar is_fixed   : 1; // this layer will be skipped during a train
		uchar is_no_bias : 1; // this layer will not use bias
		uchar dropout    : 8; // dropout level (0-255)
	};	

	// constructor
	kmDnnLyState() {};

	// copy constructor	
	kmDnnLyState(const uint64& a)	{ val = a; };

	// assignment operator
	kmDnnLyState& operator=(const uint64& a) { val = a; return *this; };

	// conversion operator... (uint64) a
	operator uint64() const { return val; };
};

///////////////////////////////////////////////////////////////
// class for layers

// base of layer
class kmDnnLy
{
public:
	kmeDnnLyType _type   = LY_BASE; // type of layer for save and load of a layer
	kmDnnLyState _state  = 0;       // state of layer
	int64        _x0_idx = 0;       // index of input node set
	int64        _x1_idx = 0;       // index of output node set	

	/////////////////////////////////////
	// basic functions

	// constructor
	kmDnnLy() {};

	// destructor
	virtual ~kmDnnLy() {};

	// get virtual funtion pointer
	void* GetVfptr() { return (void*) *((int64*)this); };

	// get type
	kmeDnnLyType GetType() { return _type; };

	// get name of class
	LPCSTR GetKmClass() const { return typeid(*this).name() + 6; };

	/////////////////////////////////////
	// general functions

	// set state
	void   SetFixed () { _state.is_fixed   = 1; };
	void ResetFixed () { _state.is_fixed   = 0; };
	void   SetNoBias() { _state.is_no_bias = 1; };
	void ResetNoBias() { _state.is_no_bias = 0; };

	void   SetDropout(float rate) {	_state.dropout = (uchar) (rate*255); };

	// get info
	bool  IsFixed ()   const { return _state.is_fixed   == 1; };
	bool  IsNoBias()   const { return _state.is_no_bias == 1; };
	float GetDropout() const { return _state.dropout/255.f; };	
	int64 GetX0()      const { return _x0_idx; };
	int64 GetX1()      const { return _x1_idx; };

	/////////////////////////////////////
	// virtual functions
			
	// get size... pure virtual
	virtual int64 GetByte() const = 0;

	// print info
	virtual void PrintInfo()
	{
		PRINTFA(" %11s ", GetKmClass());
		PRINTFA(" x0:%3lld", _x0_idx);
		PRINTFA(", x1:%3lld", _x1_idx);
		PRINTFA(", fix %d, nbs %d, drp %.1f", IsFixed(), IsNoBias(), GetDropout());
		//PRINTFA("  vfptr       : %llx\n", GetVfptr());
		//PRINTFA("  size        : %lld byte\n", GetByte());
		//PRINTFA("  type        : %lld\n", _type);
	};

	// init weight, bias and output node... pure virtual 
	virtual void  Init(kmMat1<kmMat4f32>& x, kmMat4f32& w0, kmMat2f32& b0) = 0;
	virtual void cInit(kmMat1<kcMat4f32>& x, kcMat4f32& w0, kcMat2f32& b0) {};
	virtual void cInit(kmMat1<kcMat4f16>& x, kcMat4f16& w0, kcMat2f16& b0) {};

	// init only node... pure virtual
	virtual void  Init(kmMat1<kmMat4f32>& x) = 0;
	virtual void cInit(kmMat1<kcMat4f32>& x) {};
	virtual void cInit(kmMat1<kcMat4f16>& x) {};

	// init backward... if necessary
	virtual void  InitBackward(kmMat1<kmMat4f32>& x) {};
	virtual void cInitBackward(kmMat1<kcMat4f32>& x) {};
	virtual void cInitBackward(kmMat1<kcMat4f16>& x) {};

	// dropout function... if necessary
	virtual void  InitDropout(kmMat3i8& mask0) {};
	virtual void cInitDropout(kcMat3i8& mask0) {};

	virtual void  Dropout(kmMat1<kmMat4f32>& x, const kmMat3i8& mask0) {};
	virtual void cDropout(kmMat1<kcMat4f32>& x, const kcMat3i8& mask0) {};
	virtual void cDropout(kmMat1<kcMat4f16>& x, const kcMat3i8& mask0) {};

	// forward function... pure virtual
	virtual void  Forward(kmMat1<kmMat4f32>& x,
		                   const kmMat4f32&  w0, 
		                   const kmMat2f32&  b0, const bool istrain = false) = 0;
	virtual void cForward(kmMat1<kcMat4f32>& x,
		                   const kcMat4f32&  w0, 
		                   const kcMat2f32&  b0, const bool istrain = false) {};
	virtual void cForward(kmMat1<kcMat4f16>& x,
		                   const kcMat4f16&  w0, 
		                   const kcMat2f16&  b0, const bool istrain = false) {};

	// backward function... pure virtual
	virtual void  Backward(kmMat1<kmMat4f32>& gx,
		                          kmMat4f32&  gw0,
		                          kmMat2f32&  gb0,
		             const kmMat1<kmMat4f32>& x,
		                    const kmMat4f32&  w0, 
		                    const kmMat2f32&  b0) = 0;
	virtual void cBackward(kmMat1<kcMat4f32>& gx,
		                          kcMat4f32&  gw0,
		                          kcMat2f32&  gb0,
		             const kmMat1<kcMat4f32>& x,
		                    const kcMat4f32&  w0, 
		                    const kcMat2f32&  b0) {};
	virtual void cBackward(kmMat1<kcMat4f16>& gx,
		                          kcMat4f16&  gw0,
		                          kcMat2f16&  gb0,
		             const kmMat1<kcMat4f16>& x,
		                    const kcMat4f16&  w0, 
		                    const kcMat2f16&  b0) {};
};

// fully connected layer
class kmDnnLyFull : public kmDnnLy
{
public:	
	int64 _nx1 = 0;   // number of output nodes

	/////////////////////////////////////
	// basic functions

	// constructor	
	kmDnnLyFull() { _type = LY_FULL; };
	kmDnnLyFull(int64 n_on) : kmDnnLyFull()
	{
		Set(0, 0, n_on);
	};
	kmDnnLyFull(int64 x0_idx, int64 x1_idx, int64 n_on) : kmDnnLyFull()
	{
		Set(x0_idx, x1_idx, n_on);
	};

	/////////////////////////////////////
	// general functions
	
	void Set(int64 x0_idx, int64 x1_idx, int64 nx1)
	{
		_x0_idx = x0_idx;
		_x1_idx = x1_idx;
		_nx1    = nx1;
	};

	// get size
	virtual int64 GetByte() const { return sizeof(kmDnnLyFull); };
	
	// print info
	virtual void PrintInfo()
	{
		kmDnnLy::PrintInfo();
		PRINTFA(", nx1:%3lld\n", _nx1);
	};
	
	/////////////////////////////////////
	// init functions

	virtual void Init(kmMat1<kmMat4f32>& x, kmMat4f32& w0, kmMat2f32& b0) override
	{	
		// init parameters
		const int64 nx0 = x(_x0_idx).N()/x(_x0_idx).N4();

		// create weight and bias
		w0.Recreate(_nx1, nx0, 1, 1);
		b0.Recreate(_nx1, 1);

		// set intialial value
		const float std = sqrtf(6.f/(float)nx0);

		w0.SetRand(-std, std);
		b0.SetZero();

		// create output node
		Init(x);
	};
		
	virtual void Init(kmMat1<kmMat4f32>& x) override
	{
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		x(_x1_idx).Recreate(_nx1, 1, 1, x(_x0_idx).N4());
	};

	template<typename T> 
	void _cInit(kmMat1<kcMat4<T>>& x, kcMat4<T>& w0, kcMat2<T>& b0)
	{
		// init parameters
		const int64 nx0 = x(_x0_idx).N()/x(_x0_idx).N4();

		// create weight and bias
		w0.Recreate(_nx1, nx0, 1, 1);
		b0.Recreate(_nx1, 1);

		// set intialial value
		const float std = sqrtf(6.f/(float)nx0);

		kmMat4f32 w0_(_nx1, nx0, 1, 1); w0_.SetRand(-std, std);	w0 = w0_;
		b0.SetZero();

		// create output node
		cInit(x);
	}

	template<typename T>
	void _cInit(kmMat1<kcMat4<T>>& x)
	{
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		x(_x1_idx).Recreate(_nx1, 1, 1, x(_x0_idx).N4());
	}

	virtual void cInit(kmMat1<kcMat4f32>& x, kcMat4f32& w0, kcMat2f32& b0) override { _cInit(x, w0, b0); };
	virtual void cInit(kmMat1<kcMat4f16>& x, kcMat4f16& w0, kcMat2f16& b0) override { _cInit(x, w0, b0); };

	virtual void cInit(kmMat1<kcMat4f32>& x) override {	_cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x) override { _cInit(x); };

	virtual void  InitDropout(kmMat3i8& mask) override
	{
		const float dropout = GetDropout();

		if(dropout > 0)
		{
			kmMat3f32 mat(mask.N1(), mask.N2(), mask.N3());

			mat.SetRand(0, 1.f);

			for(int64 i = 0; i < mask.N(); ++i)
			{
				mask(i) = (mat(i) < dropout) ? 1:0;
			}
		}
	};

	virtual void cInitDropout(kcMat3i8& cmask) override
	{
		if(GetDropout() > 0)
		{
			kmMat3i8 mask(cmask.N1(), cmask.N2(), cmask.N3());

			InitDropout(mask);

			cmask.Copy(mask);
		}
	};

	////////////////////////////////////////////////////
	// dropout function

	virtual void cDropout(kmMat1<kcMat4f32>& x, const kcMat3i8& mask0) override;
	virtual void cDropout(kmMat1<kcMat4f16>& x, const kcMat3i8& mask0) override;

	////////////////////////////////////////////////////
	// forward function

	virtual void Forward(kmMat1<kmMat4f32>& x,
		                  const kmMat4f32&  w0,
		                  const kmMat2f32&  b0, const bool istrain = false) override
	{
		// init variable and parameters
		const int64     nx0   = x(_x0_idx).N()/x(_x0_idx).N4();
		const int64     n_bat = x(_x0_idx).N4();
		const kmMat2f32 w0_(w0.P(), _nx1, nx0);

		for(int64 b = 0; b < n_bat; ++b)
		{
			const kmMat2f32 x0(x(_x0_idx).P(0,0,0,b),  nx0, 1);
			      kmMat2f32 x1(x(_x1_idx).P(0,0,0,b), _nx1, 1);
		
			// calc
			x1.Copy(w0_*x0 + b0);
		}
	};
	
	virtual void cForward(kmMat1<kcMat4f32>& x, const kcMat4f32& w0, const kcMat2f32& b0, const bool istrain = false) override;
	virtual void cForward(kmMat1<kcMat4f16>& x, const kcMat4f16& w0, const kcMat2f16& b0, const bool istrain = false) override;

	////////////////////////////////////////////////////
	// backward function
	virtual void Backward(kmMat1<kmMat4f32>& gx,
		                         kmMat4f32&  gw0,
		                         kmMat2f32&  gb0,
		            const kmMat1<kmMat4f32>& x,
		                   const kmMat4f32&  w0, 
		                   const kmMat2f32&  b0) override
	{
		// init variable and parameters
		const int64     nx0   = x(_x0_idx).N()/x(_x0_idx).N4();
		const int64     n_bat = x(_x0_idx).N4();
		const kmMat2f32  w0_( w0.P(), _nx1, nx0);
		      kmMat2f32 gw0_(gw0.P(), _nx1, nx0);

		// calc dl/dw, dl/db and dl/dx0... gw0, gb0, gx0
		// * Note that gx0 was already set as zero in kmDnn::Backward()
		gw0_.SetZero();
		gb0 .SetZero();

		for(int64 b = 0; b < n_bat; ++b)
		{
			const kmMat2f32  x0( x(_x0_idx).P(0,0,0,b),  nx0, 1);
			      kmMat2f32 gx0(gx(_x0_idx).P(0,0,0,b),  nx0, 1);
			const kmMat2f32 gx1(gx(_x1_idx).P(0,0,0,b), _nx1, 1);
		
			// calc dl/db... gb0
			for(int64 i =0; i < _nx1; ++i)
			{
				gb0(i,0) += gx1(i);
			}
		
			// calc dl/dw and dl/dx... gw0, gx0
			for(int64 j = 0; j <  nx0; ++j)
			for(int64 i = 0; i < _nx1; ++i)
			{	
				gw0_(i,j) += gx1(i)*x0(j);
				gx0 (j)   += gx1(i)*w0_(i,j);
			}
		}
	};
	
	virtual void cBackward(kmMat1<kcMat4f32>& gx,       kcMat4f32& gw0,       kcMat2f32& gb0,
		             const kmMat1<kcMat4f32>&  x, const kcMat4f32&  w0, const kcMat2f32&  b0) override;
	virtual void cBackward(kmMat1<kcMat4f16>& gx,       kcMat4f16& gw0,       kcMat2f16& gb0,
		             const kmMat1<kcMat4f16>&  x, const kcMat4f16&  w0, const kcMat2f16&  b0) override;
};

// convolution layer
class kmDnnLyConv : public kmDnnLy
{
public:
	int64 _nw     = 1;    // width   of filter
	int64 _nh     = 1;    // height  of filter
	int64 _nf     = 1;    // number  of filter
	int64 _strd   = 1;    // stride
	int64 _zpad   = 0;    // zero-padding

	/////////////////////////////////////
	// basic functions

	// constructor
	kmDnnLyConv() { _type = LY_CONV; };
	kmDnnLyConv(int64 nw, int64 nh, int64 nf, int64 strd, int64 zpad) : kmDnnLyConv()
	{
		Set(0, 0, nw, nh, nf, strd, zpad);
	};
	kmDnnLyConv(int64 x0_idx, int64 x1_idx, 
		        int64 nw, int64 nh, int64 nf, int64 strd, int64 zpad) : kmDnnLyConv()
	{
		Set(x0_idx, x1_idx, nw, nh, nf, strd, zpad);
	};

	/////////////////////////////////////
	// general functions
	
	void Set(int64 x0_idx, int64 x1_idx, 
		     int64 nw, int64 nh, int64 nf, int64 strd, int64 zpad)
	{
		_x0_idx = x0_idx;
		_x1_idx = x1_idx;
		_nw     = nw;
		_nh     = nh;
		_nf     = nf;
		_strd   = strd;
		_zpad   = zpad;
	};

	// get size of output node
	int64 GetX1N1(int64 x0n1) { return (x0n1 + 2*_zpad - _nw)/_strd + 1; };
	int64 GetX1N2(int64 x0n2) { return (x0n2 + 2*_zpad - _nh)/_strd + 1; };
	
	// get size
	virtual int64 GetByte() const { return sizeof(kmDnnLyConv); };

	// print info
	virtual void PrintInfo()
	{
		kmDnnLy::PrintInfo();
		PRINTFA(", (%lld, %lld, %lld, %lld, %lld)\n", _nw, _nh, _nf, _strd, _zpad);
	};

	/////////////////////////////////////
	// init function
		
	virtual void Init(kmMat1<kmMat4f32>& x, kmMat4f32& w0, kmMat2f32& b0) override
	{	
		// init parameters
		const int64 nc  = x(_x0_idx).N3(); // n of channel
		const int64 nx0 = x(_x0_idx).N()/x(_x0_idx).N4();
		const float std = sqrtf(6.f/(float) nx0);
		
		// create and initialize weight
		w0.Recreate(_nw, _nh, nc, _nf);
		w0.SetRand(-std, std);

		// create and initialize bias
		if(!IsNoBias())
		{
			b0.Recreate(_nf, 1);
			b0.SetZero();
		}
		// create output node
		Init(x);
	};
	
	virtual void Init(kmMat1<kmMat4f32>& x) override
	{
		// init parameters
		const int64 x0n1  = x(_x0_idx).N1();
		const int64 x0n2  = x(_x0_idx).N2();
		const int64 n_bat = x(_x0_idx).N4();

		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		x(_x1_idx).Recreate(GetX1N1(x0n1), GetX1N2(x0n2), _nf, n_bat);
	};

	template<typename T>
	void _cInit(kmMat1<kcMat4<T>>& x, kcMat4<T>& w0, kcMat2<T>& b0)
	{	
		// init parameters
		const int64 nc  = x(_x0_idx).N3(); // n of channel
		const int64 nx0 = x(_x0_idx).N()/x(_x0_idx).N4();
		const float std = sqrtf(6.f/(float) nx0);

		// create and initialize weight
		w0.Recreate(_nw, _nh, nc, _nf);
		kmMat4f32 w0_(_nw, _nh, nc, _nf); w0_.SetRand(-std, std); w0 = w0_;

		// create and initialize bias
		if(!IsNoBias())
		{
			b0.Recreate(_nf, 1);
			b0.SetZero();
		}
		// create output node
		cInit(x);
	}
	
	template<typename T>
	void _cInit(kmMat1<kcMat4<T>>& x)
	{
		// init parameters
		const int64 x0n1  = x(_x0_idx).N1();
		const int64 x0n2  = x(_x0_idx).N2();
		const int64 n_bat = x(_x0_idx).N4();

		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		x(_x1_idx).Recreate(GetX1N1(x0n1), GetX1N2(x0n2), _nf, n_bat);
	}

	virtual void cInit(kmMat1<kcMat4f32>& x, kcMat4f32& w0, kcMat2f32& b0) override { _cInit(x, w0, b0); };
	virtual void cInit(kmMat1<kcMat4f16>& x, kcMat4f16& w0, kcMat2f16& b0) override { _cInit(x, w0, b0); };

	virtual void cInit(kmMat1<kcMat4f32>& x) override { _cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x) override { _cInit(x); };

	////////////////////////////////////////////////////
	// forward function
	virtual void Forward(kmMat1<kmMat4f32>& x,
		                  const kmMat4f32&  w0,
		                  const kmMat2f32&  b0, const bool istrain = false) override
	{
		// init variable and parameters
		kmMat4f32& x0 = x(_x0_idx);
		kmMat4f32& x1 = x(_x1_idx); x1.SetZero();
				
		const int64 x1n1 = x1.N1(), x1n2 = x1.N2();
		const int64 x0n1 = x0.N1(), x0n2 = x0.N2(), nc = x0.N3(), nb = x0.N4();
		
		// calc convolution... main loop
		for(int64 b = 0; b < nb; ++b)
		{
			for(int64 i4 = 0; i4 < _nf; ++i4) // filter, output dim3
			{	
				for(int64 j2 = 0; j2 < x1n2; ++j2) // output dim2
				for(int64 j1 = 0; j1 < x1n1; ++j1) // output dim1
				{
					for(int64 i3 = 0; i3 < nc; ++i3) // channel, input dim3
					{
						int64 k2 = _strd*j2 - _zpad; // input dim2
		
						for(int64 i2 = 0; i2 < _nh; ++i2, ++k2) // height
						{
							// check out of boundary
							if(k2 < 0 || k2 >= x0n2) continue;
		
							int64 k1 = _strd*j1 - _zpad; // input dim1
		
							for(int64 i1 = 0; i1 < _nw; ++i1, ++k1) // width
							{
								// check out of boundary
								if(k1 < 0 || k1 >= x0n1) continue;
		
								// sum weight-multification
								x1(j1,j2,i4,b) += w0(i1,i2,i3,i4)*x0(k1,k2,i3,b);
							}
						}
					}
					// sum bias-multification
					if(!IsNoBias()) x1(j1,j2,i4,b) += b0(i4,0);
				}
			}
		}
	};

	virtual void cForward(kmMat1<kcMat4f32>& x, const kcMat4f32& w0, const kcMat2f32& b0, const bool istrain = false) override;	
	virtual void cForward(kmMat1<kcMat4f16>& x, const kcMat4f16& w0, const kcMat2f16& b0, const bool istrain = false) override;
	
	////////////////////////////////////////////////////
	// backward function
	virtual void Backward(kmMat1<kmMat4f32>& gx,
		                         kmMat4f32&  gw0,
		                         kmMat2f32&  gb0,
		            const kmMat1<kmMat4f32>& x,
		                   const kmMat4f32&  w0, 
		                   const kmMat2f32&  b0) override
	{
		// init variable and parameters
		const kmMat4f32&  x0 =  x(_x0_idx);		
		      kmMat4f32& gx0 = gx(_x0_idx);
		const kmMat4f32& gx1 = gx(_x1_idx);

		const int64 x1n1 = gx1.N1(), x1n2 = gx1.N2();	
		const int64 x0n1 =  x0.N1(), x0n2 =  x0.N2(), nc = x0.N3(), n_bat = x0.N4();

		// calc dl/dw, dl/db and dl/dx0... gw0, gb0, gx0
		// * Note that gx0 was already set as zero in kmDnn::Backward()
		gw0.SetZero(); 
		gb0.SetZero();

		// main loop
		for(int64 b = 0; b < n_bat; ++b) // batch
		{
			for(int64 i4 = 0; i4 < _nf; ++i4) // filter, output dim3
			{	
				for(int64 j2 = 0; j2 < x1n2; ++j2) // output dim2
				for(int64 j1 = 0; j1 < x1n1; ++j1) // output dim1
				{
					const float gx1_val = gx1(j1,j2,i4,b);

					for(int64 i3 = 0; i3 < nc; ++i3) // channel, input dim3
					{
						int64 k2 = _strd*j2 - _zpad; // input dim2

						for(int64 i2 = 0; i2 < _nh; ++i2, ++k2) // height
						{
							// check out of boundary
							if(k2 < 0 || k2 >= x0n2) continue;

							int64 k1 = _strd*j1 - _zpad; // input dim1

							for(int64 i1 = 0; i1 < _nw; ++i1, ++k1) // width
							{
								// check out of boundary
								if(k1 < 0 || k1 >= x0n1) continue;

								// sum gw0 and gx0
								// : x1(j1, j2, i4) += w0(i1,i2,i3,i4)*x0(k1,k2,i3)
								gw0(i1,i2,i3,i4) += gx1_val*x0(k1,k2,i3,b);
								gx0(k1,k2,i3,b)  += gx1_val*w0(i1,i2,i3,i4);
							}
						}
					}
					// sum gb0
					// : x1(j1,j2,i4,b) += b0(i4,0)
					if(!IsNoBias()) gb0(i4,0) +=  gx1_val;
				}
			}
		}
	};

	virtual void cBackward(kmMat1<kcMat4f32>& gx,       kcMat4f32& gw0,       kcMat2f32& gb0,
		             const kmMat1<kcMat4f32>&  x, const kcMat4f32&  w0, const kcMat2f32&  b0) override;
	virtual void cBackward(kmMat1<kcMat4f16>& gx,       kcMat4f16& gw0,       kcMat2f16& gb0,
		             const kmMat1<kcMat4f16>&  x, const kcMat4f16&  w0, const kcMat2f16&  b0) override;
};

// convolution filter layer
class kmDnnLyCnvf : public kmDnnLy
{
public:	
	int64 _nw     = 1;    // width   of filter
	int64 _nh     = 1;    // height  of filter
	int64 _nf     = 1;    // number  of filter
	int64 _strd   = 1;    // stride
	int64 _zpad   = 0;    // zero-padding	

	/////////////////////////////////////
	// basic functions

	// constructor
	kmDnnLyCnvf()
	{
		_type = LY_CNVF;
		Set(0,0,3,3,8,1,1);	SetNoBias(); SetFixed();
	};
	kmDnnLyCnvf(int64 nw, int64 nh, int64 nf, int64 strd, int64 zpad)
	{
		_type = LY_CNVF;
		Set(0, 0, nw, nh, nf, strd, zpad);
	};
	kmDnnLyCnvf(int64 x0_idx, int64 x1_idx, 
		        int64 nw, int64 nh, int64 nf, int64 strd, int64 zpad)
	{
		 _type = LY_CNVF;
		Set(x0_idx, x1_idx, nw, nh, nf, strd, zpad);
	};

	/////////////////////////////////////
	// general functions

	void Set(int64 x0_idx, int64 x1_idx, 
		     int64 nw, int64 nh, int64 nf, int64 strd, int64 zpad)
	{
		_x0_idx = x0_idx;
		_x1_idx = x1_idx;
		_nw     = nw;
		_nh     = nh;
		_nf     = nf;
		_strd   = strd;
		_zpad   = zpad;
	};

	// get size of output node
	int64 GetX1N1(int64 x0n1) { return (x0n1 + 2*_zpad - _nw)/_strd + 1; };
	int64 GetX1N2(int64 x0n2) { return (x0n2 + 2*_zpad - _nh)/_strd + 1; };
	
	// get size
	virtual int64 GetByte() const { return sizeof(kmDnnLyCnvf); };

	// print info
	virtual void PrintInfo()
	{
		kmDnnLy::PrintInfo();
		PRINTFA(", (%lld, %lld, %lld, %lld, %lld)\n", _nw, _nh, _nf, _strd, _zpad);
	};
	
	/////////////////////////////////////
	// init function
		
	virtual void Init(kmMat1<kmMat4f32>& x, kmMat4f32& w0, kmMat2f32& b0) override
	{	
		// init parameters		
		const int64 nx0 = x(_x0_idx).N()/x(_x0_idx).N4();
		const float std = sqrtf(6.f/(float) nx0);
		
		// create and initialize weight
		w0 = GetPreW(std);

		// create and initialize bias
		if(!IsNoBias())
		{
			b0.Recreate(_nf, 1);
			b0.SetZero();
		}
		// create output node
		Init(x);
	};
	
	virtual void Init(kmMat1<kmMat4f32>& x) override
	{
		// init parameters
		const int64 x0n1 = x(_x0_idx).N1();
		const int64 x0n2 = x(_x0_idx).N2();
		const int64 nc   = x(_x0_idx).N3();
		const int64 nb   = x(_x0_idx).N4();

		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		x(_x1_idx).Recreate(GetX1N1(x0n1), GetX1N2(x0n2), _nf*nc, nb);
	};

	template<typename T>
	void _cInit(kmMat1<kcMat4<T>>& x, kcMat4<T>& w0, kcMat2<T>& b0)
	{	
		// init parameters		
		const int64 nx0 = x(_x0_idx).N()/x(_x0_idx).N4();
		const float std = sqrtf(6.f/(float) nx0);

		// create and initialize weight
		w0 = GetPreW(std);

		// create and initialize bias
		if(!IsNoBias())
		{
			b0.Recreate(_nf, 1);
			b0.SetZero();
		}
		// create output node
		cInit(x);
	}
	
	template<typename T>
	void _cInit(kmMat1<kcMat4<T>>& x)
	{
		// init parameters
		const int64 x0n1 = x(_x0_idx).N1();
		const int64 x0n2 = x(_x0_idx).N2();
		const int64 nc   = x(_x0_idx).N3();
		const int64 nb   = x(_x0_idx).N4();

		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		x(_x1_idx).Recreate(GetX1N1(x0n1), GetX1N2(x0n2), _nf*nc, nb);
	}

	virtual void cInit(kmMat1<kcMat4f32>& x, kcMat4f32& w0, kcMat2f32& b0) override { _cInit(x, w0, b0); };
	virtual void cInit(kmMat1<kcMat4f16>& x, kcMat4f16& w0, kcMat2f16& b0) override { _cInit(x, w0, b0); };

	virtual void cInit(kmMat1<kcMat4f32>& x) override { _cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x) override { _cInit(x); };

	kmMat4f32 GetPreW(const float std)
	{
		kmMat4f32 w0(_nw, _nh, _nf, 1);

		if(_nw == 3 && _nh == 3 && _nf == 8)
		{
			const float s = sqrtf(2.f);			
			const float b = 1/9.f;
			const float c = 1/6.f;
			const float d = 1/(3*s);
			const float e = 1/(6*s);
			const float f = 0.2042f;
			const float g = 0.1238f;
			const float h = 0.0751f;

			w0.Mat2(0).Copy(kmMat2f32({ 0, 0, 0,   0, 1, 0,   0, 0, 0 })); // bypass
			w0.Mat2(1).Copy(kmMat2f32({ b, b, b,   b, b, b,   b, b, b })); // average
			w0.Mat2(2).Copy(kmMat2f32({ c, c, c,   0, 0, 0,  -c,-c,-c })); // grad y
			w0.Mat2(3).Copy(kmMat2f32({ c, 0,-c,   c, 0,-c,   c, 0,-c })); // grad x
			w0.Mat2(4).Copy(kmMat2f32({ 0,-e,-d,   e, 0,-e,   d, e, 0 })); // grad xy
			w0.Mat2(5).Copy(kmMat2f32({ d, e, 0,   e, 0,-e,   0,-e,-d })); // grad yx
			w0.Mat2(6).Copy(kmMat2f32({-b, b,-b,   b,-b, b,  -b, b,-b })); // pattern
			w0.Mat2(7).Copy(kmMat2f32({ h, g, h,   g, f, g,   h, g, h })); // gaussian
		}
		else
		{
			w0.SetRand(-std, std);
		}
		return w0;
	};

	////////////////////////////////////////////////////
	// forward function
	virtual void  Forward(kmMat1<kmMat4f32>& x, const kmMat4f32&  w0, const kmMat2f32&  b0, const bool istrain = false) override {};
	
	virtual void cForward(kmMat1<kcMat4f32>& x, const kcMat4f32&  w0, const kcMat2f32&  b0, const bool istrain = false) override;
	virtual void cForward(kmMat1<kcMat4f16>& x, const kcMat4f16&  w0, const kcMat2f16&  b0, const bool istrain = false) override;

	////////////////////////////////////////////////////
	// backward function
	virtual void Backward(kmMat1<kmMat4f32>& gx,
		                         kmMat4f32&  gw0,
		                         kmMat2f32&  gb0,
		            const kmMat1<kmMat4f32>& x,
		                   const kmMat4f32&  w0, 
	                   const kmMat2f32&  b0) override
	{};

	virtual void cBackward(kmMat1<kcMat4f32>& gx,       kcMat4f32& gw0,       kcMat2f32& gb0,
		             const kmMat1<kcMat4f32>&  x, const kcMat4f32&  w0, const kcMat2f32&  b0) override;
	virtual void cBackward(kmMat1<kcMat4f16>& gx,       kcMat4f16& gw0,       kcMat2f16& gb0,
		             const kmMat1<kcMat4f16>&  x, const kcMat4f16&  w0, const kcMat2f16&  b0) override;
};

// activation function layer
class kmDnnLyActf : public kmDnnLy
{
public:
	kmeDnnActf _actf = ACTF_RELU;

	/////////////////////////////////////
	// basic functions

	// constructor	
	kmDnnLyActf() { _type = LY_ACTF; };
	kmDnnLyActf(kmeDnnActf actf) : kmDnnLyActf()
	{
		Set(0, 0, actf);
	};
	kmDnnLyActf(int64 x0_idx, int64 x1_idx, kmeDnnActf actf) : kmDnnLyActf()
	{
		Set(x0_idx, x1_idx, actf);
	};

	/////////////////////////////////////
	// general functions
	
	void Set(int64 x0_idx, int64 x1_idx, kmeDnnActf actf)
	{
		_x0_idx = x0_idx;
		_x1_idx = x1_idx;
		_actf   = actf;
	};
			
	// get size
	virtual int64 GetByte() const { return sizeof(kmDnnLyActf); };

	// print info
	virtual void PrintInfo()
	{
		kmDnnLy::PrintInfo();
		switch(_actf)
		{
		case ACTF_RELU : PRINTFA(", ReLU\n"   ); break;
		case ACTF_SIGM : PRINTFA(", sigmoid\n"); break;
		case ACTF_SMAX : PRINTFA(", softmax\n"); break;
		}
	};

	/////////////////////////////////////
	// init function

	virtual void Init(kmMat1<kmMat4f32>& x, kmMat4f32& w0, kmMat2f32& b0) override
	{
		// create weight and bias
		// * Note that an activation layer doesn't need w0 and b0.

		// create output node
		Init(x);
	};	
	
	virtual void Init(kmMat1<kmMat4f32>& x) override
	{	
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		x(_x1_idx).Recreate(x(_x0_idx));
	};

	template<typename T>
	void _cInit(kmMat1<kcMat4<T>>& x)
	{	
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		x(_x1_idx).Recreate(x(_x0_idx));
	}

	virtual void cInit(kmMat1<kcMat4f32>& x, kcMat4f32& w0, kcMat2f32& b0) override { _cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x, kcMat4f16& w0, kcMat2f16& b0) override { _cInit(x); };

	virtual void cInit(kmMat1<kcMat4f32>& x) override { _cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x) override { _cInit(x); };

	////////////////////////////////////////////////////
	// forward function
	virtual void Forward(kmMat1<kmMat4f32>& x,
		                  const kmMat4f32&  w0, 
		                  const kmMat2f32&  b0, const bool istrain = false) override
	{
		// init variable and parameters
		const kmMat4f32& x0 = x(_x0_idx);
		      kmMat4f32& x1 = x(_x1_idx);
		
		switch(_actf)
		{
		case ACTF_RELU : x1.Copy(ReLU   (x0)); break;
		case ACTF_SIGM : x1.Copy(Sigmoid(x0)); break;
		case ACTF_SMAX : 
			for(int64 ib = 0; ib < x0.N4(); ++ib)
			{
				kmMat3f32 x0_ = x0.Mat3(ib);
				kmMat3f32 x1_ = x1.Mat3(ib);

				x1_.Copy(::Softmax(x0_));
			}
			break;
		}
	};

	template<typename T>
	void _cForward(kmMat1<kcMat4<T>>&  x, const kcMat4<T>& w0, const kcMat2<T>&  b0, const bool istrain)
	{
		// init variable and parameters
		const kcMat4<T>& x0 = x(_x0_idx);
		      kcMat4<T>& x1 = x(_x1_idx);

		switch(_actf)
		{
		case ACTF_RELU : ReLU   (x1, x0); break;
		case ACTF_SIGM : Sigmoid(x1, x0); break;
		case ACTF_SMAX : Softmax(x1, x0); break;
		}
	}

	virtual void cForward(kmMat1<kcMat4f32>& x, const kcMat4f32& w0, const kcMat2f32&  b0, const bool istrain = false) override { _cForward(x, w0, b0, istrain); };
	virtual void cForward(kmMat1<kcMat4f16>& x, const kcMat4f16& w0, const kcMat2f16&  b0, const bool istrain = false) override { _cForward(x, w0, b0, istrain); };

	void Softmax(kcMat4f32& x1, const kcMat4f32& x0, cudaStream_t s = 0);
	void Softmax(kcMat4f16& x1, const kcMat4f16& x0, cudaStream_t s = 0);

	////////////////////////////////////////////////////
	// backward function
	virtual void Backward(kmMat1<kmMat4f32>& gx,
		                         kmMat4f32&  gw0,
		                         kmMat2f32&  gb0,
		            const kmMat1<kmMat4f32>& x,
		                   const kmMat4f32&  w0, 
		                   const kmMat2f32&  b0) override
	{
		// init variable and parameters		
		const kmMat4f32&  x1 =  x(_x1_idx);
		      kmMat4f32& gx0 = gx(_x0_idx);
		const kmMat4f32& gx1 = gx(_x1_idx);
		const int64        n = gx0.N();
				
		switch(_actf)
		{
		case ACTF_RELU: 
			for(int64 i = 0; i < n; ++i)
			{
				gx0(i) += (x1(i) > 0)? gx1(i):0.f;
			}
			break;

		case ACTF_SIGM:
			for(int64 i = 0; i < n; ++i)
			{
				gx0(i) += x1(i)*(1.f - x1(i))*gx1(i);
			}
			break;

		case ACTF_SMAX:

			kmMat1f32 x1gx1(x1.N4()); x1gx1.SetZero();

			for(int64 i4 = 0; i4 < x1.N4(); ++i4)
			for(int64 i3 = 0; i3 < x1.N3(); ++i3)
			for(int64 i2 = 0; i2 < x1.N2(); ++i2)
			for(int64 i1 = 0; i1 < x1.N1(); ++i1)
			{
				x1gx1(i4) += x1(i1,i2,i3,i4)*gx1(i1,i2,i3,i4);
			}
			for(int64 i4 = 0; i4 < x1.N4(); ++i4)
			for(int64 i3 = 0; i3 < x1.N3(); ++i3)
			for(int64 i2 = 0; i2 < x1.N2(); ++i2)
			for(int64 i1 = 0; i1 < x1.N1(); ++i1)
			{
				gx0(i1,i2,i3,i4) += x1(i1,i2,i3,i4)*(gx1(i1,i2,i3,i4) - x1gx1(i4));
			}
			break;
		}
	};

	// backward function
	virtual void cBackward(kmMat1<kcMat4f32>& gx,        kcMat4f32& gw0,       kcMat2f32&  gb0,
		             const kmMat1<kcMat4f32>&  x,  const kcMat4f32&  w0, const kcMat2f32&   b0) override
	{
		// init variable and parameters		
		const kcMat4f32&  x1 =  x(_x1_idx);
		      kcMat4f32& gx0 = gx(_x0_idx);
		const kcMat4f32& gx1 = gx(_x1_idx);
		const int64        n = gx0.N();
		
		// * Note that gradient of softmax is the same with that of sigmoids
		switch(_actf)
		{
		case ACTF_RELU: cBackward_ReLU(gx0, gx1, x1); break;
		case ACTF_SIGM: cBackward_Sigm(gx0, gx1, x1); break;
		case ACTF_SMAX: cBackward_Smax(gx0, gx1, x1); break;
		}
	};

	virtual void cBackward(kmMat1<kcMat4f16>& gx,        kcMat4f16& gw0,       kcMat2f16&  gb0,
		             const kmMat1<kcMat4f16>&  x,  const kcMat4f16&  w0, const kcMat2f16&   b0) override
	{
		// init variable and parameters		
		const kcMat4f16&  x1 =  x(_x1_idx);
		      kcMat4f16& gx0 = gx(_x0_idx);
		const kcMat4f16& gx1 = gx(_x1_idx);
		const int64        n = gx0.N();
		
		// * Note that gradient of softmax is the same with that of sigmoids
		switch(_actf)
		{
		case ACTF_RELU: cBackward_ReLU(gx0, gx1, x1); break;
		case ACTF_SIGM: cBackward_Sigm(gx0, gx1, x1); break;
		case ACTF_SMAX: cBackward_Smax(gx0, gx1, x1); break;
		}
	};

	void cBackward_ReLU(kcMat4f32& gx0, const kcMat4f32& gx1, const kcMat4f32& x1);
	void cBackward_Sigm(kcMat4f32& gx0, const kcMat4f32& gx1, const kcMat4f32& x1);
	void cBackward_Smax(kcMat4f32& gx0, const kcMat4f32& gx1, const kcMat4f32& x1);

	void cBackward_ReLU(kcMat4f16& gx0, const kcMat4f16& gx1, const kcMat4f16& x1);
	void cBackward_Sigm(kcMat4f16& gx0, const kcMat4f16& gx1, const kcMat4f16& x1);
	void cBackward_Smax(kcMat4f16& gx0, const kcMat4f16& gx1, const kcMat4f16& x1);
};

// pooling layer
class kmDnnLyPool : public kmDnnLy
{
public:
	int64      _nw   = 1;         // width   of filter
	int64      _nh   = 1;         // height  of filter		
	int64      _strd = 1;         // stride	
	kmeDnnPool _pool = POOL_MAX;  // type of pooling

	/////////////////////////////////////
	// basic functions

	// constructor
	kmDnnLyPool() { _type = LY_POOL; };
	kmDnnLyPool(int64 nw, int64 nh, int64 strd, kmeDnnPool pool) : kmDnnLyPool()
	{
		Set(0, 0, nw, nh, strd, pool);
	};
	kmDnnLyPool(int64 x0_idx, int64 x1_idx, int64 nw, int64 nh, int64 strd, kmeDnnPool pool) 
	: kmDnnLyPool()
	{
		Set(x0_idx, x1_idx, nw, nh, strd, pool);
	};

	/////////////////////////////////////
	// general functions
	
	void Set(int64 x0_idx, int64 x1_idx, int64 nw, int64 nh, int64 strd, kmeDnnPool pool)
	{
		_x0_idx = x0_idx;
		_x1_idx = x1_idx;
		_nw     = nw;
		_nh     = nh;
		_strd   = strd;
		_pool   = pool;
	};

	// get size of output node
	int64 GetX1N1(int64 in_n1) { return (in_n1 - _nw)/_strd + 1; };
	int64 GetX1N2(int64 in_n2) { return (in_n2 - _nh)/_strd + 1; };
		
	// get size
	virtual int64 GetByte() const { return sizeof(kmDnnLyPool); };

	// print info
	virtual void PrintInfo()
	{
		kmDnnLy::PrintInfo();
		PRINTFA(", (%lld, %lld, %lld)", _nw, _nh, _strd);
		switch(_pool)
		{
		case POOL_MAX : PRINTFA(", max\n"   ); break;
		case POOL_AVG : PRINTFA(", average\n"); break;
		}
	};

	/////////////////////////////////////
	// init function
	
	virtual void Init(kmMat1<kmMat4f32>& x, kmMat4f32& w0, kmMat2f32& b0) override
	{
		// create weight and bias
		// * Note that pooling layer doesn't need w0 and b0.

		// create output node
		Init(x);
	};

	// init output node
	virtual void Init(kmMat1<kmMat4f32>& x) override
	{	
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		const kmMat4f32& x0 = x(_x0_idx);

		const int64 n1 = GetX1N1(x0.N1());
		const int64 n2 = GetX1N2(x0.N2());

		x(_x1_idx).Recreate(n1, n2, x0.N3(), x0.N4());
	};

	template<typename T>
	void _cInit(kmMat1<kcMat4<T>>& x)
	{	
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		const kcMat4f32& x0 = x(_x0_idx);

		const int64 n1 = GetX1N1(x0.N1());
		const int64 n2 = GetX1N2(x0.N2());

		x(_x1_idx).Recreate(n1, n2, x0.N3(), x0.N4());
	}

	virtual void cInit(kmMat1<kcMat4f32>& x, kcMat4f32& w0, kcMat2f32& b0) override { _cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x, kcMat4f16& w0, kcMat2f16& b0) override { _cInit(x); };

	virtual void cInit(kmMat1<kcMat4f32>& x) override { _cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x) override { _cInit(x); };

	/////////////////////////////////////
	// forward function

	// forward function... max pooling
	void Forward_Max(kmMat4f32& x1, const kmMat4f32& x0)
	{
		// init variable and parameters		
		const int64 x1n1 = x1.N1(), x1n2 = x1.N2();	
		const int64 x0n1 = x0.N1(), x0n2 = x0.N2(), nc = x0.N3(), nb = x0.N4();

		// calc pooling... main loop
		for(int64 b  = 0; b  < nb  ; ++b ) // batch
		for(int64 i3 = 0; i3 < nc  ; ++i3) // output, intput dim3
		for(int64 j2 = 0; j2 < x1n2; ++j2) // output dim2
		for(int64 j1 = 0; j1 < x1n1; ++j1) // output dim1
		{
			float val = FLOAT_MIN;

			for(int64 i2 = 0, k2 = _strd*j2; i2 < _nh; ++i2, ++k2) // height
			{	
				if(k2 >= x0n2) continue; // check out of boundary

				for(int64 i1 = 0, k1 = _strd*j1; i1 < _nw; ++i1, ++k1) // width
				{	
					if(k1 >= x0n1) continue; // check out of boundary

					// find max
					val = MAX(val, x0(k1,k2,i3,b));
				}
			}
			x1(j1,j2,i3,b) = val;
		}
	};

	// forward function... average pooling
	void Forward_Avg(kmMat4f32& x1, const kmMat4f32& x0)
	{
		// init variable and parameters		
		const int64 x1n1 = x1.N1(), x1n2 = x1.N2();	
		const int64 x0n1 = x0.N1(), x0n2 = x0.N2(), nc = x0.N3(), nb = x0.N4();		

		// calc pooling... main loop
		for(int64 b  = 0; b  < nb  ; ++b ) // batch
		for(int64 i3 = 0; i3 < nc  ; ++i3) // output, intput dim3
		for(int64 j2 = 0; j2 < x1n2; ++j2) // output dim2
		for(int64 j1 = 0; j1 < x1n1; ++j1) // output dim1
		{
			float sum = 0; int64 num = 0;

			for(int64 i2 = 0, k2 = _strd*j2; i2 < _nh; ++i2, ++k2) // height
			{	
				if(k2 >= x0n2) continue; // check out of boundary

				for(int64 i1 = 0, k1 = _strd*j1; i1 < _nw; ++i1, ++k1) // width
				{	
					if(k1 >= x0n1) continue; // check out of boundary

					// sum
					sum += x0(k1,k2,i3,b); ++num;
				}
			}
			x1(j1,j2,i3,b) = sum/(float) num;
		}
	};

	virtual void Forward(kmMat1<kmMat4f32>& x,
		                  const kmMat4f32&  w0,
		                  const kmMat2f32&  b0, const bool istrain = false) override
	{
		// init variable and parameters
		const kmMat4f32& x0 = x(_x0_idx);
		      kmMat4f32& x1 = x(_x1_idx); x1.SetZero();

		switch(_pool)
		{
		case POOL_MAX : Forward_Max(x1, x0); break;
		case POOL_AVG : Forward_Avg(x1, x0); break;
		}
	};

	virtual void cForward(kmMat1<kcMat4f32>& x, const kcMat4f32& w0, const kcMat2f32& b0, const bool istrain = false) override;
	virtual void cForward(kmMat1<kcMat4f16>& x, const kcMat4f16& w0, const kcMat2f16& b0, const bool istrain = false) override;

	////////////////////////////////////////////////////
	// backward function

	// backward function.... max pooling
	void Backward_Max(kmMat4f32&  gx0, const kmMat4f32& x0, const kmMat4f32&  gx1)
	{
		// init variable and parameters
		const int64 x1n1 = gx1.N1(), x1n2 = gx1.N2();	
		const int64 x0n1 = gx0.N1(), x0n2 = gx0.N2(), nc = gx0.N3(), nb = gx0.N4();

		// calc pooling... main loop
		for(int64 b  = 0; b  < nb  ; ++b ) // batch
		for(int64 i3 = 0; i3 < nc  ; ++i3) // output, intput dim3
		for(int64 j2 = 0; j2 < x1n2; ++j2) // output dim2
		for(int64 j1 = 0; j1 < x1n1; ++j1) // output dim1
		{
			float val = FLOAT_MIN;
			int64 max_k1 = 0, max_k2 = 0;

			for(int64 i2 = 0, k2 = _strd*j2; i2 < _nh; ++i2, ++k2) // height
			{	
				if(k2 >= x0n2) continue; // check out of boundary

				for(int64 i1 = 0, k1 = _strd*j1; i1 < _nw; ++i1, ++k1) // width
				{	
					if(k1 >= x0n1) continue; // check out of boundary

					// find max
					if(x0(k1,k2,i3,b) > val)
					{
						val = x0(k1,k2,i3,b); max_k1 = k1, max_k2 = k2;
					}
				}
			}
			gx0(max_k1,max_k2,i3,b) += gx1(j1,j2,i3,b);
		}
	};

	// backward function.... average pooling
	void Backward_Avg(kmMat4f32&  gx0, const kmMat4f32& gx1)
	{
		// init variable and parameters
		const int64 x1n1 = gx1.N1(), x1n2 = gx1.N2();	
		const int64 x0n1 = gx0.N1(), x0n2 = gx0.N2(), nc = gx0.N3(), nb = gx0.N4();

		// calc pooling... main loop
		for(int64 b  = 0; b  < nb  ; ++b ) // batch
		for(int64 i3 = 0; i3 < nc  ; ++i3) // output, intput dim3
		for(int64 j2 = 0; j2 < x1n2; ++j2) // output dim2
		for(int64 j1 = 0; j1 < x1n1; ++j1) // output dim1
		{
			// calc num
			int64 num = 0;

			for(int64 i2 = 0, k2 = _strd*j2; i2 < _nh; ++i2, ++k2) // height
			{
				if(k2 >= x0n2) continue;

				for(int64 i1 = 0, k1 = _strd*j1; i1 < _nw; ++i1, ++k1) // width
				{
					if(k1 >= x0n1) continue; ++num;
				}
			}

			// calc gx0
			const float val_gx0 = gx1(j1,j2,i3,b)/(float) num;

			for(int64 i2 = 0, k2 = _strd*j2; i2 < _nh; ++i2, ++k2) // height
			{
				if(k2 >= x0n2) continue;

				for(int64 i1 = 0, k1 = _strd*j1; i1 < _nw; ++i1, ++k1) // width
				{
					if(k1 >= x0n1) continue;

					gx0(k1,k2,i3,b) += val_gx0;
				}
			}
		}
	};
		
	virtual void Backward(kmMat1<kmMat4f32>& gx,
		                         kmMat4f32&  gw0,
		                         kmMat2f32&  gb0,
		            const kmMat1<kmMat4f32>& x,
		                   const kmMat4f32&  w0, 
		                   const kmMat2f32&  b0) override
	{
		// init variable and parameters
		      kmMat4f32& gx0 = gx(_x0_idx);
		const kmMat4f32& gx1 = gx(_x1_idx), x0 = x(_x0_idx);

		switch(_pool)
		{
		case POOL_MAX : Backward_Max(gx0, x0, gx1); break;
		case POOL_AVG : Backward_Avg(gx0,     gx1); break;
		}
	};

	virtual void cBackward(kmMat1<kcMat4f32>& gx,       kcMat4f32& gw0,       kcMat2f32& gb0,
		             const kmMat1<kcMat4f32>&  x, const kcMat4f32&  w0, const kcMat2f32&  b0) override;
	virtual void cBackward(kmMat1<kcMat4f16>& gx,       kcMat4f16& gw0,       kcMat2f16& gb0,
		             const kmMat1<kcMat4f16>&  x, const kcMat4f16&  w0, const kcMat2f16&  b0) override;
};

// combining layer
class kmDnnLyComb : public kmDnnLy
{
public:
	int64 _xc_idx = 0; // index of cominbing node set

	/////////////////////////////////////
	// basic functions

	// constructor	
	kmDnnLyComb() { _type = LY_COMB; };
	kmDnnLyComb(int64 xc_idx) : kmDnnLyComb()
	{		
		Set(0, 0, xc_idx);
	};
	kmDnnLyComb(int64 x0_idx, int64 x1_idx, int64 xc_idx) : kmDnnLyComb()
	{		
		Set(x0_idx, x1_idx, xc_idx);
	};

	/////////////////////////////////////
	// general functions
	
	void Set(int64 x0_idx, int64 x1_idx, int64 xc_idx)
	{
		_x0_idx = x0_idx;
		_x1_idx = x1_idx;
		_xc_idx = xc_idx;
	};

	// get size
	virtual int64 GetByte() const { return sizeof(kmDnnLyComb); };

	/////////////////////////////////////
	// init function
		
	virtual void Init(kmMat1<kmMat4f32>& x, kmMat4f32& w0, kmMat2f32& b0) override
	{
		// create weight and bias
		// * Note that an comb layer doesn't need w0 and b0.
		
		// init variables and parameters
		Init(x);
	};
		
	virtual void Init(kmMat1<kmMat4f32>& x) override
	{
		// init variables and parameters
		const kmMat4f32& x0 = x(_x0_idx);
		const kmMat4f32& xc = x(_xc_idx);

		const int64 x0_n1 = x0.N1(), x0_n2 = x0.N2(), x0_n3 = x0.N3();
		const int64 xc_n1 = xc.N1(), xc_n2 = xc.N2(), xc_n3 = xc.N3();
		const int64 n_bat = x0.N4();
				
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		kmMat4f32& x1 = x(_x1_idx);

		// determine the combining dimension
		if(x0_n1 == xc_n1 && x0_n2 == xc_n2) // combining in 3rd-dim for conv layer
		{
			x1.Recreate(x0_n1, x0_n2, x0_n3 + xc_n3, n_bat);
		}
		else if(x0_n2 == 1 && x0_n3 == 1 && xc_n2 == 1 && xc_n3 == 1) // combining in 1st-dim for full layer
		{
			x1.Recreate(x0_n1 + xc_n1, 1, 1, n_bat);
		}
		else
		{
			PRINTFA("[kmDnnLyComb::Init] the combining rule was violated\n");
			throw KE_WRONG_CONFIG;
		}
	};

	// init backward to realloc _x
	virtual void InitBackward(kmMat1<kmMat4f32>& x)  override
	{		
		kmMat4f32& x0 = x(_x0_idx);
		kmMat4f32& xc = x(_xc_idx);
		kmMat4f32& x1 = x(_x1_idx);

		const int64 x0_n1 = x0.N1(), x0_n2 = x0.N2(), x0_n3 = x0.N3();
		const int64 xc_n1 = xc.N1(), xc_n2 = xc.N2(), xc_n3 = xc.N3();
		const int64 n_bat = x0.N4(), x1_p3 = x1.P3();

		x0.Release();
		xc.Release();

		// * Note that 4th dimension is for mini batch, 
		// * so 3rd pitch is set as  3rd pitch of x1
		x0.SetP(&x1(0,0,    0, 0), x0_n1, x0_n2, x0_n3, n_bat, x0_n1, x0_n2, x1_p3);
		xc.SetP(&x1(0,0,x0_n3, 0), xc_n1, xc_n2, xc_n3, n_bat, xc_n1, xc_n2, x1_p3);
	};
	
	template<typename T>
	void _cInit(kmMat1<kcMat4<T>>& x)
	{
		// init variables and parameters
		const kcMat4f32& x0 = x(_x0_idx);
		const kcMat4f32& xc = x(_xc_idx);

		const int64 x0_n1 = x0.N1(), x0_n2 = x0.N2(), x0_n3 = x0.N3();
		const int64 xc_n1 = xc.N1(), xc_n2 = xc.N2(), xc_n3 = xc.N3();
		const int64 n_bat = x0.N4();
				
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		kcMat4<T>& x1 = x(_x1_idx);

		// determine the combining dimension
		if(x0_n1 == xc_n1 && x0_n2 == xc_n2) // combining in 3rd-dim for conv layer
		{
			x1.Recreate(x0_n1, x0_n2, x0_n3 + xc_n3, n_bat);
		}
		else if(x0_n2 == 1 && x0_n3 == 1 && xc_n2 == 1 && xc_n3 == 1) // combining in 1st-dim for full layer
		{
			x1.Recreate(x0_n1 + xc_n1, 1, 1, n_bat);
		}
		else
		{
			PRINTFA("[kcDnnLyComb::Init] the combining rule was violated\n");
			throw KE_WRONG_CONFIG;
		}
	}

	virtual void cInit(kmMat1<kcMat4f32>& x, kcMat4f32& w0, kcMat2f32& b0) override { _cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x, kcMat4f16& w0, kcMat2f16& b0) override { _cInit(x); };

	virtual void cInit(kmMat1<kcMat4f32>& x) override { _cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x) override { _cInit(x); };

	// init backward to realloc _x
	template<typename T>
	void _cInitBackward(kmMat1<kcMat4<T>>& x)
	{		
		kcMat4<T>& x0 = x(_x0_idx);
		kcMat4<T>& xc = x(_xc_idx);
		kcMat4<T>& x1 = x(_x1_idx);

		const int64 x0_n1 = x0.N1(), x0_n2 = x0.N2(), x0_n3 = x0.N3();
		const int64 xc_n1 = xc.N1(), xc_n2 = xc.N2(), xc_n3 = xc.N3();
		const int64 n_bat = x0.N4(), x1_p3 = x1.P3();

		x0.Release();
		xc.Release();

		// * Note that 4th dimension is for mini batch, 
		// * so 3rd pitch is set as  3rd pitch of x1
		x0.SetP(x1.P(0,0,    0, 0), x0_n1, x0_n2, x0_n3, n_bat, x0_n1, x0_n2, x1_p3);
		xc.SetP(x1.P(0,0,x0_n3, 0), xc_n1, xc_n2, xc_n3, n_bat, xc_n1, xc_n2, x1_p3);
	}

	virtual void cInitBackward(kmMat1<kcMat4f32>& x)  override { _cInitBackward(x); };
	virtual void cInitBackward(kmMat1<kcMat4f16>& x)  override { _cInitBackward(x); };

	////////////////////////////////////////////////////
	// forward function
	virtual void Forward(kmMat1<kmMat4f32>& x,
		                  const kmMat4f32&  w0, 
		                  const kmMat2f32&  b0, const bool istrain = false) override
	{
		// * Note that the comb layer doesn't need a forward process 
		// * since x(_x0_idx) and x(_cns_idx) shared a memory with x(_x1_idx)
	};

	////////////////////////////////////////////////////
	// backward function
	virtual void Backward(kmMat1<kmMat4f32>& gx,
		                         kmMat4f32&  gw0,
		                         kmMat2f32&  gb0,
		            const kmMat1<kmMat4f32>& x,
		                   const kmMat4f32&  w0, 
		                   const kmMat2f32&  b0) override
	{
		// * Note that the comb layer doesn't need a backward process 
		// * since gx(_x0_idx) and gx(_cns_idx) shared a memory with gx(_x1_idx)
	};

	// print info
	virtual void PrintInfo()
	{
		kmDnnLy::PrintInfo();
		PRINTFA(", xc: %3lld\n", _xc_idx);
	};
};

// adding layer
class kmDnnLyAdd : public kmDnnLy
{
public:
	int64 _xa_idx = 0; // index of adding node set

	/////////////////////////////////////
	// basic functions

	// constructor	
	kmDnnLyAdd() { _type = LY_ADD; };
	kmDnnLyAdd(int64 xa_idx) : kmDnnLyAdd()
	{		
		Set(0, 0, xa_idx);
	};
	kmDnnLyAdd(int64 x0_idx, int64 x1_idx, int64 xa_idx) : kmDnnLyAdd()
	{		
		Set(x0_idx, x1_idx, xa_idx);
	};

	/////////////////////////////////////
	// general functions
	
	void Set(int64 x0_idx, int64 x1_idx, int64 xa_idx)
	{
		_x0_idx = x0_idx;
		_x1_idx = x1_idx;
		_xa_idx = xa_idx;
	};
		
	// get size
	virtual int64 GetByte() const { return sizeof(kmDnnLyAdd); };

	/////////////////////////////////////
	// init function

	virtual void Init(kmMat1<kmMat4f32>& x, kmMat4f32& w0, kmMat2f32& b0) override
	{
		// create weight and bias
		// * Note that an add layer doesn't need w0 and b0.
		
		// create output node
		Init(x);
	};

	virtual void Init(kmMat1<kmMat4f32>& x) override
	{
		// init variables and parameters
		const kmMat4f32& x0 = x(_x0_idx);
		const kmMat4f32& xa = x(_xa_idx);

		ASSERTA( x0.IsEqualSizeDim(xa), "[kmDnnLyAdd::Init in 960]");
		
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		kmMat4f32& x1 = x(_x1_idx);

		x1.Recreate(x0);
	}

	template<typename T>
	void _cInit(kmMat1<kcMat4<T>>& x)
	{
		// init variables and parameters
		const kcMat4f32& x0 = x(_x0_idx);
		const kcMat4f32& xa = x(_xa_idx);

		ASSERTA( x0.IsEqualSizeDim(xa), "[kmDnnLyAdd::cInit in 1458]");
		
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		kcMat4<T>& x1 = x(_x1_idx);

		x1.Recreate(x0);
	}

	virtual void cInit(kmMat1<kcMat4f32>& x, kcMat4f32& w0, kcMat2f32& b0) override { _cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x, kcMat4f16& w0, kcMat2f16& b0) override { _cInit(x); };

	virtual void cInit(kmMat1<kcMat4f32>& x) override { _cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x) override { _cInit(x); };

	////////////////////////////////////////////////////
	// forward function

	virtual void Forward(kmMat1<kmMat4f32>& x,
		                  const kmMat4f32&  w0, 
		                  const kmMat2f32&  b0, const bool istrain = false) override
	{
		// init variables and parameters
		const kmMat4f32& x0 = x(_x0_idx);
		const kmMat4f32& xa = x(_xa_idx);
		      kmMat4f32& x1 = x(_x1_idx);

		x1.Copy(x0 + xa);
	};

	virtual void cForward(kmMat1<kcMat4f32>& x,
		                   const kcMat4f32&  w0, 
		                   const kcMat2f32&  b0, const bool istrain = false) override
	{
		// init variables and parameters
		const kcMat4f32& x0 = x(_x0_idx);
		const kcMat4f32& xa = x(_xa_idx);
		      kcMat4f32& x1 = x(_x1_idx);

		Add(x1, x0, xa); // x1 = x0 + xa
	};	

	virtual void cForward(kmMat1<kcMat4f16>& x,
		                   const kcMat4f16&  w0, 
		                   const kcMat2f16&  b0, const bool istrain = false) override
	{
		// init variables and parameters
		const kcMat4f16& x0 = x(_x0_idx);
		const kcMat4f16& xa = x(_xa_idx);
		      kcMat4f16& x1 = x(_x1_idx);

		Add(x1, x0, xa); // x1 = x0 + xa
	};

	////////////////////////////////////////////////////
	// backward function

	virtual void Backward(kmMat1<kmMat4f32>& gx,
		                         kmMat4f32&  gw0,
		                         kmMat2f32&  gb0,
		            const kmMat1<kmMat4f32>& x,
		                   const kmMat4f32&  w0, 
		                   const kmMat2f32&  b0) override
	{
		// init variables and parameters
		      kmMat4f32& gx0 = gx(_x0_idx);
			  kmMat4f32& gxa = gx(_xa_idx);
		const kmMat4f32& gx1 = gx(_x1_idx);
		
		gx0 += gx1;
		gxa += gx1;
	};

	virtual void cBackward(kmMat1<kcMat4f32>& gx,       kcMat4f32& gw0,       kcMat2f32& gb0,
		             const kmMat1<kcMat4f32>&  x, const kcMat4f32&  w0, const kcMat2f32&  b0) override
	{
		// init variables and parameters
		      kcMat4f32& gx0 = gx(_x0_idx);
			  kcMat4f32& gxa = gx(_xa_idx);
		const kcMat4f32& gx1 = gx(_x1_idx);
				
		gx0 += gx1;
		gxa += gx1;
	};
	virtual void cBackward(kmMat1<kcMat4f16>& gx,       kcMat4f16& gw0,       kcMat2f16& gb0,
		             const kmMat1<kcMat4f16>&  x, const kcMat4f16&  w0, const kcMat2f16&  b0) override
	{
		// init variables and parameters
		      kcMat4f16& gx0 = gx(_x0_idx);
			  kcMat4f16& gxa = gx(_xa_idx);
		const kcMat4f16& gx1 = gx(_x1_idx);
				
		gx0 += gx1;
		gxa += gx1;
	};

	// print info
	virtual void PrintInfo()
	{
		kmDnnLy::PrintInfo();
		PRINTFA(", xa: %lld\n", _xa_idx);
	};
};

// batch norminization layer (BN)
class kmDnnLyBnrm : public kmDnnLy
{
public:
	/////////////////////////////////////
	// basic functions

	// constructor	
	kmDnnLyBnrm() { _type = LY_BNRM; };
	kmDnnLyBnrm(int64 x0_idx, int64 x1_idx) : kmDnnLyBnrm()
	{
		Set(x0_idx, x1_idx);
	};

	/////////////////////////////////////
	// general functions
	
	void Set(int64 x0_idx, int64 x1_idx)
	{
		_x0_idx = x0_idx;
		_x1_idx = x1_idx;
	};
		
	// get size
	virtual int64 GetByte() const { return sizeof(kmDnnLyBnrm); };

	// print info
	virtual void PrintInfo()
	{
		kmDnnLy::PrintInfo();
		PRINTFA("\n");
	};

	/////////////////////////////////////
	// init function
		
	virtual void Init(kmMat1<kmMat4f32>& x, kmMat4f32& w0, kmMat2f32& b0) override
	{
		// init parameters
		const int64 nc = x(_x0_idx).N3(); // n of channel

		// create weight (bias isn't used)
		// * Note that w0(2:7) are mean, var and n of nodes, not conventional weights.
		// * So, they will be updated while training 
		// * and will be used only for inference (in case that n_bat is 1)
		w0.Recreate(8,nc,1,1); // w(0:7,ich) :{ gamma, beta, mean, var, m, ... }

		// set initial value
		for(int64 ic = 0; ic < nc; ++ic) 
		{
			w0(0,ic,0,0) = 1.f; // gamma
			w0(1,ic,0,0) = 0;   // beta
			w0(2,ic,0,0) = 0;   // mean for inference
			w0(3,ic,0,0) = 1.f; // var  for inference
			w0(4,ic,0,0) = 0;   // m    for inference
			w0(5,ic,0,0) = 0;   // mean of mini-batch
			w0(6,ic,0,0) = 1.f; // var  of mini-batch
			w0(7,ic,0,0) = 0;   // m    of mini-batch
		}

		// create output node
		Init(x);
	};
		
	virtual void Init(kmMat1<kmMat4f32>& x) override
	{	
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		x(_x1_idx).Recreate(x(_x0_idx));
	};

	template<typename T>
	void _cInit(kmMat1<kcMat4<T>>& x, kcMat4<T>& w0, kcMat2<T>& b0)
	{
		// init parameters
		const int64 nc = x(_x0_idx).N3(); // n of channel

		// create weight (bias isn't used)
		// * Note that w0(2:7) are mean, var and n of nodes, not conventional weights.
		// * So, they will be updated while training 
		// * and will be used only for inference (in case that n_bat is 1)
		w0.Recreate(8,nc,1,1); // w(0:4,ich) :{ gamma, beta, mean, var, m, ... }
		
		// set initial value
		kmMat4f32 w0_(8,nc,1,1);

		for(int64 ic = 0; ic < nc; ++ic) 
		{
			w0_(0,ic,0,0) = 1.f; // gamma
			w0_(1,ic,0,0) = 0;   // beta
			w0_(2,ic,0,0) = 0;   // mean for inference
			w0_(3,ic,0,0) = 1.f; // var  for inference
			w0_(4,ic,0,0) = 0;   // m    for inference
			w0_(5,ic,0,0) = 0;   // mean of mini-batch
			w0_(6,ic,0,0) = 1.f; // var  of mini-batch
			w0_(7,ic,0,0) = 0;   // m    of mini-batch
		}		
		w0 = kcMat4f32(w0_);

		// create output node
		cInit(x);
	}
		
	template<typename T>
	void _cInit(kmMat1<kcMat4<T>>& x)
	{	
		// create output node
		if(x.N() < _x1_idx + 1) x.SetN1(_x1_idx + 1);

		x(_x1_idx).Recreate(x(_x0_idx));
	}

	virtual void cInit(kmMat1<kcMat4f32>& x, kcMat4f32& w0, kcMat2f32& b0) override { _cInit(x, w0, b0); };
	virtual void cInit(kmMat1<kcMat4f16>& x, kcMat4f16& w0, kcMat2f16& b0) override { _cInit(x, w0, b0); };

	virtual void cInit(kmMat1<kcMat4f32>& x) override { _cInit(x); };
	virtual void cInit(kmMat1<kcMat4f16>& x) override { _cInit(x); };

	////////////////////////////////////////////////////
	// forward function
	virtual void Forward(kmMat1<kmMat4f32>& x,
		                  const kmMat4f32&  w0,
		                  const kmMat2f32&  b0, const bool istrain = false) override
	{
		// init variable and parameters
		const kmMat4f32& x0 = x(_x0_idx);
		      kmMat4f32& x1 = x(_x1_idx);

		const int64 n1 = x0.N1(), n2 = x0.N2();
		const int64 nc = x0.N3(); // n of channel
		const int64 nb = x0.N4(); // n of batch

		// main loop
		for(int64 ic = 0; ic < nc; ++ic)
		{
			// get mean and var.... mean, var
			float mean = 0, var = 0;

			if(istrain) // mini-batch while training
			{
				// calc mean, var, m of mini-batch
				for(int64 ib = 0; ib < nb; ++ib)
				for(int64 i2 = 0; i2 < n2; ++i2)
				for(int64 i1 = 0; i1 < n1; ++i1)
				{
					const float val = x0(i1,i2,ic,ib);
					mean += val;
					var  += val*val;
				}
				const float m = (float) n1*n2*nb;
				
				mean = mean/m;
				var  = var/m - mean*mean;

				// calc mean, var, m for inference
				const float mean0 = w0(2,ic,0,0);
				const float var0  = w0(3,ic,0,0);
				const float m0    = floorf(w0(4,ic,0,0)*0.9f);

				const float m1    = m + m0;
				const float mean1 = (m*mean + m0*mean0)/m1;
				const float var1  = (m*(var + mean*mean) + m0*(var0 + mean0*mean0))/m1 
					                - mean1*mean1;
				// update to w0
				w0(2,ic,0,0) = mean1; w0(3,ic,0,0) = var1; w0(4,ic,0,0) = m1; // for inference
				w0(5,ic,0,0) = mean;  w0(6,ic,0,0) = var;  w0(7,ic,0,0) = m;  // of mini-batch
 			}
			else // for inference
			{
				mean = w0(2,ic,0,0);
				var  = w0(3,ic,0,0);
			}

			// update x1(:,:,ic,:)
			const kmMat4f32 x0c = x0.Mat4(kmI(), kmI(), kmI(ic), kmI());
			      kmMat4f32 x1c = x1.Mat4(kmI(), kmI(), kmI(ic), kmI());

			const float a = w0(0,ic,0,0)/sqrtf(var + FLOAT_SMALL);
			const float b = w0(1,ic,0,0) - a*mean;

			x1c.Copy(a*x0c + b);
		}
	};

	virtual void cForward(kmMat1<kcMat4f32>& x, const kcMat4f32& w0, const kcMat2f32& b0, const bool istrain = false) override;
	virtual void cForward(kmMat1<kcMat4f16>& x, const kcMat4f16& w0, const kcMat2f16& b0, const bool istrain = false) override;

	////////////////////////////////////////////////////
	// backward function
	virtual void Backward(kmMat1<kmMat4f32>& gx,
		                         kmMat4f32&  gw0,
		                         kmMat2f32&  gb0,
		            const kmMat1<kmMat4f32>& x,
		                   const kmMat4f32&  w0, 
		                   const kmMat2f32&  b0) override
	{
		// init variable and parameters
		const kmMat4f32&  x0 =  x(_x0_idx);
		const kmMat4f32&  x1 =  x(_x1_idx);
		      kmMat4f32& gx0 = gx(_x0_idx);
		const kmMat4f32& gx1 = gx(_x1_idx);

		const int64 n1 = x0.N1(), n2 = x0.N2();
		const int64 nc = x0.N3();     // n of channel
		const int64 nb = x0.N4();     // n of batch
		const int64 m  = n1*n2*nb;    // n of nodes per a channel
		const float inv_m = 1.f/(float) m;

		// calc dl/dw and dl/dx0... gw0(2,nc,1,1), gx0(n1,n2,nc,nb)
		// * Note that gx0 was already set as zero in kmDnn::Backward()
		gw0.SetZero();

		// main loop
		for(int64 ic = 0; ic < nc; ++ic)
		{
			// get gamman, mean, var
			const float gamma = w0(0,ic,0,0);			
			const float mean  = w0(5,ic,0,0);
			const float var   = w0(6,ic,0,0) + FLOAT_SMALL;

			// calc sum of gyi(sum_gy), sum of xmi(sum_xm), sum of gyxmi(sum_gyxm)
			float sum_gy = 0, sum_xm = 0, sum_gyxm = 0;

			for(int64 ib = 0; ib < nb; ++ib)
			{
				for(int64 i2 = 0; i2 < n2; ++i2)
				for(int64 i1 = 0; i1 < n1; ++i1)
				{
					const float gyi = gx1(i1,i2,ic,ib);
					const float xmi =  x0(i1,i2,ic,ib) - mean;

					sum_gy   += gyi;
					sum_xm   += xmi;
					sum_gyxm += gyi*xmi;
				}
			}
			// calc gv, gm
			const float inv_sqrt_v       = 1.f/sqrtf(var);
			const float gamma_inv_sqrt_v = gamma*inv_sqrt_v;

			const float gv = -gamma_inv_sqrt_v/var*0.5f*sum_gyxm;
			const float gm = -gamma_inv_sqrt_v*sum_gy - gv*2.f*inv_m*sum_xm;
			
			// update gw (gg, gb)
			gw0(0,ic,0,0) = inv_sqrt_v*sum_gyxm; // grad of gamma
			gw0(1,ic,0,0) = sum_gy;              // grad of beta

			// calc and update gx0
			for(int64 ib = 0; ib < nb; ++ib)
			{
				for(int64 i2 = 0; i2 < n2; ++i2)
				for(int64 i1 = 0; i1 < n1; ++i1)
				{
					const float gyi = gx1(i1,i2,ic,ib);
					const float xmi =  x0(i1,i2,ic,ib) - mean;

					gx0(i1,i2,ic,ib) += gyi*gamma_inv_sqrt_v + inv_m*(2.f*gv*xmi + gm);
				}
			}
		}
	};
	virtual void cBackward(kmMat1<kcMat4f32>& gx,       kcMat4f32& gw0,       kcMat2f32& gb0,
		             const kmMat1<kcMat4f32>&  x, const kcMat4f32&  w0, const kcMat2f32&  b0) override;
	virtual void cBackward(kmMat1<kcMat4f16>& gx,       kcMat4f16& gw0,       kcMat2f16& gb0,
		             const kmMat1<kcMat4f16>&  x, const kcMat4f16&  w0, const kcMat2f16&  b0) override;
};

///////////////////////////////////////////////////////////////
// optimization class

// algorithm of optimzation for kmOpt1 (1st order optimzation)
enum kmeOpt1
{					  
	OPT1_GD,          // 0, gradient descent
	OPT1_MOMENTUM,    // 1, momentum
	OPT1_RMSPROP,     // 2, RMSProp
	OPT1_ADAM,        // 3, Adaptive moment estimation	
	OPT1_ADAGRAD,     // 4, Adaptive gradient	
	OPT1_ADADELTA,    // 5, Adaptive delta
	OPT1_NAG,         // 6, Nesterov accelerate gradient
	OPT1_SQN          // 7, simple quasi-Newton 
};

// base class of 1st order optimzation
class kmOpt1Base
{
public:
	int _itr = 0;       // number of iterations

	// init function
	//  n : number of design variables
	virtual void  Init(int64 n, const kmMat1f32& param = {}) {};
	virtual void cInit(int64 n, const kmMat1f32& param = {}) {}; // for cuda

	// update function
	//  step  : step size
	//  n_bat : batch size (gw is sum of batch)
	//          * Note that n_bat must be 1 if gw is average of batch
	virtual void  Update(kmMat1f32& w, const kmMat1f32& gw, float step) {};
	virtual void cUpdate(kcMat1f32& w, const kcMat1f32& gw, float step) {};
	virtual void cUpdate(kcMat1f16& w, const kcMat1f16& gw, float step) {};
};

// Gradient descent
class kmOpt1Gd : public kmOpt1Base
{
public:
	//////////////////////////////////////
	// virtual functions
	
	// update	
	virtual void cUpdate(kcMat1f32& w, const kcMat1f32& gw, float step = 0) override;
	virtual void cUpdate(kcMat1f16& w, const kcMat1f16& gw, float step = 0) override;
};

// ADAM 
class kmOpt1Adam : public kmOpt1Base
{
public:
	float _beta1  = 0.9f;   // general param1... forgetting factor of gradient
	float _beta2  = 0.999f; // general param2... forgetting factor of square of gradient
	
	kmMat1f32 _v;           // gradient with moment
	kmMat1f32 _s;           // square of gradient
	kcMat1f32 _cv;          // gradient with moment
	kcMat1f32 _cs;          // square of gradient

	//////////////////////////////////////
	// virtual functions

	// init
	virtual void Init(int64 n, const kmMat1f32& param = {}) override
	{
		_v.Recreate(n); _v.SetZero();
		_s.Recreate(n); _s.SetZero();

		Initparam(param);
	};	

	virtual void cInit(int64 n, const kmMat1f32& param = {}) override
	{
		_cv.Recreate(n); _cv.SetZero();
		_cs.Recreate(n); _cs.SetZero();

		Initparam(param);
	};

	void Initparam(const kmMat1f32& param)
	{
		ASSERTFA(param.N() == 0 || param.N() == 2, "[kmOpt1Adam] the number of param is not correct");
		
		if(param.N() == 2) { _beta1 = param(0); _beta2 = param(1); };
	};

	// update	
	virtual void cUpdate(kcMat1f32& w, const kcMat1f32& gw, float step = 0) override;
	virtual void cUpdate(kcMat1f16& w, const kcMat1f16& gw, float step = 0) override;
};

// Adaptive Delta (AdaDelta)
class kmOpt1Adadel : public kmOpt1Base
{
public:
	kmMat1f32 _s;      // square of gradient
	kmMat1f32 _d;      // square of delta w
	kcMat1f32 _cs;
	kcMat1f32 _cd;

	//////////////////////////////////////
	// virtual functions

	// init
	virtual void Init(int64 n, const kmMat1f32& param = {}) override
	{
		_d.Recreate(n); _d.SetZero();
		_s.Recreate(n); _s.SetZero();
	};	

	virtual void cInit(int64 n, const kmMat1f32& param = {}) override
	{
		_cd.Recreate(n); _cd.SetZero();
		_cs.Recreate(n); _cs.SetZero();
	};

	// update	
	virtual void cUpdate(kcMat1f32& w, const kcMat1f32& gw, float step = 0) override;
	virtual void cUpdate(kcMat1f16& w, const kcMat1f16& gw, float step = 0) override;
};

// SQN (simple quasi-Newton)
class kmOpt1Sqn : public kmOpt1Base
{
public:	
	kmMat1f32 _gm;       // gradient on t-1
	kmMat1f32 _s;        // square of gradient
	kmMat1f32 _d;		 // square of delta w

	kcMat1f32 _cgm;	
	kcMat1f32 _cs;
	kcMat1f32 _cd;

	//////////////////////////////////////
	// virtual functions

	// init
	virtual void Init(int64 n, const kmMat1f32& param = {}) override
	{
		_gm.Recreate(n); _gm.SetZero();
		_s .Recreate(n); _s .SetZero();
		_d .Recreate(n); _d .SetZero();
	};	

	virtual void cInit(int64 n, const kmMat1f32& param = {}) override
	{
		_cgm.Recreate(n); _cgm.SetZero();
		_cs .Recreate(n); _cs .SetZero();
		_cd .Recreate(n); _cd .SetZero();
	};

	// update	
	virtual void cUpdate(kcMat1f32& w, const kcMat1f32& gw, float step = 0) override;
	virtual void cUpdate(kcMat1f16& w, const kcMat1f16& gw, float step = 0) override;
};

// cover class of 1st order optimzation
class kmOpt1
{
public:
	kmOpt1Base*  _opt  = nullptr;

	// constructor
	kmOpt1() {};
	//kmOpt1(kmeProcType pt, kmeOpt1 alg, int64 n) { Set(pt, alg, n); };

	// destructor
	~kmOpt1() { Release(); };

	// release
	void Release() { if(_opt != nullptr) { delete _opt; _opt = nullptr; }};

	// init
	void Init(kmeProcType pt, kmeOpt1 alg, int64 n, const kmMat1f32& param = {})
	{		
		// check
		if(_opt != nullptr) Release();

		// init algorithm
		switch(alg)
		{
		case OPT1_GD       : _opt = (kmOpt1Base*) new kmOpt1Gd;     break;
		case OPT1_ADAM     : _opt = (kmOpt1Base*) new kmOpt1Adam;   break;		
		case OPT1_SQN      : _opt = (kmOpt1Base*) new kmOpt1Sqn;    break;
		case OPT1_ADADELTA : _opt = (kmOpt1Base*) new kmOpt1Adadel; break;
		default : 
			PRINTFA(" * [kmOpt1::Init] kmeOpt1(%d) was not supported\n", alg);
			throw KE_ASSERTION_FAILED;
		}

		// init class
		switch(pt)
		{
		case PT_CPU  : _opt-> Init(n, param); break;
		case PT_CUDA : _opt->cInit(n, param); break;
		default : 
			PRINTFA(" * [kmOpt1::Init] kmeProcType(%d) was not supported\n", pt); 
			throw KE_ASSERTION_FAILED;
		}
	};

	// update
	void  Update(kmMat1f32& w, const kmMat1f32& gw, float step = 0)
	{
		_opt->Update(w, gw, step);
	};
	void cUpdate(kcMat1f32& w, const kcMat1f32& gw, float step = 0)
	{
		_opt->cUpdate(w, gw, step);
	};
	void cUpdate(kcMat1f16& w, const kcMat1f16& gw, float step = 0)
	{
		_opt->cUpdate(w, gw, step);
	};
};

///////////////////////////////////////////////////////////////
// main class for dnn

// class of Deep learning Neural Networks 
// * Note that 4th dimension of _x is only for mini-batch
template<typename T>
class kmDnn
{
public:
	kmQue<kmDnnLy>     _ly;       // layers
	kmMat1<kmMat4f32>  _x;        // node (including input and output)
	kmMat1<kmMat4f32>  _w;        // weight
	kmMat1<kmMat2f32>  _b;        // bias

	kmeDnnLoss  _loss  = LOSS_MSE; // loss type
	int64       _n_bat = 1;        // number of (mini) batch	
	kmeProcType _pt    = PT_CPU;   // PT_CPU, PT_CUDA

	kmMat1<kcMat4<T>> _cx;         // only for cuda
	kmMat1<kcMat4<T>> _cw;         // only for cuda
	kmMat1<kcMat2<T>> _cb;	       // only for cuda

	kmMat1f32         _hol;        // history of loss (optional)

	/////////////////////////////////////
	// basic functions

	// constructor
	kmDnn() {};
	kmDnn(LPCWSTR file_name)                     { Load(file_name); };
	kmDnn(int64 max_n_ly, int64 max_ly_byte = 0) { Create(max_n_ly, max_ly_byte); };	

	// destructor
	virtual ~kmDnn() {};

	/////////////////////////////////////
	// general functions

	// get name of class
	virtual LPCSTR GetKmClass() const { return "kmDnn"; };

	// create memory for layers
	void Create(int64 max_n_ly, int64 max_ly_byte = 0)
	{
		if(max_ly_byte == 0) max_ly_byte = 256*max_n_ly;

		_ly.Create(max_n_ly, max_ly_byte);
	};

	// add layer into memory of layers
	template<class Y> int64 Add(Y& ly)
	{
		// check index of input node set
		if(ly._x0_idx == 0 && _ly.N() > 0)
		{
			ly._x0_idx = _ly.End()->_x1_idx;
		}

		// check index of output node set
		if(ly._x1_idx == 0)
		{
			ly._x1_idx = ly._x0_idx + 1;
		}
		
		return _ly.PushBack(ly);
	}

	// get layer
	kmDnnLy& operator()(int64 i) const { return *_ly(i); };

	// get number of layer
	int64 N() const { return _ly.N(); };
	
	/////////////////////////////////////
	// init functions

	// init node, weight and bias	
	void Init(int64 x0n1, int64 x0n2, int64 x0n3, int64 n_bat = 1)
	{
		// set number of batch
		_n_bat = n_bat;

		// init parameters
		int64 n_ly = _ly.N();

		// create kmMat for node, weight and bias
		_x.Recreate(0, n_ly + 1);
		_w.Recreate(n_ly);
		_b.Recreate(n_ly);

		// init input node _x(0)
		_x.PushBack();
		_x(0).Recreate(x0n1, x0n2, x0n3, _n_bat);

		// init _w, _b, _x
		for(int64 i = 0; i < n_ly; ++i)
		{
			_ly(i)->Init(_x, _w(i), _b(i));
		}

		// init backward to realloc _x for kmDnnLyComb
		InitBackward();

		// init cuda
		if(_pt == PT_CUDA) cInit(x0n1, x0n2, x0n3, n_bat);

		// init history of loss
		_hol.Recreate(0,128);
	};
		
	void cInit(int64 x0n1, int64 x0n2, int64 x0n3, int64 n_bat = 1)
	{
		// set number of batch
		_n_bat = n_bat;

		// init parameters
		int64 n_ly = _ly.N();

		// create kcMat
		_cx.Recreate(0, n_ly + 1);
		_cw.Recreate(n_ly);
		_cb.Recreate(n_ly);

		// init input node _cx(0)
		_cx.PushBack();
		_cx(0).Recreate(x0n1, x0n2, x0n3, _n_bat);

		// init _w, _b, _x
		for(int64 i = 0; i < n_ly; ++i)
		{
			_ly(i)->cInit(_cx, _cw(i), _cb(i));
		}

		// init backward to realloc _x for kmDnnLyComb
		cInitBackward();
	};

	// realloc _x for kmDnnLyComb
	void  InitBackward() { for(int64 i = _ly.N(); i--;) _ly(i)-> InitBackward( _x); };
	void cInitBackward() { for(int64 i = _ly.N(); i--;) _ly(i)->cInitBackward(_cx); };

	// init only node
	void InitNode(int64 x0n1, int64 x0n2, int64 x0n3, int64 n_bat = 1)
	{
		// set number of batch
		_n_bat = n_bat;
		
		// init parameters
		const int64 n_ly = _ly.N();

		// create kmMat for node
		_x.Recreate(0, n_ly + 1);

		// init input node _x(0)
		_x.PushBack();
		_x(0).Recreate(x0n1, x0n2, x0n3, _n_bat);

		// init _x
		for(int64 i = 0; i < n_ly; ++i) _ly(i)->Init(_x);

		// init backward to realloc _x for kmDnnLyComb
		InitBackward();
	};

	void cInitNode(int64 x0n1, int64 x0n2, int64 x0n3, int64 n_bat = 1)
	{
		// set number of batch
		_n_bat = n_bat;
		
		// init parameters
		const int64 n_ly = _ly.N();

		// create kcMat for node
		_cx.Recreate(0, n_ly + 1);

		// init input node _cx(0)
		_cx.PushBack();
		_cx(0).Recreate(x0n1, x0n2, x0n3, _n_bat);

		// init _x
		for(int64 i = 0; i < n_ly; ++i) _ly(i)->cInit(_cx);

		// init backward to realloc _x for kmDnnLyComb
		cInitBackward();
	};
	
	// init node for batch
	void InitNodeBatch(int64 n_bat)
	{	
		if(n_bat != _n_bat)
		{
			PRINTFA("[kmDnn::InitNodeBatch] batch %d --> %d\n", _n_bat, n_bat);

			InitNode(_x(0).N1(), _x(0).N2(), _x(0).N3(), n_bat);

			if(_pt == PT_CUDA) cInitNodeBatch(n_bat);
		}
	};
	
	void cInitNodeBatch(int64 n_bat)
	{	
		cInitNode(_cx(0).N1(), _cx(0).N2(), _cx(0).N3(), n_bat);
	};

	// set nodes as zeros
	void SetNodeZero()
	{
		for(int64 i =  _x.N(); i--; ) _x(i).SetZero();

		if(_pt == PT_CUDA) for(int64 i = _cx.N(); i--; ) _cx(i).SetZero();
	};
	
	// set type of loss
	void SetLoss(const kmeDnnLoss loss)	{ _loss = loss;	};

	// set type of process
	void SetProcType(const kmeProcType pt) { _pt = pt; };

	// init dropout mask
	void InitDropout(const kmMat1<kmMat3i8>& mask)
	{
		for(int64 i = 0; i < _ly.N(); ++i) _ly(i)->InitDropout(mask(i));
	};

	void cInitDropout(const kmMat1<kcMat3i8>& mask)
	{
		for(int64 i = 0; i < _ly.N(); ++i) _ly(i)->cInitDropout(mask(i));
	};
	
	/////////////////////////////////////////
	// loss functions

	// get loss
	float GetLoss(const kmMat4f32& t1) const
	{	
		switch(_loss)
		{
		case LOSS_MSE : return Mse(*_x.End(), t1);
		case LOSS_CEE : return Cee(*_x.End(), t1)/float(t1.N4());
		}
		return Mse(*_x.End(), t1);
	};

	float cGetLoss(const kcMat4f32& ct1) const
	{	
		switch(_loss)
		{
		case LOSS_MSE : return Mse(*_cx.End(), ct1);
		case LOSS_CEE : return Cee(*_cx.End(), ct1)/float(ct1.N4());
		}
		return Mse(*_cx.End(), ct1);
	};

	// get gradient of loss
	void GetGradLoss(kmMat4f32& gx, const kmMat4f32& t1) const
	{
		ASSERTA(gx.IsEqualSizeDim(t1), "[kmDnn::GetGradLoss in 416]");

		kmMat3f32& x1 = *_x.End();

		switch(_loss)
		{
		case LOSS_MSE: 
			for(int64 i = 0; i < t1.N(); ++i)
			{
				gx(i) = 2.f*(x1(i) - t1(i)); 
			}
			break;
		case LOSS_CEE:
			for(int64 i = 0; i < t1.N(); ++i)
			{
				gx(i) = -t1(i)/MAX(x1(i), 1e-10f);
			}
			break;
		}
	};

	void cGetGradLoss(kcMat4<T>& cgx, const kcMat4<T>& ct1) const
	{
		ASSERTA(cgx.IsEqualSizeDim(ct1), "[kmDnn::cGetGradLoss in 2029]");
		
		const kcMat4<T>& cx1 = *_cx.End();

		switch(_loss)
		{
		case LOSS_MSE: GradMse(cgx, cx1, ct1); break;
		case LOSS_CEE: GradCee(cgx, cx1, ct1); break;
		}
	};

	/////////////////////////////////////////
	// forward functions

	kmMat4f32 Forward(const kmMat4f32& x0, bool istrain = false, 
		              const kmMat1<kmMat3i8>& mask = 0)
	{
		// check num of batch
		if(x0.N4() != _x(0).N4()) InitNodeBatch(x0.N4());

		// copy input to the first node
		ASSERTA(x0.N() == _x(0).N(), "[kmDnn::Forward in 350] %d = %d", x0.N(), _x(0).N());
		
		// init parameters
		const int64 n_ly = _ly.N();

		// main loop
		if(_pt == PT_CUDA) // CUDA processing only for inference
		{
			if(_cx(0).GetType() == typeid(float)) _cx(0).Copy(x0);
			else                                  _cx(0) = kcMat4f32(x0);

			KC_CHECK_TIME_START;
			for(int64 i = 0; i < n_ly; ++i)
			{
				_ly(i)->cForward(_cx, _cw(i), _cb(i), istrain);
			}
			KC_CHECK_TIME_END("forwarding");

			if(_cx(-1).GetType() == typeid(float))          _cx(-1) .CopyTo(_x(-1));
			else                                  kcMat4f32(_cx(-1)).CopyTo(_x(-1));
		}
		else // CPU processing
		{
			_x(0).Copy(x0);

			for(int64 i = 0; i < n_ly; ++i)
			{	
				_ly(i)->Forward(_x, _w(i), _b(i), istrain);
			}
		}

		// copy output from the last node
		return _x(-1);
	};

	void cForward(const kcMat4<T>& cx0, bool istrain = false, 
		          const kmMat1<kcMat3i8>& cmask = 0)
	{
		// check num of batch
		if(cx0.N4() != _cx(0).N4()) InitNodeBatch(cx0.N4());
	
		// copy input to the first node
		ASSERTA(cx0.N() == _cx(0).N(), "[kmDnn::cForward in 2091] %d = %d", cx0.N(), _cx(0).N());
		
		// init parameters
		const int64 n_ly = _ly.N();
	
		_cx(0) = cx0;
	
		for(int64 i = 0; i < n_ly; ++i)
		{	
			// forward
			_ly(i)->cForward(_cx, _cw(i), _cb(i), istrain);
	
			// dropout
			if(cmask.N() > 0 && istrain && _ly(i)->GetDropout() > 0)
			{
				_ly(i)->cDropout(_cx, cmask(i));
			}
		}
	}

	/////////////////////////////////////////
	// backward functions

	void Backward(kmMat1<kmMat4f32>& gx, 
		          kmMat1<kmMat4f32>& gw, 
		          kmMat1<kmMat2f32>& gb,
		          const  kmMat4f32 & x0, 
		          const  kmMat4f32 & t1,
		          const  kmMat1<kmMat3i8>& mask = 0)
	{
		// init dropout
		InitDropout(mask);

		// forward
		Forward(x0, true);

		// set gx as zero
		// * Note that a node set can be connected to several node sets.
		// * So, gx should be set as zero before the backward process.
		for(int64 i = gx.N(); i--; ) gx(i).SetZero();

		// calc the last gx... *gx.End()		
		GetGradLoss(gx(-1), t1);
		
		// main loop
		for(int64 i = _ly.N(); i--;)
		{
			_ly(i)->Backward(gx, gw(i), gb(i), _x, _w(i), _b(i));
		};
	};

	void cBackward(kmMat1<kcMat4<T>>& gx, 
		           kmMat1<kcMat4<T>>& gw, 
		           kmMat1<kcMat2<T>>& gb,
		           const  kcMat4<T> & x0, 
		           const  kcMat4<T> & t1,
	               const kmMat1<kcMat3i8>& mask = 0)
	{
		// init dropout
		if(mask.N() > 0) cInitDropout(mask);

		// forward
		cForward(x0, true, mask);

		KC_CHECK_ERROR("kmDnn::cBackward/cForward");

		// set gx as zero
		// * Note that a node set can be connected to several node sets.
		// * So, gx should be set as zero before the backward process.
		for(int64 i = gx.N(); i--; ) gx(i).SetZero();
		
		// calc the last gx... *gx.End()
		cGetGradLoss(gx(-1), t1);

		// main loop
		for(int64 i = _ly.N(); i--;)
		{
			// dropout
			if(mask.N() > 0 && _ly(i)->GetDropout() > 0)
			{
				_ly(i)->cDropout(gx, mask(i));
			}

			// forward
			_ly(i)->cBackward(gx, gw(i), gb(i), _cx, _cw(i), _cb(i));
		};
		KC_CHECK_ERROR("kmDnn::cBackward/main loop");
	}

	////////////////////////////////////
	// load and save functions

	// save dnn
	void Save(const wchar* name)
	{
		// copy weight to CPU
		if(_pt == PT_CUDA) CopyWbToHost();

		// save data
		kmFile file(name, KF_NEW);
		
		const int64 v[] = {7,1,1,2}; // version
		const int64 n[] = {_x(0).N1(), _x(0).N2(), _x(0).N3() };

		file.Write   (v, 4);
		file.Write   (n, 3);
		file.WriteQue(&_ly);
		file.WriteMat(&_w);
		file.WriteMat(&_b);
		file.Write   (&_loss);
		file.Write   (&_pt);
		file.WriteMat(&_hol);

		file.Close();
	};

	// load dnn
	void Load(const wchar* name, const int64 n_bat = 1)
	{
		// load data
		PRINTFW(L"* %s", name);

		kmFile file(name, KF_READ);

		int64 v[4] = {0,}; // version
		int64 n[3] = {0,}; // x0.n1, x0.n2, x0.n3

		file.Read   (v, 4);
		file.Read   (n, 3);
		file.ReadQue(&_ly);
		file.ReadMat(&_w);
		file.ReadMat(&_b);
		file.Read   (&_loss);
		file.Read   (&_pt);
		file.ReadMat(&_hol);

		file.Close();

		PRINTFW(L" was loaded : ver%lld.%lld.%lld.%lld\n",v[0],v[1],v[2],v[3]);

		// reset virtual function pointer (__vfptr)
		for(int64 i = 0; i < _ly.N(); ++i)
		{
			switch(_ly(i)->GetType())
			{
			case LY_FULL: *((void**)_ly(i)) = GetVfptr<kmDnnLyFull>(); break;
			case LY_CONV: *((void**)_ly(i)) = GetVfptr<kmDnnLyConv>(); break;
			case LY_POOL: *((void**)_ly(i)) = GetVfptr<kmDnnLyPool>(); break;
			case LY_ACTF: *((void**)_ly(i)) = GetVfptr<kmDnnLyActf>(); break;
			case LY_COMB: *((void**)_ly(i)) = GetVfptr<kmDnnLyComb>(); break;
			case LY_ADD : *((void**)_ly(i)) = GetVfptr<kmDnnLyAdd >(); break;
			case LY_BNRM: *((void**)_ly(i)) = GetVfptr<kmDnnLyBnrm>(); break;
			case LY_CNVF: *((void**)_ly(i)) = GetVfptr<kmDnnLyCnvf>(); break;
			}
		}

		// init
		InitNode(n[0], n[1], n[2], n_bat); // init only node

		if(_pt == PT_CUDA)
		{
			cInit(n[0], n[1], n[2], n_bat); // init node and wb
			CopyWbToDevice();
		}
	};

	////////////////////////////////////
	// loss functions
	void PushBackLoss(const float loss)
	{
		_hol.PushBack(loss);
	};

	void ClearLoss()
	{
		_hol.SetN1(0);
	};

	/////////////////////////////////////
	// cuda functions

	void CopyWbToHost()	
	{
		for(int64 i = 0; i < _cw.N(); ++i)
		{
			if(_cw(i).N() > 0) _cw(i).CopyTo(_w(i));
			if(_cb(i).N() > 0) _cb(i).CopyTo(_b(i));
		}
		cudaDeviceSynchronize();
	};

	void CopyWbToDevice()
	{
		for(int64 i = 0; i < _cw.N(); ++i)
		{
			if(_cw(i).N() > 0) _cw(i).Copy(_w(i));
			if(_cb(i).N() > 0) _cb(i).Copy(_b(i));
		}
		cudaDeviceSynchronize();
	};

	void CopyNodeToHost()
	{
		for(int64 i = 0; i < _cx.N(); ++i)
		{
			if(_cx(i).N() > 0) _cx(i).CopyTo(_x(i));
		}
		cudaDeviceSynchronize();
	};

	void CopyNodeToDevice()	
	{
		for(int64 i = 0; i < _cx.N(); ++i)
		{
			if(_cx(i).N() > 0) _cx(i).Copy(_x(i));
		}
		cudaDeviceSynchronize();
	};

	//////////////////////////////////////////////////////
	// output result

	// print info of dnn
	void PrintInfo() const
	{
		PRINTFA("---------- kmDnn ----------\n");

		// processing type
		PRINTFA("Processing Type : %d (0: CPU, 1: CUDA)\n\n", _pt);

		// layer
		for(int64 i = 0; i < _ly.N(); ++i)
		{
			PRINTFA("[ly%3lld]", i); _ly(i)->PrintInfo();
		}
		PRINTFA("\n");

		// node
		int64 n = 0;

		for(int64 i = 0; i < _x.N(); ++i)
		{
			PRINTFA("[x%3lld]", i); _x(i).PrintDim(); n += _x(i).N();			
		}
		PRINTFA("------ total %d nodes (%d KB)\n\n", n, n*sizeof(float)>>10);

		// weight
		n = 0;

		for(int64 i = 0; i < _w.N(); ++i)
		{
			PRINTFA("[w%3lld]", i); _w(i).PrintDim(); n += _w(i).N();
		}
		PRINTFA("------ total %d weights (%d KB)\n\n", n, n*sizeof(float)>>10);

		// bias
		n = 0;

		for(int64 i = 0; i < _b.N(); ++i)
		{
			PRINTFA("[b%3lld]", i); _b(i).PrintDim(); n += _b(i).N();
		}
		PRINTFA("------ total %d bias (%d KB)\n\n", n, n*sizeof(float)>>10);

		PRINTFA("--------------------------\n");
	};
	
	// get label... label(n_bat)
	kmMat1u32 GetLabel()
	{
		// init variables
		const kmMat4f32& y = *_x.End();
		const int64     n1 = y.N1(), n_bat = y.N4();

		// check size
		ASSERTA(y.N2() == 1 && y.N3() == 1, "[kmDnn::GetLabel in 2035]");

		// get label
		kmMat1u32 label(n_bat);

		for(int64 ib = 0; ib < n_bat; ++ib)
		{
			float val = y(0,0,0,ib); int64 idx = 0;
			for(int64 i1 = 1; i1 < n1; ++i1)
			{
				const float yv = y(i1,0,0,ib);
				if(val < yv) { val = yv; idx = i1; }
			}
			label(ib) = (uint) idx;
		}
		return label;
	};

	// get accuracy (%)
	float GetAccuracy(const kmMat4f32& x0, const kmMat1u32& tlabel, const int64 n_batch = 0)
	{
		// check size
		ASSERTA(x0.N4() == tlabel.N1(),"[kmDnn::GetAccuracy in 2328]");

		// init parameters
		const int64 n_bat = (n_batch == 0)? MIN(1000, x0.N4()) : n_batch;		
		const int64 n_itr = x0.N4()/n_bat;

		// calc accuracy
		int64 n_win = 0;

		for(int64 itr = 0; itr < n_itr; ++itr)
		{
			const kmI i4 = {itr*n_bat, (itr + 1)*n_bat - 1};

			// predict
			Forward(x0.Mat4(kmI(), kmI(), kmI(), i4));

			// get label
			kmMat1u32 ylabel = GetLabel();

			for(int64 i = 0; i < i4.Len(); ++i)
			{
				if(ylabel(i) == tlabel(i + i4.s)) ++n_win;
			}
		}
		const float acc = n_win/(float)(n_itr*n_bat)*100.f;

		PRINTFA("** score : %.2f%% (%d/%d, batch %d)\n", acc, n_win, n_itr*n_bat, n_bat);

		return acc;
	};
};

// class of training networks
template<typename T>
class kmDnnTrain
{
public:
	int64              _n_set;              // number of training set
	kmMat4f32          _x0;                 // training set of input data (n1,n2,n3,n_set)
	kmMat4f32          _t1;                 // training set of ground true data (n1,n2,n3,n_set)
								            
	kmMat1<kmMat4f32>  _gx;                 // gradient of node
	kmMat1<kmMat4f32>  _gw;                 // gradient of weight
	kmMat1<kmMat2f32>  _gb;                 // gradient of bias
								            
	kmMat1<kmMat3i8>   _mask;               // mask for dropout (0: not, 1: dropout)
								            
	kmMat1<kmOpt1>     _optw;               // optimization for weight
	kmMat1<kmOpt1>     _optb;               // optimization for bias
	kmeOpt1            _optalg = OPT1_ADAM; // optimization algorithm
	kmMat1f32          _optparam;           // optimziation parameters

	kmMat1f32          _hol;                // history of loss

	// for cuda version
	kcMat4<T>          _cx0;    // training set of input data (n1,n2,n3,n_set)
	kcMat4<T>          _ct1;    // training set of ground true data (n1,n2,n3,n_set)

	kmMat1<kcMat4<T>>  _cgx;    // gradient of node
	kmMat1<kcMat4<T>>  _cgw;    // gradient of weight
	kmMat1<kcMat2<T>>  _cgb;    // gradient of bias
	kmMat1<kcMat3i8>   _cmask;  // mask for dropout (0: not, 1: dropout)

	/////////////////////////////////////
	// basic functions

	// constructor
	kmDnnTrain() {};

	kmDnnTrain(const kmMat4f32& x0, const kmMat4f32& t1, const kmeOpt1 alg = OPT1_ADAM)
	{
		Set(x0, t1, alg);
	};

	// destructor
	virtual ~kmDnnTrain() {};

	/////////////////////////////////////
	// general functions

	// get name of class
	virtual LPCSTR GetKmClass() const { return "kmDnnTrain"; };

	// set data
	void Set(const kmMat4f32& x0, const kmMat4f32& t1, const kmeOpt1 alg = OPT1_ADAM)
	{
		// init number of training sets
		_n_set = x0.N4();

		// set training data
		_x0.Set(x0);
		_t1.Set(t1);

		// set algorithm
		_optalg = alg;
	};

	// set algorithm
	void SetAlg(const kmeOpt1 alg, const kmMat1f32& param = {}) 
	{
		_optalg = alg;
		if(param.N() > 0) _optparam = param;
	};

	// init class
	void Init(kmDnn<T>& net, const int64 n_bat = 1)
	{
		// reset net for batch
		net.InitNodeBatch(n_bat);
		net.SetNodeZero();

		// allocate memory for gradient data		
		_gx.Recreate(net._x);
		_gw.Recreate(net._w);
		_gb.Recreate(net._b);		

		for(int64 i = 0; i < _gx.N(); ++i) { _gx(i).Recreate(net._x(i)); _gx(i).SetZero();}
		for(int64 i = 0; i < _gw.N(); ++i) { _gw(i).Recreate(net._w(i)); _gw(i).SetZero();}
		for(int64 i = 0; i < _gb.N(); ++i) { _gb(i).Recreate(net._b(i)); _gb(i).SetZero();}

		// realloc _gx for kmDnnLyComb
		for(int64 i = net._ly.N(); i--;)
		{
			(net._ly(i))->InitBackward(_gx);
		}

		// allocate _mask for dropout
		_mask.Recreate(net._ly.N());

		for(int64 i = 0; i < _mask.N(); ++i) if(net(i).GetDropout() > 0)
		{
			const kmMat4f32& x1 = net._x(net(i).GetX1());

			_mask(i).Recreate(x1.N1(), x1.N2(), x1.N3());
		}

		// init cuda memory
		if(net._pt == PT_CUDA)
		{
			// copy data to gpu memory
			_cx0 = _x0;
			_ct1 = _t1;

			// allocate memory for gradient data
			_cgx.Recreate(net._x);
			_cgw.Recreate(net._w);
			_cgb.Recreate(net._b);

			for(int64 i = 0; i < _cgx.N(); ++i) { _cgx(i).Recreate(net._x(i)); _cgx(i).SetZero();}
			for(int64 i = 0; i < _cgw.N(); ++i)	{ _cgw(i).Recreate(net._w(i)); _cgw(i).SetZero();}
			for(int64 i = 0; i < _cgb.N(); ++i) { _cgb(i).Recreate(net._b(i)); _cgb(i).SetZero();}

			// realloc _cgx for kmDnnLyComb
			for(int64 i = net.N(); i--;)
			{
				(net._ly(i))->cInitBackward(_cgx);
			}

			// allocate _mask for dropout
			_cmask.Recreate(net.N());

			for(int64 i = 0; i < _cmask.N(); ++i) if(net(i).GetDropout() > 0)
			{
				const kmMat4f32& x1 = net._x(net(i).GetX1());

				_cmask(i).Recreate(x1.N1(), x1.N2(), x1.N3());
			}

			// allocate opt
			_optw.Recreate(net.N());
			_optb.Recreate(net.N());

			for(int64 i = 0; i < net.N(); ++i)
			{
				//if(net._w(i).N() > 0) { _optw(i).Set(net._pt, _optalg, net._w(i).N()); _optw(i).Init();}
				//if(net._b(i).N() > 0) { _optb(i).Set(net._pt, _optalg, net._b(i).N()); _optb(i).Init();}

				if(net._w(i).N() > 0) { _optw(i).Init(net._pt, _optalg, net._w(i).N(), _optparam);} 
				if(net._b(i).N() > 0) { _optb(i).Init(net._pt, _optalg, net._b(i).N(), _optparam);} 
			}
		}
	};
	
	void Train(kmDnn<T>& net, float lrate = 0.1f,
		       const int64 n_bat = 1,                // number of batch
		       const int64 n_epc = 16,               // number of epooch
		       const int64 n_rep = 1)                // number of repitition
	{
		// init parameters
		const int64 n_ly   = net._ly.N();
		const int64 n_itr  = _n_set/n_bat;
		const char* pt_str = (net._pt == PT_CUDA) ? "CUDA":"CPU";

		PRINTFA("* ----------------------------- \n");
		PRINTFA("* start of training ----------- \n");
		PRINTFA("    proc  : %s\n", pt_str);
		PRINTFA("    n_epc : %d\n", n_epc);
		PRINTFA("    n_set : %d\n",_n_set);
		PRINTFA("    n_bat : %d\n", n_bat);
		PRINTFA("    n_itr : %d\n", n_itr);		
		PRINTFA("    n_rep : %d\n", n_rep);
		PRINTFA("    rate  : %f\n", lrate);
		PRINTFA("    alg   : %d\n",_optalg);
		PRINTFA("    loss  : %d\n", net._loss);

		float loss = 0;
								
		// main loop
		KM_CHECK_TIME_START;

		for(int64 i_epc = 0; i_epc < n_epc; ++i_epc)
		{
			PRINTFA(" [epc %d] ", i_epc);

			for(int64 i_itr = 0; i_itr < n_itr; ++i_itr)
			{
				PRINTFA(".");

				int64 set_s = i_itr*n_bat, set_e = (i_itr+1)*n_bat - 1;

				if(net._pt == PT_CUDA)
				{
					// set input and output
					kcMat4f32 cx0 = _cx0.Mat4(kmI(), kmI(), kmI(), kmI(set_s, set_e));
					kcMat4f32 ct1 = _ct1.Mat4(kmI(), kmI(), kmI(), kmI(set_s, set_e));

					for(int64 i_rep = n_rep; i_rep--;)
					{	
						// backward... gx, gw, gb
						if(i_epc >= n_epc - 3) net.cBackward(_cgx, _cgw, _cgb, cx0, ct1);
						else                   net.cBackward(_cgx, _cgw, _cgb, cx0, ct1, _cmask);
						
						KC_CHECK_ERROR("Train/net.cBackward");

						// loop of layers
						for(int64 k = 0; k < n_ly; ++k) if(!net(k).IsFixed())
						{	
							// update weight
							if(_cgw(k).N() > 0)
							{
								kcMat1<T> w = net._cw(k).Flat(), gw = _cgw(k).Flat();
							
								_optw(k).cUpdate(w, gw, lrate);
							}
							
							// update bias
							if(_cgb(k).N() > 0) 
							{
								kcMat1<T> b = net._cb(k).Flat(), gb = _cgb(k).Flat();
							
								_optb(k).cUpdate(b, gb, lrate);
							}
						}
						KC_CHECK_ERROR("Train/updating weight");
					}
					if(i_itr == n_itr - 1) loss = net.cGetLoss(ct1);

					KC_CHECK_ERROR("Train/net.cGetLoss");
				}
				else // PT_CPU
				{
					// set input and output
					kmMat4f32 x0 = _x0.Mat4(kmI(), kmI(), kmI(), kmI(set_s, set_e));
					kmMat4f32 t1 = _t1.Mat4(kmI(), kmI(), kmI(), kmI(set_s, set_e));

					for(int64 i_rep = n_rep; i_rep--;)
					{
						// backward... gx, gw, gb
						net.Backward(_gx, _gw, _gb, x0, t1);
						
						// update weight
						for(int64 k = 0; k < n_ly; ++k) if(!net(k).IsFixed())
						{
							if(_gw(k).N() > 0) net._w(k) -= lrate*_gw(k);
							if(_gb(k).N() > 0) net._b(k) -= lrate*_gb(k);
						}						
					}
					if(i_itr == n_itr - 1) loss = net.GetLoss(t1);
				}
			}
			//if(i_epc == 1) system("pause");

			// update hisory of loss
			net.PushBackLoss(loss);	
			PRINTFA(" %.8f\n", loss);
		}
		KM_CHECK_TIME_END_SEC("kmDnnTrain");

		PRINTFA("* end of training \n");
	};

	/////////////////////////////////////
	// cuda functions

	void CopyGradToHost()
	{
		for(int64 i = 0; i < _cgw.N(); ++i)
		{
			if(typeid(T) == typeid(float))
			{
				if(_cgw(i).N() > 0) _cgw(i).CopyTo(_gw(i));
				if(_cgb(i).N() > 0) _cgb(i).CopyTo(_gb(i));
			}
			else
			{
				if(_cgw(i).N() > 0) kcMat4f32(_cgw(i)).CopyTo(_gw(i));
				if(_cgb(i).N() > 0) kcMat2f32(_cgb(i)).CopyTo(_gb(i));
			}
		}

		for(int64 i = 0; i < _cgx.N(); ++i)
		{
			if(_cgx(i).N() > 0) _cgx(i).CopyTo(_gx(i));
		}
	};

	//////////////////////////////////////////////////////
	// load test data

	// load cifar10 data
	static void LoadCifar10(kmMat4f32& train_x, kmMat4f32& train_t, kmMat1u8& train_l,
		                    kmMat4f32& test_x , kmMat4f32& test_t,  kmMat1u8& test_l,
		                    LPCWSTR folder_name)
	{
		///////////////////////////////////////////////
		// load the kmf file if it exists
		{
			kmStrw file_name(L"%s/cifar10.kmf", folder_name);

			if(kmFile::Exist(file_name))
			{
				PRINTFW(L"* loading(%s)\n", file_name.P());

				kmFile file(file_name);

				file.ReadMat(&train_x);
				file.ReadMat(&train_t);
				file.ReadMat(&train_l);
				file.ReadMat(&test_x);
				file.ReadMat(&test_t);
				file.ReadMat(&test_l);

				file.Close();

				return;
			}
		}
		///////////////////////////////////////////////
		// load training data set 
		// : [label(1) r(1024) g (1024) b(1024) ] (1 + 3072 byte) x 10000 x 5
				
		// create data		
		train_l.Recreate(50000);

		// load data
		kmMat4u8 train_i(32,32,3,50000);
		
		for(int i = 0; i < 5; ++i)
		{
			kmStrw file_name(L"%s/data_batch_%d.bin", folder_name, i + 1);

			PRINTFW(L"* loading(%s)\n", file_name.P());

			kmFile file(file_name, KF_READ);

			for(int j = 0; j < 10000; ++j)
			{
				const int64 idx = j + i*10000;

				file.Read(train_l.P(idx), 1);	
				file.Read(train_i.P(0,0,0,idx), 3072);
			}
			file.Close();
		}

		///////////////////////////////////////////////
		// load test data set 
		// : [label(1) r(1024) g (1024) b(1024) ] (1 + 3072 byte) x 10000

		// create data		
		test_l.Recreate(10000);

		// load data
		kmMat4u8 test_i(32,32,3,10000);

		{
			kmStrw file_name(L"%s/test_batch.bin", folder_name);

			PRINTFW(L"* loading(%s)\n", file_name.P());

			kmFile file(file_name, KF_READ);

			for(int64 idx = 0; idx < 10000; ++idx)
			{
				file.Read(test_l.P(idx), 1);	
				file.Read(test_i.P(0,0,0,idx), 3072);
			}
			file.Close();
		}

		///////////////////////////////////////////////
		// convert to output

		PRINTFA("* converting data\n");

		train_x.Recreate(train_i);
		train_x.CopyTCast(train_i.P());

		kcMat4f32 cx = train_x; cx *= (1.f/255.f); cx.CopyTo(train_x);	//train_x *= (1.f/255.f);

		train_t.Recreate(10,1,1,50000);
		train_t.SetZero();
		for(int64 i = 50000; i--;) train_t(train_l(i),0,0,i) = 1.f;

		test_x.Recreate(test_i);
		test_x.CopyTCast(test_i.P());
		cx = test_x; cx *= (1.f/255.f); cx.CopyTo(test_x);	//test_x *= (1.f/255.f);

		test_t.Recreate(10,1,1,10000);
		test_t.SetZero();
		for(int64 i = 10000; i--;) test_t(test_l(i),0,0,i) = 1.f;

		// save data
		{
			PRINTFA("* saving data to cifar10.kmf\n");

			kmStrw file_name(L"%s/cifar10.kmf", folder_name);
			kmFile file(file_name, KF_NEW);

			file.WriteMat(&train_x);
			file.WriteMat(&train_t);
			file.WriteMat(&train_l);
			file.WriteMat(&test_x);
			file.WriteMat(&test_t);
			file.WriteMat(&test_l);
		}
	};

	// load mnist data
	static void LoadMnist(kmMat4f32& train_x, kmMat4f32& train_t, kmMat1u8& train_l,
		                  kmMat4f32& test_x , kmMat4f32& test_t,  kmMat1u8& test_l,
		                  LPCWSTR folder_name)
	{
		///////////////////////////////////////////////
		// load training image set of mnist... img

		kmStrw file_name(L"%s/train-images.idx3-ubyte", folder_name);
		kmFile file(file_name, KF_READ);

		int header[4] = {0, }, magic, n_img, n_row, n_col, n_item;

		file.Read(header, 4);

		magic = ENDIAN32(header[0]); // 2051
		n_img = ENDIAN32(header[1]); // number of images
		n_row = ENDIAN32(header[2]); // number of rows
		n_col = ENDIAN32(header[3]); // number of colums

		kmMat3u8 img(n_col, n_row, n_img);	file.Read(img.Begin(), img.GetSize());

		file.Close();

		PRINTFA("* training set : %d, %d, %d, %d\n", magic, n_img, n_row, n_col);

		///////////////////////////////////////////////
		// load training label file of mnist.... label

		file_name.SetStr(L"%s/train-labels.idx1-ubyte", folder_name);
		file.Open(file_name, KF_READ);
		file.Read(header, 2);

		magic  = ENDIAN32(header[0]); // 2049
		n_item = ENDIAN32(header[1]); // number of items

		train_l.Recreate(n_item); file.Read(train_l.P(), train_l.GetSize());

		file.Close();

		PRINTFA("* training label set : %d, %d\n", magic, n_item);

		///////////////////////////////////////////////
		// load test image set file of mnist.... test_img

		file_name.SetStr(L"%s/t10k-images.idx3-ubyte", folder_name);
		file.Open(file_name, KF_READ);
		file.Read(header, 4);

		magic = ENDIAN32(header[0]); // 2051
		n_img = ENDIAN32(header[1]); // number of images
		n_row = ENDIAN32(header[2]); // number of rows
		n_col = ENDIAN32(header[3]); // number of colums

		kmMat3u8 test_img(n_col, n_row, n_img); file.Read(test_img.P(), test_img.GetSize());

		file.Close();

		PRINTFA("* test image set : %d, %d, %d, %d\n", magic, n_img, n_row, n_col);

		///////////////////////////////////////////////
		// load training label file of mnist... test_label

		file_name.SetStr(L"%s/t10k-labels.idx1-ubyte", folder_name);
		file.Open(file_name, KF_READ);
		file.Read(header, 2);

		magic  = ENDIAN32(header[0]); // 2049
		n_item = ENDIAN32(header[1]); // number of items

		test_l.Recreate(n_item); file.Read(test_l.P(), test_l.GetSize());

		file.Close();

		PRINTFA("* test label set : %d, %d\n", magic, n_item);

		///////////////////////////////////////////////
		// convert to output

		train_x.Recreate(img.N1(), img.N2(), 1, img.N3());
		train_x.CopyTCast(img.P());

		kcMat4f32 cx = train_x; cx *= (1.f/255.f); cx.CopyTo(train_x);	//train_x *= (1.f/255.f);

		train_t.Recreate(10,1,1,img.N3());
		train_t.SetZero();
		for(int64 i = img.N3(); i--;) train_t(train_l(i),0,0,i) = 1.f;

		test_x.Recreate(test_img.N1(), test_img.N2(), 1, test_img.N3());
		test_x.CopyTCast(test_img.P());
		cx = test_x; cx *= (1.f/255.f); cx.CopyTo(test_x);	//test_x *= (1.f/255.f);

		test_t.Recreate(10,1,1,test_img.N3());
		test_t.SetZero();
		for(int64 i = test_img.N3(); i--;) test_t(test_l(i),0,0,i) = 1.f;
	};
};

// define type for kmDnn and kmDnnTrain
typedef kmDnn<float>         kmDnnf;
typedef kmDnn<half>          kmDnnh;
typedef kmDnnTrain<float>    kmDnnTrainf;
typedef kmDnnTrain<half>     kmDnnTrainh;

#endif /* __km7Dnn_H_INCLUDED_2019_03_05__ */
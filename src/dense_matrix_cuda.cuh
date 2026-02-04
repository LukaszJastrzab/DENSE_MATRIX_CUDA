#pragma once

#include <vector>
#include <stdexcept>
#include <type_traits>

#include <thrust/complex.h>

#include "utilities.cuh"

template< typename T >
class dense_matrix_cuda
{
	// Type definition for state of dense_matrix
	// =========================================
	enum class DYNAMIC_STATE : int
	{
		INIT,
		QR_DECOMPOSED
	};

public:
	/// constructors
	dense_matrix_cuda() = default;
	dense_matrix_cuda( const dense_matrix_cuda& );
	dense_matrix_cuda( dense_matrix_cuda&& ) = default;
	dense_matrix_cuda( size_t rows, size_t cols );

	/// destructor
	~dense_matrix_cuda();

	/// sets matrix sizes and allocates memory
	void init( size_t rows, size_t cols );
	/// adds elements and throws exception if row / col is out of range
	void set_element( T value, size_t row, size_t col );

	/// performs QR decomposition using CUDA data
	void QR_decomposition();
	/// decomposes matrix "in situ" to factors QR using Householder method
	//__device__
	//void QR_step( const int step );

private:
	/// current state of matrix
	DYNAMIC_STATE m_dynamic_state{ DYNAMIC_STATE::INIT };

	/// amount of rows
	size_t m_rows{ 0 };
	/// amount of columns
	size_t m_cols{ 0 };

	/// flattened matrix data
	std::vector< T > m_matrix;

	/// flattened matrix data for GPU device
	T* m_d_matrix{ nullptr };
	/// for QU decomposition (on device)
	T* m_d_betas{ nullptr };
	T* m_d_v_firsts{ nullptr };
};

template< typename T >
dense_matrix_cuda< T >::dense_matrix_cuda( size_t rows, size_t cols )
{
	init( rows, cols );
}

template< typename T >
dense_matrix_cuda< T >::dense_matrix_cuda( const dense_matrix_cuda& A )
	:
	m_dynamic_state( A.m_dynamic_state ),
	m_rows( A.m_rows ),
	m_cols( A.m_cols ),
	m_matrix( A.m_matrix )
{
}

template< typename T >
dense_matrix_cuda< T >::~dense_matrix_cuda()
{
	if( m_d_matrix )
		cudaFree( m_d_matrix );
	if( m_d_betas )
		cudaFree( m_d_betas );
	if( m_d_v_firsts )
		cudaFree( m_d_v_firsts );
}

template< typename T >
void dense_matrix_cuda< T >::init( size_t rows, size_t cols )
{
	m_rows = rows;
	m_cols = cols;

	m_matrix.resize( m_rows * m_cols, T{} );
}

template< typename T >
void dense_matrix_cuda< T >::set_element( T value, size_t row, size_t col )
{
	auto elem_idx = calc_elem_idx( row, col, m_cols );

	if( elem_idx >= m_matrix.size() )
		throw std::out_of_range( "dense_matrix_cuda< T >::set_element - elem_idx >= m_matrix.size()" );

	m_matrix[ elem_idx ] = value;
}

template< typename T >
__global__
void QR_first_step( T* A_in, T* A_out, T* betas, T* v_firsts, const int A_rows, const int A_cols, const int step )
{
	int tid = threadIdx.x;
	int block_size = blockDim.x;

	extern __shared__ unsigned char sdata_raw[];
	double* ndata = reinterpret_cast< double* >( sdata_raw );
	T* vTv = reinterpret_cast< T* >( sdata_raw + block_size * sizeof( double ) );

	double sum{};
	T vTv_sum{};

	int row = step + tid;
	while( row < A_rows )
	{
		size_t a_idx = calc_elem_idx( row, step, A_cols );
		T a_rs = A_in[ a_idx ];
		sum += norm2( a_rs );
		vTv_sum += conjugate( a_rs ) * a_rs;
		row += block_size;
	}

	ndata[ tid ] = sum;
	vTv[ tid ] = vTv_sum;
	__syncthreads();

	// sum up using reduction
	// ======================
	for( unsigned int s = block_size / 2; s > 0; s >>= 1 )
	{
		if( tid < s )
		{
			ndata[ tid ] += ndata[ tid + s ];
			vTv[ tid ] += vTv[ tid + s ];
		}
		__syncthreads();
	}

	if( tid == 0 )
	{
		size_t a_idx = calc_elem_idx( step, step, A_cols );
		T a_ss = A_in[ a_idx ];
		double alpha_abs = abs_val( a_ss );
		T sign = ( alpha_abs != 0.0 ? -( a_ss ) / T( alpha_abs ) : T{ -1 } );
		T sign_norm = sign * sqrt( ndata[ 0 ] );

		vTv[ 0 ] -= conjugate( a_ss ) * a_ss; // subtract wrong element;
		a_ss -= sign_norm;
		vTv[ 0 ] += conjugate( a_ss ) * a_ss; // add right one

		A_in[ a_idx ] = sign_norm;

		betas[ step ] = 2.0 / vTv[ 0 ];
		v_firsts[ step ] = a_ss;
	}
	__syncthreads();
}


template< typename T >
__global__
void QR_compute_vTA( const T* A_in, T* vTA, const T* v_firsts, const int A_rows, const int A_cols, const int step )
{
	int col = step + threadIdx.x + blockDim.x * blockIdx.x + 1;

	if( col >= A_cols )
		return;

	T sum{ conjugate( v_firsts[ step ] ) * A_in[ calc_elem_idx( step, col, A_cols ) ] };

	for( int s = step + 1; s < A_rows; ++s )
		sum += conjugate( A_in[ calc_elem_idx( s, step, A_cols ) ] ) * A_in[ calc_elem_idx( s, col, A_cols ) ];

	vTA[ col ] = sum;
}

template< typename T >
__global__
void QR_compute_reflections( const T* A_in, T* A_out, const T* vTA, const T* betas, const T* v_firsts, const int A_rows, const int A_cols, const int step )
{
	int col = step + threadIdx.x + blockDim.x * blockIdx.x;
	int row = step + threadIdx.y + blockDim.y * blockIdx.y - 1;

	if( row >= A_rows || col >= A_cols || row < 0 )
		return;

	T beta = betas[ step ];
	T vta = vTA[ col ];

	int a_idx = calc_elem_idx( row, col, A_cols );

	if( row < step || col <= step)
		A_out[ a_idx ] = A_in[ a_idx ];
	else
	{
		if( row > step )
			A_out[ a_idx ] = A_in[ a_idx ] - beta * A_in[ calc_elem_idx( row, step, A_cols ) ] * vta;
		else
			A_out[ a_idx ] = A_in[ a_idx ] - beta * v_firsts[ step ] * vta;
	}
}

template< typename T >
void dense_matrix_cuda< T >::QR_decomposition()
{
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::exception( "dense_matrix_cuda< T >::QR_decomposition() - m_dynamic_state != DYNAMIC_STATE::INIT" );

	cudaMalloc( &m_d_matrix, m_matrix.size() * sizeof( T ) );
	cudaMemcpy( m_d_matrix, m_matrix.data(), m_matrix.size() * sizeof( T ), cudaMemcpyHostToDevice );

	auto max_steps = std::min( m_rows - 1, m_cols );

	cudaMalloc( &m_d_betas, max_steps * sizeof( T ) );
	cudaMalloc( &m_d_v_firsts, max_steps * sizeof( T ) );

	T* d_matrix_out{ nullptr };
	cudaMalloc( &d_matrix_out, m_matrix.size() * sizeof( T ) );
	T* d_vTA{ nullptr };
	cudaMalloc( &d_vTA, m_cols * sizeof( T ) );

	const int TX1 = 256;
	const int TX2 = 256;
	const int TX3 = 16, TY3 = 16;

	const int st1_lmem_size = TX1 * ( sizeof( double ) + sizeof( T ) );
	const dim3 blockSize1( TX1 );
	const dim3 gridSize1( 1 );
	const dim3 blockSize2( TX2 );
	const dim3 blockSize3( TX3, TY3 );

	const int d_rows{ static_cast< int >( m_rows ) };
	const int d_cols{ static_cast< int >( m_cols ) };

	for ( int step{ 0 }; step < max_steps; ++step )
	{
		QR_first_step <<< gridSize1, blockSize1, st1_lmem_size >>>
			( m_d_matrix, d_matrix_out, m_d_betas, m_d_v_firsts, d_rows, d_cols, step );

		//const dim3 gridSize2( div_up( m_cols, TX2 ) );
		const dim3 gridSize2( div_up( m_cols - step - 1, TX2 ) );  // to be checked

		QR_compute_vTA <<< gridSize2, blockSize2 >>>
			( m_d_matrix, d_vTA, m_d_v_firsts, d_rows, d_cols, step );

		//const dim3 gridSize3( div_up( m_cols, TX3 ), div_up( m_rows, TY3 ) );
		const dim3 gridSize3( div_up( m_cols - step, TX3 ), div_up( m_rows - step + 1, TY3 ) ); // to be checked

		QR_compute_reflections <<< gridSize3, blockSize3 >>>
			( m_d_matrix, d_matrix_out, d_vTA, m_d_betas, m_d_v_firsts, d_rows, d_cols, step );

		std::swap( m_d_matrix, d_matrix_out );
	}

	// test
	//const int size = 7 * 7;
	//T AAA[ size ];
	//cudaMemcpy( AAA, m_d_matrix, size * sizeof( T ), cudaMemcpyDeviceToHost );
	//AAA[ 0 ] = AAA[ 0 ];
	// test


	cudaFree( d_vTA );
	cudaFree( d_matrix_out );

	m_dynamic_state = DYNAMIC_STATE::QR_DECOMPOSED;
}

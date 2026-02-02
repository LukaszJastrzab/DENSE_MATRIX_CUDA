#pragma once

#include <vector>
#include <stdexcept>
#include <type_traits>

#include <thrust/complex.h>

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

	/// for LU decomposition
	//std::vector< T > m_pivots;
	/// for QR decomposition
	//std::vector< T > m_betas;
	//std::vector< T > m_v_firsts;

	/// flattened matrix data for GPU device
	T* m_d_matrix{ nullptr };
	/// for QU decomposition (on device)
	T* m_d_betas{ nullptr };
	T* m_d_v_firsts{ nullptr };
};

__host__ __device__ __forceinline__
int div_up( int a, int b )
{
	return ( a + b - 1 ) / b;
}

__host__ __device__ __forceinline__
size_t calc_elem_idx( size_t row, size_t col, size_t A_cols )
{
	return col + row * A_cols;
}

template< typename T >
__host__ __device__ __forceinline__
double norm2( const T& x )
{
	return x * x;
}

template< typename T >
__host__ __device__ __forceinline__
double norm2( const thrust::complex< T >& x )
{
	return thrust::norm( x );
}

template< typename T >
__host__ __device__ __forceinline__
double abs_val( const T& x )
{
	return x >= T( 0 ) ? x : -x;
}

template< typename T >
__host__ __device__ __forceinline__
double abs_val( const thrust::complex< T >& x )
{
	return thrust::abs( x );
}

template< typename T >
__host__ __device__ __forceinline__
T conjugate( const T& x )
{
	return x;
}

template< typename T >
__host__ __device__ __forceinline__
thrust::complex< T > conjugate( const thrust::complex< T >& x )
{
	return thrust::conj( x );
}

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
void QR_step( T* A, T* betas, T* v_firsts, const int A_rows, const int A_cols, const int step )
{
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int block_size = blockDim.x * blockDim.y;

	extern __shared__ unsigned char sdata_raw[];
	double* ndata = reinterpret_cast< double* >( sdata_raw );
	T* vTv = reinterpret_cast< T* >( sdata_raw + block_size * sizeof( double ) );
	T* v = reinterpret_cast< T* >( sdata_raw + block_size * ( sizeof( double ) + sizeof( T ) ) );

	// first calculate sub column norm
	// and multiplication vTv
	// ===============================
	int col_len = A_rows - step;
	int col_len_per_thread = div_up( col_len, block_size );

	double sum{};
	T vTv_sum{};

	int row = step + tid;
	while( row < A_rows )
	{
		v[ row ] = A[ calc_elem_idx( row, step, A_cols ) ];
		sum += norm2( v[ row ] );
		vTv_sum += conjugate( v[ row ] ) * v[ row ];
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
		T a_kk = v[ 0 ];
		double alpha_abs = abs_val( a_kk );
		T sign = ( alpha_abs != 0.0 ? -( a_kk ) / T( alpha_abs ) : T{ -1 } );
		T sign_norm = sign * sqrt( ndata[ 0 ] );

		vTv[ 0 ] -= conjugate( v[ 0 ] ) * v[ 0 ]; // subtract wrong element;
		v[ 0 ] -= sign_norm;
		vTv[ 0 ] += conjugate( v[ 0 ] ) * v[ 0 ]; // add right one

		if( blockIdx.x == 0 && blockIdx.y == 0 )
		{
			A[ calc_elem_idx( step, step, A_cols ) ] = sign_norm;
			betas[ step ] = 2.0 / vTv[ 0 ];
			v_firsts[ step ] = v[ 0 ];
		}
	}
	__syncthreads();



}

template< typename T >
void dense_matrix_cuda< T >::QR_decomposition()
{
	cudaMalloc( &m_d_matrix, m_matrix.size() * sizeof( T ) );
	cudaMemcpy( m_d_matrix, m_matrix.data(), m_matrix.size() * sizeof( T ), cudaMemcpyHostToDevice );

	auto max_steps = std::min( m_rows - 1, m_cols );

	cudaMalloc( &m_d_betas, max_steps * sizeof( T ) );
	cudaMalloc( &m_d_v_firsts, max_steps * sizeof( T ) );

	const int TX = 16, TY = 8;
	const int v_size = m_rows - 0; // - step
	const int lmem_size = TX * TY * ( sizeof( double ) + sizeof( T ) ) + v_size * sizeof( T );
	const dim3 blockSize( TX, TY );
	const dim3 gridSize( div_up( m_cols, TX ), div_up( m_rows, TY ) );
	QR_step << < gridSize, blockSize, lmem_size >> > ( m_d_matrix, m_d_betas, m_d_v_firsts, static_cast< int >( m_rows ), static_cast< int >( m_cols ), 0 );
}

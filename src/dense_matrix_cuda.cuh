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
	/// initializes CUDA data
	void init_cuda();
	/// adds elements and throws exception if row / col is out of range
	void set_element( T value, size_t row, size_t col );

	/// decomposes matrix "in situ" to factors QR using Householder method
	__device__
	void QR_step( const int step );

private:

	/// divide to upper value
	__host__ __device__
	int div_up( int a, int b ) const;

	/// calculates index to flattened matrix element
	__host__ __device__
	size_t calc_elem_idx( size_t row, size_t col );

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

template< typename T >
__host__ __device__
int dense_matrix_cuda< T >::div_up( int a, int b ) const
{
    return ( a + b - 1 ) / b;
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
void dense_matrix_cuda< T >::init_cuda()
{
	cudaMalloc( &m_d_matrix, m_matrix.size() * sizeof( T ) );
	cudaMemcpy( m_d_matrix, m_matrix.data(), m_matrix.size() * sizeof( T ), cudaMemcpyHostToDevice );

	auto max_steps = std::min( m_rows - 1, m_cols );

	cudaMalloc( &m_d_betas, max_steps * sizeof( T ) );
	cudaMalloc( &m_d_v_firsts, max_steps * sizeof( T ) );
}

template< typename T >
__host__ __device__
size_t dense_matrix_cuda< T >::calc_elem_idx( size_t row, size_t col )
{
	return col + row * m_cols;
}


template< typename T >
void dense_matrix_cuda< T >::set_element( T value, size_t row, size_t col )
{
	auto elem_idx = calc_elem_idx( row, col );

	if( elem_idx >= m_matrix.size() )
		throw std::out_of_range( "dense_matrix_cuda< T >::set_element - elem_idx >= m_matrix.size()" );

	m_matrix[ elem_idx ] = value;
}

template< typename T >
__device__
void dense_matrix_cuda< T >::QR_step( const int step )
{
	extern __shared__ T sdata[]; // size = blocksize + m_rows

	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int block_size = blockDim.x * blockDim.y;

	// first calculate sub column norm
	// ===============================
	int col_len = m_rows - step;
	int col_len_per_thread = div_up( col_len, block_size );

	T sum{};

	int row = step + tid;
	while( row < m_rows )
	{
		sum += thrust::norm( m_d_matrix[ calc_elem_idx( row, step ) ] );
		row += block_size;
	}

	sdata[ tid ] = sum;
	__syncthreads();

	// sum up using reduction
	// ======================
	for( unsigned int s = block_size / 2; s > 0; s >>= 1 )
	{
		if( tid < s )
			sdata[ tid ] += sdata[ tid + s ];
		__syncthreads();
	}

	if( tid == 0 )
	{
		T a_kk = m_d_matrix[ calc_elem_idx( step, step ) ];
		double alpha_abs = std::abs( a_kk );
		T sign = ( alpha_abs != 0.0 ? -( a_kk ) / alpha_abs : T{ -1 } );
		T sign_norm = sign * col_norm;
	}

	__syncthreads();





}

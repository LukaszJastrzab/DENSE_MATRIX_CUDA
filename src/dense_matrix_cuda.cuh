#pragma once

#include <vector>
#include <stdexcept>
#include <type_traits>

#include <thrust/complex.h>
#include <cublas_v2.h>

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

	// it counts value r := Ax - b
	void count_residual_vector( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const;

	/// performs QR decomposition using CUDA data
	void QR_decomposition();
	/// solves equation Ax=b, where A is decomposed to factors QR (by Householders method)
	void solve_QR( std::vector< T >& x, const std::vector< T >& b ) const;
	/// solves equation Ax=b, where A is decomposed to factors QR (by Householders method)
	void solve_QR_blocked( std::vector< T >& x, const std::vector< T >& b, const size_t block_size ) const;

private:
	/// creates triangular factor T for blocked QR decoposition (Q = I - VTV*)
	void create_QR_triangular_factor_T( std::vector< std::vector< T > >& Tmx, const size_t step, const size_t step_offset ) const;



private:
	/// current state of matrix
	DYNAMIC_STATE m_dynamic_state{ DYNAMIC_STATE::INIT };

	/// amount of rows
	size_t m_rows{ 0 };
	/// amount of columns
	size_t m_cols{ 0 };

	/// flattened matrix data
	std::vector< T > m_matrix;
	/// additional data for QR decomposition
	std::vector< T > m_betas, m_v_firsts;
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

	if( row < step || col <= step )
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

	T* d_matrix_in{ nullptr }, * d_matrix_out{ nullptr };
	T* d_betas{ nullptr }, * d_v_firsts{ nullptr }, * d_vTA{ nullptr };

	cudaMalloc( &d_matrix_in, m_matrix.size() * sizeof( T ) );
	cudaMemcpy( d_matrix_in, m_matrix.data(), m_matrix.size() * sizeof( T ), cudaMemcpyHostToDevice );
	cudaMalloc( &d_matrix_out, m_matrix.size() * sizeof( T ) );

	auto max_steps = std::min( m_rows - 1, m_cols );

	cudaMalloc( &d_betas, max_steps * sizeof( T ) );
	cudaMalloc( &d_v_firsts, max_steps * sizeof( T ) );
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

	for( int step{ 0 }; step < max_steps; ++step )
	{
		QR_first_step << < gridSize1, blockSize1, st1_lmem_size >> >
			( d_matrix_in, d_matrix_out, d_betas, d_v_firsts, d_rows, d_cols, step );

		//const dim3 gridSize2( div_up( m_cols, TX2 ) );
		const dim3 gridSize2( div_up( m_cols - step - 1, TX2 ) );  // to be checked

		QR_compute_vTA << < gridSize2, blockSize2 >> >
			( d_matrix_in, d_vTA, d_v_firsts, d_rows, d_cols, step );

		//const dim3 gridSize3( div_up( m_cols, TX3 ), div_up( m_rows, TY3 ) );
		const dim3 gridSize3( div_up( m_cols - step, TX3 ), div_up( m_rows - step + 1, TY3 ) ); // to be checked

		QR_compute_reflections << < gridSize3, blockSize3 >> >
			( d_matrix_in, d_matrix_out, d_vTA, d_betas, d_v_firsts, d_rows, d_cols, step );

		std::swap( d_matrix_in, d_matrix_out );
	}

	cudaMemcpy( m_matrix.data(), d_matrix_in, m_matrix.size() * sizeof( T ), cudaMemcpyDeviceToHost );

	m_betas.resize( max_steps );
	cudaMemcpy( m_betas.data(), d_betas, max_steps * sizeof( T ), cudaMemcpyDeviceToHost );
	m_v_firsts.resize( max_steps );
	cudaMemcpy( m_v_firsts.data(), d_v_firsts, max_steps * sizeof( T ), cudaMemcpyDeviceToHost );

	cudaFree( d_matrix_in );
	cudaFree( d_matrix_out );
	cudaFree( d_betas );
	cudaFree( d_v_firsts );
	cudaFree( d_vTA );

	m_dynamic_state = DYNAMIC_STATE::QR_DECOMPOSED;
}


template< typename T >
void dense_matrix_cuda< T >::count_residual_vector( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const
{
	switch( m_dynamic_state )
	{
	case DYNAMIC_STATE::INIT:
		if( x.size() != m_cols || b.size() != m_rows || r.size() != m_rows )
			throw std::invalid_argument( "dense_matrix_cuda< T >::count_residual_vector - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );

		for( size_t row{ 0 }; row < m_rows; ++row )
			r[ row ] = -b[ row ];
		for( size_t row{ 0 }; row < m_rows; ++row )
			for( size_t col{ 0 }; col < m_cols; ++col )
				r[ row ] += ( x[ col ] * m_matrix[ calc_elem_idx( row, col, m_cols ) ] );
		break;

	default:
		throw std::exception( "dense_matrix_cuda< T >::count_residual_vector - not supported dynamic state" );
	}
}


template< typename T >
void dense_matrix_cuda< T >::solve_QR( std::vector< T >& x, const std::vector< T >& b ) const
{
	if( b.size() != m_rows )
		throw std::invalid_argument( "dense_matrix_cuda< T >::solve_QR - b.size() != m_rows" );

	if( m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix_cuda< T >::solve_QR() - m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED" );

	auto max_steps = std::min( m_rows - 1, m_cols );

	// first x := Q^T * b = H_1 * H_2 * ... * H_k * b
	// ==============================================
	x = b;
	for( size_t step{ 0 }; step < max_steps; ++step )
	{
		T vTb{ conjugate( m_v_firsts[ step ] ) * x[ step ] };
		for( size_t r{ step + 1 }; r < m_rows; ++r )
			vTb += conjugate( m_matrix[ calc_elem_idx( r, step, m_cols ) ] ) * x[ r ];

		x[ step ] -= m_betas[ step ] * m_v_firsts[ step ] * vTb;
		for( size_t r{ step + 1 }; r < m_rows; ++r )
			x[ r ] -= m_betas[ step ] * m_matrix[ calc_elem_idx( r, step, m_cols ) ] * vTb;
	}

	// then solve Rx = Q^T * b by back substitution
	// ============================================
	for( auto r = static_cast< int >( m_cols ) - 1; r >= 0; --r )
	{
		T sum{ T{} };
		for( int c{ r + 1 }; c < m_cols; ++c )
			sum += m_matrix[ calc_elem_idx( r, c, m_cols ) ] * x[ c ];

		x[ r ] = ( x[ r ] - sum ) / m_matrix[ calc_elem_idx( r, r, m_cols ) ];
	}
}


template< typename T >
void dense_matrix_cuda< T >::create_QR_triangular_factor_T( std::vector< std::vector< T > >& Tmx, const size_t step, const size_t step_offset ) const
{
	const auto lstep = step_offset + step;

	if( lstep >= m_betas.size() )
		throw std::out_of_range( "dense_matrix_cuda< T >::create_QR_triangular_factor_T - lstep >= m_betas.size()" );

	Tmx[ step ][ step ] = m_betas[ lstep ];

	if( step > 0 )
	{
		std::vector< T > VTv( step );
		for( size_t s{ step_offset }; s < lstep; ++s )
		{
			auto s_in{ s - step_offset };
			VTv[ s_in ] = conjugate( m_matrix[ calc_elem_idx( lstep, s, m_cols ) ] ) * m_v_firsts[ lstep ];
			for( size_t r{ lstep + 1 }; r < m_rows; ++r )
				VTv[ s_in ] += conjugate( m_matrix[ calc_elem_idx( r, s, m_cols ) ] ) * m_matrix[ calc_elem_idx( r, lstep, m_cols ) ];
		}

		for( size_t sr{ 0 }; sr < step; ++sr )
		{
			for( size_t sc{ 0 }; sc < step; ++sc )
				Tmx[ sr ][ step ] -= Tmx[ sr ][ sc ] * VTv[ sc ];

			Tmx[ sr ][ step ] *= m_betas[ lstep ];
		}
	}
}

template< typename T >
void dense_matrix_cuda< T >::solve_QR_blocked( std::vector< T >& x, const std::vector< T >& b, const size_t block_size ) const
{
	if( b.size() != m_rows )
		throw std::invalid_argument( "dense_matrix_cuda< T >::solve_QR_blocked - b.size() != m_rows" );

	auto max_steps{ std::min( m_rows - 1, m_cols ) };
	size_t step_offset{ 0 };

	x = b;
	while( step_offset < max_steps )
	{
		auto b_size{ std::min( block_size, max_steps - step_offset ) };
		auto b_end{ step_offset + b_size };

		std::vector< T > VTb( b_size, T{} );
		for( size_t step{ 0 }; step < b_size; ++step )
		{
			const auto lstep{ step_offset + step };
			VTb[ step ] += conjugate( m_v_firsts[ lstep ] ) * x[ lstep ];
			for( size_t r{ lstep + 1 }; r < m_rows; ++r )
				VTb[ step ] += conjugate( m_matrix[ calc_elem_idx( r, lstep, m_cols ) ] ) * x[ r ];
		}

		std::vector< std::vector < T > > Tmx( b_size, std::vector< T >( b_size, T{} ) );

		for( size_t s{ 0 }; s < b_size; ++s )
			create_QR_triangular_factor_T( Tmx, s, step_offset );

		std::vector< T > TVTb( b_size, T{} );

		for( size_t r{ 0 }; r < b_size; ++r )
			for( size_t s{ 0 }; s < b_size; ++s )
				TVTb[ r ] += conjugate( Tmx[ s ][ r ] ) * VTb[ s ];

		for( size_t r{ step_offset }; r < m_rows; ++r )
		{
			size_t s_in{ 0 };
			size_t s{ step_offset };
			for( ; s < std::min( b_end, r ); ++s )
				x[ r ] -= m_matrix[ calc_elem_idx( r, s, m_cols ) ] * TVTb[ s_in++ ];

			if( s == r && s < b_end )
				x[ r ] -= m_v_firsts[ s ] * TVTb[ s_in ];
		}

		step_offset += b_size;
	}

	// then solve Rx = Q^T * b by back substitution
	// ============================================
	for( auto r = static_cast< int >( m_cols ) - 1; r >= 0; --r )
	{
		T sum{ T{} };
		for( int c{ r + 1 }; c < m_cols; ++c )
			sum += m_matrix[ calc_elem_idx( r, c, m_cols ) ] * x[ c ];

		x[ r ] = ( x[ r ] - sum ) / m_matrix[ calc_elem_idx( r, r, m_cols ) ];
	}
}


//cublasHandle_t handle;
//cublasCreate( &handle );
//
//T* d_x{ nullptr };
//cudaMalloc( d_x, b.size() * sizeof( T ) );
//cudaMemxpy( d_x, b.data(), b.size() );
//
//auto quare_size = std::min( m_rows, m_cols );
//cublasDtrsv(
//	handle,
//	CUBLAS_FILL_MODE_UPPER,
//	CUBLAS_OP_N,
//	CUBLAS_DIAG_NON_UNIT,
//	quare_size,
//	m_d_matrix,
//	m_rows,
//	d_x,
//	1
//);
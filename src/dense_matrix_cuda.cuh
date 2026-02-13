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
	/// performs blocked QR decomposition using CUDA data
	void QR_decomposition_blocked( const size_t block_size  );
	/// solves equation Ax=b, where A is decomposed to factors QR (by Householders method)
	void solve_QR( std::vector< T >& x, const std::vector< T >& b ) const;
	/// solves equation Ax=b, where A is decomposed to factors QR (by Householders method)
	void solve_QR_blocked( std::vector< T >& x, const std::vector< T >& b, const size_t block_size ) const;

private:
	/// creates triangular factor T for blocked QR decoposition (Q = I - VTV*)
	void create_QR_triangular_factor_T( T* Tmx, const size_t block_size, const size_t step, const size_t step_offset ) const;
	/// decomposes block on cpu
	void QR_decomposition_blocked_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps );


private:
	/// current state of matrix
	DYNAMIC_STATE m_dynamic_state{ DYNAMIC_STATE::INIT };

	/// amount of rows
	size_t m_rows{ 0 };
	/// amount of columns
	size_t m_cols{ 0 };

	/// flattened matrix data
	std::vector< T > m_matrix;
	T* m_d_matrix{ nullptr };
	/// additional data for QR decomposition
	std::vector< T > m_betas, m_v_firsts;
	T* m_d_betas{ nullptr }, * m_d_v_firsts{ nullptr };

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
	auto elem_idx = calc_elem_idx( row, col, m_rows );

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
		size_t a_idx = calc_elem_idx( row, step, A_rows );
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
		size_t a_idx = calc_elem_idx( step, step, A_rows );
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

	T sum{ conjugate( v_firsts[ step ] ) * A_in[ calc_elem_idx( step, col, A_rows ) ] };

	for( int s = step + 1; s < A_rows; ++s )
		sum += conjugate( A_in[ calc_elem_idx( s, step, A_rows ) ] ) * A_in[ calc_elem_idx( s, col, A_rows ) ];

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

	int a_idx = calc_elem_idx( row, col, A_rows );

	if( row < step || col <= step )
		A_out[ a_idx ] = A_in[ a_idx ];
	else
	{
		if( row > step )
			A_out[ a_idx ] = A_in[ a_idx ] - beta * A_in[ calc_elem_idx( row, step, A_rows ) ] * vta;
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
	T* d_vTA{ nullptr };

	cudaMalloc( &d_matrix_in, m_matrix.size() * sizeof( T ) );
	cudaMemcpy( d_matrix_in, m_matrix.data(), m_matrix.size() * sizeof( T ), cudaMemcpyHostToDevice );
	cudaMalloc( &d_matrix_out, m_matrix.size() * sizeof( T ) );

	auto max_steps = std::min( m_rows - 1, m_cols );

	cudaMalloc( &m_d_betas, max_steps * sizeof( T ) );
	cudaMalloc( &m_d_v_firsts, max_steps * sizeof( T ) );
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
			( d_matrix_in, d_matrix_out, m_d_betas, m_d_v_firsts, d_rows, d_cols, step );

		const dim3 gridSize2( div_up( m_cols - step - 1, TX2 ) );

		QR_compute_vTA << < gridSize2, blockSize2 >> >
			( d_matrix_in, d_vTA, m_d_v_firsts, d_rows, d_cols, step );

		const dim3 gridSize3( div_up( m_cols - step, TX3 ), div_up( m_rows - step + 1, TY3 ) );

		QR_compute_reflections << < gridSize3, blockSize3 >> >
			( d_matrix_in, d_matrix_out, d_vTA, m_d_betas, m_d_v_firsts, d_rows, d_cols, step );

		std::swap( d_matrix_in, d_matrix_out );
	}

	std::swap( m_d_matrix, d_matrix_in );
	cudaMemcpy( m_matrix.data(), m_d_matrix, m_matrix.size() * sizeof( T ), cudaMemcpyDeviceToHost );

	m_betas.resize( max_steps );
	cudaMemcpy( m_betas.data(), m_d_betas, max_steps * sizeof( T ), cudaMemcpyDeviceToHost );
	m_v_firsts.resize( max_steps );
	cudaMemcpy( m_v_firsts.data(), m_d_v_firsts, max_steps * sizeof( T ), cudaMemcpyDeviceToHost );

	cudaFree( d_matrix_out );
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
				r[ row ] += ( x[ col ] * m_matrix[ calc_elem_idx( row, col, m_rows ) ] );
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
			vTb += conjugate( m_matrix[ calc_elem_idx( r, step, m_rows ) ] ) * x[ r ];

		x[ step ] -= m_betas[ step ] * m_v_firsts[ step ] * vTb;
		for( size_t r{ step + 1 }; r < m_rows; ++r )
			x[ r ] -= m_betas[ step ] * m_matrix[ calc_elem_idx( r, step, m_rows ) ] * vTb;
	}

	// then solve Rx = Q^T * b by back substitution
	// ============================================
	for( auto r = static_cast< int >( m_cols ) - 1; r >= 0; --r )
	{
		T sum{ T{} };
		for( int c{ r + 1 }; c < m_cols; ++c )
			sum += m_matrix[ calc_elem_idx( r, c, m_rows ) ] * x[ c ];

		x[ r ] = ( x[ r ] - sum ) / m_matrix[ calc_elem_idx( r, r, m_rows ) ];
	}
}


template< typename T >
void dense_matrix_cuda< T >::create_QR_triangular_factor_T( T* Tmx, const size_t block_size, const size_t step, const size_t step_offset ) const
{
	const auto lstep = step_offset + step;

	if( lstep >= m_betas.size() )
		throw std::out_of_range( "dense_matrix_cuda< T >::create_QR_triangular_factor_T - lstep >= m_betas.size()" );

	Tmx[ calc_elem_idx( step, step, block_size ) ] = m_betas[ lstep ];

	if( step > 0 )
	{
		std::vector< T > VTv( step );
		for( size_t s{ step_offset }; s < lstep; ++s )
		{
			auto s_in{ s - step_offset };
			VTv[ s_in ] = conjugate( m_matrix[ calc_elem_idx( lstep, s, m_rows ) ] ) * m_v_firsts[ lstep ];
			for( size_t r{ lstep + 1 }; r < m_rows; ++r )
				VTv[ s_in ] += conjugate( m_matrix[ calc_elem_idx( r, s, m_rows ) ] ) * m_matrix[ calc_elem_idx( r, lstep, m_rows ) ];
		}

		for( size_t sr{ 0 }; sr < step; ++sr )
		{
			for( size_t sc{ 0 }; sc < step; ++sc )
				Tmx[ calc_elem_idx( sr, step, block_size ) ] -= Tmx[ calc_elem_idx( sr, sc, block_size ) ] * VTv[ sc ];

			Tmx[ calc_elem_idx( sr, step, block_size ) ] *= m_betas[ lstep ];
		}
	}
}

template< typename T >
__global__
void QR_compute_blocked_TVTb( const T* A_in, const T* v_firsts, const int A_rows, const int A_cols, const T* Tmx, const T* b, T* TVTb, const int step_offset  )
{
	const int tid = threadIdx.x;
	const int b_size = blockDim.x;

	extern __shared__ unsigned char sdata_raw[];
	T* VTb = reinterpret_cast< T* >( sdata_raw );

	int lstep{ step_offset + tid };

	VTb[ tid ] = conjugate( v_firsts[ lstep ] ) * b[ lstep ];
	for( int r{ lstep + 1 }; r < A_rows; ++r )
		VTb[ tid ] += conjugate( A_in[ calc_elem_idx( r, lstep, A_rows ) ] ) * b[ r ];

	__syncthreads();

	T sum{};
	for( size_t s{ 0 }; s < b_size; ++s )
		sum += conjugate( Tmx[ calc_elem_idx( s, tid, b_size ) ] ) * VTb[ s ];

	TVTb[ tid ] = sum;
}

template< typename T >
__global__
void QR_compute_blocked_VTVTb( const T* A_in, const T* v_firsts, const int A_rows, const int A_cols, T* b, const T* TVTb, const int step_offset )
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int block_size = blockDim.x;

	int row{ step_offset + tid };
	int block_end = step_offset + block_size;
	int col_end = block_end < row ? block_end : row;

	int row_in{ 0 };
	int col{ step_offset };

	for( ; col < col_end; ++col )
		b[ row ] -= A_in[ calc_elem_idx( row, col, A_rows ) ] * TVTb[ row_in++ ];

	if( col == row && col < block_end )
		b[ row ] -= v_firsts[ col ] * TVTb[ row_in ];
}


template< typename T >
void dense_matrix_cuda< T >::solve_QR_blocked( std::vector< T >& x, const std::vector< T >& b, const size_t block_size ) const
{
	if( b.size() != m_rows )
		throw std::invalid_argument( "dense_matrix_cuda< T >::solve_QR_blocked - b.size() != m_rows" );

	auto max_steps{ std::min( m_rows - 1, m_cols ) };
	size_t step_offset{ 0 };

	T* d_Tmx{ nullptr };
	T* d_b{ nullptr };
	T* d_TVTb{ nullptr };

	cudaMallocManaged( &d_Tmx, block_size * block_size * sizeof( T ) );
	cudaMalloc( &d_b, m_rows * sizeof( T ) );
	cudaMemcpy( d_b, b.data(), b.size() * sizeof( T ), cudaMemcpyHostToDevice );
	cudaMalloc( &d_TVTb, block_size * sizeof( T ) );

	auto qb = b;

	while( step_offset < max_steps )
	{
		auto b_size{ std::min( block_size, max_steps - step_offset ) };

		cudaMemset( d_Tmx, 0, b_size * b_size * sizeof( T ) );
		for( size_t s{ 0 }; s < b_size; ++s )
			create_QR_triangular_factor_T( d_Tmx, b_size, s, step_offset );

		QR_compute_blocked_TVTb << < dim3( 1 ), dim3( b_size ), b_size * sizeof( T ) >> > ( m_d_matrix, m_d_v_firsts, m_rows, m_cols, d_Tmx, d_b, d_TVTb, step_offset );

		QR_compute_blocked_VTVTb << < div_up( m_rows - step_offset, b_size ), dim3( b_size ) >> > ( m_d_matrix, m_d_v_firsts, m_rows, m_cols, d_b, d_TVTb, step_offset );

		step_offset += b_size;
	}

	cublasHandle_t handle;
	cublasCreate( &handle );

	auto quare_size = std::min( m_rows, m_cols );
	CublasTrsv<T>::call(
		handle,
		CUBLAS_FILL_MODE_UPPER,
		CUBLAS_OP_N,
		CUBLAS_DIAG_NON_UNIT,
		quare_size,
		m_d_matrix,
		m_cols,
		d_b,
		1
	);

	x.resize( b.size() );
	cudaMemcpy( x.data(), d_b, b.size() * sizeof( T ), cudaMemcpyDeviceToHost );

	cudaFree( d_Tmx );
	cudaFree( d_b );
	cudaFree( d_TVTb );
}


template< typename T >
void dense_matrix_cuda< T >::QR_decomposition_blocked_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps )
{
	size_t block_end{ block_size + step_offset };
	size_t l_max_steps{ std::min( max_steps, block_end ) };
	size_t l_max_col{ std::min( block_end, m_cols ) };

	std::vector< T > vTA( l_max_col, T{} );

	for( size_t step{ step_offset }; step < l_max_steps; ++step )
	{
		double col_norm{ 0.0 };

		// calcualte norm
		// ==============
		for( size_t r{ step }; r < m_rows; ++r )
		{
			double abs_v = abs_val( m_matrix[ calc_elem_idx( r, step, m_rows ) ] );
			col_norm += abs_v * abs_v;
		}
		col_norm = std::sqrt( col_norm );

		// stabilization sign calculation
		// ==============================
		const size_t step_idx = calc_elem_idx( step, step, m_rows );

		double alpha_abs = abs_val( m_matrix[ step_idx ] );
		T sign = ( alpha_abs != 0.0 ? -( m_matrix[ step_idx ] ) / alpha_abs : T{ -1 } );
		T sign_norm = sign * T{ col_norm };

		m_v_firsts[ step ] = m_matrix[ step_idx ] - sign_norm;

		T vTv{ conjugate( m_v_firsts[ step ] ) * m_v_firsts[ step ] };

		for( size_t r{ step + 1 }; r < m_rows; ++r )
			vTv += conjugate( m_matrix[ calc_elem_idx( r, step, m_rows ) ] ) * m_matrix[ calc_elem_idx( r, step, m_rows ) ];

		m_betas[ step ] = 2.0 / vTv;

		m_matrix[ step_idx ] = sign_norm;

		// calculate vTA ( v*A in case of complex )
		// ========================================
		for( size_t c{ step + 1 }; c < l_max_col; ++c )
		{
			const size_t c_in{ c - step_offset };

			vTA[ c_in ] = conjugate( m_v_firsts[ step ] ) * m_matrix[ calc_elem_idx( step, c, m_rows ) ];
			for( size_t r{ step + 1 }; r < m_rows; ++r )
				vTA[ c_in ] += conjugate( m_matrix[ calc_elem_idx( r, step, m_rows ) ] ) * m_matrix[ calc_elem_idx( r, c, m_rows ) ];
		}

		// calculate (I-bvvT)A = A - b(v(vTA)) only for first block_size columns
		// =====================================================================
		for( size_t c{ step + 1 }; c < l_max_col; ++c )
			m_matrix[ calc_elem_idx( step, c, m_rows ) ] -= m_betas[ step ] * m_v_firsts[ step ] * vTA[ c - step_offset ];

		for( size_t r{ step + 1 }; r < m_rows; ++r )
			for( size_t c{ step + 1 }; c < l_max_col; ++c )
				m_matrix[ calc_elem_idx( r, c, m_rows ) ] -= m_betas[ step ] * m_matrix[ calc_elem_idx( r, step, m_rows ) ] * vTA[ c - step_offset ];
	}
}

template< typename T >
__global__
void QR_decomposition_blocked_TVTA_gpu( T* TVTA,
										const T* A_in,
										const T* Tmx,
										const T* v_firsts,
										const int A_rows,
										const int A_cols,
										const int block_size,
										const int row_offset,
										const int col_offset )
{
	const int col = col_offset + threadIdx.x + blockDim.x * blockIdx.x;
	const int row = row_offset + threadIdx.y;

	if( col >= A_cols || row >= row_offset + block_size  )
		return;

	const int t_row = threadIdx.y;

	T sum{ conjugate( v_firsts[ row ] ) * A_in[ calc_elem_idx( row, col, A_rows ) ] };
	for( int r{ row + 1 }; r < A_rows; ++r )
		sum += conjugate( A_in[ calc_elem_idx( r, row, A_rows ) ] ) * A_in[ calc_elem_idx( r, col, A_rows ) ];

	TVTA[ calc_elem_idx( t_row, col, block_size ) ] = sum;

	__syncthreads();

	sum = conjugate(  Tmx[ calc_elem_idx( 0, t_row, block_size ) ] ) * TVTA[ calc_elem_idx( 0, col, block_size ) ];
	for( int r{ 1 }; r < block_size; ++r )
		sum += conjugate( Tmx[ calc_elem_idx( r, t_row, block_size ) ] ) * TVTA[ calc_elem_idx( r, col, block_size ) ];

	__syncthreads();

	TVTA[ calc_elem_idx( t_row, col, block_size ) ] = sum;
}

template< typename T >
__global__
void QR_decomposition_blocked_VTVTA_gpu( const T* TVTA,
										 T* A_out,
										 const T* v_firsts,
										 const int A_rows,
										 const int A_cols,
										 const int block_size,
										 const int row_offset,
										 const int col_offset )
{

	const int col = col_offset + threadIdx.x + blockDim.x * blockIdx.x;
	const int row = row_offset + threadIdx.y + blockDim.y * blockIdx.y;

	if( col >= A_cols || row >= A_rows )
		return;

	const int t_col = row_offset;
	const int t_row = row - row_offset;

	int sum_range = ( block_size < t_row + 1 ? block_size : t_row + 1 );

	T sum{};

	for( int c{ 0 }; c < sum_range; ++c )
	{
		const int c_i = t_col + c;
		T v_i = ( c == t_row ? v_firsts[ c_i ] : A_out[ calc_elem_idx( row, c_i, A_rows ) ] );
		sum += v_i * TVTA[ calc_elem_idx( c, col, block_size ) ];
	}

	A_out[ calc_elem_idx( row, col, A_rows ) ] -= sum;
}

template< typename T >
void dense_matrix_cuda< T >::QR_decomposition_blocked( const size_t block_size )
{
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::exception( "dense_matrix_cuda< T >::QR_decomposition() - m_dynamic_state != DYNAMIC_STATE::INIT" );

	const auto max_steps{ std::min( m_rows - 1, m_cols ) };
	size_t step_offset{ 0 }, row_offset{ 0 };

	m_betas.resize( max_steps );
	m_v_firsts.resize( max_steps );	

	cudaMalloc( &m_d_matrix, m_matrix.size() * sizeof( T ) );
	cudaMemcpy( m_d_matrix, m_matrix.data(), m_matrix.size() * sizeof( T ), cudaMemcpyHostToDevice ); // to be more optimized

	cudaMalloc( &m_d_betas, max_steps * sizeof( T ) );
	cudaMalloc( &m_d_v_firsts, max_steps * sizeof( T ) );

	T* d_Tmx{ nullptr };
	cudaMallocManaged( &d_Tmx, block_size * block_size * sizeof( T ) );
	T* d_TVTA{ nullptr };
	cudaMalloc( &d_TVTA, block_size * m_cols * sizeof( T ) );

	while( step_offset < max_steps )
	{
		auto b_size{ block_size };

		QR_decomposition_blocked_cpu( b_size, step_offset, max_steps );

		size_t rows_to_copy = m_rows - row_offset;
		size_t cols_to_copy = std::min( b_size, m_cols - step_offset );

		cudaMemcpy2D(
			m_d_matrix + row_offset + step_offset * m_rows,       // dst
			m_rows * sizeof( T ),                                 // dst pitch
			m_matrix.data() + row_offset + step_offset * m_rows,  // src
			m_rows * sizeof( T ),                                 // src pitch
			rows_to_copy * sizeof( T ),                           // width (bytes)
			cols_to_copy,                                         // height (kolumny)
			cudaMemcpyHostToDevice
		);

		size_t v_data_size = std::min( b_size, max_steps - row_offset );
		cudaMemcpy( m_d_v_firsts + step_offset, m_v_firsts.data() + step_offset, v_data_size * sizeof( T ), cudaMemcpyHostToDevice );

		if( step_offset + b_size >= m_cols )
			break;

		cudaMemset( d_Tmx, 0, b_size * b_size * sizeof( T ) );
		for( size_t s{ 0 }; s < b_size; ++s )
			create_QR_triangular_factor_T( d_Tmx, b_size, s, step_offset );

		step_offset += b_size;

		dim3 blockDim( b_size, b_size );
		dim3 grid1Dim( div_up( m_cols - step_offset, b_size ), 1 );
		QR_decomposition_blocked_TVTA_gpu<<< grid1Dim, blockDim >>>( d_TVTA, m_d_matrix, d_Tmx, m_d_v_firsts, m_rows, m_cols, b_size, row_offset, step_offset );

		dim3 grid2Dim( div_up( m_cols - step_offset, b_size ), div_up( m_rows - row_offset, b_size ) );
		QR_decomposition_blocked_VTVTA_gpu<<< grid2Dim, blockDim >>>( d_TVTA, m_d_matrix, m_d_v_firsts, m_rows, m_cols, b_size, row_offset, step_offset );

		rows_to_copy = m_rows - row_offset;
		cols_to_copy = m_cols - step_offset;

		cudaMemcpy2D(
			m_matrix.data() + row_offset + step_offset * m_rows,  // dst
			m_rows * sizeof( T ),                                 // dst pitch
			m_d_matrix + row_offset + step_offset * m_rows,       // src
			m_rows * sizeof( T ),                                 // src pitch
			rows_to_copy * sizeof( T ),                           // width (bytes)
			cols_to_copy,                                         // height (kolumny)
			cudaMemcpyDeviceToHost
		);

		row_offset += b_size;
	}

	cudaMemcpy( m_d_betas, m_betas.data(), max_steps * sizeof( T ), cudaMemcpyHostToDevice );

	cudaFree( d_Tmx );
	cudaFree( d_TVTA );

	m_dynamic_state = DYNAMIC_STATE::QR_DECOMPOSED;
}

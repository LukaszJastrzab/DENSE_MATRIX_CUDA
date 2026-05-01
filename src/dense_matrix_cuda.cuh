#pragma once

#include <vector>
#include <stdexcept>
#include <type_traits>
#include <numeric>

#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cublas_v2.h>

#include "utilities.cuh"


// Type definition for state of dense_matrix_cuda
// ==============================================
enum class DYNAMIC_STATE : int
{
	NONE,
	ROL_INIT,
	COL_INIT,
	LU_DECOMPOSED,
	QR_DECOMPOSED,
	QHQ_DECOMPOSED,
	QUASI_QR
};

template< typename T >
class dense_matrix_cuda
{
public:
	/// constructors
	dense_matrix_cuda() = default;
	dense_matrix_cuda( const dense_matrix_cuda& ) = default;
	dense_matrix_cuda( dense_matrix_cuda&& ) = default;
	dense_matrix_cuda( DYNAMIC_STATE init_state, size_t rows, size_t cols );

	/// destructor
	~dense_matrix_cuda() = default;

	/// double type used in this template
	using DT = typename double_type< T >::type;
	/// real type used in this template
	using RT = typename real_type< T >::type;

	/// sets matrix sizes and allocates memory
	void init( DYNAMIC_STATE init_state, size_t rows, size_t cols );
	/// adds elements and throws exception if row / col is out of range
	void set_element( T value, size_t row, size_t col );

	/// it counts value r := Ax - b
	void count_residual_vector( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const;
	/// Method improves the accuracy of the solution
	void iterative_refinement( std::vector< DT >& x, const std::vector< DT >& b, const double acc, const size_t max_it, const dense_matrix_cuda< T >* A_orig = nullptr ) const;

	/// performs blocked QR decomposition by Householder algorithm "in situ" using CUDA
	void QR_decomposition( bool scaling, const size_t block_size = 8 );
	/// solves equation Ax=b, where A is decomposed to factors QR (by Householders method)
	void solve_QR( std::vector< DT >& x, const std::vector< DT >& b ) const;

	/// decomposes matrix "in situ" to factors LU using Gauss elimination using CUDA
	/// with partial pivoting (one column search)
	void LU_decomposition( bool scaling, const size_t block_size = 32 );
	/// solves equation Ax=b, where A is decomposed to factors LU (by Gauss elimination)
	void solve_LU( std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >* y = nullptr ) const;
	/// rows scaling
	void rows_scaling( T* d_A );
	/// cols scaling
	void cols_scaling( T* d_A );

	/// performs blocked QHQ decomposition using Householder algorithm, H is in Hessenberg form
	void QHQ_decomposition( const size_t block_size = 8 );

private:
	/// function calculates index in one of initial matrix state
	size_t calc_elem_idx( size_t row, size_t col ) const;
	/// creates triangular factor T for blocked QR decoposition (Q = I - VTV*)
	void create_QR_triangular_factor_T( T* Tmx, const size_t block_size, const size_t step, const size_t step_offset, const size_t row_shift = 0 ) const;
	/// decomposes block on cpu
	void QR_block_decomposition_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps );
	/// decomposes block on cpu
	void LU_block_decomposition_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps );
	/// decomposes block on cpu to QHQ, where H is in Hessenber form
	void QHQ_block_decomposition_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps );
	/// partial pivoting for Gauss elimination
	void choose_pivot( const size_t step );
	/// counts residual vector functions that depends of dynamic matrix state
	void count_residual_Ax_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const;
	void count_residual_LUx_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const;
	void count_residual_QRx_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const;

private:
	/// current state of matrix
	DYNAMIC_STATE m_dynamic_state{ DYNAMIC_STATE::NONE };

	/// amount of rows
	size_t m_rows{ 0 };
	/// amount of columns
	size_t m_cols{ 0 };

	/// flattened matrix data
	std::vector< T > m_matrix;
	/// additional data for QR decomposition
	std::vector< T > m_betas, m_v_firsts;
	/// row permutation
	std::vector< size_t > m_p_row;

	/// row / column scaling parameters
	std::vector< double > m_scalars;

};

__host__ __device__ __forceinline__
size_t calc_elem_idx_RLD( size_t row, size_t col, size_t cols )
{
	// row majority
	return col + row * cols;
}

__host__ __device__ __forceinline__
size_t calc_elem_idx_CLD( size_t row, size_t col, size_t rows )
{
	// column majority
	return row + col * rows;
}

template< typename T >
inline size_t dense_matrix_cuda< T >::calc_elem_idx( size_t row, size_t col ) const
{
	switch( m_dynamic_state )
	{
	case DYNAMIC_STATE::COL_INIT:
		return calc_elem_idx_CLD( row, col, m_rows );

	case DYNAMIC_STATE::ROL_INIT:
		return calc_elem_idx_RLD( row, col, m_cols );

	default:
		throw std::invalid_argument( "dense_matrix_cuda< T >::calc_elem_idx - state not supported" );
	}
}

template< typename T >
dense_matrix_cuda< T >::dense_matrix_cuda( DYNAMIC_STATE init_state, size_t rows, size_t cols )
{
	init( init_state, rows, cols );
}


template< typename T >
void dense_matrix_cuda< T >::init( DYNAMIC_STATE init_state, size_t rows, size_t cols )
{
	m_dynamic_state = init_state;

	m_rows = rows;
	m_cols = cols;

	m_matrix.resize( m_rows * m_cols, T{} );
}


template< typename T >
void dense_matrix_cuda< T >::set_element( T value, size_t row, size_t col )
{
	size_t elem_idx{ calc_elem_idx( row, col ) };

	if( elem_idx >= m_matrix.size() )
		throw std::out_of_range( "dense_matrix_cuda< T >::set_element - elem_idx >= m_matrix.size()" );

	m_matrix[ elem_idx ] = value;
}


template< typename T >
void dense_matrix_cuda< T >::count_residual_vector( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const
{
	switch( m_dynamic_state )
	{
	case DYNAMIC_STATE::COL_INIT:
	case DYNAMIC_STATE::ROL_INIT:
		count_residual_Ax_b( x, b, r );
		break;

	case DYNAMIC_STATE::LU_DECOMPOSED:
		count_residual_LUx_b( x, b, r );
		break;

	case DYNAMIC_STATE::QR_DECOMPOSED:
		count_residual_QRx_b( x, b, r );
		break;

	default:
		throw std::invalid_argument( "dense_matrix_cuda< T >::count_residual_vector - not supported dynamic state" );
	}
}


template < typename T >
void dense_matrix_cuda< T >::choose_pivot( const size_t step )
{
	size_t ROW{ 0 };
	double ABS_VAL{ 0.0 };

	for( size_t row{ step }; row < m_rows; ++row )
	{
		const double new_abs{ abs_val( m_matrix[ calc_elem_idx_RLD( m_p_row[ row ], step, m_cols ) ] ) };

		if( new_abs > ABS_VAL )
		{
			ABS_VAL = new_abs;
			ROW = row;
		}
	}

	std::swap( m_p_row[ ROW ], m_p_row[ step ] );
}


template< typename T >
void dense_matrix_cuda< T >::LU_block_decomposition_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps )
{
	const auto block_steps = std::min( max_steps, step_offset + block_size );

	for( size_t step{ step_offset }; step < block_steps; ++step )
	{
		choose_pivot( step );

		const size_t eliminating_row = m_p_row[ step ];
		const auto pivot{ m_matrix[ calc_elem_idx_RLD( eliminating_row, step, m_cols ) ] };

		for( size_t row{ step + 1 }; row < m_rows; ++row )
		{
			const size_t eliminated_row = m_p_row[ row ];

			const size_t elimintor_idx{ calc_elem_idx_RLD( eliminated_row, step, m_cols ) };

			m_matrix[ elimintor_idx ] /= pivot;
			const auto eliminator = m_matrix[ elimintor_idx ];

			for( size_t col{ step + 1 }; col < std::min( m_cols, step_offset + block_size ); ++col )
				m_matrix[ calc_elem_idx_RLD( eliminated_row, col, m_cols ) ] -= eliminator * m_matrix[ calc_elem_idx_RLD( eliminating_row, col, m_cols ) ];
		}
	}
}


template< typename T >
__global__
void L_block_update(
	T* A_in,
	const size_t* p_row,
	const int A_cols,
	const int row_offset,
	const int col_offset )
{
	const int col = col_offset + threadIdx.x + blockDim.x * blockIdx.x;
	const int mx_size = blockDim.y;

	extern __shared__ unsigned char sdata_raw[];
	T* L = reinterpret_cast< T* >( sdata_raw );
	T* U_i = reinterpret_cast< T* >( sdata_raw + ( ( mx_size + threadIdx.x ) * mx_size ) * sizeof( T ) );

	if( threadIdx.y > threadIdx.x )
		L[ calc_elem_idx_RLD( threadIdx.y, threadIdx.x, mx_size ) ]	=
			A_in[ calc_elem_idx_RLD( p_row[ row_offset + threadIdx.y ], row_offset + threadIdx.x, A_cols ) ];

	__syncthreads();

	if( col >= A_cols || threadIdx.y > 0 )
		return;

	for( int r{ 0 }; r < mx_size; ++r )
	{
		const size_t in_idx = calc_elem_idx_RLD( p_row[ row_offset + r ], col, A_cols );

		U_i[ r ] = A_in[ in_idx ];

		for( int c{ 0 }; c < r; ++c )
			U_i[ r ] -= L[ calc_elem_idx_RLD( r, c, mx_size ) ] * U_i[ c ];

		A_in[ in_idx ] = U_i[ r ];
	}
}

template< typename T >
__global__
void LU_Schur_complement(
	T* A_in,
	const size_t* p_row,
	const int A_rows,
	const int A_cols,
	const int row_offset,
	const int col_offset )
{
	const int col = col_offset + threadIdx.x + blockDim.x * blockIdx.x;
	const int row = col_offset + threadIdx.y + blockDim.y * blockIdx.y;

	const int mx_size = blockDim.x;
	const int mx_count = blockDim.x * blockDim.y;

	const size_t row_orig{ p_row[ row ] };

	extern __shared__ unsigned char sdata_raw[];
	T* L = reinterpret_cast< T* >( sdata_raw );
	T* U = reinterpret_cast< T* >( sdata_raw + mx_count * sizeof( T ) );

	if ( row < A_rows )
		L[ calc_elem_idx_RLD( threadIdx.y, threadIdx.x, mx_size ) ] = A_in[ calc_elem_idx_RLD( row_orig, row_offset + threadIdx.x, A_cols ) ];
	if ( col < A_cols )
		U[ calc_elem_idx_RLD( threadIdx.y, threadIdx.x, mx_size ) ] = A_in[ calc_elem_idx_RLD( p_row[ row_offset + threadIdx.y ], col, A_cols ) ];

	__syncthreads();

	if( row >= A_rows || col >= A_cols )
		return;

	const size_t in_idx = calc_elem_idx_RLD( row_orig, col, A_cols );

	T res{ A_in[ in_idx ] };

	for( int i{ 0 }; i < blockDim.x; ++i )
		res -= L[ calc_elem_idx_RLD( threadIdx.y, i, mx_size ) ] * U[ calc_elem_idx_RLD( i, threadIdx.x, mx_size ) ];

	A_in[ in_idx ] = res;
}

template< typename T >
void dense_matrix_cuda< T >::LU_decomposition( bool scaling, const size_t block_size )
{
	if( m_dynamic_state != DYNAMIC_STATE::ROL_INIT )
		throw std::invalid_argument( "dense_matrix_cuda< T >::LU_decomposition() - m_dynamic_state != DYNAMIC_STATE::ROL_INIT" );
	if( m_rows != m_cols )
		throw std::invalid_argument( "dense_matrix_cuda< T >::LU_decomposition: m_rows != m_cols" );

	m_p_row.resize( m_rows );
	std::iota( m_p_row.begin(), m_p_row.end(), 0 );

	T* d_matrix{ nullptr };
	size_t *d_p_row{ nullptr };

	cudaMalloc( &d_matrix, m_matrix.size() * sizeof( T ) );
	cudaMemcpy( d_matrix, m_matrix.data(), m_matrix.size() * sizeof( T ), cudaMemcpyHostToDevice );
	cudaMalloc( &d_p_row, m_rows * sizeof( size_t ) );

	const size_t max_steps{ m_rows - 1 };
	size_t step_offset{ 0 }, col_offset{ 0 };

	if( scaling )
	{
		rows_scaling( d_matrix );

		cudaMemcpy2D(
			m_matrix.data(),                                 // dst
			m_cols * sizeof( T ),                            // pitch dst
			d_matrix,                                        // src
			m_cols * sizeof( T ),                            // pitch src
			std::min( block_size, m_cols ) * sizeof( T ),    // bytes per row
			m_rows,                                          // all rows
			cudaMemcpyDeviceToHost
		);
	}
	else
		m_scalars.resize( m_rows, 1.0 );

	while( step_offset < max_steps )
	{
		const auto b_size = std::min( block_size, max_steps - step_offset );

		LU_block_decomposition_cpu( b_size, step_offset, max_steps );

		cudaMemcpy( d_p_row + step_offset, m_p_row.data() + step_offset, ( m_rows - step_offset ) * sizeof( size_t ), cudaMemcpyHostToDevice );

		for( size_t r{ step_offset }; r < m_rows; ++r )
		{
			const auto start_offset{ step_offset + m_p_row[ r ] * m_cols };
			cudaMemcpy( d_matrix + start_offset, m_matrix.data() + start_offset, b_size * sizeof( T ), cudaMemcpyHostToDevice );
		}

		col_offset += b_size;

		const dim3 block1_dim( b_size, b_size );
		const dim3 grid1_dim( div_up( m_cols - col_offset, b_size ), 1 );
		const size_t smem1_size{ 2ull * b_size * b_size * sizeof( T ) };
		L_block_update <<< grid1_dim, block1_dim, smem1_size >>> ( d_matrix, d_p_row, m_cols, step_offset, col_offset );

		const dim3 block2_dim( b_size, b_size );
		const dim3 grid2_dim( div_up( m_cols - col_offset, b_size ), div_up( m_rows - col_offset, b_size ) );
		const size_t smem2_size{ 2ull * b_size * b_size * sizeof( T ) };
		LU_Schur_complement <<< grid2_dim, block2_dim, smem2_size >>> ( d_matrix, d_p_row, m_rows, m_cols, step_offset, col_offset );

		for( size_t r{ step_offset }; r < m_rows; ++r )
		{
			const auto start_offset{ col_offset + m_p_row[ r ] * m_cols };
			cudaMemcpy( m_matrix.data() + start_offset, d_matrix + start_offset, ( m_cols - col_offset ) * sizeof( T ), cudaMemcpyDeviceToHost );
		}

		step_offset += b_size;
	}

	cudaFree( d_p_row );
	cudaFree( d_matrix );

	m_dynamic_state = DYNAMIC_STATE::LU_DECOMPOSED;
}


template< typename T >
void dense_matrix_cuda< T >::solve_LU( std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >* y ) const
{
	if( b.size() != m_rows )
		throw std::invalid_argument( "dense_matrix_cuda< T >::solve_LU - b.size() != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::LU_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix_cuda< T >::solve_LU() - m_dynamic_state != DYNAMIC_STATE::LU_DECOMPOSED" );

	std::vector< DT > y_alloc;

	if( y == nullptr )
	{
		y_alloc.resize( m_rows );
		y = &y_alloc;
	}

	// first solve the equation Ly = b
	// ===============================
	y->at( 0 ) = ( b[ m_p_row[ 0 ] ] * static_cast< DT >( m_scalars[ m_p_row[ 0 ] ] ) );

	for( size_t row{ 1 }; row < m_cols; ++row )
	{
		const int p_row = m_p_row[ row ];
		y->at( row ) = ( b[ p_row ] * static_cast< DT >( m_scalars[ p_row ] ) );

		for( size_t col{ 0 }; col < row; ++col )
			y->at( row ) -= static_cast< DT >( m_matrix[ calc_elem_idx_RLD( p_row, col, m_cols ) ] ) * y->at( col );
	}

	// second solve the equation Ux = y
	// ================================
	for( int row{ static_cast< int >( m_cols ) - 1 }; row >= 0; --row )
	{
		x[ row ] = y->at( row );

		for( int col{ row + 1 }; col < m_cols; ++col )
			x[ row ] -= static_cast< DT >( m_matrix[ calc_elem_idx_RLD( m_p_row[ row ], col, m_cols ) ] ) * x[ col ];

		x[ row ] /= m_matrix[ calc_elem_idx_RLD( m_p_row[ row ], row, m_cols ) ];
	}
}


template< typename T >
void dense_matrix_cuda< T >::solve_QR( std::vector< DT >& x, const std::vector< DT >& b ) const
{
	if( b.size() != m_rows )
		throw std::invalid_argument( "dense_matrix_cuda< T >::solve_QR - b.size() != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix_cuda< T >::solve_QR() - m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED" );

	auto max_steps{ m_rows - 1 };

	// first x := Q^T * b = H_1 * H_2 * ... * H_k * b
	// ==============================================
	x = b;
	for( size_t step{ 0 }; step < max_steps; ++step )
	{
		DT vTb{ conjugate( static_cast< DT >( m_v_firsts[ step ] ) ) * x[ step ] };
		for( size_t r{ step + 1 }; r < m_rows; ++r )
			vTb += conjugate( static_cast< DT >( m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ] ) ) * x[ r ];

		x[ step ] -= static_cast< DT >( m_betas[ step ] * m_v_firsts[ step ] ) * vTb;
		for( size_t r{ step + 1 }; r < m_rows; ++r )
			x[ r ] -= static_cast< DT >( m_betas[ step ] * m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ] ) * vTb;
	}

	// then solve Rx = Q^T * b by back substitution
	// ============================================
	for( auto r = static_cast< int >( m_cols ) - 1; r >= 0; --r )
	{
		DT sum{ T{} };
		for( int c{ r + 1 }; c < m_cols; ++c )
			sum += static_cast< DT >( m_matrix[ calc_elem_idx_CLD( r, c, m_rows ) ] ) * x[ c ];

		x[ r ] = ( x[ r ] - sum ) / m_matrix[ calc_elem_idx_CLD( r, r, m_rows ) ];
	}

	for( size_t c{ 0 }; c < m_cols; ++c )
		x[ c ] *= static_cast< DT >( m_scalars[ c ] );
}


template< typename T >
void dense_matrix_cuda< T >::create_QR_triangular_factor_T( T* Tmx, const size_t block_size, const size_t step, const size_t step_offset, const size_t row_shift ) const
{
	const auto lstep = step_offset + step;

	if( lstep >= m_betas.size() )
		throw std::out_of_range( "dense_matrix_cuda< T >::create_QR_triangular_factor_T - lstep >= m_betas.size()" );

	Tmx[ calc_elem_idx_CLD( step, step, block_size ) ] = m_betas[ lstep ];

	if( step > 0 )
	{
		std::vector< T > VTv( step );
		for( size_t s{ step_offset }; s < lstep; ++s )
		{
			auto s_in{ s - step_offset };
			VTv[ s_in ] = conjugate( m_matrix[ calc_elem_idx_CLD( lstep + row_shift, s, m_rows ) ] ) * m_v_firsts[ lstep ];
			for( size_t r{ lstep + row_shift + 1 }; r < m_rows; ++r )
				VTv[ s_in ] += conjugate( m_matrix[ calc_elem_idx_CLD( r, s, m_rows ) ] ) * m_matrix[ calc_elem_idx_CLD( r, lstep, m_rows ) ];
		}

		for( size_t sr{ 0 }; sr < step; ++sr )
		{
			for( size_t sc{ 0 }; sc < step; ++sc )
				Tmx[ calc_elem_idx_CLD( sr, step, block_size ) ] -= Tmx[ calc_elem_idx_CLD( sr, sc, block_size ) ] * VTv[ sc ];

			Tmx[ calc_elem_idx_CLD( sr, step, block_size ) ] *= m_betas[ lstep ];
		}
	}
}

template< typename T >
void dense_matrix_cuda< T >::QHQ_block_decomposition_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps )
{	// 	const auto max_steps = m_rows - 2;
	size_t block_end{ block_size + step_offset };
	size_t l_max_steps{ std::min( max_steps, block_end ) };
	size_t l_max_col{ std::min( block_end, m_cols ) };

	std::vector< T > Av( m_rows, T{} ), vTA( l_max_col, T{} );

	for( size_t step{ step_offset }; step < l_max_steps; ++step )
	{
		double col_norm{ 0.0 };
		const size_t row_step{ step + 1 };

		// calcualte norm
		// ==============
		for( size_t r{ row_step }; r < m_rows; ++r )
		{
			double abs_v = abs_val( m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ] );
			col_norm += abs_v * abs_v;
		}
		col_norm = std::sqrt( col_norm );

		// stabilization sign calculation
		// ==============================
		auto lead_elem_idx{ calc_elem_idx_CLD( row_step, step, m_rows ) };
		double alpha_abs = abs_val( m_matrix[ lead_elem_idx ] );
		T sign = ( alpha_abs != 0.0 ? -( m_matrix[ lead_elem_idx ] ) / T{ static_cast< RT >( alpha_abs ) } : T{ -1 } );
		T sign_norm = sign * T{ static_cast< RT >( col_norm ) };

		m_v_firsts[ step ] = m_matrix[ lead_elem_idx ] - sign_norm;
		const auto v1{ m_v_firsts[ step ] };
		const auto v1T{ conjugate( v1 ) };

		T vTv{ v1T * v1 };
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
		{
			const auto elem_idx{ calc_elem_idx_CLD( r, step, m_rows ) };
			vTv += conjugate( m_matrix[ elem_idx ] ) * m_matrix[ elem_idx ];
		}

		if( vTv == T{ 0 } )
		{
			m_betas[ step ] = T{ 0 };
			continue;
		}

		m_betas[ step ] = static_cast< RT >( 2.0 ) / vTv;
		m_matrix[ lead_elem_idx ] = sign_norm;

		if( row_step == l_max_steps )
			break;

		const auto beta{ m_betas[ step ] };

		// calculate Av
		//=============
		for( size_t r{ 0 }; r < m_rows; ++r )
		{
			Av[ r ] = m_matrix[ calc_elem_idx_CLD( r, row_step, m_rows ) ] * v1;
			for( size_t c{ row_step + 1 }; c < m_cols; ++c )
				Av[ r ] += m_matrix[ calc_elem_idx_CLD( r, c, m_rows ) ] * m_matrix[ calc_elem_idx_CLD( c, step, m_rows ) ];
		}

		// calculate vTA ( v*A in case of complex )
		// ========================================
		for( size_t c{ step }; c < l_max_col; ++c )
		{			
			vTA[ c ] = v1T * m_matrix[ calc_elem_idx_CLD( row_step, c, m_rows ) ];
			for( size_t r{ row_step + 1 }; r < m_rows; ++r )
				vTA[ c ] += conjugate( m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ] ) * m_matrix[ calc_elem_idx_CLD( r, c, m_rows ) ];
		}

		// alpha = v*Av
		// ============
		T alpha{ v1T * Av[ row_step ] };
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
			alpha += conjugate( m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ] ) * Av[ r ];

		// update those part of matrix that are changed only by right mult by QT
		// =====================================================================
		for( size_t r{ 0 }; r < row_step; ++r )
		{
			const auto Av_{ Av[ r ] };			
			m_matrix[ calc_elem_idx_CLD( r, row_step, m_rows ) ] -= beta * Av_ * v1T;

			for( size_t c{ row_step + 1 }; c < l_max_col; ++c )
				m_matrix[ calc_elem_idx_CLD( r, c, m_rows ) ] -= beta * Av_ * conjugate( m_matrix[ calc_elem_idx_CLD( c, step, m_rows ) ] );
		}

		// update left-upper corner of submatrix
		// =====================================
		m_matrix[ calc_elem_idx_CLD( row_step, row_step, m_rows ) ] -=
				beta * ( v1 * vTA[ row_step ] + Av[ row_step ] * v1T - beta * alpha * v1 * v1T );

		// update fiest modificated sub row
		// ================================
		for( size_t c{ row_step + 1 }; c < l_max_col; ++c )
		{
			const auto v1T{ conjugate( m_matrix[ calc_elem_idx_CLD( c, step, m_rows ) ] ) };			
			m_matrix[ calc_elem_idx_CLD( row_step, c, m_rows ) ] -= beta * ( v1 * vTA[ c ] + Av[ row_step ] * v1T - beta * alpha * v1 * v1T );
		}

		// update fiest modificated sub col
		// ================================
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
		{
			const auto v1{ m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ] };
			m_matrix[ calc_elem_idx_CLD( r, row_step, m_rows ) ] -= beta * ( v1 * vTA[ row_step ] + Av[ r ] * v1T - beta * alpha * v1 * v1T );
		}

		// update rest part of sub matrix
		// ==============================
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
		{
			const auto v1{ m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ] };
			const auto Av_{ Av[ r ] };

			for( size_t c{ row_step + 1 }; c < l_max_col; ++c )
			{
				const auto v1T{ conjugate( m_matrix[ calc_elem_idx_CLD( c, step, m_rows ) ] ) };
				m_matrix[ calc_elem_idx_CLD( r, c, m_rows ) ] -= beta * ( v1 * vTA[ c ] + Av_ * v1T - beta * alpha * v1 * v1T );
			}
		}
	}
}

template< typename T >
void dense_matrix_cuda< T >::QHQ_decomposition( const size_t block_size )
{
	if( m_dynamic_state != DYNAMIC_STATE::COL_INIT )
		throw std::invalid_argument( "dense_matrix_cuda< T >::QR_decomposition() - m_dynamic_state != DYNAMIC_STATE::COL_INIT" );
	if( m_rows != m_cols )
		throw std::invalid_argument( "dense_matrix_cuda< T >::QHQ_decomposition() - m_rows != m_cols" );

	const auto max_steps{ m_rows - 2 };
	//size_t step_offset{ 0 }, row_offset{ 0 };

	std::vector< T > Tmx( block_size * block_size, T{} );

	m_betas.resize( max_steps );
	m_v_firsts.resize( max_steps );

	QHQ_block_decomposition_cpu( block_size, 0, max_steps );

	const auto b_size = std::min( block_size, max_steps - 0 ); // - step_offset

	memset( Tmx.data(), 0, b_size * b_size * sizeof( T ) );
	for( size_t s{ 0 }; s < b_size; ++s )
		//create_QR_triangular_factor_T( Tmx.data(), b_size, s, step_offset );
		create_QR_triangular_factor_T( Tmx.data(), b_size, s, 0, 1 );

	// to do

	m_dynamic_state = DYNAMIC_STATE::QHQ_DECOMPOSED;
}

template< typename T >
void dense_matrix_cuda< T >::QR_block_decomposition_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps )
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
			double abs_v = abs_val( m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ] );
			col_norm += abs_v * abs_v;
		}
		col_norm = std::sqrt( col_norm );

		// stabilization sign calculation
		// ==============================
		const size_t step_idx = calc_elem_idx_CLD( step, step, m_rows );

		double alpha_abs = abs_val( m_matrix[ step_idx ] );
		T sign = ( alpha_abs != 0.0 ? -( m_matrix[ step_idx ] ) / alpha_abs : T{ -1 } );
		T sign_norm = sign * T{ static_cast< RT >( col_norm ) };

		m_v_firsts[ step ] = m_matrix[ step_idx ] - sign_norm;

		T vTv{ conjugate( m_v_firsts[ step ] ) * m_v_firsts[ step ] };

		for( size_t r{ step + 1 }; r < m_rows; ++r )
			vTv += conjugate( m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ] ) * m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ];

		m_betas[ step ] = 2.0 / vTv;

		m_matrix[ step_idx ] = sign_norm;

		// calculate vTA ( v*A in case of complex )
		// ========================================
		for( size_t c{ step + 1 }; c < l_max_col; ++c )
		{
			const size_t c_in{ c - step_offset };

			vTA[ c_in ] = conjugate( m_v_firsts[ step ] ) * m_matrix[ calc_elem_idx_CLD( step, c, m_rows ) ];
			for( size_t r{ step + 1 }; r < m_rows; ++r )
				vTA[ c_in ] += conjugate( m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ] ) * m_matrix[ calc_elem_idx_CLD( r, c, m_rows ) ];
		}

		// calculate (I-bvvT)A = A - b(v(vTA)) only for first block_size columns
		// =====================================================================
		for( size_t c{ step + 1 }; c < l_max_col; ++c )
			m_matrix[ calc_elem_idx_CLD( step, c, m_rows ) ] -= m_betas[ step ] * m_v_firsts[ step ] * vTA[ c - step_offset ];

		for( size_t r{ step + 1 }; r < m_rows; ++r )
			for( size_t c{ step + 1 }; c < l_max_col; ++c )
				m_matrix[ calc_elem_idx_CLD( r, c, m_rows ) ] -= m_betas[ step ] * m_matrix[ calc_elem_idx_CLD( r, step, m_rows ) ] * vTA[ c - step_offset ];
	}
}


template< typename T >
__global__
void QR_decomposition_blocked_TVTA_gpu( T* TVTA,
	const T* A_in,
	const T* v_firsts,
	const int A_rows,
	const int A_cols,
	const int block_size,
	const int row_offset,
	const int col_offset )
{
	const int col = col_offset + threadIdx.x + blockDim.x * blockIdx.x;
	const int row = row_offset + threadIdx.y;
	T sum{};

	bool active = !( col >= A_cols || row >= row_offset + block_size );

	if( active )
	{
		sum = conjugate( v_firsts[ row ] ) * A_in[ calc_elem_idx_CLD( row, col, A_rows ) ];
		for( int r{ row + 1 }; r < A_rows; ++r )
			sum += conjugate( A_in[ calc_elem_idx_CLD( r, row, A_rows ) ] ) * A_in[ calc_elem_idx_CLD( r, col, A_rows ) ];

		TVTA[ calc_elem_idx_CLD( threadIdx.y, col, block_size ) ] = sum;
	}

	__syncthreads();

	if( active )
	{
		sum = conjugate( TVTA[ calc_elem_idx_CLD( 0, threadIdx.y, block_size ) ] ) * TVTA[ calc_elem_idx_CLD( 0, col, block_size ) ];
		for( int r{ 1 }; r <= threadIdx.y; ++r )
			sum += conjugate( TVTA[ calc_elem_idx_CLD( r, threadIdx.y, block_size ) ] ) * TVTA[ calc_elem_idx_CLD( r, col, block_size ) ];
	}

	__syncthreads();

	if( active )
		TVTA[ calc_elem_idx_CLD( threadIdx.y, col, block_size ) ] = sum;
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
		T v_i = ( c == t_row ? v_firsts[ c_i ] : A_out[ calc_elem_idx_CLD( row, c_i, A_rows ) ] );
		sum += v_i * TVTA[ calc_elem_idx_CLD( c, col, block_size ) ];
	}

	A_out[ calc_elem_idx_CLD( row, col, A_rows ) ] -= sum;
}

template< typename T >
void dense_matrix_cuda< T >::QR_decomposition( bool scaling, const size_t block_size )
{
	if( m_dynamic_state != DYNAMIC_STATE::COL_INIT )
		throw std::invalid_argument( "dense_matrix_cuda< T >::QR_decomposition() - m_dynamic_state != DYNAMIC_STATE::COL_INIT" );
	if( m_rows != m_cols )
		throw std::invalid_argument( "dense_matrix_cuda< T >::QR_decomposition() -  m_rows != m_cols" );

	const auto max_steps{ m_rows - 1 };
	size_t step_offset{ 0 }, row_offset{ 0 };

	m_betas.resize( max_steps );
	m_v_firsts.resize( max_steps );

	T *d_matrix{ nullptr }, *d_v_firsts{ nullptr }, *d_TVTA{ nullptr };

	cudaMalloc( &d_matrix, m_matrix.size() * sizeof( T ) );
	cudaMemcpy( d_matrix, m_matrix.data(), m_matrix.size() * sizeof( T ), cudaMemcpyHostToDevice );

	if( scaling )
	{
		cols_scaling( d_matrix );
		cudaMemcpy( m_matrix.data(), d_matrix, std::min( block_size, m_rows ) * m_cols * sizeof( T ), cudaMemcpyDeviceToHost );
	}
	else
		m_scalars.resize( m_cols, 1.0 );

	cudaMalloc( &d_v_firsts, max_steps * sizeof( T ) );

	std::vector< T > Tmx( block_size * block_size, T{} );

	cudaMalloc( &d_TVTA, block_size * m_cols * sizeof( T ) );

	while( step_offset < max_steps )
	{
		auto b_size = std::min( block_size, max_steps - step_offset );

		QR_block_decomposition_cpu( b_size, step_offset, max_steps );

		size_t rows_to_copy = m_rows - row_offset;
		size_t cols_to_copy = std::min( b_size, m_cols - step_offset );

		cudaMemcpy2D(
			d_matrix + row_offset + step_offset * m_rows,         // dst
			m_rows * sizeof( T ),                                 // dst pitch
			m_matrix.data() + row_offset + step_offset * m_rows,  // src
			m_rows * sizeof( T ),                                 // src pitch
			rows_to_copy * sizeof( T ),                           // width (bytes)
			cols_to_copy,                                         // height (cols)
			cudaMemcpyHostToDevice
		);

		size_t v_data_size = std::min( b_size, max_steps - step_offset );
		cudaMemcpy( d_v_firsts + step_offset, m_v_firsts.data() + step_offset, v_data_size * sizeof( T ), cudaMemcpyHostToDevice );

		if( step_offset + b_size >= m_cols )
			break;

		memset( Tmx.data(), 0, b_size * b_size * sizeof( T ) );
		for( size_t s{ 0 }; s < b_size; ++s )
			create_QR_triangular_factor_T( Tmx.data(), b_size, s, step_offset );

		cudaMemcpy2D(
			d_TVTA,                     // dst
			b_size * sizeof( T ),       // dst pitch
			Tmx.data(),                 // src
			b_size * sizeof( T ),       // src pitch
			b_size * sizeof( T ),       // width (bytes)
			b_size,                     // height (cols)
			cudaMemcpyHostToDevice
		);

		step_offset += b_size;

		dim3 blockDim( b_size, b_size );
		dim3 grid1Dim( div_up( m_cols - step_offset, b_size ), 1 );
		QR_decomposition_blocked_TVTA_gpu << < grid1Dim, blockDim >> > ( d_TVTA, d_matrix, d_v_firsts, m_rows, m_cols, b_size, row_offset, step_offset );

		dim3 grid2Dim( div_up( m_cols - step_offset, b_size ), div_up( m_rows - row_offset, b_size ) );
		QR_decomposition_blocked_VTVTA_gpu << < grid2Dim, blockDim >> > ( d_TVTA, d_matrix, d_v_firsts, m_rows, m_cols, b_size, row_offset, step_offset );

		cols_to_copy = std::min( b_size, m_cols - step_offset );

		cudaMemcpy2D(
			m_matrix.data() + step_offset * m_rows,  // dst
			m_rows * sizeof( T ),                    // dst pitch
			d_matrix + step_offset * m_rows,         // src
			m_rows * sizeof( T ),                    // src pitch
			m_rows * sizeof( T ),                    // width (bytes)
			cols_to_copy,                            // height (cols)
			cudaMemcpyDeviceToHost
		);

		row_offset += b_size;
	}

	cudaFree( d_TVTA );
	cudaFree( d_v_firsts );
	cudaFree( d_matrix );

	m_dynamic_state = DYNAMIC_STATE::QR_DECOMPOSED;
}

template< typename T >
void dense_matrix_cuda< T >::count_residual_Ax_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const
{
	if( x.size() != m_cols || b.size() != m_rows || r.size() != m_rows )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_Ax_b - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::ROL_INIT && m_dynamic_state != DYNAMIC_STATE::COL_INIT )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_Ax_b - m_dynamic_state != DYNAMIC_STATE::INIT" );

	for( size_t row{ 0 }; row < m_rows; ++row )
		r[ row ] = -b[ row ];
	for( size_t row{ 0 }; row < m_rows; ++row )
		for( size_t col{ 0 }; col < m_cols; ++col )
			r[ row ] += ( x[ col ] * static_cast< DT >( m_matrix[ calc_elem_idx( row, col ) ] ) );
}

template< typename T >
void dense_matrix_cuda< T >::count_residual_LUx_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const
{
	if( x.size() != m_cols || b.size() != m_rows || r.size() != m_rows )
		throw std::invalid_argument( "dense_matrix_cuda< T >::count_residual_LUx_b - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::LU_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix_cuda< T >::count_residual_LUx_b - m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED" );

	std::vector< DT > w( m_rows, T{} );

	// compute w=Ux
	// ============
	for( size_t row{ 0 }; row < m_rows; ++row )
		for( size_t col{ row }; col < m_cols; ++col )
			w[ row ] += ( x[ col ] * static_cast< DT >( m_matrix[ calc_elem_idx_RLD( m_p_row[ row ], col, m_cols ) ] ) );


	// compute r = Lw - b
	// ==================
	for( size_t row{ 0 }; row < m_rows; ++row )
	{
		r[ m_p_row[ row ] ] = w[ row ] - ( b[ m_p_row[ row ] ] * static_cast< DT >( m_scalars[ m_p_row[ row ] ] ) );

		for( size_t col{ 0 }; col < row; ++col )
			r[ m_p_row[ row ] ] += w[ col ] * static_cast< DT >( m_matrix[ calc_elem_idx_RLD( m_p_row[ row ], col, m_cols ) ] );

		r[ m_p_row[ row ] ] /= static_cast< DT >( m_scalars[ m_p_row[ row ] ] );
	}
}

template< typename T >
void dense_matrix_cuda< T >::count_residual_QRx_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const
{
	if( x.size() != m_cols || b.size() != m_rows || r.size() != m_rows )
		throw std::invalid_argument( "dense_matrix_cuda< T >::count_residual_QRx_b - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix_cuda< T >::count_residual_QRx_b - m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED" );

	const int max_steps{ static_cast< int >( m_rows ) - 1 };

	for( size_t row{ 0 }; row < m_rows; ++row )
	{
		r[ row ] = DT{};
		for( size_t col{ row }; col < m_cols; ++col )
			r[ row ] += ( x[ col ] * static_cast< DT >( m_matrix[ calc_elem_idx_CLD( row, col, m_rows ) ] )
				/ static_cast< DT >( m_scalars[ col ] ) );
	}

	for( int step{ max_steps - 1 }; step >= 0; --step )
	{
		DT vRx{ conjugate( static_cast< DT >( m_v_firsts[ step ] ) ) * r[ step ] };
		for( int s{ step + 1 }; s < static_cast< int >( m_rows ); ++s )
			vRx += conjugate( static_cast< DT >( m_matrix[ calc_elem_idx_CLD( s, step, m_rows ) ] ) ) * r[ s ];

		r[ step ] -= static_cast< DT >( m_betas[ step ] * m_v_firsts[ step ] ) * vRx;
		for( int s{ step + 1 }; s < static_cast< int >( m_rows ); ++s )
			r[ s ] -= static_cast< DT >( m_betas[ step ] * m_matrix[ calc_elem_idx_CLD( s, step, m_rows ) ] ) * vRx;
	}

	for( size_t row{ 0 }; row < m_rows; ++row )
		r[ row ] -= b[ row ];
}

template < typename T >
void dense_matrix_cuda< T >::iterative_refinement( std::vector< DT >& x, const std::vector< DT >& b, const double acc, const size_t max_it, const dense_matrix_cuda< T >* A_orig ) const
{
	if( m_rows < m_cols )
		throw std::exception( "dense_matrix_cuda< T >::iterative_refinement - m_rows < m_cols" );

	const size_t N = m_rows;

	std::vector< DT > d( N );
	std::vector< DT > r( N );
	std::vector< DT > y( N );

	size_t iteration = 0;

	if( A_orig != nullptr )
		A_orig->count_residual_vector( x, b, r );
	else
		count_residual_vector( x, b, r );

	double v_norm = l2_norm( r );
	double new_v_norm;

	// int while condition are contained 2 conditions to stop the calculations,
	// third condition is implemented inside the loop
	// =======================================================================
	while( iteration < max_it && v_norm > acc )
	{
		switch( m_dynamic_state )
		{
		case DYNAMIC_STATE::LU_DECOMPOSED:
			solve_LU( d, r, &y );
			break;

		case DYNAMIC_STATE::QR_DECOMPOSED:
			solve_QR( d, r );
			break;

		default:
			throw std::invalid_argument( "dense_matrix< T >::iterative_refinement - dynamic state not supported" );
		}

		for( size_t i = 0; i < N; ++i )
			d[ i ] = x[ i ] - d[ i ];

		if( A_orig != nullptr )
			A_orig->count_residual_vector( d, b, r );
		else
			count_residual_vector( d, b, r );

		new_v_norm = l2_norm( r );

		// if norm of new residual vector is less then previous then accept new solution
		// =============================================================================
		if( new_v_norm < v_norm )
		{
			for( size_t i = 0; i < N; ++i )
				x[ i ] = d[ i ];
			v_norm = new_v_norm;
			iteration++;
		}
		// otherwise keep previous solution
		// ================================
		else
			break;
	}
}


template< typename T >
__global__
void scaling_compute_norms( double* d_scalars, const T* d_A, const int md_size )
{
	extern __shared__ double sdata[];

	const unsigned int lead_dim{ blockIdx.x };
	const unsigned int tid{ threadIdx.x };

	double sum{};

	const unsigned int lead_idx{ lead_dim * md_size };

	for( unsigned int j = tid + lead_idx; j < md_size + lead_idx; j += blockDim.x )
		sum += abs_val( d_A[ j ] );

	sdata[ tid ] = sum;

	__syncthreads();

	for( unsigned int s = blockDim.x / 2; s > 0; s >>= 1 )
	{
		if( tid < s )
			sdata[ tid ] += sdata[ tid + s ];

		__syncthreads();
	}

	if( tid == 0 )
		d_scalars[ lead_dim ] = sdata[ 0 ];
}

template< typename T >
__global__
void matrix_scaling( double* d_scalars, T* d_A, const int ld_size, const int md_size )
{
	const int md = threadIdx.x + blockIdx.x * blockDim.x;
	const int ld = threadIdx.y + blockIdx.y * blockDim.y;

	if( md >= md_size || ld >= ld_size )
		return;

	d_A[ ld * md_size + md ] *= static_cast< T >( d_scalars[ ld ] );
}

template < typename T >
void dense_matrix_cuda< T >::rows_scaling( T* d_A )
{
	if( m_dynamic_state != DYNAMIC_STATE::ROL_INIT )
		throw std::invalid_argument( "dense_matrix_cuda< T >::rows_scaling - m_dynamic_state != DYNAMIC_STATE::ROL_INIT" );

	double* d_scalars{ nullptr };
	cudaMalloc( &d_scalars, m_rows * sizeof( double ) );

	int threads{ 256 };
	scaling_compute_norms <<< m_rows, threads, threads * sizeof( double ) >>> ( d_scalars, d_A, m_cols );

	thrust::device_ptr< double > ptr( d_scalars );
	double max_scalar = *thrust::max_element( ptr, ptr + m_rows );
	thrust::transform( ptr, ptr + m_cols, ptr, [ = ] __device__( double x ) { return max_scalar / x; } );

	dim3 block( 16, 16 );
	dim3 grid( div_up( m_cols, block.x ), div_up( m_rows, block.y ) );
	matrix_scaling <<< grid, block >>> ( d_scalars, d_A, m_rows, m_cols );

	m_scalars.resize( m_rows );
	cudaMemcpy( m_scalars.data(), d_scalars, m_rows * sizeof( double ), cudaMemcpyDeviceToHost );
	cudaFree( d_scalars );
}

template < typename T >
void dense_matrix_cuda< T >::cols_scaling( T* d_A )
{
	if( m_dynamic_state != DYNAMIC_STATE::COL_INIT )
		throw std::invalid_argument( "dense_matrix_cuda< T >::cols_scaling - m_dynamic_state != DYNAMIC_STATE::COL_INIT" );

	double* d_scalars{ nullptr };
	cudaMalloc( &d_scalars, m_cols * sizeof( double ) );

	int threads{ 256 };
	scaling_compute_norms <<< m_cols, threads, threads * sizeof( double ) >>> ( d_scalars, d_A, m_rows );

	thrust::device_ptr< double > ptr( d_scalars );
	double max_scalar = *thrust::max_element( ptr, ptr + m_cols );
	thrust::transform( ptr, ptr + m_cols, ptr, [ = ] __device__( double x ) { return max_scalar / x; } );

	dim3 block( 16, 16 );
	dim3 grid( div_up( m_rows, block.x ), div_up( m_cols, block.y ) );
	matrix_scaling <<< grid, block >>> ( d_scalars, d_A, m_cols, m_rows );

	m_scalars.resize( m_cols );
	cudaMemcpy( m_scalars.data(), d_scalars, m_cols * sizeof( double ), cudaMemcpyDeviceToHost );
	cudaFree( d_scalars );
}
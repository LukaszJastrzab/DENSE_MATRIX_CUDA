#pragma once

#include <vector>
#include <stdexcept>
#include <type_traits>
#include <numeric>

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
		LU_DECOMPOSED,
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

	/// performs blocked QR decomposition by Householder algorithm "in situ" using CUDA
	void QR_decomposition( const size_t block_size = 8 );
	/// solves equation Ax=b, where A is decomposed to factors QR (by Householders method)
	void solve_QR( std::vector< T >& x, const std::vector< T >& b ) const;

	/// decomposes matrix "in situ" to factors LU using Gauss elimination using CUDA
	/// with partial pivoting (one column search)
	void LU_decomposition( const size_t block_size );
	/// solves equation Ax=b, where A is decomposed to factors LU (by Gauss elimination)
	void solve_LU( std::vector< T >& x, const std::vector< T >& b ) const;

private:
	/// creates triangular factor T for blocked QR decoposition (Q = I - VTV*)
	void create_QR_triangular_factor_T( T* Tmx, const size_t block_size, const size_t step, const size_t step_offset ) const;
	/// decomposes block on cpu
	void QR_block_decomposition_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps );
	/// decomposes block on cpu
	void LU_block_decomposition_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps );
	/// partial pivoting for Gauss elimination
	void choose_pivot( const size_t step );


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

	/// row permutation
	std::vector< size_t > m_p_row;		/// under i-th index : original row number

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
		const double new_abs{ abs_val( m_matrix[ m_p_row[ row ] ][ step ] ) };

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
}


template< typename T >
void dense_matrix_cuda< T >::LU_decomposition( const size_t block_size )
{
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix_cuda< T >::LU_decomposition() - m_dynamic_state != DYNAMIC_STATE::INIT" );

	if( m_rows < m_cols )
		throw std::invalid_argument( "dense_matrix< T >::LU_decomposition: m_rows < m_cols" );

	m_p_row.resize( m_rows );
	std::iota( m_p_row.begin(), m_p_row.end(), 0 );

	//const auto max_steps{ std::min( m_rows - 1, m_cols ) };
	//size_t step_offset{ 0 }, row_offset{ 0 };

	// to be continued
}


template< typename T >
void dense_matrix_cuda< T >::solve_LU( std::vector< T >& x, const std::vector< T >& b ) const
{
	if( b.size() != m_rows )
		throw std::invalid_argument( "dense_matrix_cuda< T >::solve_LU - b.size() != m_rows" );

	if( m_dynamic_state != DYNAMIC_STATE::LU_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix_cuda< T >::solve_LU() - m_dynamic_state != DYNAMIC_STATE::LU_DECOMPOSED" );

	// to be continued
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
void dense_matrix_cuda< T >::QR_block_decomposition_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps )
{
	using real_t = typename real_type< T >::type;

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
		T sign_norm = sign * T{ static_cast< real_t >( col_norm ) };

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
		sum = conjugate( v_firsts[ row ] ) * A_in[ calc_elem_idx( row, col, A_rows ) ];
		for( int r{ row + 1 }; r < A_rows; ++r )
			sum += conjugate( A_in[ calc_elem_idx( r, row, A_rows ) ] ) * A_in[ calc_elem_idx( r, col, A_rows ) ];

		TVTA[ calc_elem_idx( threadIdx.y, col, block_size ) ] = sum;
	}

	__syncthreads();

	if( active )
	{
		sum = conjugate( TVTA[ calc_elem_idx( 0, threadIdx.y, block_size ) ] ) * TVTA[ calc_elem_idx( 0, col, block_size ) ];
		for( int r{ 1 }; r <= threadIdx.y; ++r )
			sum += conjugate( TVTA[ calc_elem_idx( r, threadIdx.y, block_size ) ] ) * TVTA[ calc_elem_idx( r, col, block_size ) ];
	}

	__syncthreads();

	if ( active )
		TVTA[ calc_elem_idx( threadIdx.y, col, block_size ) ] = sum;	
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
void dense_matrix_cuda< T >::QR_decomposition( const size_t block_size )
{
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix_cuda< T >::QR_decomposition() - m_dynamic_state != DYNAMIC_STATE::INIT" );

	const auto max_steps{ std::min( m_rows - 1, m_cols ) };
	size_t step_offset{ 0 }, row_offset{ 0 };

	m_betas.resize( max_steps );
	m_v_firsts.resize( max_steps );	

	cudaMalloc( &m_d_matrix, m_matrix.size() * sizeof( T ) );
	cudaMemcpy( m_d_matrix, m_matrix.data(), m_matrix.size() * sizeof( T ), cudaMemcpyHostToDevice );

	cudaMalloc( &m_d_betas, max_steps * sizeof( T ) );
	cudaMalloc( &m_d_v_firsts, max_steps * sizeof( T ) );

	std::vector< T > Tmx( block_size * block_size, T{} );

	T* d_TVTA{ nullptr };
	cudaMalloc( &d_TVTA, block_size * m_cols * sizeof( T ) );

	while( step_offset < max_steps )
	{
		auto b_size = std::min( block_size, max_steps - step_offset );

		QR_block_decomposition_cpu( b_size, step_offset, max_steps );

		size_t rows_to_copy = m_rows - row_offset;
		size_t cols_to_copy = std::min( b_size, m_cols - step_offset );

		cudaMemcpy2D(
			m_d_matrix + row_offset + step_offset * m_rows,       // dst
			m_rows * sizeof( T ),                                 // dst pitch
			m_matrix.data() + row_offset + step_offset * m_rows,  // src
			m_rows * sizeof( T ),                                 // src pitch
			rows_to_copy * sizeof( T ),                           // width (bytes)
			cols_to_copy,                                         // height (cols)
			cudaMemcpyHostToDevice
		);

		size_t v_data_size = std::min( b_size, max_steps - step_offset );
		cudaMemcpy( m_d_v_firsts + step_offset, m_v_firsts.data() + step_offset, v_data_size * sizeof( T ), cudaMemcpyHostToDevice );

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
		QR_decomposition_blocked_TVTA_gpu<<< grid1Dim, blockDim >>>( d_TVTA, m_d_matrix, m_d_v_firsts, m_rows, m_cols, b_size, row_offset, step_offset );

		dim3 grid2Dim( div_up( m_cols - step_offset, b_size ), div_up( m_rows - row_offset, b_size ) );
		QR_decomposition_blocked_VTVTA_gpu<<< grid2Dim, blockDim >>>( d_TVTA, m_d_matrix, m_d_v_firsts, m_rows, m_cols, b_size, row_offset, step_offset );

		cols_to_copy = std::min( b_size, m_cols - step_offset );

		cudaMemcpy2D(
			m_matrix.data() + step_offset * m_rows,  // dst
			m_rows * sizeof( T ),                    // dst pitch
			m_d_matrix + step_offset * m_rows,       // src
			m_rows * sizeof( T ),                    // src pitch
			m_rows * sizeof( T ),                    // width (bytes)
			cols_to_copy,                            // height (cols)
			cudaMemcpyDeviceToHost
		);

		row_offset += b_size;
	}

	cudaMemcpy( m_d_betas, m_betas.data(), max_steps * sizeof( T ), cudaMemcpyHostToDevice );

	cudaFree( d_TVTA );

	m_dynamic_state = DYNAMIC_STATE::QR_DECOMPOSED;
}

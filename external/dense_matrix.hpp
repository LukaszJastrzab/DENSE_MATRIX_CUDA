#pragma once

#include <vector>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <numeric>

#include <utilities.cuh>

template< typename T >
class dense_matrix
{
private:
	// Type definition for state of dense_matrix
	// =========================================
	enum class DYNAMIC_STATE : int
	{
		INIT,
		ITERATIVE,
		LU_DECOMPOSED,
		QR_DECOMPOSED
	};

public:
	/// constructors
	dense_matrix() = default;
	dense_matrix( const dense_matrix& ) = default;
	dense_matrix( dense_matrix&& ) = default;
	dense_matrix( size_t rows, size_t cols );

	///destructor
	~dense_matrix() = default;

	/// defualt assign operator
	dense_matrix< T >& operator=( const dense_matrix< T >& ) = default;

	/// sets matrix sizes and allocates memory
	void init( size_t rows, size_t cols );
	/// adds elements and throws exception if row / col is out of range
	void set_element( T value, size_t row, size_t col );

	// it counts value r := Ax - b
	void count_residual_Ax_b( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const;
	void count_residual_LUx_b( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const;
	void count_residual_QRx_b( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const;
	void count_residual_vector( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const;

	/// decomposes matrix "in situ" to factors LU using Gauss elimination
	void LU_decomposition( size_t pivoting_rows = 0 );
	/// Method solves LU problem (LU_decomposition is needed to call before)
	void solve_LU( std::vector< T >& x, const std::vector< T >& b, std::vector< T >* y = nullptr ) const;

	/// decomposes matrix "in situ" to factors QR using Householder method
	void QR_decomposition();
	/// solves equation Ax=b, where A is decomposed to factors QR (by Householders method)
	void solve_QR( std::vector< T >& x, const std::vector< T >& b, std::vector< T >* y = nullptr ) const;

	/// Method improves the accuracy of the solution
	void iterative_refinement( std::vector< T >& x, const std::vector< T >& b, const double acc, const size_t max_it, const dense_matrix< T >* A_orig = nullptr ) const;

	/// mult operator that mutliplise matrix A by vector x
	template< typename U >
	friend std::vector< U > operator*( const dense_matrix< U >& A, const std::vector< U >& x );


private:
	/// current state of matrix
	DYNAMIC_STATE m_dynamic_state{ DYNAMIC_STATE::INIT };

	/// amount of rows
	size_t m_rows{ 0 };
	/// amount of columns
	size_t m_cols{ 0 };
	/// matrix data
	std::vector< std::vector< T > > m_matrix;

	/// for QR decomposition
	std::vector< T > m_betas;
	std::vector< T > m_v_firsts;


	// row permutation
	std::vector< size_t > m_p_row;		/// under i-th index : original row number
	// column permutation
	std::vector< size_t > m_p_col;		/// under i-th index : original column number

	/// function for pivoting during LU decomposition
	void choose_pivot( const size_t stage, const size_t search );

};

template< typename T >
dense_matrix< T >::dense_matrix( size_t rows, size_t cols )
{
	init( rows, cols );
}

template< typename T >
void dense_matrix< T >::init( size_t rows, size_t cols )
{
	m_rows = rows;
	m_cols = cols;

	m_matrix.resize( m_rows, std::vector< T >( cols, T{} ) );
}

template< typename T >
void dense_matrix< T >::set_element( T value, size_t row, size_t col )
{
	if( row >= m_rows || col >= m_cols )
		throw std::out_of_range( "dense_matrix< T >::set_element - row >= m_rows || col >= m_cols" );

	m_matrix[ row ][ col ] = value;
}

template< typename U >
std::vector< U > operator*( const dense_matrix< U >& A, const std::vector< U >& x )
{
	if( x.size != A.m_cols )
		throw std::invalid_argument( "operator* - x.size != A.m_cols" );

	std::vector< U > result( A.m_rows, U{} );

	for( size_t r{ 0 }; r < A.m_rows; ++r )
		for( size_t c{ 0 }; c < A.m_cols; ++c )
			result[ r ] += A.m_matrix[ r ][ c ] * x[ c ];

	return result;
}

template < typename T >
void dense_matrix< T >::choose_pivot( const size_t step, const size_t search )
{
	const size_t LastSearch = ( step + search < m_cols ? step + search : m_cols );

	size_t ROW{ 0 }, COL{ 0 };
	double ABS_VAL{ 0.0 };

	for( size_t col{ step }; col < LastSearch; ++col )
		for( size_t row{ step }; row < m_rows; ++row )
		{
			const double new_abs{ abs_val( m_matrix[ m_p_row[ row ] ][ m_p_col[ col ] ] ) };

			if( new_abs > ABS_VAL )
			{
				ABS_VAL = new_abs;
				ROW = row;
				COL = col;
			}
		}

	std::swap( m_p_row[ ROW ], m_p_row[ step ] );
	std::swap( m_p_col[ COL ], m_p_col[ step ] );
}

template< typename T >
void dense_matrix< T >::LU_decomposition( size_t pivoting_rows )
{
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix< T >::LU_decomposition: INIT state is required" );

	if( m_rows < m_cols )
		throw std::invalid_argument( "dense_matrix< T >::LU_decomposition: m_rows < m_cols" );

	// 0 means max pivoting strategy ( search through all active parts of active rows )
	// ================================================================================
	if( pivoting_rows == 0 )
		pivoting_rows = m_rows;

	m_p_row.resize( m_rows );
	std::iota( m_p_row.begin(), m_p_row.end(), 0 );

	m_p_col.resize( m_cols );
	std::iota( m_p_col.begin(), m_p_col.end(), 0 );

	const size_t max_steps = std::min( m_rows - 1, m_cols );

	for( size_t step{ 0 }; step < max_steps; ++step )
	{
		choose_pivot( step, pivoting_rows );

		const size_t eliminating_row = m_p_row[ step ];
		const size_t stage_col = m_p_col[ step ];

		const auto pivot{ m_matrix[ eliminating_row ][ stage_col ] };

		for( size_t row{ step + 1 }; row < m_rows; ++row )
		{
			const size_t eliminated_row = m_p_row[ row ];

			m_matrix[ eliminated_row ][ stage_col ] /= pivot;
			const auto eliminator = m_matrix[ eliminated_row ][ stage_col ];

			for( size_t col{ step + 1 }; col < m_cols; ++col )
			{
				const size_t p_col{ m_p_col[ col ] };
				m_matrix[ eliminated_row ][ p_col ] -= eliminator * m_matrix[ eliminating_row ][ p_col ];
			}
		}
	}

	m_dynamic_state = DYNAMIC_STATE::LU_DECOMPOSED;
}

template< typename T >
void dense_matrix< T >::solve_LU( std::vector< T >& x, const std::vector< T >& b, std::vector< T >* y ) const
{
	if( m_dynamic_state != DYNAMIC_STATE::LU_DECOMPOSED )
		throw std::invalid_argument( " dense_matrix< T >::solve_LU: LU_decomposition is needed before" );

	if( m_cols > m_rows )
		throw std::invalid_argument( " dense_matrix< T >::solve_LU: m_cols > m_rows" );

	const size_t max_step{ std::min( m_rows - 1, m_cols ) };
	std::vector< T > y_alloc;

	if( y == nullptr )
	{
		y_alloc.resize( m_rows, T{ 666.0 } );
		y = &y_alloc;
	}

	// first solve the equation Ly = b
	// ===============================
	y->at( 0 ) = b[ m_p_row[ 0 ] ];

	for( size_t row{ 1 }; row < m_cols; ++row )
	{
		const int p_row = m_p_row[ row ];
		y->at( row ) = b[ p_row ];

		for( size_t col{ 0 }; col < row; ++col )
			y->at( row ) -= m_matrix[ p_row ][ m_p_col[ col ] ] * y->at( col );
	}

	// second solve the equation Ux = y
	// ================================
	for( int row{ static_cast< int >( m_cols ) - 1 }; row >= 0; --row )
	{
		x[ m_p_col[ row ] ] = y->at( row );

		for( int col{ row + 1 }; col < m_cols; ++col )
			x[ m_p_col[ row ] ] -= m_matrix[ m_p_row[ row ] ][ m_p_col[ col ] ] * x[ m_p_col[ col ] ];

		x[ m_p_col[ row ] ] /= m_matrix[ m_p_row[ row ] ][ m_p_col[ row ] ];
	}
}

template< typename T >
void dense_matrix< T >::QR_decomposition()
{
	using real_t = typename real_type< T >::type;

	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix< T >::QR_decomposition() - m_dynamic_state != DYNAMIC_STATE::INIT" );

	if( m_rows < m_cols )
		throw std::invalid_argument( "dense_matrix< T >::QR_decomposition() - m_rows < m_cols" );

	const auto max_steps = std::min( m_rows - 1, m_cols );

	// additioanl stored elements needed to recreated Householder vectors v
	// ====================================================================
	m_betas.resize( max_steps, T{} );
	m_v_firsts.resize( max_steps, T{} );

	std::vector< T > vTA( m_cols, T{} );

	for( size_t step{ 0 }; step < max_steps; ++step )
	{
		double col_norm{ 0.0 };

		// calcualte norm
		// ==============
		for( size_t r{ step }; r < m_rows; ++r )
		{
			double abs_val = std::abs( m_matrix[ r ][ step ] );
			col_norm += abs_val * abs_val;
		}
		col_norm = std::sqrt( col_norm );

		// stabilization sign calculation
		// ==============================
		double alpha_abs = std::abs( m_matrix[ step ][ step ] );
		T sign = ( alpha_abs != 0.0 ? -( m_matrix[ step ][ step ] ) / T{ static_cast< real_t >( alpha_abs ) } : T{ -1 } );
		T sign_norm = sign * T{ static_cast< real_t >( col_norm ) };

		m_v_firsts[ step ] = m_matrix[ step ][ step ] - sign_norm;

		T vTv{ conjugate( m_v_firsts[ step ] ) * m_v_firsts[ step ] };

		for( size_t r{ step + 1 }; r < m_rows; ++r )
			vTv += conjugate( m_matrix[ r ][ step ] ) * m_matrix[ r ][ step ];

		// store additional required by QR decomposition data 
		// ==================================================
		m_betas[ step ] = static_cast< real_t >( 2.0 ) / vTv;


		// apply the Householder transformation to the remaining submatrix
		// only needed operations "in situ"
		// ===============================================================
		m_matrix[ step ][ step ] = sign_norm;

		// ==============================================================
		// now we should perform operations A := A - beta( v( vT( A ) ) )
		// above parathesis shows how this operations should be treated
		// ==============================================================

		// calculate vTA ( v*A in case of complex )
		// ========================================
		for( size_t c{ step + 1 }; c < m_cols; ++c )
		{
			vTA[ c ] = conjugate( m_v_firsts[ step ] ) * m_matrix[ step ][ c ];
			for( size_t r{ step + 1 }; r < m_rows; ++r )
				vTA[ c ] += conjugate( m_matrix[ r ][ step ] ) * m_matrix[ r ][ c ];
		}

		// calculate (I-bvvT)A = A - b(v(vTA))
		// ===================================
		for( size_t c{ step + 1 }; c < m_cols; ++c )
			m_matrix[ step ][ c ] -= m_betas[ step ] * m_v_firsts[ step ] * vTA[ c ];

		for( size_t r{ step + 1 }; r < m_rows; ++r )
			for( size_t c{ step + 1 }; c < m_cols; ++c )
				m_matrix[ r ][ c ] -= m_betas[ step ] * m_matrix[ r ][ step ] * vTA[ c ];
	}

	m_dynamic_state = DYNAMIC_STATE::QR_DECOMPOSED;
}

template< typename T >
void dense_matrix< T >::solve_QR( std::vector< T >& x, const std::vector< T >& b, std::vector< T >* y ) const
{
	if( b.size() != m_rows )
		throw std::invalid_argument( "dense_matrix< T >::solve_QR - b.size() != m_rows" );

	if( m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix< T >::solve_QR() - m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED" );

	const auto max_steps = std::min( m_rows - 1, m_cols );

	// first x := Q^T * b = H_1 * H_2 * ... * H_k * b
	// ==============================================
	x = b;
	for( size_t step{ 0 }; step < max_steps; ++step )
	{
		T vTb{ conjugate( m_v_firsts[ step ] ) * x[ step ] };
		for( size_t r{ step + 1 }; r < m_rows; ++r )
			vTb += conjugate( m_matrix[ r ][ step ] ) * x[ r ];

		x[ step ] -= m_betas[ step ] * m_v_firsts[ step ] * vTb;
		for( size_t r{ step + 1 }; r < m_rows; ++r )
			x[ r ] -= m_betas[ step ] * m_matrix[ r ][ step ] * vTb;
	}

	// then solve Rx = Q^T * b by back substitution
	// ============================================
	for( auto r = static_cast< int >( m_cols ) - 1; r >= 0; --r )
	{
		T sum{ T{} };
		for( int c{ r + 1 }; c < m_cols; ++c )
			sum += m_matrix[ r ][ c ] * x[ c ];

		x[ r ] = ( x[ r ] - sum ) / m_matrix[ r ][ r ];
	}
}

template< typename T >
void dense_matrix< T >::count_residual_Ax_b( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const
{
	if( x.size() != m_cols || b.size() != m_rows || r.size() != m_rows )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_Ax_b - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );
	if ( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_Ax_b - m_dynamic_state != DYNAMIC_STATE::INIT" );

	for( size_t row{ 0 }; row < m_rows; ++row )
		r[ row ] = -b[ row ];
	for( size_t row{ 0 }; row < m_rows; ++row )
		for( size_t col{ 0 }; col < m_cols; ++col )
			r[ row ] += ( x[ col ] * m_matrix[ row ][ col ] );
}

template< typename T >
void dense_matrix< T >::count_residual_LUx_b( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const
{
	if( x.size() != m_cols || b.size() != m_rows || r.size() != m_rows )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_LUx_b - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::LU_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_Ax_b - m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED" );

	std::vector< T > w( m_rows, T{} );

	// compute w=Ux
	// ============
	for( size_t row{ 0 }; row < m_rows; ++row )
		for( size_t col{ row }; col < m_cols; ++col )
			w[ row ] += ( x[ m_p_col[ col ] ] * m_matrix[ m_p_row[ row ] ][ m_p_col[ col ] ] );

	// compute r = Lw - b
	// ==================
	for( size_t row{ 0 }; row < m_rows; ++row )
	{
		r[ m_p_row[ row ] ] = w[ row ] - b[ m_p_row[ row ] ];

		for( size_t col{ 0 }; col < row; ++col )
			r[ m_p_row[ row ] ] += w[ col ] * m_matrix[ m_p_row[ row ] ][ m_p_col[ col ] ];
	}
}

template< typename T >
void dense_matrix< T >::count_residual_QRx_b( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const
{
	if( x.size() != m_cols || b.size() != m_rows || r.size() != m_rows )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_QRx_b - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_Ax_b - m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED" );

	const int max_steps = std::min( m_rows - 1, m_cols );

	for( size_t row{ 0 }; row < m_rows; ++row )
	{
		r[ row ] = T{};
		for( size_t col{ row }; col < m_cols; ++col )
			r[ row ] += ( x[ col ] * m_matrix[ row ][ col ] );
	}

	for( int step{ max_steps - 1 }; step >= 0; --step )
	{
		T vRx{ conjugate( m_v_firsts[ step ] ) * r[ step ] };
		for( int s{ step + 1 }; s < static_cast< int >( m_rows ); ++s )
			vRx += conjugate( m_matrix[ s ][ step ] ) * r[ s ];

		r[ step ] -= m_betas[ step ] * m_v_firsts[ step ] * vRx;
		for( int s{ step + 1 }; s < static_cast< int >( m_rows ); ++s )
			r[ s ] -= m_betas[ step ] * m_matrix[ s ][ step ] * vRx;
	}

	for( size_t row{ 0 }; row < m_rows; ++row )
		r[ row ] -= b[ row ];
}

template< typename T >
void dense_matrix< T >::count_residual_vector( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const
{
	switch( m_dynamic_state )
	{
	case DYNAMIC_STATE::INIT:
		count_residual_Ax_b( x, b, r );
		break;

	case DYNAMIC_STATE::LU_DECOMPOSED:
		count_residual_LUx_b( x, b, r );
		break;

	case DYNAMIC_STATE::QR_DECOMPOSED:
		count_residual_QRx_b( x, b, r );
		break;

	default:
		throw std::invalid_argument( "dense_matrix< T >::count_residual_vector - state not supported" );
	}
}

template < typename T >
void dense_matrix< T >::iterative_refinement( std::vector< T >& x, const std::vector< T >& b, const double acc, const size_t max_it, const dense_matrix< T >* A_orig ) const
{
	if( m_rows < m_cols )
		throw std::exception( "dense_matrix< T >::iterative_refinement - m_rows < m_cols" );

	const size_t N = m_rows;

	std::vector< T > d( N );
	std::vector< T > r( N );
	std::vector< T > y( N );

	size_t iteration = 0;

	if ( A_orig != nullptr )
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
			solve_QR( d, r, &y );
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
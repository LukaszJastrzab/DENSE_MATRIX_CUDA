#pragma once

#include <vector>
#include <cmath>
#include <type_traits>
#include <stdexcept>

template< typename T >
double l2_norm( const std::vector< T >& v )
{
	double sum{ 0.0 };

	for( const auto& vi : v )
		sum += norm( vi );

	return std::sqrt( sum );
}


template< typename T >
T conj_if_complex( const T& x )
{
	if constexpr( std::is_floating_point_v< T > )
		return x;
	else
		return std::conj( x );
}


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

	/// sets matrix sizes and allocates memory
	void init( size_t rows, size_t cols );
	/// adds elements and throws exception if row / col is out of range
	void set_element( T value, size_t row, size_t col );

	// it counts value r := Ax - b
	void count_residual_vector( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const;

	/// decomposes matrix "in situ" to factors QR using Householder method
	void QR_decomposition();
	/// solves equation Ax=b, where A is decomposed to factors QR (by Householders method)
	void solve_QR( std::vector< T >& x, const std::vector< T >& b ) const;


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

	/// for LU decomposition
	std::vector< T > m_pivots;
	/// for QR decomposition
	std::vector< T > m_betas;
	std::vector< T > m_v_firsts;


	// row permutation
	std::vector< T > m_p_row;	/// under i-th index : original row number
	// row rev permutation
	std::vector< T > m_rp_row;	/// under i-th index : position of i-th original row
	// column permutation
	std::vector< T > m_p_col;	/// under i-th index : original column number
	// column rev permutation
	std::vector< T > m_rp_col;	/// under i-th index : position of i-th original column

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

template< typename T >
void dense_matrix< T >::QR_decomposition()
{
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix< T >::QR_decomposition() - m_dynamic_state != DYNAMIC_STATE::INIT" );

	if( m_rows < m_cols )
		throw std::invalid_argument( "dense_matrix< T >::QR_decomposition() - m_rows < m_cols" );

	auto max_steps = std::min( m_rows - 1, m_cols );

	// additioanl stored elements needed to recreated Householder vectors v
	// ====================================================================
	m_betas.resize( max_steps, T{} );
	m_v_firsts.resize( max_steps, T{} );

	std::vector< T > v( m_rows, T{} );
	std::vector< T > vTA( m_cols, T{} );

	for( size_t step{ 0 }; step < max_steps; ++step )
	{
		T col_norm{ 0.0 };

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
		T sign = ( alpha_abs != 0.0 ? -( m_matrix[ step ][ step ] ) / alpha_abs : T{ -1 } );
		T sign_norm = sign * col_norm;

		v[ step ] = m_matrix[ step ][ step ] - sign_norm;
		T vTv{ conj_if_complex( v[ step ] ) * v[ step ] };

		for( size_t r{ step + 1 }; r < m_rows; ++r )
		{
			v[ r ] = m_matrix[ r ][ step ];
			vTv += conj_if_complex( v[ r ] ) * v[ r ];
		}

		// store additional required by QR decomposition data 
		// ==================================================
		m_betas[ step ] = 2.0 / vTv;
		m_v_firsts[ step ] = v[ step ];

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
			vTA[ c ] = T{};
			for( size_t r{ step }; r < m_rows; ++r )
				vTA[ c ] += conj_if_complex( v[ r ] ) * m_matrix[ r ][ c ];
		}

		for( size_t c{ step + 1 }; c < m_cols; ++c )
			for( size_t r{ step }; r < m_rows; ++r )
				m_matrix[ r ][ c ] -= m_betas[ step ] * v[ r ] * vTA[ c ];
	}

	m_dynamic_state = DYNAMIC_STATE::QR_DECOMPOSED;
}

template< typename T >
void dense_matrix< T >::solve_QR( std::vector< T >& x, const std::vector< T >& b ) const
{
	if( b.size() != m_rows )
		throw std::invalid_argument( "dense_matrix< T >::solve_QR - b.size() != m_rows" );

	if( m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix< T >::solve_QR() - m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED" );

	auto max_steps = std::min( m_rows - 1, m_cols );

	// first x := Q^T * b = H_1 * H_2 * ... * H_k * b
	// ==============================================
	x = b;
	for( size_t step{ 0 }; step < max_steps; ++step )
	{
		T vTb{ conj_if_complex( m_v_firsts[ step ] ) * x[ step ] };
		for( size_t r{ step + 1 }; r < m_rows; ++r )
			vTb += conj_if_complex( m_matrix[ r ][ step ] ) * x[ r ];

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
void dense_matrix< T >::count_residual_vector( const std::vector< T >& x, const std::vector< T >& b, std::vector< T >& r ) const
{
	if( x.size() != m_cols || b.size() != m_rows || r.size() != m_rows )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_vector - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );

	for( size_t row{ 0 }; row < m_rows; ++row )
		r[ row ] = -b[ row ];
	for( size_t row{ 0 }; row < m_rows; ++row )
		for( size_t col{ 0 }; col < m_cols; ++col )
			r[ row ] += ( x[ col ] * m_matrix[ row ][ col ] );
}
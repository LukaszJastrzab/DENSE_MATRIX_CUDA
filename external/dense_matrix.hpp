#pragma once

#include <vector>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <numeric>

#include <utilities.cuh>

namespace dm
{

// Type definition for state of dense_matrix
// =========================================
enum class DYNAMIC_STATE : int
{
	INIT,
	ITERATIVE,
	LU_DECOMPOSED,
	QR_DECOMPOSED,
	QHQ_DECOMPOSED,
	QUASI_QR
};

template< typename T >
class dense_matrix
{
public:
	/// constructors
	dense_matrix() = default;
	dense_matrix( const dense_matrix& ) = default;
	dense_matrix( dense_matrix&& ) = default;
	dense_matrix( size_t rows, size_t cols );

	///destructor
	~dense_matrix() = default;

	/// double type used by this template
	using DT = typename double_type< T >::type;
	/// real type used by this template
	using RT = typename real_type< T >::type;

	/// defualt assign operator
	dense_matrix< T >& operator=( const dense_matrix< T >& ) = default;

	/// sets matrix sizes and allocates memory
	void init( size_t rows, size_t cols );
	/// adds elements and throws exception if row / col is out of range
	void set_element( T value, size_t row, size_t col );

	/// it counts value r := Ax - b
	void count_residual_vector( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const;

	/// decomposes matrix "in situ" to factors LU using Gauss elimination
	void LU_decomposition( const bool scaling, size_t pivoting_rows = 0, const RT singularity_acc = std::numeric_limits< RT >::epsilon() );
	/// method solves LU problem (LU_decomposition is needed to call before)
	void solve_LU( std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >* y = nullptr ) const;
	/// method computes determinant using LU decomposition
	typename double_type< T >::type det() const;

	/// decomposes matrix "in situ" to factors QR using Householder method
	void QR_decomposition( const bool scaling, const RT singularity_acc = std::numeric_limits< RT >::epsilon() );
	/// solves equation Ax=b, where A is decomposed to factors QR (by Householders method)
	void solve_QR( std::vector< DT >& x, const std::vector< DT >& b ) const;

	/// decomposes matrix "in situ" to QHQ (using Householder) where H is in Hessenberg form
	void QHQ_decomposition();
	/// computes eqigen values using QR algorithm
	void compute_eigenvalues_QR( std::vector< std::complex< double > >& l, const size_t max_it, const bool Francis = true,
								 const double acc = std::numeric_limits< RT >::epsilon() );

	/// Method improves the accuracy of the solution
	void iterative_refinement( std::vector< DT >& x, const std::vector< DT >& b, const double acc, const size_t max_it, const dense_matrix< T >* A_orig = nullptr ) const;

	/// addition operator
	template< typename U, typename V >
	friend dense_matrix< std::common_type_t< U, V > > operator+( const dense_matrix< U >& A, const dense_matrix< V >& B );
	/// subtraction operator
	template< typename U, typename V >
	friend dense_matrix< std::common_type_t< U, V > > operator-( const dense_matrix< U >& A, const dense_matrix< V >& B );
	/// multiplication operators
	template< typename U, typename V >
	friend dense_matrix< std::common_type_t< U, V > > operator*( const dense_matrix< U >& A, const dense_matrix< V >& B );
	/// multiplication operator
	template< typename U, typename V >
	friend dense_matrix< std::common_type_t< U, V > > operator*( const  V& b, const dense_matrix< U >& A );
	/// mult operator that mutliplise matrix A by vector x
	template< typename U, typename V >
	friend dense_matrix< std::common_type_t< U, V > > operator*( const V& b, const dense_matrix< U >& A );

private:
	/// current state of matrix
	DYNAMIC_STATE m_dynamic_state{ DYNAMIC_STATE::INIT };

	/// amount of rows
	size_t m_rows{ 0 };
	/// amount of columns
	size_t m_cols{ 0 };
	/// matrix data
	std::vector< std::vector< T > > m_matrix;

	/// sing for determinant
	int m_dsign{ 1 };

	/// for QR decomposition
	std::vector< T > m_betas;
	std::vector< T > m_v_firsts;

	/// row permutation
	std::vector< size_t > m_p_row;		/// under i-th index : original row number
	/// column permutation
	std::vector< size_t > m_p_col;		/// under i-th index : original column number

	/// row / column scaling parameters
	std::vector< double > m_scalars;

	/// accuracy used for finding complex blocks in QR algrithm
	inline static const double DEFLATION_ACC{ std::numeric_limits< RT >::epsilon() };

	/// Function permuts row lying on pos1 position with row lying on pos2 position
	void permute_rows( size_t pos1, size_t pos2 );
	/// Function permuts column lying on pos1 position with column lying on pos2 position
	void permute_cols( size_t pos1, size_t pos2 );
	/// function for pivoting during LU decomposition
	void choose_pivot( const size_t stage, const size_t search );
	/// rows scaling
	void rows_scaling();
	/// cols scaling
	void cols_scaling();
	/// it counts value r := Ax - b for different dynamic state of the matrix
	void count_residual_Ax_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const;
	void count_residual_LUx_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const;
	void count_residual_QRx_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const;
	/// method dumps eigen values during QR algorithm
	void QR_get_eigenvalues( std::vector< std::complex< double > >& l );
	/// method used buble racing in QR algorithm for eigenvalues problem
	bool QHQ_NxN_with_shifts( const size_t row_shift, const size_t col_shift, const size_t block_end, size_t block_size, std::vector< T > v = {} );
	/// method returns Francis step column
	std::vector< T > get_Francis_v( const size_t shift, const size_t block_end );
	/// method used in QR alogoritm, it gets eigenvalues from 2x2 Shur block
	void QR_get_eigenvalues_from_block( const size_t shift, std::vector< std::complex< double > >&l );

	/// test methods
	template< typename U >
	friend std::vector< dense_matrix< U > > get_factors( const dense_matrix< U >& A );

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

template< typename U, typename V >
dense_matrix< std::common_type_t< U, V > > operator+( const dense_matrix< U >& A, const dense_matrix< V >& B )
{
	if( A.m_rows != B.m_rows || A.m_cols != B.m_cols )
		throw std::invalid_argument( "dense_matrix: operator+ - A.m_rows != B.m_rows || A.m_cols != B.m_cols" );

	using R = std::common_type_t< U, V >;

	dense_matrix< R > result( A.m_rows, B.m_cols );

	for( size_t r{ 0 }; r < A.m_rows; ++r )
		for( size_t c{ 0 }; c < A.m_cols; ++c )
			result.set_element( static_cast< R >( A.m_matrix[ r ][ c ] ) + static_cast< R >( B.m_matrix[ r ][ c ] ), r, c );

	return result;
}

template< typename U, typename V >
dense_matrix< std::common_type_t< U, V > > operator-( const dense_matrix< U >& A, const dense_matrix< V >& B )
{
	if( A.m_rows != B.m_rows || A.m_cols != B.m_cols )
		throw std::invalid_argument( "dense_matrix: operator- - A.m_rows != B.m_rows || A.m_cols != B.m_cols" );

	using R = std::common_type_t< U, V >;

	dense_matrix< R > result( A.m_rows, B.m_cols );

	for( size_t r{ 0 }; r < A.m_rows; ++r )
		for( size_t c{ 0 }; c < A.m_cols; ++c )
			result.set_element( static_cast< R >( A.m_matrix[ r ][ c ] ) - static_cast< R >( B.m_matrix[ r ][ c ] ), r, c );

	return result;
}

template< typename U, typename V >
dense_matrix< std::common_type_t< U, V > > operator*( const dense_matrix< U >& A, const dense_matrix< V >& B )
{
	if( A.m_cols != B.m_rows )
		throw std::invalid_argument( "dense_matrix: operator* - A.m_cols != B.m_rows" );

	using R = std::common_type_t< U, V >;

	dense_matrix< R > result( A.m_rows, B.m_cols );

	for( size_t r{ 0 }; r < A.m_rows; ++r )
		for( size_t c{ 0 }; c < B.m_cols; ++c )
		{
			R mult_sum{};
			for( size_t i{ 0 }; i < A.m_cols; ++i )
				mult_sum += A.m_matrix[ r ][ i ] * B.m_matrix[ i ][ c ];
			result.set_element( mult_sum, r, c );
		}

	return result;
}

template< typename U, typename V >
std::vector< std::common_type_t< U, V > > operator*( const dense_matrix< U >& A, const std::vector< U >& v )
{
	if( v.size != A.m_cols )
		throw std::invalid_argument( "operator* - v.size != A.m_cols" );

	std::vector< std::common_type_t< U, V > > result( A.m_rows, U{} );

	for( size_t r{ 0 }; r < A.m_rows; ++r )
		for( size_t c{ 0 }; c < A.m_cols; ++c )
			result[ r ] += A.m_matrix[ r ][ c ] * v[ c ];

	return result;
}

template< typename U, typename V >
dense_matrix< std::common_type_t< U, V > > operator*( const  V& b, const dense_matrix< U >& A )
{
	using R = std::common_type_t< U, V >;

	dense_matrix< R > result( A.m_rows, A.m_cols );

	for( size_t r{ 0 }; r < A.m_rows; ++r )
		for( size_t c{ 0 }; c < A.m_cols; ++c )
			result.m_matrix[ r ][ c ] = A.m_matrix[ r ][ c ] * b;

	return result;
}

template < typename T >
void dense_matrix< T >::permute_rows( size_t pos1, size_t pos2 )
{
	if( pos1 != pos2 )
		m_dsign = -m_dsign;

	std::swap( m_p_row[ pos1 ], m_p_row[ pos2 ] );
}

template < typename T >
void dense_matrix< T >::permute_cols( size_t pos1, size_t pos2 )
{
	if( pos1 != pos2 )
		m_dsign = -m_dsign;

	std::swap( m_p_col[ pos1 ], m_p_col[ pos2 ] );
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

	permute_rows( ROW, step );
	permute_cols( COL, step );
}

template< typename T >
void dense_matrix< T >::LU_decomposition( const bool scaling, size_t pivoting_rows, const RT singularity_acc )
{
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix< T >::LU_decomposition: INIT state is required" );
	if( m_rows != m_cols )
		throw std::invalid_argument( "dense_matrix< T >::LU_decomposition: m_rows != m_cols" );

	// 0 means max pivoting strategy ( search through all active parts of active rows )
	// ================================================================================
	if( pivoting_rows == 0 )
		pivoting_rows = m_rows;

	if( scaling )
		rows_scaling();
	else
		m_scalars.resize( m_rows, 1.0 );

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

		if( abs_val( pivot ) <= singularity_acc )
			throw singularity_error( "dense_matrix< T >::LU_decomposition - a singular matrix was obtained" );

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

	if( abs_val( m_matrix[ m_p_row[ max_steps ] ][ m_p_col[ max_steps ] ] ) <= singularity_acc )
		throw singularity_error( "dense_matrix< T >::LU_decomposition - a singular matrix was obtained" );

	m_dynamic_state = DYNAMIC_STATE::LU_DECOMPOSED;
}

template< typename T >
void dense_matrix< T >::solve_LU( std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >* y ) const
{
	if( m_cols != m_rows )
		throw std::invalid_argument( " dense_matrix< T >::solve_LU: m_cols != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::LU_DECOMPOSED )
		throw std::invalid_argument( " dense_matrix< T >::solve_LU: LU_decomposition is needed before" );

	const size_t max_step{ std::min( m_rows - 1, m_cols ) };

	if( x.size() < m_cols )
		x.resize( m_cols );

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
			y->at( row ) -= static_cast< DT >( m_matrix[ p_row ][ m_p_col[ col ] ] ) * y->at( col );
	}

	// second solve the equation Ux = y
	// ================================
	for( int row{ static_cast< int >( m_cols ) - 1 }; row >= 0; --row )
	{
		x[ m_p_col[ row ] ] = y->at( row );

		for( int col{ row + 1 }; col < m_cols; ++col )
			x[ m_p_col[ row ] ] -= static_cast< DT >( m_matrix[ m_p_row[ row ] ][ m_p_col[ col ] ] ) * x[ m_p_col[ col ] ];

		x[ m_p_col[ row ] ] /= static_cast< DT >( m_matrix[ m_p_row[ row ] ][ m_p_col[ row ] ] );
	}
}

template< typename T >
typename double_type< T >::type dense_matrix< T >::det() const
{
	if( m_cols != m_rows )
		throw std::invalid_argument( " dense_matrix< T >::det: m_cols != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::LU_DECOMPOSED )
		throw std::invalid_argument( " dense_matrix< T >::det: LU_decomposition is needed before" );

	DT result{ static_cast< DT >( m_dsign ) };

	for( size_t rc{ 0 }; rc < m_rows; ++rc )
		result *= ( static_cast< DT >( m_matrix[ m_p_row[ rc ] ][ m_p_col[ rc ] ] ) / static_cast< DT >( m_scalars[ m_p_row[ rc ] ] ) );

	return result;
}

template< typename T >
void dense_matrix< T >::QHQ_decomposition()
{
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix< T >::QHQ_decomposition() - m_dynamic_state != DYNAMIC_STATE::INIT" );
	if( m_rows != m_cols )
		throw std::invalid_argument( "dense_matrix< T >::QHQ_decomposition() - m_rows != m_cols" );

	const auto max_steps = m_rows - 2;

	// additioanl stored elements needed to recreated Householder vectors v
	// ====================================================================
	m_betas.resize( max_steps, T{} );
	m_v_firsts.resize( max_steps, T{} );

	std::vector< T > Av( m_rows, T{} ), vTA( m_cols, T{} );

	for( size_t step{ 0 }; step < max_steps; ++step )
	{
		double col_norm{ 0.0 };
		const size_t row_step{ step + 1 };

		// calcualte norm
		// ==============
		for( size_t r{ row_step }; r < m_rows; ++r )
		{
			double abs_v = abs_val( m_matrix[ r ][ step ] );
			col_norm += abs_v * abs_v;
		}
		col_norm = std::sqrt( col_norm );

		// stabilization sign calculation
		// ==============================
		double alpha_abs = abs_val( m_matrix[ row_step ][ step ] );
		T sign = ( alpha_abs != 0.0 ? -( m_matrix[ row_step ][ step ] ) / T{ static_cast< RT >( alpha_abs ) } : T{ -1 } );
		T sign_norm = sign * T{ static_cast< RT >( col_norm ) };

		m_v_firsts[ step ] = m_matrix[ row_step ][ step ] - sign_norm;
		const auto v1{ m_v_firsts[ step ] };
		const auto v1T{ conjugate( v1 ) };

		T vTv{ v1T * v1 };
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
			vTv += conjugate( m_matrix[ r ][ step ] ) * m_matrix[ r ][ step ];

		if( vTv == T{ 0 } )
		{
			m_betas[ step ] = T{ 0 };
			continue;
		}

		m_betas[ step ] = static_cast< RT >( 2.0 ) / vTv;
		const auto beta{ m_betas[ step ] };

		// calculate Av
		//=============
		for( size_t r{ 0 }; r < m_rows; ++r )
		{
			Av[ r ] = m_matrix[ r ][ row_step ] * v1;
			for( size_t c{ row_step + 1 }; c < m_cols; ++c )
				Av[ r ] += m_matrix[ r ][ c ] * m_matrix[ c ][ step ];
		}

		// calculate vTA ( v*A in case of complex )
		// ========================================
		for( size_t c{ step }; c < m_cols; ++c )
		{
			vTA[ c ] = v1T * m_matrix[ row_step ][ c ];
			for( size_t r{ row_step + 1 }; r < m_rows; ++r )
				vTA[ c ] += conjugate( m_matrix[ r ][ step ] ) * m_matrix[ r ][ c ];
		}

		// alpha = v*Av
		// ============
		T alpha{ v1T * Av[ row_step ] };
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
			alpha += conjugate( m_matrix[ r ][ step ] ) * Av[ r ];

		// apply the Householder transformation QAQ to the remaining submatrix
		// only needed operations "in situ"
		// ===================================================================
		m_matrix[ row_step ][ step ] = sign_norm;

		// update those part of matrix that are changed only by right mult by QT
		// =====================================================================
		for( size_t r{ 0 }; r < row_step; ++r )
		{
			const auto Av_{ Av[ r ] };
			m_matrix[ r ][ row_step ] -= beta * Av_ * v1T;

			for( size_t c{ row_step + 1 }; c < m_cols; ++c )
				m_matrix[ r ][ c ] -= beta * Av_ * conjugate( m_matrix[ c ][ step ] );
		}

		// update left-upper corner of submatrix
		// =====================================
		m_matrix[ row_step ][ row_step ] -= beta * ( v1 * vTA[ row_step ] + Av[ row_step ] * v1T - beta * alpha * v1 * v1T );

		// update fiest modificated sub row
		// ================================
		for( size_t c{ row_step + 1 }; c < m_cols; ++c )
		{
			const auto v1T{ conjugate( m_matrix[ c ][ step ] ) };
			m_matrix[ row_step ][ c ] -= beta * ( v1 * vTA[ c ] + Av[ row_step ] * v1T - beta * alpha * v1 * v1T );
		}

		// update fiest modificated sub col
		// ================================
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
		{
			const auto v1{ m_matrix[ r ][ step ] };
			m_matrix[ r ][ row_step ] -= beta * ( v1 * vTA[ row_step ] + Av[ r ] * v1T - beta * alpha * v1 * v1T );
		}

		// update rest part of sub matrix
		// ==============================
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
		{
			const auto v1{ m_matrix[ r ][ step ] };
			const auto Av_{ Av[ r ] };

			for( size_t c{ row_step + 1 }; c < m_cols; ++c )
			{
				const auto v1T{ conjugate( m_matrix[ c ][ step ] ) };
				m_matrix[ r ][ c ] -= beta * ( v1 * vTA[ c ] + Av_ * v1T - beta * alpha * v1 * v1T );
			}
		}
	}

	m_dynamic_state = DYNAMIC_STATE::QHQ_DECOMPOSED;
}


template< typename T >
void dense_matrix< T >::QR_get_eigenvalues_from_block( const size_t shift, std::vector< std::complex< double > >& l )
{
	std::complex< double > a{ m_matrix[ shift ][ shift ] }, b{ m_matrix[ shift ][ shift + 1 ] },
		c{ m_matrix[ shift + 1 ][ shift ] }, d{ m_matrix[ shift + 1 ][ shift + 1 ] };

	std::complex< double > tr{ a + d };
	std::complex< double > det{ a * d - b * c };
	std::complex< double > disc{ std::sqrt( tr * tr - std::complex< double >( 4.0 ) * det ) };

	l[ shift ] = ( tr + disc ) / std::complex< double >( 2.0 );
	l[ shift + 1 ] = ( tr - disc ) / std::complex< double >( 2.0 );
}


template< typename T >
void dense_matrix< T >::QR_get_eigenvalues( std::vector< std::complex< double > >& l )
{
	for( size_t i{ 0 }; i < m_cols; )
	{
		if( i < m_cols - 1 && abs_val( m_matrix[ i + 1 ][ i ] ) > DEFLATION_ACC )
		{
			QR_get_eigenvalues_from_block( i, l );
			i += 2;
		}
		else
		{
			l[ i ] = std::complex< double >( m_matrix[ i ][ i ] );
			++i;
		}
	}
}


template< typename T >
bool dense_matrix< T >::QHQ_NxN_with_shifts( const size_t row_shift, const size_t col_shift, const size_t block_end, const size_t block_size, std::vector< T > v )
{
	const auto row_nshift{ row_shift + 1 };
	const auto col_nshift{ col_shift + 1 };

	bool use_spec_v{ v.size() > 0 };

	if( !use_spec_v )
	{
		v.resize( block_size );
		for( size_t i{ row_shift }; i < min( m_rows, row_shift + block_size ); ++i )
			v[ i - row_shift ] = m_matrix[ i ][ col_shift ];
	}

	double col_norm{ 0.0 };
	std::vector< double > abs_v( block_size );
	for( size_t i{ 0 }; i < block_size; ++i )
	{	
		double vabs{ abs_val( v[ i ] ) };
		abs_v[ i ] = vabs;
		col_norm += ( vabs * vabs );
	}
	col_norm = std::sqrt( col_norm );

	T sign = ( abs_v[ 0 ] != 0.0 ? -v[ 0 ] / T{ static_cast< RT >( abs_v[ 0 ] ) } : T{ -1 } );
	T sign_norm = sign * T{ static_cast< RT >( col_norm ) };

	v[ 0 ] -= sign_norm;

	std::vector< T > vT( block_size );
	for( size_t i{ 0 }; i < block_size; ++i )
		vT[ i ] = conjugate( v[ i ] );

	T vTv{};
	for( size_t i{ 0 }; i < block_size; ++i )
		vTv += v[ i ] * vT[ i ];
	
	if( !use_spec_v )
	{
		m_matrix[ row_shift ][ col_shift ] = sign_norm;
		for( size_t i{ row_shift + 1 }; i < min( m_rows, row_shift + block_size ); ++i )
			m_matrix[ i ][ col_shift ] = T{};
	}

	if( abs_val( vTv ) < std::numeric_limits< RT >::epsilon() )
		return false;

	const auto beta{ static_cast< RT >( 2.0 ) / vTv };

	// A' <- A - beta * v * ( vT * A )
	// ===============================
	std::vector< T > vTA( block_end, T{} );

	for( size_t c{ use_spec_v ? col_shift : col_nshift }; c < block_end; ++c )
		for ( size_t i{ row_shift }; i < min( m_rows, row_shift + block_size ); ++i )
			vTA[ c ] += vT[ i - row_shift ] * m_matrix[ i ][ c ];

	for( size_t c{ use_spec_v ? col_shift : col_nshift }; c < block_end; ++c )
		for( size_t i{ row_shift }; i < min( m_rows, row_shift + block_size ); ++i)
			m_matrix[ i ][ c ] -= beta * v[ i - row_shift ] * vTA[ c ];

	// A <- A' - beta * ( A' v ) vT
	// ============================
	std::vector< T > Av( block_end, T{} );

	for( size_t r{ 0 }; r < std::min( row_nshift + block_size, block_end ); ++r )
		for( size_t i{ row_shift }; i < min( m_cols, row_shift + block_size ); ++i )
			Av[ r ] += m_matrix[ r ][ i ] * v[ i - row_shift ];

	for( size_t r{ 0 }; r < std::min( row_nshift + block_size, block_end ); ++r )
		for( size_t i{ row_shift }; i < min( m_rows, row_shift + block_size ); ++i )
			m_matrix[ r ][ i ] -= beta * Av[ r ] * vT[ i - row_shift ];

	return true;
}

template< typename T >
std::vector< T > dense_matrix< T >::get_Francis_v( const size_t shift, const size_t block_end )
{
	const T a{ m_matrix[ block_end - 1 ][ block_end - 1 ] },
		b{ m_matrix[ block_end - 1 ][ block_end ] },
		c{ m_matrix[ block_end ][ block_end - 1 ] },
		d{ m_matrix[ block_end ][ block_end ] };
	const T tr{ a + d }, det{ a * d - b * c };
	const T a11{ m_matrix[ shift ][ shift ] }, a12{ m_matrix[ shift ][ shift + 1 ] },
		a21{ m_matrix[ shift + 1 ][ shift ] }, a22{ m_matrix[ shift + 1 ][ shift + 1 ] };
	T a32{};

	if ( shift < m_rows - 2 )
		a32 = m_matrix[ shift + 2 ][ shift + 1 ];

	std::vector< T > v{ a11 * a11 + a21 * a12 - tr * a11 + det,
		                a11 * a21 + a21 * a22 - tr * a21,
	                                a21 * a32 };

	return v;
}

template< typename T >
void dense_matrix< T >::compute_eigenvalues_QR( std::vector< std::complex< double > >& l, const size_t max_it, const bool Francis, const double acc )
{
	if( m_rows != m_cols )
		throw std::invalid_argument( "dense_matrix< T >::compute_eigenvalues_QR - m_rows != m_cols" );
	if( m_dynamic_state == DYNAMIC_STATE::INIT )
		QHQ_decomposition();
	if( m_dynamic_state != DYNAMIC_STATE::QHQ_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix< T >::compute_eigenvalues_QR - m_dynamic_state != DYNAMIC_STATE::QHQ_DECOMPOSED" );

	using ull_pair = std::pair< size_t, size_t >;
	std::map< size_t, ull_pair > blocks;

	const auto block_size = Francis ? 3 : 2;
	l.resize( m_rows, std::complex< double >{} );

	// vanish elements under lower diag of Hessenberg form
	// ===================================================
	for( size_t i{ 0 }; i < m_rows - 2; ++i )
		m_matrix[ i + 2 ][ i ] = T{};

	// vanish elements yet under lower diag of Hessenberg form
	// in case of double shift usage
	// =======================================================
	if ( Francis )
		for( size_t i{ 0 }; i < m_rows - 3; ++i )
			m_matrix[ i + 3 ][ i ] = T{};

	size_t block_idx{ 0 };
	blocks[ block_idx++ ] = ull_pair( 0, m_rows );

	for( size_t iter{ 0 }; iter < max_it; ++iter )
	{
		// deflection
		// ==========
		for( auto& [ id, block ] : blocks )
		{
			for( auto i{ block.first }; i < block.second - 1; ++i )
			{
				auto ii{ i + 1 };
				const double a{ abs_val( m_matrix[ i ][ i ] ) },
					b{ abs_val( m_matrix[ ii ][ ii ] ) },
					c{ abs_val( m_matrix[ ii ][ i ] ) };

				if( c <= acc * ( a + b ) )
				{
					m_matrix[ ii ][ i ] = T{};
					blocks[ block_idx++ ] = ull_pair( ii, block.second );
					block.second = ii;
					break;
				}
			}
		}

		// remove block which sizes are <= 2
		// =================================
		for( auto it = blocks.begin(); it != blocks.end(); )
		{
			if( it->second.second - it->second.first <= 2 )
				it = blocks.erase( it );
			else
				++it;
		}

		if( blocks.size() == 0 )
			break;

		for( const auto& [ id, block ] : blocks )
		{
			// Rayleigh shifting
			// =================
			T mu{ m_matrix[ block.second - 1 ][ block.second - 1 ] };
			for( auto i{ block.first }; i < block.second; ++i )
				m_matrix[ i ][ i ] -= mu;

			// QR Hessenberg reduction
			// =======================
			if( QHQ_NxN_with_shifts( block.first, block.first, block.second, block_size,
				Francis ? get_Francis_v( block.first, block.second - 1 ) : std::vector< T >() ) )
			{
				for( auto i{ block.first }; i < block.second - 1; ++i )
					QHQ_NxN_with_shifts( i + 1, i, block.second, block_size );
			}

			// Rayleigh shifting back
			// ======================
			for( auto i{ block.first }; i < block.second; ++i )
				m_matrix[ i ][ i ] += mu;
		}
	}

	QR_get_eigenvalues( l );

	m_dynamic_state = DYNAMIC_STATE::QUASI_QR;
}


template< typename T >
void dense_matrix< T >::QR_decomposition( const bool scaling, const RT singularity_acc )
{
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix< T >::QR_decomposition() - m_dynamic_state != DYNAMIC_STATE::INIT" );
	if( m_rows != m_cols )
		throw std::invalid_argument( "dense_matrix< T >::QR_decomposition() - m_rows != m_cols" );

	if( scaling )
		cols_scaling();
	else
		m_scalars.resize( m_cols, 1.0 );

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
			double abs_v = abs_val( m_matrix[ r ][ step ] );
			col_norm += abs_v * abs_v;
		}
		col_norm = std::sqrt( col_norm );

		if ( col_norm < singularity_acc )
			throw singularity_error( "dense_matrix< T >::QR_decomposition - a singular matrix was obtained" );

		// stabilization sign calculation
		// ==============================
		double alpha_abs = abs_val( m_matrix[ step ][ step ] );
		T sign = ( alpha_abs != 0.0 ? -( m_matrix[ step ][ step ] ) / T{ static_cast< RT >( alpha_abs ) } : T{ -1 } );
		T sign_norm = sign * T{ static_cast< RT >( col_norm ) };

		m_v_firsts[ step ] = m_matrix[ step ][ step ] - sign_norm;

		T vTv{ conjugate( m_v_firsts[ step ] ) * m_v_firsts[ step ] };

		for( size_t r{ step + 1 }; r < m_rows; ++r )
			vTv += conjugate( m_matrix[ r ][ step ] ) * m_matrix[ r ][ step ];

		// store additional required by QR decomposition data 
		// ==================================================
		m_betas[ step ] = static_cast< RT >( 2.0 ) / vTv;

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

	if( abs_val( m_matrix[ max_steps ][ max_steps ] ) < singularity_acc )
		throw singularity_error( "dense_matrix< T >::QR_decomposition - a singular matrix was obtained" );

	m_dynamic_state = DYNAMIC_STATE::QR_DECOMPOSED;
}

template< typename T >
void dense_matrix< T >::solve_QR( std::vector< DT >& x, const std::vector< DT >& b ) const
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
		DT vTb{ conjugate( static_cast< DT >( m_v_firsts[ step ] ) ) * x[ step ] };
		for( size_t r{ step + 1 }; r < m_rows; ++r )
			vTb += conjugate( static_cast< DT >( m_matrix[ r ][ step ] ) ) * x[ r ];

		x[ step ] -= static_cast< DT >( m_betas[ step ] ) * static_cast< DT >( m_v_firsts[ step ] ) * vTb;
		for( size_t r{ step + 1 }; r < m_rows; ++r )
			x[ r ] -= static_cast< DT >( m_betas[ step ] ) * static_cast< DT >( m_matrix[ r ][ step ] ) * vTb;
	}

	// then solve Rx = Q^T * b by back substitution
	// ============================================
	for( auto r = static_cast< int >( m_cols ) - 1; r >= 0; --r )
	{
		DT sum{ 0.0 };
		for( int c{ r + 1 }; c < m_cols; ++c )
			sum += static_cast< DT >( m_matrix[ r ][ c ] ) * x[ c ];

		x[ r ] = ( x[ r ] - sum ) / static_cast< DT >( m_matrix[ r ][ r ] );
	}

	for( size_t c{ 0 }; c < m_cols; ++c )
		x[ c ] *= static_cast< DT >( m_scalars[ c ] );
}

template< typename T >
void dense_matrix< T >::count_residual_Ax_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const
{
	if( x.size() != m_cols || b.size() != m_rows || r.size() != m_rows )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_Ax_b - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_Ax_b - m_dynamic_state != DYNAMIC_STATE::INIT" );

	for( size_t row{ 0 }; row < m_rows; ++row )
		r[ row ] = -b[ row ];
	for( size_t row{ 0 }; row < m_rows; ++row )
		for( size_t col{ 0 }; col < m_cols; ++col )
			r[ row ] += ( x[ col ] * static_cast< DT >( m_matrix[ row ][ col ] ) );
}

template< typename T >
void dense_matrix< T >::count_residual_LUx_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const
{
	if( x.size() != m_cols || b.size() != m_rows || r.size() != m_rows )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_LUx_b - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::LU_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_LUx_b - m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED" );

	std::vector< DT > w( m_rows, T{} );

	// compute w=Ux
	// ============
	for( size_t row{ 0 }; row < m_rows; ++row )
		for( size_t col{ row }; col < m_cols; ++col )
			w[ row ] += ( x[ m_p_col[ col ] ] * static_cast< DT >( m_matrix[ m_p_row[ row ] ][ m_p_col[ col ] ] ) );

	// compute r = Lw - b
	// ==================
	for( size_t row{ 0 }; row < m_rows; ++row )
	{
		r[ m_p_row[ row ] ] = w[ row ] - ( b[ m_p_row[ row ] ] * static_cast< DT >( m_scalars[ m_p_row[ row ] ] ) );

		for( size_t col{ 0 }; col < row; ++col )
			r[ m_p_row[ row ] ] += w[ col ] * static_cast< DT >( m_matrix[ m_p_row[ row ] ][ m_p_col[ col ] ] );

		r[ m_p_row[ row ] ] /= static_cast< DT >( m_scalars[ m_p_row[ row ] ] );
	}
}

template< typename T >
void dense_matrix< T >::count_residual_QRx_b( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const
{
	if( x.size() != m_cols || b.size() != m_rows || r.size() != m_rows )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_QRx_b - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix< T >::count_residual_QRx_b - m_dynamic_state != DYNAMIC_STATE::QR_DECOMPOSED" );

	const int max_steps = std::min( m_rows - 1, m_cols );

	for( size_t row{ 0 }; row < m_rows; ++row )
	{
		r[ row ] = DT{};
		for( size_t col{ row }; col < m_cols; ++col )
			r[ row ] += ( x[ col ] * static_cast< DT >( m_matrix[ row ][ col ] )
				/ static_cast< DT >( m_scalars[ col ] ) );
	}

	for( int step{ max_steps - 1 }; step >= 0; --step )
	{
		DT vRx{ conjugate( static_cast< DT >( m_v_firsts[ step ] ) ) * r[ step ] };
		for( int s{ step + 1 }; s < static_cast< int >( m_rows ); ++s )
			vRx += conjugate( static_cast< DT >( m_matrix[ s ][ step ] ) ) * r[ s ];

		r[ step ] -= static_cast< DT >( m_betas[ step ] ) * static_cast< DT >( m_v_firsts[ step ] ) * vRx;
		for( int s{ step + 1 }; s < static_cast< int >( m_rows ); ++s )
			r[ s ] -= static_cast< DT >( m_betas[ step ] ) * static_cast< DT >( m_matrix[ s ][ step ] ) * vRx;
	}

	for( size_t row{ 0 }; row < m_rows; ++row )
		r[ row ] -= b[ row ];
}

template< typename T >
void dense_matrix< T >::count_residual_vector( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const
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
void dense_matrix< T >::iterative_refinement( std::vector< DT >& x, const std::vector< DT >& b, const double acc, const size_t max_it, const dense_matrix< T >* A_orig ) const
{
	if( m_rows != m_cols )
		throw std::exception( "dense_matrix< T >::iterative_refinement - m_rows != m_cols" );

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

template < typename T >
void dense_matrix< T >::rows_scaling()
{
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix< T >::rows_scaling: m_dynamic_state != DYNAMIC_STATE::INIT" );

	double max_scalar{ 0.0 };
	m_scalars.resize( m_rows, 0.0 );

	for( size_t row{ 0 }; row < m_rows; ++row )
	{
		for( size_t col{ 0 }; col < m_cols; ++col )
			m_scalars[ row ] += abs_val( m_matrix[ row ][ col ] );

		max_scalar = std::max( max_scalar, m_scalars[ row ] );
	}

	for( size_t row{ 0 }; row < m_rows; ++row )
	{
		m_scalars[ row ] = ( max_scalar / m_scalars[ row ] );

		for( size_t col{ 0 }; col < m_cols; ++col )
			m_matrix[ row ][ col ] *= static_cast< T >( m_scalars[ row ] );
	}
}

template < typename T >
void dense_matrix< T >::cols_scaling()
{
	if( m_dynamic_state != DYNAMIC_STATE::INIT )
		throw std::invalid_argument( "dense_matrix< T >::cols_scaling: m_dynamic_state != DYNAMIC_STATE::INIT" );

	double max_scalar{ 0.0 };
	m_scalars.resize( m_cols, 0.0 );

	for( size_t col{ 0 }; col < m_cols; ++col )
	{
		for( size_t row{ 0 }; row < m_rows; ++row )
			m_scalars[ col ] += abs_val( m_matrix[ row ][ col ] );

		max_scalar = std::max( max_scalar, m_scalars[ col ] );
	}

	for( size_t col{ 0 }; col < m_cols; ++col )
	{
		m_scalars[ col ] = ( max_scalar / m_scalars[ col ] );

		for( size_t row{ 0 }; row < m_rows; ++row )
			m_matrix[ row ][ col ] *= static_cast< T >( m_scalars[ col ] );
	}
}



// just test function
template< typename U >
std::vector< dense_matrix< U > > get_factors( const dense_matrix< U >& A )
{
	std::vector< dense_matrix< U > > factors;

	switch( A.m_dynamic_state )
	{
	case DYNAMIC_STATE::QHQ_DECOMPOSED:
		dense_matrix< U > H( A.m_rows, A.m_cols ), I( A.m_rows, A.m_cols ), Q( A.m_rows, A.m_cols ), QT( A.m_rows, A.m_cols );

		for( size_t i{ 0 }; i < A.m_rows; ++i )
		{
			I.set_element( U{ 1.0 }, i, i );
			Q.set_element( U{ 1.0 }, i, i );
			QT.set_element( U{ 1.0 }, i, i );
		}

		for( int r{ 0 }; r < static_cast< int >( A.m_rows ); ++r )
			for( int c{ std::max( 0, r - 1 ) }; c < A.m_cols; ++c )
				H.set_element( A.m_matrix[ r ][ c ], r, c );

		factors.push_back( std::move( H ) );

		const auto max_steps = A.m_rows - 2;

		for( size_t step{ 0 }; step < max_steps; ++step )
		{
			const size_t nstep{ step + 1 };
			dense_matrix< U > v( A.m_rows, 1 ), vT( 1, A.m_cols ), Q_k( A.m_rows, A.m_cols );

			v.set_element( A.m_v_firsts[ step ], nstep, 0 );
			for( size_t i{ nstep + 1 }; i < A.m_rows; ++i )
				v.set_element( A.m_matrix[ i ][ step ], i, 0 );

			vT.set_element( conjugate( A.m_v_firsts[ step ] ), 0, nstep );
			for( size_t i{ nstep + 1 }; i < A.m_cols; ++i )
				vT.set_element( conjugate( A.m_matrix[ i ][ step ] ), 0, i );

			Q_k = I - A.m_betas[ step ] * ( v * vT );

			Q = Q * Q_k;
			QT = Q_k * QT;
		}

		factors.push_back( std::move( Q ) );
		factors.push_back( std::move( QT ) );

		break;
	}

	return factors;
}
}
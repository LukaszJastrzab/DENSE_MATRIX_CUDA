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

namespace dmc
{

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
	SCHUR_FORM,
	SCHUR_VECTORS,
	EIGEN_VECTORS
};

template< typename T >
class dense_matrix_cuda
{
	/// each template calss should be friend to each other
	template< typename > friend class dense_matrix_cuda;

public:
	/// constructors
	dense_matrix_cuda() = default;
	dense_matrix_cuda( const dense_matrix_cuda& ) = default;
	dense_matrix_cuda( dense_matrix_cuda&& ) = default;
	dense_matrix_cuda( DYNAMIC_STATE init_state, size_t rows, size_t cols );

	/// destructor
	~dense_matrix_cuda() = default;
	/// assign operator of the same type T
	dense_matrix_cuda& operator=( const dense_matrix_cuda& ) = default;

	/// return amount of rows
	size_t get_rows_amount() const;
	/// return amount of columns
	size_t get_cols_amount() const;

	/// double type used in this template
	using DT = typename double_type< T >::type;
	/// real type used in this template
	using RT = typename real_type< T >::type;
	/// double complex type
	using DC = thrust::complex< double >;

	/// sets matrix sizes and allocates memory
	void init( DYNAMIC_STATE init_state, size_t rows, size_t cols );
	/// adds elements and throws exception if row / col is out of range
	void set_element( T value, size_t row, size_t col );

	/// max norm
	double norm_max() const;
	/// inf norm
	double norm_inf() const;

	/// Function permuts row lying on pos1 position with row lying on pos2 position
	void permute_rows( size_t pos1, size_t pos2 );
	/// Function permuts column lying on pos1 position with column lying on pos2 position
	void permute_cols( size_t pos1, size_t pos2 );

	/// it counts value r := Ax - b
	void count_residual_vector( const std::vector< DT >& x, const std::vector< DT >& b, std::vector< DT >& r ) const;
	/// Method improves the accuracy of the solution
	void iterative_refinement( std::vector< DT >& x, const std::vector< DT >& b, const double acc, const size_t max_it, const dense_matrix_cuda< T >* A_orig = nullptr ) const;

	/// tarnsposition
	void transpose();
	/// conjugates all matrix elements
	void conjugation();
	/// performs hermitan transposition
	void hermitian_transpose();

	/// assign operator for different template types
	template< typename U >
	dense_matrix_cuda< T >& operator=( const dense_matrix_cuda< U >& );
	/// addition operator
	template< typename U, typename V >
	friend dense_matrix_cuda< std::common_type_t< U, V > > operator+( const dense_matrix_cuda< U >& A, const dense_matrix_cuda< V >& B );
	/// subtraction operator
	template< typename U, typename V >
	friend dense_matrix_cuda< std::common_type_t< U, V > > operator-( const dense_matrix_cuda< U >& A, const dense_matrix_cuda< V >& B );
	/// multiplication operators
	template< typename U, typename V >
	friend dense_matrix_cuda< std::common_type_t< U, V > > operator*( const dense_matrix_cuda< U >& A, const dense_matrix_cuda< V >& B );
	/// mult operator that multiply matrix A by vector v
	template< typename U, typename V >
	friend std::vector< std::common_type_t< U, V > > operator*( const dense_matrix_cuda< U >& A, const std::vector< U >& v );
	/// mult operator that multiply vector v by matrix A
	template< typename U, typename V >
	friend std::vector< std::common_type_t< U, V > > operator*( const std::vector< U >& v, const dense_matrix_cuda< U >& A );
	/// mult operator that multiply matrix A by scalar b
	template< typename U, typename V >
	friend dense_matrix_cuda< std::common_type_t< U, V > > operator*( const V& b, const dense_matrix_cuda< U >& A );
	/// mult operator that multiply matrix A by scalar b
	template< typename U, typename V >
	friend dense_matrix_cuda< std::common_type_t< U, V > > operator*( const dense_matrix_cuda< U >& A, const V& b );

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
	void QHQ_decomposition();
	/// computes eqigen values using QR algorithm
	template< typename C1, typename C2 >
	void compute_eigenvalues_QR( std::vector< thrust::complex< C1 > >& l, dense_matrix_cuda* SV, dense_matrix_cuda< thrust::complex< C2 > >* EV, const size_t max_it = 1000, const bool Francis = true, const double acc = std::numeric_limits< RT >::epsilon() );

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
	/// method dumps eigen values during QR algorithm
	template< typename C1 >
	void QR_get_eigenvalues( std::vector< thrust::complex< C1 > >& l, std::map< size_t, size_t >& final_blocks ) const;
	/// method used buble racing in QR algorithm for eigenvalues problem
	bool QHQ_NxN_with_shifts( const size_t row_shift, const size_t col_shift, const size_t block_end, size_t block_size, dense_matrix_cuda* V, std::vector< T > v = {} );
	/// method returns Francis step column (double shift needed for real matrices with complex eigens)
	std::vector< T > get_Francis_v( const size_t shift, const size_t block_end ) const;
	/// method used in QR alogoritm, it gets eigenvalues from 2x2 Schur block
	template< typename C1 >
	void QR_get_eigenvalues_from_block( const size_t shift, std::vector< thrust::complex< C1 > >& l ) const;
	/// method used for creation of Schur vectors during QR algorithm
	void apply_VQ_step( dense_matrix_cuda& SV, const size_t row_shift, const size_t col_shift, const size_t col_len, std::vector< T >* v, T beta, size_t block_end = std::numeric_limits< size_t >::max() ) const;
	/// computes eigen vectors from given SCHUR_FORM matrix, Schur vectors and eigen values ( used in QR algorithm)
	template< typename C1, typename C2 >
	void compute_eigenvectors( dense_matrix_cuda< thrust::complex< C2 > >& EV, const dense_matrix_cuda< T >& SV, const std::vector< thrust::complex< C1 > >& l, const std::map< size_t, size_t >& blocks ) const;

private:
	/// current state of matrix
	DYNAMIC_STATE m_dynamic_state{ DYNAMIC_STATE::NONE };

	/// amount of rows
	size_t m_rows{ 0 };
	/// amount of columns
	size_t m_cols{ 0 };

	/// sing for determinant
	int m_dsign{ 1 };

	/// flattened matrix data
	std::vector< T > m_matrix;
	/// additional data for QR decomposition
	std::vector< T > m_betas, m_v_firsts;
	/// row permutation
	std::vector< size_t > m_p_row, m_p_col;

	/// row / column scaling parameters
	std::vector< double > m_scalars;

};

__host__ __device__ __forceinline__
size_t RLD( size_t row, size_t col, size_t cols )
{
	// row majority
	return col + row * cols;
}

__host__ __device__ __forceinline__
size_t CLD( size_t row, size_t col, size_t rows )
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
	case DYNAMIC_STATE::QR_DECOMPOSED:
	case DYNAMIC_STATE::QHQ_DECOMPOSED:
	case DYNAMIC_STATE::SCHUR_FORM:
	case DYNAMIC_STATE::SCHUR_VECTORS:
	case DYNAMIC_STATE::EIGEN_VECTORS:
		return CLD( row, col, m_rows );

	case DYNAMIC_STATE::ROL_INIT:
		return RLD( row, col, m_cols );

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
size_t dense_matrix_cuda< T >::get_rows_amount() const
{
	return m_rows;
}

template< typename T >
size_t dense_matrix_cuda< T >::get_cols_amount() const
{
	return m_cols;
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
double dense_matrix_cuda< T >::norm_max() const
{
	double result{ 0 };

	for( size_t r{ 0 }; r < m_rows; ++r )
		for( size_t c{ 0 }; c < m_cols; ++c )
			result = std::max( result, abs_val( m_matrix[ calc_elem_idx( r, c ) ] ) );

	return result;
}

template< typename T >
double dense_matrix_cuda< T >::norm_inf() const
{
	double result{ 0 };

	for( size_t r{ 0 }; r < m_rows; ++r )
	{
		double row_norm{ 0 };
		for( size_t c{ 0 }; c < m_cols; ++c )
			row_norm += abs_val( m_matrix[ calc_elem_idx( r, c ) ] );
		result = std::max( row_norm, result );
	}

	return result;
}

template < typename T >
void dense_matrix_cuda< T >::permute_rows( size_t pos1, size_t pos2 )
{
	if( pos1 != pos2 )
		m_dsign = -m_dsign;

	std::swap( m_p_row[ pos1 ], m_p_row[ pos2 ] );
}

template < typename T >
void dense_matrix_cuda< T >::permute_cols( size_t pos1, size_t pos2 )
{
	if( pos1 != pos2 )
		m_dsign = -m_dsign;

	std::swap( m_p_col[ pos1 ], m_p_col[ pos2 ] );
}

template< typename T >
void dense_matrix_cuda< T >::transpose()
{
	std::vector< T > t_matrix( m_cols * m_rows );

	for( size_t r{ 0 }; r < m_rows; ++r )
		for( size_t c{ 0 }; c < m_cols; ++c )
			t_matrix[ calc_elem_idx( c, r ) ] = m_matrix[ calc_elem_idx( r, c ) ];

	std::swap( m_rows, m_cols );
	std::swap( m_p_row, m_p_col );

	m_matrix = std::move( t_matrix );
}

template< typename T >
void dense_matrix_cuda< T >::conjugation()
{
	for( size_t r{ 0 }; r < m_rows; ++r )
		for( size_t c{ 0 }; c < m_cols; ++c )
			m_matrix[ calc_elem_idx( r, c ) ] = conjugate( m_matrix[ calc_elem_idx( r, c ) ] );
}

template< typename T >
void dense_matrix_cuda< T >::hermitian_transpose()
{
	transpose();
	conjugation();
}

template< typename T >
template< typename U >
dense_matrix_cuda< T >& dense_matrix_cuda< T >::operator=( const dense_matrix_cuda< U >& other )
{
	m_rows = other.m_rows;
	m_cols = other.m_cols;
	m_dynamic_state = other.m_dynamic_state;

	m_matrix.resize( m_rows * m_cols );
	for( size_t r{ 0 }; r < m_rows; ++r )
		for( size_t c{ 0 }; c < m_cols; ++c )
			m_matrix[ calc_elem_idx( r, c ) ] = static_cast< T >( other.m_matrix[ calc_elem_idx( r, c ) ] );

	m_betas.resize( other.m_betas.size() );
	for( size_t i{ 0 }; m_betas.size(); ++i )
		m_betas[ i ] = static_cast< T >( other.m_betas[ i ] );

	m_v_firsts.resize( other.m_v_firsts.size() );
	for( size_t i{ 0 }; m_v_firsts.size(); ++i )
		m_v_firsts[ i ] = static_cast< T >( other.m_v_firsts[ i ] );

	m_dsign = other.m_dsign;
	m_p_row = other.m_p_row;
	m_p_col = other.m_p_col;
	m_scalars = other.m_scalars;

	return *this;
}

template< typename U, typename V >
dense_matrix_cuda< std::common_type_t< U, V > > operator+( const dense_matrix_cuda< U >& A, const dense_matrix_cuda< V >& B )
{
	if( A.m_rows != B.m_rows || A.m_cols != B.m_cols )
		throw std::invalid_argument( "dense_matrix_cuda: operator+ - A.m_rows != B.m_rows || A.m_cols != B.m_cols" );

	using R = std::common_type_t< U, V >;

	dense_matrix_cuda< R > result( A.m_dynamic_state, A.m_rows, A.m_cols );

	for( size_t r{ 0 }; r < A.m_rows; ++r )
		for( size_t c{ 0 }; c < A.m_cols; ++c )
			result.set_element( static_cast< R >( A.m_matrix[ A.calc_elem_idx( r, c ) ] ) + static_cast< R >( B.m_matrix[ B.calc_elem_idx( r, c ) ] ), r, c );

	return result;
}

template< typename U, typename V >
dense_matrix_cuda< std::common_type_t< U, V > > operator-( const dense_matrix_cuda< U >& A, const dense_matrix_cuda< V >& B )
{
	if( A.m_rows != B.m_rows || A.m_cols != B.m_cols )
		throw std::invalid_argument( "dense_matrix_cuda: operator- - A.m_rows != B.m_rows || A.m_cols != B.m_cols" );

	using R = std::common_type_t< U, V >;

	dense_matrix_cuda< R > result( A.m_dynamic_state, A.m_rows, A.m_cols );

	for( size_t r{ 0 }; r < A.m_rows; ++r )
		for( size_t c{ 0 }; c < A.m_cols; ++c )
			result.set_element( static_cast< R >( A.m_matrix[ A.calc_elem_idx( r, c ) ] ) - static_cast< R >( B.m_matrix[ B.calc_elem_idx( r, c ) ] ), r, c );

	return result;
}

template< typename U, typename V >
dense_matrix_cuda< std::common_type_t< U, V > > operator*( const dense_matrix_cuda< U >& A, const dense_matrix_cuda< V >& B )
{
	if( A.m_cols != B.m_rows )
		throw std::invalid_argument( "dense_matrix_cuda: operator* - A.m_cols != B.m_rows" );

	using R = std::common_type_t< U, V >;

	dense_matrix_cuda< R > result( A.m_dynamic_state, A.m_rows, B.m_cols );

	for( size_t r{ 0 }; r < A.m_rows; ++r )
		for( size_t c{ 0 }; c < B.m_cols; ++c )
		{
			R mult_sum{};
			for( size_t i{ 0 }; i < A.m_cols; ++i )
				mult_sum += static_cast< R >( A.m_matrix[ A.calc_elem_idx( r, i ) ] ) * static_cast< R >( B.m_matrix[ B.calc_elem_idx( i, c ) ] );
			result.set_element( mult_sum, r, c );
		}

	return result;
}

template< typename U, typename V >
std::vector< std::common_type_t< U, V > > operator*( const dense_matrix_cuda< U >& A, const std::vector< U >& v )
{
	if( v.size != A.m_cols )
		throw std::invalid_argument( "dense_matrix_cuda: operator* - v.size != A.m_cols" );

	using R = std::common_type_t< U, V >;

	std::vector< R > result( A.m_rows, R{} );

	for( size_t r{ 0 }; r < A.m_rows; ++r )
		for( size_t c{ 0 }; c < A.m_cols; ++c )
			result[ r ] += static_cast< R >( A.m_matrix[ A.calc_elem_idx( r, c ) ] ) * static_cast< R >( v[ c ] );

	return result;
}

template< typename U, typename V >
std::vector< std::common_type_t< U, V > > operator*( const std::vector< U >& v, const dense_matrix_cuda< U >& A )
{
	if( v.size != A.m_rows )
		throw std::invalid_argument( "dense_matrix_cuda: operator* - v.size != A.m_rows" );

	using R = std::common_type_t< U, V >;

	std::vector< R > result( A.m_cols, R{} );

	for( size_t c{ 0 }; c < A.m_cols; ++c )
		for( size_t r{ 0 }; r < A.m_rows; ++r )
			result[ c ] += static_cast< R >( v[ r ] ) * static_cast< R >( A.m_matrix[ A.calc_elem_idx( r, c ) ] );

	return result;
}

template< typename U, typename V >
dense_matrix_cuda< std::common_type_t< U, V > > operator*( const V& b, const dense_matrix_cuda< U >& A )
{
	using R = std::common_type_t< U, V >;

	dense_matrix_cuda< R > result( A.m_rows, A.m_cols );

	for( size_t r{ 0 }; r < A.m_rows; ++r )
		for( size_t c{ 0 }; c < A.m_cols; ++c )
			result.m_matrix[ result.calc_elem_idx( r, c ) ] = static_cast< R >( A.m_matrix[ A.calc_elem_idx( r, c ) ] ) * static_cast< R >( b );

	return result;
}

template< typename U, typename V >
dense_matrix_cuda< std::common_type_t< U, V > > operator*( const dense_matrix_cuda< U >& A, const V& b )
{
	return b * A;
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
		const double new_abs{ abs_val( m_matrix[ RLD( m_p_row[ row ], step, m_cols ) ] ) };

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
		const auto pivot{ m_matrix[ RLD( eliminating_row, step, m_cols ) ] };

		for( size_t row{ step + 1 }; row < m_rows; ++row )
		{
			const size_t eliminated_row = m_p_row[ row ];

			const size_t elimintor_idx{ RLD( eliminated_row, step, m_cols ) };

			m_matrix[ elimintor_idx ] /= pivot;
			const auto eliminator = m_matrix[ elimintor_idx ];

			for( size_t col{ step + 1 }; col < std::min( m_cols, step_offset + block_size ); ++col )
				m_matrix[ RLD( eliminated_row, col, m_cols ) ] -= eliminator * m_matrix[ RLD( eliminating_row, col, m_cols ) ];
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
		L[ RLD( threadIdx.y, threadIdx.x, mx_size ) ]	=
			A_in[ RLD( p_row[ row_offset + threadIdx.y ], row_offset + threadIdx.x, A_cols ) ];

	__syncthreads();

	if( col >= A_cols || threadIdx.y > 0 )
		return;

	for( int r{ 0 }; r < mx_size; ++r )
	{
		const size_t in_idx = RLD( p_row[ row_offset + r ], col, A_cols );

		U_i[ r ] = A_in[ in_idx ];

		for( int c{ 0 }; c < r; ++c )
			U_i[ r ] -= L[ RLD( r, c, mx_size ) ] * U_i[ c ];

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
		L[ RLD( threadIdx.y, threadIdx.x, mx_size ) ] = A_in[ RLD( row_orig, row_offset + threadIdx.x, A_cols ) ];
	if ( col < A_cols )
		U[ RLD( threadIdx.y, threadIdx.x, mx_size ) ] = A_in[ RLD( p_row[ row_offset + threadIdx.y ], col, A_cols ) ];

	__syncthreads();

	if( row >= A_rows || col >= A_cols )
		return;

	const size_t in_idx = RLD( row_orig, col, A_cols );

	T res{ A_in[ in_idx ] };

	for( int i{ 0 }; i < blockDim.x; ++i )
		res -= L[ RLD( threadIdx.y, i, mx_size ) ] * U[ RLD( i, threadIdx.x, mx_size ) ];

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
			y->at( row ) -= static_cast< DT >( m_matrix[ RLD( p_row, col, m_cols ) ] ) * y->at( col );
	}

	// second solve the equation Ux = y
	// ================================
	for( int row{ static_cast< int >( m_cols ) - 1 }; row >= 0; --row )
	{
		x[ row ] = y->at( row );

		for( int col{ row + 1 }; col < m_cols; ++col )
			x[ row ] -= static_cast< DT >( m_matrix[ RLD( m_p_row[ row ], col, m_cols ) ] ) * x[ col ];

		x[ row ] /= m_matrix[ RLD( m_p_row[ row ], row, m_cols ) ];
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
			vTb += conjugate( static_cast< DT >( m_matrix[ CLD( r, step, m_rows ) ] ) ) * x[ r ];

		x[ step ] -= static_cast< DT >( m_betas[ step ] * m_v_firsts[ step ] ) * vTb;
		for( size_t r{ step + 1 }; r < m_rows; ++r )
			x[ r ] -= static_cast< DT >( m_betas[ step ] * m_matrix[ CLD( r, step, m_rows ) ] ) * vTb;
	}

	// then solve Rx = Q^T * b by back substitution
	// ============================================
	for( auto r = static_cast< int >( m_cols ) - 1; r >= 0; --r )
	{
		DT sum{ T{} };
		for( int c{ r + 1 }; c < m_cols; ++c )
			sum += static_cast< DT >( m_matrix[ CLD( r, c, m_rows ) ] ) * x[ c ];

		x[ r ] = ( x[ r ] - sum ) / m_matrix[ CLD( r, r, m_rows ) ];
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

	Tmx[ CLD( step, step, block_size ) ] = m_betas[ lstep ];

	if( step > 0 )
	{
		std::vector< T > VTv( step );
		for( size_t s{ step_offset }; s < lstep; ++s )
		{
			auto s_in{ s - step_offset };
			VTv[ s_in ] = conjugate( m_matrix[ CLD( lstep + row_shift, s, m_rows ) ] ) * m_v_firsts[ lstep ];
			for( size_t r{ lstep + row_shift + 1 }; r < m_rows; ++r )
				VTv[ s_in ] += conjugate( m_matrix[ CLD( r, s, m_rows ) ] ) * m_matrix[ CLD( r, lstep, m_rows ) ];
		}

		for( size_t sr{ 0 }; sr < step; ++sr )
		{
			for( size_t sc{ 0 }; sc < step; ++sc )
				Tmx[ CLD( sr, step, block_size ) ] -= Tmx[ CLD( sr, sc, block_size ) ] * VTv[ sc ];

			Tmx[ CLD( sr, step, block_size ) ] *= m_betas[ lstep ];
		}
	}
}

template< typename T >
void dense_matrix_cuda< T >::QHQ_block_decomposition_cpu( const size_t block_size, const size_t step_offset, const size_t max_steps )
{
	size_t block_end{ block_size + step_offset };
	size_t l_max_steps{ std::min( max_steps, block_end ) };
	size_t l_max_col{ std::min( block_end, m_cols ) };

	std::vector< T > Av( m_rows, T{} );
	T vTA{};

	for( size_t step{ step_offset }; step < l_max_steps; ++step )
	{
		double col_norm{ 0.0 };
		const size_t row_step{ step + 1 };

		// calcualte norm
		// ==============
		for( size_t r{ row_step }; r < m_rows; ++r )
		{
			double abs_v = abs_val( m_matrix[ CLD( r, step, m_rows ) ] );
			col_norm += abs_v * abs_v;
		}
		col_norm = std::sqrt( col_norm );

		// stabilization sign calculation
		// ==============================
		auto lead_elem_idx{ CLD( row_step, step, m_rows ) };
		double alpha_abs = abs_val( m_matrix[ lead_elem_idx ] );
		T sign = ( alpha_abs != 0.0 ? -( m_matrix[ lead_elem_idx ] ) / T{ static_cast< RT >( alpha_abs ) } : T{ -1 } );
		T sign_norm = sign * T{ static_cast< RT >( col_norm ) };

		m_v_firsts[ step ] = m_matrix[ lead_elem_idx ] - sign_norm;
		const auto v1{ m_v_firsts[ step ] };
		const auto v1T{ conjugate( v1 ) };

		T vTv{ v1T * v1 };
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
		{
			const auto elem_idx{ CLD( r, step, m_rows ) };
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
			Av[ r ] = m_matrix[ CLD( r, row_step, m_rows ) ] * v1;
			for( size_t c{ row_step + 1 }; c < m_cols; ++c )
				Av[ r ] += m_matrix[ CLD( r, c, m_rows ) ] * m_matrix[ CLD( c, step, m_rows ) ];
		}

		// calculate vTA ( v*A in case of complex )
		// ========================================
		vTA = v1T * m_matrix[ CLD( row_step, row_step, m_rows ) ];
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
			vTA += conjugate( m_matrix[ CLD( r, step, m_rows ) ] ) * m_matrix[ CLD( r, row_step, m_rows ) ];

		// alpha = v*Av
		// ============
		T alpha{ v1T * Av[ row_step ] };
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
			alpha += conjugate( m_matrix[ CLD( r, step, m_rows ) ] ) * Av[ r ];

		// update those part of matrix that are changed only by right mult by QT
		// =====================================================================
		for( size_t r{ 0 }; r < row_step; ++r )
			m_matrix[ CLD( r, row_step, m_rows ) ] -= beta * Av[ r ] * v1T;

		// update left-upper corner of submatrix
		// =====================================
		m_matrix[ CLD( row_step, row_step, m_rows ) ] -=
				beta * ( v1 * vTA + Av[ row_step ] * v1T - beta * alpha * v1 * v1T );

		// update fiest modificated sub col
		// ================================
		for( size_t r{ row_step + 1 }; r < m_rows; ++r )
		{
			const auto v1{ m_matrix[ CLD( r, step, m_rows ) ] };
			m_matrix[ CLD( r, row_step, m_rows ) ] -= beta * ( v1 * ( vTA - beta * alpha * v1T ) + Av[ r ] * v1T );
		}
	}
}

template< typename T >
__global__
void QR_decomposition_blocked_AVT_gpu( T* AVT,
	const T* TVTA,
	const T* A_in,
	const T* v_firsts,
	const T* V,
	const int A_rows,
	const int A_cols,
	const int block_size,
	const int col_offset,
	const int row_shift )
{
	const int col = threadIdx.x + col_offset;
	const int row = threadIdx.y + blockDim.y * blockIdx.y;

	T sum{};

	bool active = !( threadIdx.x >= block_size || row >= A_rows );

	if( active )
	{
		int col_idx{ col + row_shift };
		sum = A_in[ CLD( row, col_idx++, A_rows ) ] * v_firsts[ col ];

		for( ; col_idx < A_cols; ++col_idx )
			sum += A_in[ CLD( row, col_idx, A_rows ) ] *  V[ CLD( col_idx, threadIdx.x, A_rows ) ];

		AVT[ CLD( row, threadIdx.x, A_rows ) ] = sum;
	}

	__syncthreads();

	if( active )
	{
		sum = AVT[ CLD( row, 0, A_rows ) ] * TVTA[ CLD( 0, threadIdx.x, block_size ) ];
		for( int c{ 1 }; c <= threadIdx.x; ++c )
			sum += AVT[ CLD( row, c, A_rows ) ] * TVTA[ CLD( c, threadIdx.x, block_size ) ];
	}

	__syncthreads();

	if( active )
		AVT[ CLD( row, threadIdx.x, A_rows ) ] = sum;
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
	const int col_offset,
	const int row_shift )
{
	const int col = col_offset + threadIdx.x + blockDim.x * blockIdx.x;
	const int col_b = row_offset + threadIdx.x;
	const int row = row_offset + threadIdx.y + row_shift;
	const int row_b = row - row_shift;
	const int b_size_sq{ block_size * block_size };
	T sum{};

	extern __shared__ unsigned char sdata_raw[];
	T* Tmx = reinterpret_cast< T* >( sdata_raw );
	T* Vblock = Tmx + b_size_sq;
	T* Ablock = Vblock + b_size_sq;
	T* VTA    = Ablock + b_size_sq;

	const size_t sharedIdxXY{ CLD( threadIdx.x, threadIdx.y, block_size ) };
	const size_t sharedIdxYX{ CLD( threadIdx.y, threadIdx.x, block_size ) };	

	Tmx[ sharedIdxXY ] = TVTA[ sharedIdxXY ];

	bool active = !( col >= A_cols || row_b >= row_offset + block_size );

	if( threadIdx.x < threadIdx.y && row < A_rows )
		Vblock[ sharedIdxYX ] = conjugate( A_in[ CLD( row, col_b, A_rows ) ] );
	else if( threadIdx.x == threadIdx.y )
		Vblock[ sharedIdxYX ] = conjugate( v_firsts[ row_b ] );

	if( active && row < A_rows )
		Ablock[ sharedIdxYX ] = A_in[ CLD( row, col, A_rows ) ];

	__syncthreads();

	if( active )
		for( unsigned int r{ threadIdx.y }; r < block_size; ++r )
			sum += Vblock[ CLD( r, threadIdx.y, block_size ) ] * Ablock[ CLD( r, threadIdx.x, block_size ) ];

	int block_offset{ col_offset + row_shift };
	while( block_offset < A_rows )
	{
		__syncthreads();

		int t_row{ block_offset + ( int )threadIdx.y };

		if( t_row < A_rows )
			Vblock[ sharedIdxYX ] = conjugate( A_in[ CLD( t_row, col_b, A_rows ) ] );
		else
			Vblock[ sharedIdxYX ] = T{};

		if( active && t_row < A_rows )
			Ablock[ sharedIdxYX ] = A_in[ CLD( t_row, col, A_rows ) ];
		else
			Ablock[ sharedIdxYX ] = T{};

		__syncthreads();

		for( int r{ 0 }; r < block_size; ++r )
			sum += Vblock[ CLD( r, threadIdx.y, block_size ) ] * Ablock[ CLD( r, threadIdx.x, block_size ) ];

		block_offset += block_size;
	}

	VTA[ sharedIdxYX ] = sum;

	__syncthreads();

	if( active )
	{
		sum = T{};
		for( int r{ 0 }; r <= threadIdx.y; ++r )
			sum += conjugate( Tmx[ CLD( r, threadIdx.y, block_size ) ] ) * VTA[ CLD( r, threadIdx.x, block_size ) ];

		TVTA[ CLD( threadIdx.y, col, block_size ) ] = sum;
	}
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
	const int col_offset,
	const int row_shift )
{
	const int t_row = threadIdx.y + blockDim.y * blockIdx.y;
	const int col = col_offset + threadIdx.x + blockDim.x * blockIdx.x;
	const int row = row_offset + t_row + row_shift;

	if( col >= A_cols || row >= A_rows )
		return;

	int sum_range = ( block_size < t_row + 1 ? block_size : t_row + 1 );

	T sum{};

	for( int c{ 0 }; c < sum_range; ++c )
	{
		const int c_i = row_offset + c;

		const T v_i = ( c == t_row ? v_firsts[ c_i ] : A_out[ CLD( row, c_i, A_rows ) ] );
		sum += v_i * TVTA[ CLD( c, col, block_size ) ];
	}

	A_out[ CLD( row, col, A_rows ) ] -= sum;
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
	const int col_offset,
	const int row_shift )
{
	const int t_row = threadIdx.y + blockDim.y * blockIdx.y;
	const int col = col_offset + threadIdx.x + blockDim.x * blockIdx.x;
	const int row = row_offset + t_row + row_shift;

	extern __shared__ unsigned char sdata_raw[];
	T* Vblock = reinterpret_cast< T* >( sdata_raw );
	T* TVTAblock = Vblock + ( block_size * block_size );

	const size_t sharedIdx{ CLD( threadIdx.y, threadIdx.x, block_size ) };

	if( row < A_rows )
	{
		if( threadIdx.y < t_row || threadIdx.y > threadIdx.x )
			Vblock[ sharedIdx ] = A_out[ CLD( row, threadIdx.x + row_offset, A_rows ) ];
		else if( threadIdx.x == threadIdx.y )
			Vblock[ sharedIdx ] = v_firsts[ threadIdx.y + row_offset ];
		else
			Vblock[ sharedIdx ] = T{};
	}
	else
		Vblock[ sharedIdx ] = T{};

	if( col < A_cols )
		TVTAblock[ sharedIdx ] = TVTA[ CLD( threadIdx.y, col, block_size ) ];
	else
		TVTAblock[ sharedIdx ] = T{};

	__syncthreads();

	if( col >= A_cols || row >= A_rows )
		return;

	T sum{};
	for( int i{ 0 }; i < block_size; ++i )
		sum += Vblock[ CLD( threadIdx.y, i , block_size ) ] * TVTAblock[ CLD( i, threadIdx.x, block_size ) ];

	A_out[ CLD( row, col, A_rows ) ] -= sum;
}

template< typename T >
__global__
void QR_decomposition_blocked_AVTVT_gpu( const T* AVT,
	T* A_out,
	const T* v_firsts,
	const int A_rows,
	const int A_cols,
	const int block_size,
	const int row_offset,
	const int col_offset )
{
	const int col = col_offset + threadIdx.x + blockDim.x * blockIdx.x;
	const int row = threadIdx.y + blockDim.y * blockIdx.y;

	if( col >= A_cols || row >= A_rows )
		return;

	T sum{};

	for( int c{ 0 }; c < block_size; ++c )
	{
		const T v_i = conjugate( col == col_offset && c == block_size - 1 ?
				  v_firsts[ row_offset + c ] : A_out[ CLD( col, c + row_offset, A_rows ) ] );

		sum += AVT[ CLD( row, c, A_rows ) ] * v_i;
	}

	A_out[ CLD( row, col, A_rows ) ] -= sum;
}

template< typename T >
void dense_matrix_cuda< T >::QHQ_decomposition()
{
	if( m_dynamic_state != DYNAMIC_STATE::COL_INIT )
		throw std::invalid_argument( "dense_matrix_cuda< T >::QR_decomposition() - m_dynamic_state != DYNAMIC_STATE::COL_INIT" );
	if( m_rows != m_cols )
		throw std::invalid_argument( "dense_matrix_cuda< T >::QHQ_decomposition() - m_rows != m_cols" );

	// ======================================================================================================
	// here QHQ decomposition using blocked version can work only for blocks sizes 1 or 2
	// this is becouse higher size involves modifications whole matrix by cpu part
	// which does not make a sense
	// luckily this algorithm still works correctly for block size = 2 which give us some acceleration anyway
	// ======================================================================================================
	const size_t block_size{ 2 };

	const auto max_steps{ m_rows - 2 };
	size_t step_offset{ 0 }, row_offset{ 0 };

	m_betas.resize( max_steps );
	m_v_firsts.resize( max_steps );

	T* d_matrix{ nullptr }, * d_v_firsts{ nullptr }, * d_TVTA{ nullptr }, * d_AVT{ nullptr }, *d_V{ nullptr };

	cudaMalloc( &d_matrix, m_matrix.size() * sizeof( T ) );
	cudaMemcpy( d_matrix, m_matrix.data(), m_matrix.size() * sizeof( T ), cudaMemcpyHostToDevice );

	cudaMalloc( &d_v_firsts, max_steps * sizeof( T ) );
	cudaMalloc( &d_TVTA, block_size * m_cols * sizeof( T ) );
	cudaMalloc( &d_AVT, block_size * m_rows * sizeof( T ) );
	cudaMalloc( &d_V, block_size * m_rows * sizeof( T ) );

	std::vector< T > Tmx( block_size * block_size, T{} );

	auto b_size = std::min( block_size, max_steps );

	while( step_offset < max_steps )
	{
		const dim3 blockDim( b_size, b_size );

		QHQ_block_decomposition_cpu( b_size, step_offset, max_steps );

		size_t rows_to_copy{ m_rows - row_offset };
		size_t cols_to_copy{ std::min( b_size, m_cols - step_offset ) };
		size_t src_cpy_offset{ row_offset + step_offset * m_rows };

		cudaMemcpy2D(
			d_V + row_offset,                  // dst
			m_rows * sizeof( T ),              // dst pitch
			m_matrix.data() + src_cpy_offset,  // src
			m_rows * sizeof( T ),              // src pitch
			rows_to_copy * sizeof( T ),        // width (bytes)
			cols_to_copy,                      // height (cols)
			cudaMemcpyHostToDevice
		);

		size_t v_data_size = std::min( b_size, max_steps - step_offset );
		cudaMemcpy( d_v_firsts + step_offset, m_v_firsts.data() + step_offset, v_data_size * sizeof( T ), cudaMemcpyHostToDevice );

		memset( Tmx.data(), 0, b_size * b_size * sizeof( T ) );
		for( size_t s{ 0 }; s < b_size; ++s )
			create_QR_triangular_factor_T( Tmx.data(), b_size, s, step_offset, 1 );

		cudaMemcpy2D(
			d_TVTA,                     // dst
			b_size * sizeof( T ),       // dst pitch
			Tmx.data(),                 // src
			b_size * sizeof( T ),       // src pitch
			b_size * sizeof( T ),       // width (bytes)
			b_size,                     // height (cols)
			cudaMemcpyHostToDevice
		);

		{
			dim3 gridDim( 1, div_up( m_rows, b_size ) );
			QR_decomposition_blocked_AVT_gpu<<< gridDim, blockDim >>>( d_AVT, d_TVTA, d_matrix, d_v_firsts, d_V, m_rows, m_cols, b_size, step_offset, 1 );
		}

		cudaMemcpy2D(
			d_matrix + src_cpy_offset,         // dst
			m_rows * sizeof( T ),              // dst pitch
			m_matrix.data() + src_cpy_offset,  // src
			m_rows * sizeof( T ),              // src pitch
			rows_to_copy * sizeof( T ),        // width (bytes)
			cols_to_copy,                      // height (cols)
			cudaMemcpyHostToDevice
		);

		step_offset += b_size;

		{
			dim3 gridDim( div_up( m_cols - step_offset, b_size ), div_up( m_rows, b_size ) );
			QR_decomposition_blocked_AVTVT_gpu <<< gridDim, blockDim >>> ( d_AVT, d_matrix, d_v_firsts, m_rows, m_cols, b_size, row_offset, step_offset );
		}

		{
			dim3 gridDim( div_up( m_cols - step_offset, b_size ), 1 );
			size_t lmem_size{ 4 * b_size * b_size * sizeof( T ) };
			QR_decomposition_blocked_TVTA_gpu <<< gridDim, blockDim, lmem_size >>> ( d_TVTA, d_matrix, d_v_firsts, m_rows, m_cols, b_size, row_offset, step_offset, 1 );
		}

		{
			dim3 gridDim( div_up( m_cols - step_offset, b_size ), div_up( m_rows - row_offset, b_size ) );
			size_t lmem_size{ 2 * b_size * b_size * sizeof( T ) };
			QR_decomposition_blocked_VTVTA_gpu_new <<< gridDim, blockDim, lmem_size >>> ( d_TVTA, d_matrix, d_v_firsts, m_rows, m_cols, b_size, row_offset, step_offset, 1 );
		}

		b_size = std::min( block_size, max_steps - step_offset );
		cols_to_copy = m_cols - step_offset;

		cudaMemcpy2D(
			m_matrix.data() + step_offset * m_rows,  // dst
			m_rows * sizeof( T ),                    // dst pitch
			d_matrix + step_offset * m_rows,         // src
			m_rows * sizeof( T ),                    // src pitch
			m_rows * sizeof( T ),                    // width (bytes)
			cols_to_copy,                            // height (cols)
			cudaMemcpyDeviceToHost
		);

		row_offset = step_offset;
	}

	cudaFree( d_matrix );
	cudaFree( d_v_firsts );
	cudaFree( d_TVTA );
	cudaFree( d_AVT );
	cudaFree( d_V );

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
			double abs_v = abs_val( m_matrix[ CLD( r, step, m_rows ) ] );
			col_norm += abs_v * abs_v;
		}
		col_norm = std::sqrt( col_norm );

		// stabilization sign calculation
		// ==============================
		const size_t step_idx = CLD( step, step, m_rows );

		double alpha_abs = abs_val( m_matrix[ step_idx ] );
		T sign = ( alpha_abs != 0.0 ? -( m_matrix[ step_idx ] ) / alpha_abs : T{ -1 } );
		T sign_norm = sign * T{ static_cast< RT >( col_norm ) };

		m_v_firsts[ step ] = m_matrix[ step_idx ] - sign_norm;

		T vTv{ conjugate( m_v_firsts[ step ] ) * m_v_firsts[ step ] };

		for( size_t r{ step + 1 }; r < m_rows; ++r )
			vTv += conjugate( m_matrix[ CLD( r, step, m_rows ) ] ) * m_matrix[ CLD( r, step, m_rows ) ];

		m_betas[ step ] = 2.0 / vTv;

		m_matrix[ step_idx ] = sign_norm;

		// calculate vTA ( v*A in case of complex )
		// ========================================
		for( size_t c{ step + 1 }; c < l_max_col; ++c )
		{
			const size_t c_in{ c - step_offset };

			vTA[ c_in ] = conjugate( m_v_firsts[ step ] ) * m_matrix[ CLD( step, c, m_rows ) ];
			for( size_t r{ step + 1 }; r < m_rows; ++r )
				vTA[ c_in ] += conjugate( m_matrix[ CLD( r, step, m_rows ) ] ) * m_matrix[ CLD( r, c, m_rows ) ];
		}

		// calculate (I-bvvT)A = A - b(v(vTA)) only for first block_size columns
		// =====================================================================
		for( size_t c{ step + 1 }; c < l_max_col; ++c )
			m_matrix[ CLD( step, c, m_rows ) ] -= m_betas[ step ] * m_v_firsts[ step ] * vTA[ c - step_offset ];

		for( size_t r{ step + 1 }; r < m_rows; ++r )
			for( size_t c{ step + 1 }; c < l_max_col; ++c )
				m_matrix[ CLD( r, c, m_rows ) ] -= m_betas[ step ] * m_matrix[ CLD( r, step, m_rows ) ] * vTA[ c - step_offset ];
	}
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
		const auto b_size{ std::min( block_size, max_steps - step_offset ) };

		QR_block_decomposition_cpu( b_size, step_offset, max_steps );

		size_t rows_to_copy{ m_rows - row_offset };
		size_t cols_to_copy{ std::min( b_size, m_cols - step_offset ) };
		size_t cpy_offset{ row_offset + step_offset * m_rows };

		cudaMemcpy2D(
			d_matrix + cpy_offset,         // dst
			m_rows * sizeof( T ),          // dst pitch
			m_matrix.data() + cpy_offset,  // src
			m_rows * sizeof( T ),          // src pitch
			rows_to_copy * sizeof( T ),    // width (bytes)
			cols_to_copy,                  // height (cols)
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

		{
			dim3 gridDim( div_up( m_cols - step_offset, b_size ), 1 );
			size_t lmem_size{ 4 * b_size * b_size * sizeof( T ) };
			QR_decomposition_blocked_TVTA_gpu << < gridDim, blockDim, lmem_size >> > ( d_TVTA, d_matrix, d_v_firsts, m_rows, m_cols, b_size, row_offset, step_offset, 0 );
		}
		{
			dim3 gridDim( div_up( m_cols - step_offset, b_size ), div_up( m_rows - row_offset, b_size ) );
			size_t lmem_size{ 2 * b_size * b_size * sizeof( T ) };
			QR_decomposition_blocked_VTVTA_gpu_new <<< gridDim, blockDim, lmem_size >>> ( d_TVTA, d_matrix, d_v_firsts, m_rows, m_cols, b_size, row_offset, step_offset, 0 );
		}

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
		throw std::invalid_argument( "dense_matrix_cuda< T >::count_residual_Ax_b - x.size() != m_cols || b.size() != m_rows || r.size() != m_rows" );
	if( m_dynamic_state != DYNAMIC_STATE::ROL_INIT && m_dynamic_state != DYNAMIC_STATE::COL_INIT )
		throw std::invalid_argument( "dense_matrix_cuda< T >::count_residual_Ax_b - m_dynamic_state != DYNAMIC_STATE::INIT" );

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
			w[ row ] += ( x[ col ] * static_cast< DT >( m_matrix[ RLD( m_p_row[ row ], col, m_cols ) ] ) );


	// compute r = Lw - b
	// ==================
	for( size_t row{ 0 }; row < m_rows; ++row )
	{
		r[ m_p_row[ row ] ] = w[ row ] - ( b[ m_p_row[ row ] ] * static_cast< DT >( m_scalars[ m_p_row[ row ] ] ) );

		for( size_t col{ 0 }; col < row; ++col )
			r[ m_p_row[ row ] ] += w[ col ] * static_cast< DT >( m_matrix[ RLD( m_p_row[ row ], col, m_cols ) ] );

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
			r[ row ] += ( x[ col ] * static_cast< DT >( m_matrix[ CLD( row, col, m_rows ) ] )
				/ static_cast< DT >( m_scalars[ col ] ) );
	}

	for( int step{ max_steps - 1 }; step >= 0; --step )
	{
		DT vRx{ conjugate( static_cast< DT >( m_v_firsts[ step ] ) ) * r[ step ] };
		for( int s{ step + 1 }; s < static_cast< int >( m_rows ); ++s )
			vRx += conjugate( static_cast< DT >( m_matrix[ CLD( s, step, m_rows ) ] ) ) * r[ s ];

		r[ step ] -= static_cast< DT >( m_betas[ step ] * m_v_firsts[ step ] ) * vRx;
		for( int s{ step + 1 }; s < static_cast< int >( m_rows ); ++s )
			r[ s ] -= static_cast< DT >( m_betas[ step ] * m_matrix[ CLD( s, step, m_rows ) ] ) * vRx;
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
			throw std::invalid_argument( "dense_matrix_cuda< T >::iterative_refinement - dynamic state not supported" );
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

template< typename T >
template< typename C1 >
void dense_matrix_cuda< T >::QR_get_eigenvalues_from_block( const size_t shift, std::vector< thrust::complex< C1 > >& l ) const
{
	using CT = thrust::complex< C1 >;

	CT a{ m_matrix[ CLD( shift, shift, m_rows ) ] }, b{ m_matrix[ CLD( shift, shift + 1, m_rows ) ] },		
		c{ m_matrix[ CLD( shift + 1, shift, m_rows ) ] }, d{ m_matrix[ CLD( shift + 1, shift + 1, m_rows ) ] };

	CT tr{ a + d };
	CT det{ a * d - b * c };
	CT disc{ thrust::sqrt( tr * tr - static_cast< CT >( 4.0 ) * det ) };

	l[ shift ] = ( tr + disc ) / CT( 2.0 );
	l[ shift + 1 ] = ( tr - disc ) / CT( 2.0 );
}


template< typename T >
template < typename C1 >
void dense_matrix_cuda< T >::QR_get_eigenvalues( std::vector< thrust::complex< C1 > >& l, std::map< size_t, size_t >& final_blocks ) const
{
	using CT = thrust::complex< C1 >;

	for( const auto [block_begin, block_end] : final_blocks )
	{
		switch( block_end - block_begin )
		{
		case 2:
			QR_get_eigenvalues_from_block( block_begin, l );
			break;

		case 1:
			l[ block_begin ] = static_cast< CT >( m_matrix[ CLD( block_begin, block_begin, m_rows ) ] );			
			break;

		default:
			throw std::runtime_error( "dense_matrix_cuda< T >::QR_get_eigenvalues - invalid block data" );
		}
	}
}


template< typename T >
bool dense_matrix_cuda< T >::QHQ_NxN_with_shifts( const size_t row_shift, const size_t col_shift, const size_t block_end, const size_t block_size, dense_matrix_cuda* V, std::vector< T > v )
{
	const auto row_nshift{ row_shift + 1 };
	const auto col_nshift{ col_shift + 1 };

	bool use_spec_v{ v.size() > 0 };

	if( !use_spec_v )
	{
		v.resize( block_size );
		for( size_t i{ row_shift }; i < min( m_rows, row_shift + block_size ); ++i )
			v[ i - row_shift ] = m_matrix[ CLD( i, col_shift, m_rows ) ];
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
		m_matrix[ CLD( row_shift, col_shift, m_rows ) ] = sign_norm;		
		for( size_t i{ row_shift + 1 }; i < min( m_rows, row_shift + block_size ); ++i )
			m_matrix[ CLD( i, col_shift, m_rows ) ] = T{};		
	}

	if( abs_val( vTv ) < std::numeric_limits< RT >::epsilon() )
		return false;

	const auto beta{ static_cast< RT >( 2.0 ) / vTv };

	if( V != nullptr )
		apply_VQ_step( *V, row_shift, col_shift, v.size(), &v, beta, block_end );

	// A' <- A - beta * v * ( vT * A )
	// ===============================
	std::vector< T > vTA( m_cols, T{} );

	for( size_t c{ use_spec_v ? col_shift : col_nshift }; c < m_cols; ++c )
		for( size_t i{ row_shift }; i < min( m_rows, row_shift + block_size ); ++i )
			vTA[ c ] += vT[ i - row_shift ] * m_matrix[ CLD( i, c, m_rows ) ];

	for( size_t c{ use_spec_v ? col_shift : col_nshift }; c < m_cols; ++c )
		for( size_t i{ row_shift }; i < min( m_rows, row_shift + block_size ); ++i )
			m_matrix[ CLD( i, c, m_rows ) ] -= beta * v[ i - row_shift ] * vTA[ c ];

	// A <- A' - beta * ( A' v ) vT
	// ============================
	std::vector< T > Av( block_end, T{} );

	for( size_t r{ 0 }; r < std::min( row_nshift + block_size, block_end ); ++r )
		for( size_t i{ row_shift }; i < min( m_cols, row_shift + block_size ); ++i )
			Av[ r ] += m_matrix[ CLD( r, i, m_rows ) ] * v[ i - row_shift ];

	for( size_t r{ 0 }; r < std::min( row_nshift + block_size, block_end ); ++r )
		for( size_t i{ row_shift }; i < min( m_cols, row_shift + block_size ); ++i )
			m_matrix[ CLD( r, i, m_rows ) ] -= beta * Av[ r ] * vT[ i - row_shift ];

	return true;
}

template< typename T >
std::vector< T > dense_matrix_cuda< T >::get_Francis_v( const size_t shift, const size_t block_end ) const
{
	const T a{ m_matrix[ CLD( block_end - 1, block_end - 1, m_rows ) ] },
		b{ m_matrix[ CLD( block_end - 1, block_end, m_rows ) ] },
		c{ m_matrix[ CLD( block_end, block_end - 1, m_rows ) ] },
		d{ m_matrix[ CLD( block_end, block_end, m_rows ) ] };
	const T tr{ a + d }, det{ a * d - b * c };
	const T a11{ m_matrix[ CLD( shift, shift, m_rows ) ] }, a12{ m_matrix[ CLD( shift, shift + 1, m_rows ) ] },
		a21{ m_matrix[ CLD( shift + 1, shift, m_rows ) ] }, a22{ m_matrix[ CLD( shift + 1, shift + 1, m_rows ) ] };
	T a32{};

	if( shift < m_rows - 2 )
		a32 = m_matrix[ CLD( shift + 2, shift + 1, m_rows ) ];

	std::vector< T > v{ a11 * a11 + a21 * a12 - tr * a11 + det,
						a11 * a21 + a21 * a22 - tr * a21,
									a21 * a32 };

	return v;
}

template< typename T >
void dense_matrix_cuda< T >::apply_VQ_step( dense_matrix_cuda& SV, const size_t row_shift, const size_t col_shift, size_t col_len, std::vector< T >* v, T beta, size_t block_end ) const
{
	if( m_rows != SV.m_rows || m_cols != SV.m_cols )
		throw std::invalid_argument( "dense_matrix_cuda< T >::apply_VQ_step - m_rows != SV.m_rows || m_cols != SV.m_cols" );

	std::vector< T > Vv( m_rows, T{} );

	std::vector< T > reflector;
	if( v == nullptr )
	{
		beta = m_betas[ col_shift ];
		reflector.resize( col_len );
		reflector[ 0 ] = m_v_firsts[ col_shift ];
		for( size_t i{ 1 }; i < col_len; ++i )
			reflector[ i ] = m_matrix[ CLD( row_shift + i, col_shift, m_rows ) ];		
		v = &reflector;
	}

	const auto col_end{ std::min( v->size() + row_shift, m_cols ) };

	// calculate Vv
	//=============
	for( size_t r{ 0 }; r < m_rows; ++r )
		for( size_t c{ row_shift }; c < col_end; ++c )
			Vv[ r ] += SV.m_matrix[ CLD( r, c, m_rows ) ] * ( *v )[ c - row_shift ];

	// update V matrix
	// ===============
	for( size_t r{ 0 }; r < m_rows; ++r )
		for( size_t c{ row_shift }; c < col_end; ++c )
			SV.m_matrix[ CLD( r, c, m_rows ) ] -= beta * Vv[ r ] * conjugate( ( *v )[ c - row_shift ] );

}

template< typename T >
template< typename C1, typename C2 >
void dense_matrix_cuda< T >::compute_eigenvectors( dense_matrix_cuda< thrust::complex< C2 > >& EV, const dense_matrix_cuda< T >& SV, const std::vector< thrust::complex< C1 > >& l, const std::map< size_t, size_t >& blocks ) const
{
	using CT2 = thrust::complex< C2 >;

	EV.init( DYNAMIC_STATE::COL_INIT, m_rows, m_cols );

	const CT2 val{ 1 };

	for( size_t i{ 0 }; i < l.size(); ++i )
	{
		const auto lambda{ l[ i ] };
		auto blockIt = blocks.find( i );
		if( blockIt == blocks.end() )
			blockIt = blocks.find( i - 1 );
		if( blockIt == blocks.end() )
			throw std::runtime_error( "dense_matrix_cuda< T >::compute_eigenvectors - wrong block data" );

		const size_t i0{ blockIt->first }, i01{ blockIt->first + 1 }, i1{ blockIt->second };
		const size_t lead_block_size{ i1 - i0 };

		switch( lead_block_size )
		{
		case 1:
			EV.m_matrix[ CLD( i, i0, m_rows ) ] = val;
			break;
		case 2:
		{
			const auto mi0{ static_cast< CT2 >( m_matrix[ CLD( i0, i0, m_rows ) ] ) - static_cast< CT2 >( lambda ) };
			const auto mi1{ static_cast< CT2 >( m_matrix[ CLD( i01, i01, m_rows ) ] ) - static_cast< CT2 >( lambda ) };
			EV.m_matrix[ CLD( i0, i, m_rows ) ] = -val * ( abs_val( mi0 ) > abs_val( m_matrix[ CLD( i01, i0, m_rows ) ] ) ?
				static_cast< CT2 >( m_matrix[ CLD( i0, i01, m_rows ) ] ) / mi0 :
				mi1 / static_cast< CT2 >( m_matrix[ CLD( i01, i0, m_rows ) ] ) );
			EV.m_matrix[ CLD( i01, i, m_rows ) ] = val;
			break;
		}
		default:
			throw std::runtime_error( "dense_matrix_cuda< T >::compute_eigenvectors - wrong block data" );
		}

		auto rit = std::make_reverse_iterator( blockIt );

		while( rit != blocks.rend() )
		{
			const size_t j0{ rit->first }, j01{ rit->first + 1 }, j1{ rit->second };
			const size_t this_block_size{ j1 - j0 };

			switch( this_block_size )
			{
			case 1:
			{
				if( abs_val( lambda - l[ j0 ] ) <= std::numeric_limits< C1 >::epsilon() )
					EV.m_matrix[ CLD( j0, i, m_rows ) ] = val;
				else
				{
					CT2 b{};
					for( size_t j{ j1 }; j < i1; ++j )
						b -= static_cast< CT2 >( m_matrix[ CLD( j0, j, m_rows ) ] ) * EV.m_matrix[ CLD( j, i, m_rows ) ];
					EV.m_matrix[ CLD( j0, i, m_rows ) ] = b / ( static_cast< CT2 >( m_matrix[ CLD( j0, j0, m_rows ) ] ) - static_cast< CT2 >( lambda ) );
				}
				break;
			}
			case 2:
			{
				if( abs_val( lambda - l[ j0 ] ) <= std::numeric_limits< C1 >::epsilon() ||
					abs_val( lambda - l[ j01 ] ) <= std::numeric_limits< C1 >::epsilon() )
				{
					const auto mj0{ static_cast< CT2 >( m_matrix[ CLD( j0, j0, m_rows ) ] ) - static_cast< CT2 >( lambda ) };
					const auto mj1{ static_cast< CT2 >( m_matrix[ CLD( j01, j01, m_rows ) ] ) - static_cast< CT2 >( lambda ) };
					EV.m_matrix[ CLD( j0, i, m_rows ) ] = -val * ( abs_val( mj0 ) > abs_val( m_matrix[ CLD( j01, j0, m_rows ) ] ) ?
						static_cast< CT2 >( m_matrix[ CLD( j0, j01, m_rows ) ] ) / mj0 :
						mj1 / static_cast< CT2 >( m_matrix[ CLD( j01, j0, m_rows ) ] ) );
					EV.m_matrix[ CLD( j01, i, m_rows ) ] = val;
					break;
				}
				else
				{
					std::vector< DC > b( this_block_size, DC{} );
					for( size_t r{ 0 }; r < this_block_size; ++r )
					{
						const size_t row{ j0 + r };
						for( size_t j{ j1 }; j < i1; ++j )
							b[ r ] -= static_cast< DC >( m_matrix[ CLD( row, j, m_rows ) ] ) * static_cast< DC >( EV.m_matrix[ CLD( j, i, m_rows ) ] );

						dense_matrix_cuda< DC > M2x2( DYNAMIC_STATE::ROL_INIT, this_block_size, this_block_size );
						M2x2.m_matrix[ RLD( 0, 0, this_block_size ) ] = static_cast< DC >( m_matrix[ CLD( j0, j0, m_rows ) ] ) - static_cast< DC >( lambda );
						M2x2.m_matrix[ RLD( 0, 1, this_block_size ) ] = static_cast< DC >( m_matrix[ CLD( j0, j01, m_rows ) ] );
						M2x2.m_matrix[ RLD( 1, 0, this_block_size ) ] = static_cast< DC >( m_matrix[ CLD( j01, j0, m_rows ) ] );
						M2x2.m_matrix[ RLD( 1, 1, this_block_size ) ] = static_cast< DC >( m_matrix[ CLD( j01, j01, m_rows ) ] ) - static_cast< DC >( lambda );

						std::vector< DC > x( this_block_size, DC{} );
						M2x2.LU_decomposition( true, this_block_size );
						M2x2.solve_LU( x, b );

						EV.m_matrix[ CLD( j0, i, m_rows ) ] = static_cast< CT2 >( x[ 0 ] );
						EV.m_matrix[ CLD( j01, i, m_rows ) ] = static_cast< CT2 >( x[ 1 ] );
					}
				}

				break;
			}
			default:
				throw std::runtime_error( "dense_matrix_cuda< T >::compute_eigenvectors - wrong block data" );
			}

			rit++;
		}
	}

	EV = SV * EV;
}

template< typename T >
template< typename C1, typename C2 >
void dense_matrix_cuda< T >::compute_eigenvalues_QR( std::vector< thrust::complex< C1 > >& l, dense_matrix_cuda* SV, dense_matrix_cuda< thrust::complex< C2 > >* EV, const size_t max_it, const bool Francis, const double acc )
{
	if( m_rows != m_cols )
		throw std::invalid_argument( "dense_matrix_cuda< T >::compute_eigenvalues_QR - m_rows != m_cols" );
	if( m_dynamic_state == DYNAMIC_STATE::COL_INIT )
		QHQ_decomposition();
	if( m_dynamic_state != DYNAMIC_STATE::QHQ_DECOMPOSED )
		throw std::invalid_argument( "dense_matrix_cuda< T >::compute_eigenvalues_QR - m_dynamic_state != DYNAMIC_STATE::QHQ_DECOMPOSED" );

	using CT1 = thrust::complex< C1 >;

	auto block_size = Francis ? 3ull : 2ull;
	l.resize( m_rows, CT1{} );
	dense_matrix_cuda< T > SV_alloc;

	// if only eigenvalues are needed then allocate Schur's only temporary ( as they are needed wnyway )
	// =================================================================================================
	if( EV != nullptr && SV == nullptr )
		SV = &SV_alloc;

	// for handling schur/eigen vectors
	// ================================
	if( SV != nullptr )
	{
		SV->init( DYNAMIC_STATE::COL_INIT, m_rows, m_cols );
		for( size_t i{ 0 }; i < m_rows; ++i )
			SV->set_element( T{ 1 }, i, i );

		for( size_t step{ 0 }; step < m_rows - 2; ++step )
			apply_VQ_step( *SV, step + 1, step, m_rows - step - 1, nullptr, T{} );
	}

	// vanish elements under lower diag of Hessenberg form
	// ===================================================
	for( size_t r{ 2 }; r < m_rows; ++r )
		for( size_t c{ 0 }; c < r - 1; ++c )
			m_matrix[ CLD( r, c, m_rows ) ] = T{};

	std::map< size_t, size_t > blocks, final_blocks;
	std::map< size_t, double > b2x2_deflacc;

	blocks[ 0 ] = m_rows;

	for( size_t iter{ 0 }; iter < max_it; ++iter )
	{
		// deflection
		// ==========
		for( auto& [block_begin, block_end] : blocks )
		{
			for( auto i{ block_begin }; i < block_end - 1; ++i )
			{
				auto ii{ i + 1 };

				if( abs_val( m_matrix[ CLD( ii, i, m_rows ) ] )
					<= acc * ( abs_val( m_matrix[ CLD( i, i, m_rows ) ] ) +
							   abs_val( m_matrix[ CLD( ii, ii, m_rows ) ] ) ) )
				{
					m_matrix[ CLD( ii, i, m_rows ) ] = T{};
					blocks[ ii ] = block_end;
					block_end = ii;
					break;
				}
			}
		}

		// remove blocks which sizes are == 1 or == 2 and cointains complex eigens
		// =======================================================================
		for( auto it = blocks.begin(); it != blocks.end(); )
		{
			const auto i0{ it->first }, i01{ it->first + 1 }, i1{ it->second };
			const auto b_size{ i1 - i0 };
			bool remove2x2{ false };

			if( b_size == 2 )
			{
				auto ndefacc{ abs_val( m_matrix[ CLD( i01, i0, m_rows ) ] ) };				
				auto defacc = b2x2_deflacc.find( i0 );

				if( defacc == b2x2_deflacc.end() )
					b2x2_deflacc[ i0 ] = ndefacc;
				else if( ndefacc >= defacc->second )
				{
					remove2x2 = true;
					b2x2_deflacc.erase( defacc );
				}
			}

			if( b_size == 1 || remove2x2 )
			{
				final_blocks[ i0 ] = i1;
				it = blocks.erase( it );
			}
			else
				++it;
		}

		if( blocks.size() == 0 )
			break;

		for( auto& [block_begin, block_end] : blocks )
		{
			const auto proc_block_size{ std::min( block_end - block_begin, block_size ) };

			// Rayleigh shifting
			// =================
			const T mu{ m_matrix[ CLD( block_end - 1, block_end - 1, m_rows ) ] };			
			for( auto i{ block_begin }; i < block_end; ++i )
				m_matrix[ CLD( i, i, m_rows ) ] -= mu;

			// QR Hessenberg reduction
			// =======================
			if( QHQ_NxN_with_shifts( block_begin, block_begin, block_end, proc_block_size, SV,
				( proc_block_size > 2 && Francis ) ? get_Francis_v( block_begin, block_end - 1 ) : std::vector< T >() ) )
			{
				for( auto i{ block_begin }; i < block_end - 1; ++i )
					QHQ_NxN_with_shifts( i + 1, i, block_end, proc_block_size, SV );
			}

			// Rayleigh shifting back
			// ======================
			for( auto i{ block_begin }; i < block_end; ++i )
				m_matrix[ CLD( i, i, m_rows ) ] += mu;
		}
	}

	for( const auto& it : blocks )
		final_blocks[ it.first ] = it.second;

	QR_get_eigenvalues( l, final_blocks );

	if( SV != nullptr )
		SV->m_dynamic_state = DYNAMIC_STATE::SCHUR_VECTORS;

	if( SV != nullptr && EV != nullptr )
		compute_eigenvectors( *EV, *SV, l, final_blocks );

	m_dynamic_state = DYNAMIC_STATE::SCHUR_FORM;
}

}
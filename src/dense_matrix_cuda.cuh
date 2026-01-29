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
	dense_matrix_cuda( const dense_matrix_cuda& ) = default;
	dense_matrix_cuda( dense_matrix_cuda&& ) = default;
	dense_matrix_cuda( size_t rows, size_t cols );

	/// sets matrix sizes and allocates memory
	void init( size_t rows, size_t cols );
	/// initializes CUDA data
	void init_cuda();
	/// adds elements and throws exception if row / col is out of range
	void set_element( T value, size_t row, size_t col );

private:

	/// calculates index to flattened matrix element
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
	std::vector< T > m_pivots;
	/// for QR decomposition
	std::vector< T > m_betas;
	std::vector< T > m_v_firsts;
};

template< typename T >
struct is_complex : std::false_type {};

template< typename T >
struct is_complex< thrust::complex< T > > : std::true_type {};

template< typename T >
struct scalar_type
{
	using type = T;
};

template< typename T >
struct scalar_type< thrust::complex< T > >
{
	using type = T;
};

template< typename T >
using scalar_t = typename scalar_type< T >::type;

template< typename T >
size_t dense_matrix_cuda< T >::calc_elem_idx( size_t row, size_t col )
{
	return col + row * m_cols;
}

template< typename T >
dense_matrix_cuda< T >::dense_matrix_cuda( size_t rows, size_t cols )
{
	init( rows, cols );
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
	auto elem_idx = calc_elem_idx( row, col );

	if( elem_idx >= m_matrix.size() )
		throw std::out_of_range( "dense_matrix_cuda< T >::set_element - elem_idx >= m_matrix.size()" );

	m_matrix[ elem_idx ] = value;
}

template< typename T >
void dense_matrix_cuda< T >::init_cuda()
{
	if constexpr( is_complex< T >::value )
	{
		auto size_of = sizeof( scalar_t< T > );
		size_of = size_of;
	}
	else
	{
		auto size_of = sizeof( T );
		size_of = size_of;
	}
}
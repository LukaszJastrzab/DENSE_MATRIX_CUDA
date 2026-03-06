#include <gtest/gtest.h>

#include <dense_matrix_cuda.cuh>
#include <functions.cuh>


// test
#include <dense_matrix.hpp>
// test

using namespace std;
constexpr double eps_float = 3e-3;
constexpr double eps_double = 1e-10;
constexpr double min_double = 0.0001;
constexpr double max_double = 10000.0;
constexpr double min_float = 0.01;
constexpr double max_float = 100.0;



TEST( test_test_test, LU_test )
{
	double val_min{ min_float }, val_max{ max_float };// , eps{ eps_float };

	dense_matrix< float > Af;
	dense_matrix_cuda< float > A;
	size_t mx_size{ 7 };

	A.init( DYNAMIC_STATE::ROL_INIT, mx_size, mx_size );
	Af.init( mx_size, mx_size );

	vector< float > b( mx_size );
	vector< float > r( mx_size );
	vector< float > x( mx_size ), xx( mx_size );

	for( size_t row{ 0 }; row < mx_size; ++row )
	{
		b[ row ] = generate_random< float >( val_min, val_max );

		for( size_t col{ 0 }; col < mx_size; ++col )
		{
			auto val = generate_random< float >( val_min, val_max );
			A.set_element( val, row, col );
			Af.set_element( val, row, col );
		}	
	}

	auto A_ = A;

	Af.LU_decomposition( 1 );

	A.LU_decomposition( 4 );
	A.solve_LU( x, b );

	A_.count_residual_vector( x, b, r );

	//EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps );
}


enum class SOLVING_TYPE : uint8_t
{
	QR_decomposition,
	LU_decomposition
};

template < typename T >
void QR_decomposition_block_test( const SOLVING_TYPE solving_type, size_t max_block_size = 32 )
{
	double val_min{ min_float }, val_max{ max_float }, eps{ eps_float };

#ifndef __CUDA_ARCH__
	if constexpr( std::is_same< typename real_type < T >::type, double >::value )
#endif
	{
		val_min = min_double;
		val_max = max_double;
		eps = eps_double;
	}


	for( size_t block_size{ max_block_size }; block_size > 0; block_size >>= 1 )
	{
		for( size_t mx_size = 500; mx_size > 1; mx_size -= 100 )
		{
			dense_matrix_cuda< T > A;

			switch( solving_type )
			{
			case SOLVING_TYPE::QR_decomposition:
				A.init( DYNAMIC_STATE::COL_INIT, mx_size, mx_size );
				break;

			case SOLVING_TYPE::LU_decomposition:
				A.init( DYNAMIC_STATE::ROL_INIT, mx_size, mx_size );
				break;
			}

			vector< T > b( mx_size );
			vector< T > r( mx_size );
			vector< T > x( mx_size );

			for( size_t row{ 0 }; row < mx_size; ++row )
			{
				b[ row ] = generate_random< T >( val_min, val_max );

				for( size_t col{ 0 }; col < mx_size; ++col )
					A.set_element( generate_random< T >( val_min, val_max ), row, col );
			}

			auto A_ = A;

			switch( solving_type )
			{
			case SOLVING_TYPE::QR_decomposition:
				A.QR_decomposition( block_size );
				A.solve_QR( x, b );
				break;

			case SOLVING_TYPE::LU_decomposition:
				A.LU_decomposition( block_size );
				A.solve_LU( x, b );
				break;
			}

			A_.count_residual_vector( x, b, r );

			EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps );
		}
	}
}


TEST( non_singular_linear_equation_real_float, QR_decomposition_blocked_Householder )
{
	QR_decomposition_block_test< float >( SOLVING_TYPE::QR_decomposition );
}

TEST( non_singular_linear_equation_real_double, QR_decomposition_blocked_Householder )
{
	QR_decomposition_block_test< double >( SOLVING_TYPE::QR_decomposition );
}

TEST( non_singular_linear_equation_complex_float, QR_decomposition_blocked_Householder )
{
	QR_decomposition_block_test< thrust::complex< float > >( SOLVING_TYPE::QR_decomposition );
}

TEST( non_singular_linear_equation_complex_double, QR_decomposition_blocked_Householder )
{
	QR_decomposition_block_test< thrust::complex< double > >( SOLVING_TYPE::QR_decomposition, 16 );
}


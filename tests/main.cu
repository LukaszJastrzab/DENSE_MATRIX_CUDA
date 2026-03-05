#include <gtest/gtest.h>

#include <dense_matrix_cuda.cuh>
#include <functions.cuh>


using namespace std;
constexpr double eps_float = 3e-3;
constexpr double eps_double = 1e-10;
constexpr double min_double = 0.0001;
constexpr double max_double = 10000.0;
constexpr double min_float = 0.01;
constexpr double max_float = 100.0;

template < typename T >
void QR_decomposition_block_test( size_t max_block_size = 32 )
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
			dense_matrix_cuda< T > A( mx_size, mx_size );
			vector< T > b( mx_size );
			vector< T > r( mx_size );
			vector< T > x( mx_size ), xx( mx_size );

			for( size_t row{ 0 }; row < mx_size; ++row )
			{
				b[ row ] = generate_random< T >( val_min, val_max );

				for( size_t col{ 0 }; col < mx_size; ++col )
				{
					auto val = generate_random< T >( val_min, val_max );
					A.set_element( val, row, col );
				}
			}

			auto A_ = A;

			A.QR_decomposition( block_size );

			A.solve_QR( x, b );
			A_.count_residual_vector( x, b, r );

			EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps );
		}
	}
}


TEST( non_singular_linear_equation_real_float, QR_decomposition_blocked_Householder )
{
	QR_decomposition_block_test< float >();
}

TEST( non_singular_linear_equation_real_double, QR_decomposition_blocked_Householder )
{
	QR_decomposition_block_test< double >();
}

TEST( non_singular_linear_equation_complex_float, QR_decomposition_blocked_Householder )
{
	QR_decomposition_block_test< thrust::complex< float > >();
}

TEST( non_singular_linear_equation_complex_double, QR_decomposition_blocked_Householder )
{
	QR_decomposition_block_test< thrust::complex< double > >( 16 );
}


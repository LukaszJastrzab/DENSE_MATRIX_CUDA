#include <gtest/gtest.h>

#include <dense_matrix_cuda.cuh>
#include <functions.cuh>

constexpr size_t MATRIX_ROW_SIZE = 300;
constexpr size_t MATRIX_COL_SIZE{ MATRIX_ROW_SIZE };

using namespace std;

TEST( non_singular_linear_equation_real, QR_decomposition_Householder )
{
	dense_matrix_cuda< double > A( MATRIX_ROW_SIZE, MATRIX_COL_SIZE );
	vector< double > b( MATRIX_ROW_SIZE );
	vector< double > r( MATRIX_ROW_SIZE );
	vector< double > x( MATRIX_COL_SIZE );

	for( size_t row{ 0 }; row < MATRIX_ROW_SIZE; ++row )
	{
		b[ row ] = generate_random< double >( 0.0001, 10000.0 );

		for( size_t col{ 0 }; col < MATRIX_COL_SIZE; ++col )
		{
			auto val = generate_random< double >( 0.0001, 10000.0 );
			A.set_element( val, row, col );
		}
	}

	auto A_ = A;

	A.QR_decomposition();
	A.solve_QR( x, b );
	A_.count_residual_vector( x, b, r );

	EXPECT_TRUE( l2_norm( r ) <= 0.00001 );
}

TEST( non_singular_linear_equation_complex, QR_decomposition_Householder )
{
	dense_matrix_cuda< thrust::complex< double > > A( MATRIX_ROW_SIZE, MATRIX_COL_SIZE );
	vector< thrust::complex< double > > b( MATRIX_ROW_SIZE );
	vector< thrust::complex< double > > r( MATRIX_ROW_SIZE );
	vector< thrust::complex< double > > x( MATRIX_COL_SIZE );

	for( size_t row{ 0 }; row < MATRIX_ROW_SIZE; ++row )
	{
		b[ row ] = ( generate_random< double >( 0.0001, 10000.0 ), generate_random< double >( 0.0001, 10000.0 ) );

		for( size_t col{ 0 }; col < MATRIX_COL_SIZE; ++col )
		{
			double real = generate_random< double >( 0.0001, 10000.0 );
			double imag = generate_random< double >( 0.0001, 10000.0 );
			A.set_element( thrust::complex< double >( real, imag), row, col );
		}
	}

	auto A_ = A;

	A.QR_decomposition();
	A.solve_QR( x, b );
	A_.count_residual_vector( x, b, r );

	EXPECT_TRUE( l2_norm( r ) <= 0.00001 );
}

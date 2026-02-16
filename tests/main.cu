#include <gtest/gtest.h>

#include <dense_matrix_cuda.cuh>
#include <functions.cuh>


using namespace std;
constexpr double eps_float = 1e-3;
constexpr double eps_double = 1e-10;

TEST( non_singular_linear_equation_real_float, QR_decomposition_Householder )
{
	const size_t MATRIX_ROW_SIZE = 500;
	const size_t MATRIX_COL_SIZE{ MATRIX_ROW_SIZE };

	dense_matrix_cuda< float > A( MATRIX_ROW_SIZE, MATRIX_COL_SIZE );
	vector< float > b( MATRIX_ROW_SIZE );
	vector< float > r( MATRIX_ROW_SIZE );
	vector< float > x( MATRIX_COL_SIZE );

	for( size_t row{ 0 }; row < MATRIX_ROW_SIZE; ++row )
	{
		b[ row ] = generate_random< float >( 0.01, 100.0 );

		for( size_t col{ 0 }; col < MATRIX_COL_SIZE; ++col )
		{
			auto val = generate_random< float >( 0.01, 100.0 );
			A.set_element( val, row, col );
		}
	}

	auto A_ = A;

	A.QR_decomposition();

	// cpu QR solver
	// =============
	A.solve_QR( x, b );
	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );

	// gpu blocked QR solver
	// =====================
	A.solve_QR_blocked( x, b, 32 );
	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );
}


TEST( non_singular_linear_equation_real_double, QR_decomposition_Householder )
{
	const size_t MATRIX_ROW_SIZE = 500;
	const size_t MATRIX_COL_SIZE{ MATRIX_ROW_SIZE };

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

	// cpu QR solver
	// =============
	A.solve_QR( x, b );
	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );

	// gpu blocked QR solver
	// =====================
	A.solve_QR_blocked( x, b, 32 );
	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );
}

TEST( non_singular_linear_equation_complex_float, QR_decomposition_Householder )
{
	const size_t MATRIX_ROW_SIZE = 500;
	const size_t MATRIX_COL_SIZE{ MATRIX_ROW_SIZE };

	dense_matrix_cuda< thrust::complex< float > > A( MATRIX_ROW_SIZE, MATRIX_COL_SIZE );
	vector< thrust::complex< float > > b( MATRIX_ROW_SIZE );
	vector< thrust::complex< float > > r( MATRIX_ROW_SIZE );
	vector< thrust::complex< float > > x( MATRIX_COL_SIZE );

	for( size_t row{ 0 }; row < MATRIX_ROW_SIZE; ++row )
	{
		b[ row ] = ( generate_random< float >( 0.01, 100.0 ), generate_random< float >( 0.01, 100.0 ) );

		for( size_t col{ 0 }; col < MATRIX_COL_SIZE; ++col )
		{
			double real = generate_random< float >( 0.01, 100.0 );
			double imag = generate_random< float >( 0.01, 100.0 );
			A.set_element( thrust::complex< float >( real, imag ), row, col );
		}
	}

	auto A_ = A;

	A.QR_decomposition();

	// cpu QR solver
	// =============
	A.solve_QR( x, b );
	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );

	// gpu blocked QR solver
	// =====================
	A.solve_QR_blocked( x, b, 32 );
	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );
}

TEST( non_singular_linear_equation_complex_double, QR_decomposition_Householder )
{
	const size_t MATRIX_ROW_SIZE = 500;
	const size_t MATRIX_COL_SIZE{ MATRIX_ROW_SIZE };

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
			A.set_element( thrust::complex< double >( real, imag ), row, col );
		}
	}

	auto A_ = A;

	A.QR_decomposition();

	// cpu QR solver
	// =============
	A.solve_QR( x, b );
	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );

	// gpu blocked QR solver
	// =====================
	A.solve_QR_blocked( x, b, 32 );
	A_.count_residual_vector( x, b, r );
	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );
}

TEST( non_singular_linear_equation_real_float, QR_decomposition_blocked_Householder )
{
	for( size_t block_size{ 32 }; block_size > 0; block_size >>= 1 )
	{
		for( size_t mx_size = 500; mx_size > 1; mx_size -= 100 )
		{
			dense_matrix_cuda< float > A( mx_size, mx_size );
			vector< float > b( mx_size );
			vector< float > r( mx_size );
			vector< float > x( mx_size ), xx( mx_size );

			for( size_t row{ 0 }; row < mx_size; ++row )
			{
				b[ row ] = generate_random< float >( 0.01, 100.0 );

				for( size_t col{ 0 }; col < mx_size; ++col )
				{
					auto val = generate_random< float >( 0.01, 100.0 );
					A.set_element( val, row, col );
				}
			}

			auto A_ = A;

			A.QR_decomposition_blocked( block_size );

			// cpu QR solver
			// =============
			A.solve_QR( x, b );
			A_.count_residual_vector( x, b, r );
			EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );

			// gpu blocked QR solver
			// =====================
			A.solve_QR_blocked( xx, b, block_size );
			A_.count_residual_vector( xx, b, r );
			EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );

		}
	}
}

TEST( non_singular_linear_equation_real_double, QR_decomposition_blocked_Householder )
{
	for( size_t block_size{ 32 }; block_size > 0; block_size >>= 1 )
	{
		for( size_t mx_size = 500; mx_size > 1; mx_size -= 100 )
		{
			dense_matrix_cuda< double > A( mx_size, mx_size );
			vector< double > b( mx_size );
			vector< double > r( mx_size );
			vector< double > x( mx_size ), xx( mx_size );

			for( size_t row{ 0 }; row < mx_size; ++row )
			{
				b[ row ] = generate_random< double >( 0.0001, 10000.0 );

				for( size_t col{ 0 }; col < mx_size; ++col )
				{
					auto val = generate_random< double >( 0.0001, 10000.0 );
					A.set_element( val, row, col );
				}
			}

			auto A_ = A;

			A.QR_decomposition_blocked( block_size );

			// cpu QR solver
			// =============
			A.solve_QR( x, b );
			A_.count_residual_vector( x, b, r );
			EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );

			// gpu blocked QR solver
			// =====================
			A.solve_QR_blocked( xx, b, block_size );
			A_.count_residual_vector( xx, b, r );
			EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );

		}
	}
}

TEST( non_singular_linear_equation_complex_float, QR_decomposition_blocked_Householder )
{
	// for complex block 32 is to big 
	// ==============================
	for( size_t block_size{ 16 }; block_size > 0; block_size >>= 1 )
	{
		for( size_t mx_size = 500; mx_size > 1; mx_size -= 100 )
		{
			dense_matrix_cuda< thrust::complex< float > > A( mx_size, mx_size );
			vector< thrust::complex< float > > b( mx_size );
			vector< thrust::complex< float > > r( mx_size );
			vector< thrust::complex< float > > x( mx_size ), xx( mx_size );

			for( size_t row{ 0 }; row < mx_size; ++row )
			{
				b[ row ] = ( generate_random< float >( 0.01, 100.0 ), generate_random< float >( 0.01, 100.0 ) );

				for( size_t col{ 0 }; col < mx_size; ++col )
				{
					double real = generate_random< float >( 0.01, 100.0 );
					double imag = generate_random< float >( 0.01, 100.0 );
					A.set_element( thrust::complex< float >( real, imag ), row, col );
				}
			}

			auto A_ = A;

			A.QR_decomposition_blocked( block_size );

			// cpu QR solver
			// =============
			A.solve_QR( x, b );
			A_.count_residual_vector( x, b, r );
			EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );

			// gpu blocked QR solver
			// =====================
			A.solve_QR_blocked( xx, b, block_size );
			A_.count_residual_vector( xx, b, r );
			EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_float );
		}
	}
}

TEST( non_singular_linear_equation_complex_double, QR_decomposition_blocked_Householder )
{
	// for complex block 32 is to big 
	// ==============================
	for( size_t block_size{ 16 }; block_size > 0; block_size >>= 1 )
	{
		for( size_t mx_size = 500; mx_size > 1; mx_size -= 100 )
		{
			dense_matrix_cuda< thrust::complex< double > > A( mx_size, mx_size );
			vector< thrust::complex< double > > b( mx_size );
			vector< thrust::complex< double > > r( mx_size );
			vector< thrust::complex< double > > x( mx_size ), xx( mx_size );

			for( size_t row{ 0 }; row < mx_size; ++row )
			{
				b[ row ] = ( generate_random< double >( 0.0001, 10000.0 ), generate_random< double >( 0.0001, 10000.0 ) );

				for( size_t col{ 0 }; col < mx_size; ++col )
				{
					double real = generate_random< double >( 0.0001, 10000.0 );
					double imag = generate_random< double >( 0.0001, 10000.0 );
					A.set_element( thrust::complex< double >( real, imag ), row, col );
				}
			}

			auto A_ = A;

			A.QR_decomposition_blocked( block_size );

			// cpu QR solver
			// =============
			A.solve_QR( x, b );
			A_.count_residual_vector( x, b, r );
			EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );

			// gpu blocked QR solver
			// =====================
			A.solve_QR_blocked( xx, b, block_size );
			A_.count_residual_vector( xx, b, r );
			EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps_double );
		}
	}
}

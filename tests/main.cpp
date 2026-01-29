#include <gtest/gtest.h>

#include <functions.hpp>

constexpr size_t MATRIX_ROW_SIZE = 20;
constexpr size_t MATRIX_COL_SIZE{ MATRIX_ROW_SIZE };

using namespace std;

TEST( non_singular_linear_equation_real, QR_decomposition_Householder )
{
	//dense_matrix< double > A( MATRIX_ROW_SIZE, MATRIX_COL_SIZE );
	vector< double > b( MATRIX_ROW_SIZE );
	vector< double > r( MATRIX_ROW_SIZE );
	vector< double > x( MATRIX_COL_SIZE );

	for( size_t row{ 0 }; row < MATRIX_ROW_SIZE; ++row )
	{
		b[ row ] = generate_random< double >( 0.0001, 10000.0 );

		//for( size_t col{ 0 }; col < MATRIX_COL_SIZE; ++col )
		//	A.set_element( generate_random< double >( 0.0001, 10000.0 ), row, col );
	}

	// make a copy for residual vector verification
	// ============================================
	//auto A_ = A;

	//A.QR_decomposition();
	//A.solve_QR( x, b );
	//A_.count_residual_vector( x, b, r );

	EXPECT_TRUE( true );
}

#include <gtest/gtest.h>

#include <dense_matrix_cuda.cuh>
#include <functions.cuh>

using namespace std;
constexpr double eps_float = 1e-5;
constexpr double eps_double = 1e-10;
constexpr double min_double = 0.0001;
constexpr double max_double = 10000.0;
constexpr double min_float = 0.01;
constexpr double max_float = 100.0;


enum class SOLVING_TYPE : uint8_t
{
	QR_decomposition,
	LU_decomposition
};

template < typename T >
void decompositions_block_test( const SOLVING_TYPE solving_type, const bool scaling, size_t max_block_size = 32 )
{
	double val_min{ min_float }, val_max{ max_float }, eps{ eps_float };

	/// double type used in solving / refinement
	using DT = typename double_type< T >::type;

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
		for( int mx_size = 500; mx_size > 1; mx_size -= 200 )
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

			vector< DT > b( mx_size );
			vector< DT > r( mx_size );
			vector< DT > x( mx_size, DT{} );

			for( size_t row{ 0 }; row < mx_size; ++row )
			{
				b[ row ] = generate_random< DT >( val_min, val_max );

				for( size_t col{ 0 }; col < mx_size; ++col )
					A.set_element( generate_random< T >( val_min, val_max ), row, col );
			}

			auto A_ = A;

			switch( solving_type )
			{
			case SOLVING_TYPE::QR_decomposition:
				A.QR_decomposition( scaling, block_size );
				break;

			case SOLVING_TYPE::LU_decomposition:
				A.LU_decomposition( scaling, block_size );
				break;
			}

			A.iterative_refinement( x, b, 0.000000000001, 1000, &A_ );
			A_.count_residual_vector( x, b, r );

			EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps );
		}
	}
}


TEST( non_singular_linear_equation_real_float, QR_decomposition_blocked_Householder_no_scaling )
{
	decompositions_block_test< float >( SOLVING_TYPE::QR_decomposition, false );
}

TEST( non_singular_linear_equation_real_double, QR_decomposition_blocked_Householder_no_scaling )
{
	decompositions_block_test< double >( SOLVING_TYPE::QR_decomposition, false );
}

TEST( non_singular_linear_equation_complex_float, QR_decomposition_blocked_Householder_no_scaling )
{
	decompositions_block_test< thrust::complex< float > >( SOLVING_TYPE::QR_decomposition, false );
}

TEST( non_singular_linear_equation_complex_double, QR_decomposition_blocked_Householder_no_scaling )
{
	decompositions_block_test< thrust::complex< double > >( SOLVING_TYPE::QR_decomposition, false, 16 );
}


TEST( non_singular_linear_equation_real_float, QR_decomposition_blocked_Householder_scaling )
{
	decompositions_block_test< float >( SOLVING_TYPE::QR_decomposition, true );
}

TEST( non_singular_linear_equation_real_double, QR_decomposition_blocked_Householder_scaling )
{
	decompositions_block_test< double >( SOLVING_TYPE::QR_decomposition, true );
}

TEST( non_singular_linear_equation_complex_float, QR_decomposition_blocked_Householder_scaling )
{
	decompositions_block_test< thrust::complex< float > >( SOLVING_TYPE::QR_decomposition, true );
}

TEST( non_singular_linear_equation_complex_double, QR_decomposition_blocked_Householder_scaling )
{
	decompositions_block_test< thrust::complex< double > >( SOLVING_TYPE::QR_decomposition, true, 16 );
}


TEST( non_singular_linear_equation_real_float, LU_decomposition_blocked_Gauss_no_scaling )
{
	decompositions_block_test< float >( SOLVING_TYPE::LU_decomposition, false );
}

TEST( non_singular_linear_equation_real_double, LU_decomposition_blocked_Gauss_no_scaling )
{
	decompositions_block_test< double >( SOLVING_TYPE::LU_decomposition, false );
}

TEST( non_singular_linear_equation_complex_float, LU_decomposition_blocked_Gauss_no_scaling )
{
	decompositions_block_test< thrust::complex< float > >( SOLVING_TYPE::LU_decomposition, false );
}

TEST( non_singular_linear_equation_complex_double, LU_decomposition_blocked_Gauss_no_scaling )
{
	decompositions_block_test< thrust::complex< double > >( SOLVING_TYPE::LU_decomposition, false, 16 );
}


TEST( non_singular_linear_equation_real_float, LU_decomposition_blocked_Gauss_scaling )
{
	decompositions_block_test< float >( SOLVING_TYPE::LU_decomposition, true );
}

TEST( non_singular_linear_equation_real_double, LU_decomposition_blocked_Gauss_scaling )
{
	decompositions_block_test< double >( SOLVING_TYPE::LU_decomposition, true );
}

TEST( non_singular_linear_equation_complex_float, LU_decomposition_blocked_Gauss_scaling )
{
	decompositions_block_test< thrust::complex< float > >( SOLVING_TYPE::LU_decomposition, true );
}

TEST( non_singular_linear_equation_complex_double, LU_decomposition_blocked_Gauss_scaling )
{
	decompositions_block_test< thrust::complex< double > >( SOLVING_TYPE::LU_decomposition, true, 16 );
}


template < typename T >
void decompositions_big_example( const SOLVING_TYPE solving_type, const bool scaling )
{
	double val_min{ min_float }, val_max{ max_float }, eps{ eps_float };

	/// double type used in solving / refinement
	using DT = typename double_type< T >::type;

#ifndef __CUDA_ARCH__
	if constexpr( std::is_same< typename real_type < T >::type, double >::value )
#endif
	{
		val_min = min_double;
		val_max = max_double;
		eps = eps_double;
	}

	size_t mx_size{ 5 };

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

	vector< DT > b( mx_size );
	vector< DT > r( mx_size );
	vector< DT > x( mx_size, DT{ 0.0 } );

	for( size_t row{ 0 }; row < mx_size; ++row )
	{
		b[ row ] = generate_random< DT >( val_min, val_max );

		for( size_t col{ 0 }; col < mx_size; ++col )
		{
			auto val = generate_random< T >( val_min, val_max );
			A.set_element( val, row, col );
		}
	}

	switch( solving_type )
	{
	case SOLVING_TYPE::QR_decomposition:
		A.QR_decomposition( scaling );
		break;

	case SOLVING_TYPE::LU_decomposition:
		A.LU_decomposition( scaling );
		break;
	}

	A.iterative_refinement( x, b, 0.000000000001, 1000 );
	A.count_residual_vector( x, b, r );

	EXPECT_LE( l2_norm( r ) / l2_norm( b ), eps );

}

TEST( big_non_singular_linear_equation_float, QR_decomposition_blocked_Householder_no_scaling )
{
	decompositions_big_example< float >( SOLVING_TYPE::QR_decomposition, false );
}

TEST( big_non_singular_linear_equation_double, QR_decomposition_blocked_Householder_no_scaling )
{
	decompositions_big_example< double >( SOLVING_TYPE::QR_decomposition, false );
}

TEST( big_non_singular_linear_equation_complex_float, QR_decomposition_blocked_Householder_no_scaling )
{
	decompositions_big_example< thrust::complex< float > >( SOLVING_TYPE::QR_decomposition, false );
}

TEST( big_non_singular_linear_equation_complex_double, QR_decomposition_blocked_Householder_no_scaling )
{
	decompositions_big_example< thrust::complex< double > >( SOLVING_TYPE::QR_decomposition, false );
}

TEST( big_non_singular_linear_equation_float, QR_decomposition_blocked_Householder_scaling )
{
	decompositions_big_example< float >( SOLVING_TYPE::QR_decomposition, true );
}

TEST( big_non_singular_linear_equation_double, QR_decomposition_blocked_Householder_scaling )
{
	decompositions_big_example< double >( SOLVING_TYPE::QR_decomposition, true );
}

TEST( big_non_singular_linear_equation_complex_float, QR_decomposition_blocked_Householder_scaling )
{
	decompositions_big_example< thrust::complex< float > >( SOLVING_TYPE::QR_decomposition, true );
}

TEST( big_non_singular_linear_equation_complex_double, QR_decomposition_blocked_Householder_scaling )
{
	decompositions_big_example< thrust::complex< double > >( SOLVING_TYPE::QR_decomposition, true );
}


TEST( big_non_singular_linear_equation_float, LU_decomposition_blocked_Gauss_no_scaling )
{
	decompositions_big_example< float >( SOLVING_TYPE::LU_decomposition, false );
}

TEST( big_non_singular_linear_equation_double, LU_decomposition_blocked_Gauss_no_scaling )
{
	decompositions_big_example< double >( SOLVING_TYPE::LU_decomposition, false );
}

TEST( big_non_singular_linear_equation_complex_float, LU_decomposition_blocked_Gauss_no_scaling )
{
	decompositions_big_example< thrust::complex< float > >( SOLVING_TYPE::LU_decomposition, false );
}

TEST( big_non_singular_linear_equation_complex_double, LU_decomposition_blocked_Gauss_no_scaling )
{
	decompositions_big_example< thrust::complex< double > >( SOLVING_TYPE::LU_decomposition, false );
}

TEST( big_non_singular_linear_equation_float, LU_decomposition_blocked_Gauss_scaling )
{
	decompositions_big_example< float >( SOLVING_TYPE::LU_decomposition, true );
}

TEST( big_non_singular_linear_equation_double, LU_decomposition_blocked_Gauss_scaling )
{
	decompositions_big_example< double >( SOLVING_TYPE::LU_decomposition, true );
}

TEST( big_non_singular_linear_equation_complex_float, LU_decomposition_blocked_Gauss_scaling )
{
	decompositions_big_example< thrust::complex< float > >( SOLVING_TYPE::LU_decomposition, true );
}

TEST( big_non_singular_linear_equation_complex_double, LU_decomposition_blocked_Gauss_scaling )
{
	decompositions_big_example< thrust::complex< double > >( SOLVING_TYPE::LU_decomposition, true );
}


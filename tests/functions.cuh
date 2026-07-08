#include <random>

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <thrust/complex.h>

template< typename T >
T generate_random( double min_val, double max_val )
{
	static std::random_device rd;
	static std::mt19937 gen( rd() );

	std::uniform_real_distribution< double > dis( min_val, max_val );
	std::uniform_int_distribution< int > sign_dis( 0, 1 );

	T sign = ( sign_dis( gen ) == 0 ) ? static_cast< T >( 1.0 ) : static_cast< T >( -1.0 );

	return static_cast< T >( dis( gen ) ) * sign;
}

template <>
thrust::complex< float > generate_random( double min_val, double max_val )
{
	return thrust::complex< float >( generate_random< float >( min_val, max_val ), generate_random< float >( min_val, max_val ) );
}

template <>
thrust::complex< double > generate_random( double min_val, double max_val )
{
	return thrust::complex< double >( generate_random< double >( min_val, max_val ), generate_random< double >( min_val, max_val ) );
}


template < typename T, typename Layout >
void print_matrix( const std::vector< T >& matrix, const size_t rows, const size_t cols, const Layout idx, const size_t follow_dim )
{
	std::ofstream out( "matrix.tsv" );

	for( size_t r = 0; r < rows; ++r )
	{
		for( size_t c = 0; c < cols; ++c )
		{
			auto value = matrix[ idx( r, c, follow_dim ) ];

			std::ostringstream ss;
			ss << std::setprecision( 15 ) << value;

			std::string str = ss.str();
			std::replace( str.begin(), str.end(), '.', ',' );

			out << str;

			if( c + 1 < cols )
				out << '\t';
		}

		out << '\n';
	}

	out.close();
}



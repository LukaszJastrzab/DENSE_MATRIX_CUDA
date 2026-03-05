#include <random>
#include <complex>

template< typename T >
T generate_random( T min_val, T max_val )
{
	static std::random_device rd;
	static std::mt19937 gen( rd() );

	std::uniform_real_distribution< T > dis( min_val, max_val );
	std::uniform_int_distribution< int > sign_dis( 0, 1 );

	T sign = ( sign_dis( gen ) == 0 ) ? static_cast< T >( 1.0 ) : static_cast< T >( -1.0 );

	return dis( gen ) * sign;
}

template< typename T >
std::complex<T> generate_complex_random( T min_val, T max_val )
{
	return std::complex<T>( generate_random<T>( min_val, max_val ), generate_random<T>( min_val, max_val ) );
}
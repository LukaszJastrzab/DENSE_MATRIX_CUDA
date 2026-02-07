#pragma once

#include <complex>

#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
int div_up( int a, int b )
{
	return ( a + b - 1 ) / b;
}

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
size_t calc_elem_idx( size_t row, size_t col, size_t A_cols )
{
	return col + row * A_cols;
}

template< typename T >
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
double norm2( const T& x )
{
	return x * x;
}

template< typename T >
inline double norm2( const std::complex< T >& x )
{
	return std::norm( x );
}

template< typename T >
inline double l2_norm( const std::vector< T >& v )
{
	double sum{};

	for( const auto& item : v )
		sum += norm2( item );

	return std::sqrt( sum );
}

template< typename T >
inline double l2_norm( const std::vector< std::complex< T > >& v )
{
	double sum{};

	for( const auto& item : v )
		sum += norm2( item );

	return std::sqrt( sum );
}

template< typename T >
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
double abs_val( const T& x )
{
	return std::abs( x );
}

template< typename T >
double abs_val( const std::complex< T >& x )
{
	return std::abs( x );
}

template< typename T >
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
T conjugate( const T& x )
{
	return x;
}

template< typename T >
std::complex< T > conjugate( const std::complex< T >& x )
{
	return std::conj( x );
}


#ifdef __CUDACC__

template< typename T >
__host__ __device__ __forceinline__
double norm2( const thrust::complex< T >& x )
{
	return thrust::norm( x );
}

template< typename T >
inline double l2_norm( const std::vector< thrust::complex< T > >& v )
{
	double sum{};

	for( const auto& item : v )
		sum += norm2( item );

	return std::sqrt( sum );
}

template< typename T >
__host__ __device__ __forceinline__
double abs_val( const thrust::complex< T >& x )
{
	return thrust::abs( x );
}

template< typename T >
__host__ __device__ __forceinline__
thrust::complex< T > conjugate( const thrust::complex< T >& x )
{
	return thrust::conj( x );
}

#endif
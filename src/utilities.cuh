#pragma once

#include <thrust/complex.h>

__host__ __device__ __forceinline__
int div_up( int a, int b )
{
	return ( a + b - 1 ) / b;
}

__host__ __device__ __forceinline__
size_t calc_elem_idx( size_t row, size_t col, size_t A_cols )
{
	return col + row * A_cols;
}

template< typename T >
__host__ __device__ __forceinline__
double norm2( const T& x )
{
	return x * x;
}

template< typename T >
__host__ __device__ __forceinline__
double norm2( const thrust::complex< T >& x )
{
	return thrust::norm( x );
}

template< typename T >
__host__ __device__ __forceinline__
double abs_val( const T& x )
{
	return x >= T( 0 ) ? x : -x;
}

template< typename T >
__host__ __device__ __forceinline__
double abs_val( const thrust::complex< T >& x )
{
	return thrust::abs( x );
}

template< typename T >
__host__ __device__ __forceinline__
T conjugate( const T& x )
{
	return x;
}

template< typename T >
__host__ __device__ __forceinline__
thrust::complex< T > conjugate( const thrust::complex< T >& x )
{
	return thrust::conj( x );
}
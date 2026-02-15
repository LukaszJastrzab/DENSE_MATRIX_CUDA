#pragma once

#include <type_traits>
#include <complex>

#ifdef __CUDACC__
#include <thrust/complex.h>
#include <cublas_v2.h>
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
size_t calc_elem_idx( size_t row, size_t col, size_t rows )
{
    // column majority
	return row + col * rows;

}

template< typename T >
struct real_type
{
    using type = T;
};

template< typename T >
struct real_type< std::complex< T > >
{
    using type = T;
};

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
struct real_type< thrust::complex< T > >
{
    using type = T;
};

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

template<typename T>
struct CublasTrsv;

template<>
struct CublasTrsv<float>
{
    static cublasStatus_t call(
        cublasHandle_t handle,
        cublasFillMode_t uplo,
        cublasOperation_t trans,
        cublasDiagType_t diag,
        int n,
        const float* A,
        int lda,
        float* x,
        int incx
    )
    {
        return cublasStrsv( handle, uplo, trans, diag, n, A, lda, x, incx );
    }
};

template<>
struct CublasTrsv<double>
{
    static cublasStatus_t call(
        cublasHandle_t handle,
        cublasFillMode_t uplo,
        cublasOperation_t trans,
        cublasDiagType_t diag,
        int n,
        const double* A,
        int lda,
        double* x,
        int incx
    )
    {
        return cublasDtrsv( handle, uplo, trans, diag, n, A, lda, x, incx );
    }
};

template<>
struct CublasTrsv<thrust::complex<float>>
{
    static cublasStatus_t call(
        cublasHandle_t handle,
        cublasFillMode_t uplo,
        cublasOperation_t trans,
        cublasDiagType_t diag,
        int n,
        const thrust::complex<float>* A,
        int lda,
        thrust::complex<float>* x,
        int incx
    )
    {
        return cublasCtrsv( handle, uplo, trans, diag, n,
            reinterpret_cast< const cuComplex* >( A ), lda,
            reinterpret_cast< cuComplex* >( x ), incx );
    }
};

template<>
struct CublasTrsv<thrust::complex<double>>
{
    static cublasStatus_t call(
        cublasHandle_t handle,
        cublasFillMode_t uplo,
        cublasOperation_t trans,
        cublasDiagType_t diag,
        int n,
        const thrust::complex<double>* A,
        int lda,
        thrust::complex<double>* x,
        int incx
    )
    {
        return cublasZtrsv( handle, uplo, trans, diag, n,
            reinterpret_cast< const cuDoubleComplex* >( A ), lda,
            reinterpret_cast< cuDoubleComplex* >( x ), incx );
    }
};

#endif
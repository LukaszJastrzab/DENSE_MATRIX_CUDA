template< typename T >
__global__
void L_block_update_new(
	T* A_in,
	const size_t* p_row,
	const int A_cols,
	const int row_offset,
	const int col_offset )
{
	const int col = col_offset + threadIdx.x + blockDim.x * blockIdx.x;
	const int mx_size = blockDim.y;

	extern __shared__ unsigned char sdata_raw[];
	T* L = reinterpret_cast< T* >( sdata_raw );
	T* U_i = reinterpret_cast< T* >( sdata_raw + ( ( mx_size + threadIdx.x ) * mx_size ) * sizeof( T ) );

	if( threadIdx.y > threadIdx.x )
		L[ RLD( threadIdx.y, threadIdx.x, mx_size ) ] =
		A_in[ RLD( p_row[ row_offset + threadIdx.y ], row_offset + threadIdx.x, A_cols ) ];

	const size_t in_idx = RLD( p_row[ row_offset + threadIdx.y ], col, A_cols );
	U_i[ threadIdx.y ] = A_in[ in_idx ];

	__syncthreads();

	bool active{ col < A_cols };

	for( int c{ 0 }; c < mx_size - 1; ++c )
	{
		if ( active && threadIdx.y > c )
			U_i[ threadIdx.y ] -= L[ RLD( threadIdx.y, c, mx_size ) ] * U_i[ c ];

		__syncthreads();
	}

	if ( active )
		A_in[ in_idx ] = U_i[ threadIdx.y ];
}
/** Implementation Description
 *  Perform LUD in several iterations of two CUDA kernels. The firest kernel
 *  "ludcmp_peri_col" computer peri-column of the matrix, the second kernel
 *  "ludcmp_internal" update all the internal elements of the matrix.
 *
 *   Use texture memory and share memory
 **/


#define BLOCK_SIZE 16
texture<float, 1, cudaReadModeElementType> texRef;


//<<<n/BLOCK_SIZE,BLOCK_SIZE>>>
__global__ static void 
ludcmp_peri_col(float *a, int x, int sp, int n)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x+x+1;
  idx -= sp;

  if (idx >= x+1) {
    a[(idx)*n+x] = tex1Dfetch(texRef, idx*n+x)/tex1Dfetch(texRef, x*n+x);
  }
}

//<<<n/BLOCK_SIZE*BLOCK_SIZE, BLOCK_SIZE*BLOCK_SIZE>>>
__global__ static void 
ludcmp_internal(float *a, int x, int sp, int n)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x+x;
  int idy = blockIdx.y*blockDim.y+threadIdx.y+x;

  idx -= sp;
  idy -= sp;

  __shared__ float srow[BLOCK_SIZE];
  __shared__ float scol[BLOCK_SIZE];

  if (idx >= x && idy >= x){
    if (threadIdx.x == BLOCK_SIZE-1)
      srow[threadIdx.y] = tex1Dfetch(texRef, (x-1)*n+idy);
    if (threadIdx.y == BLOCK_SIZE-1)
      scol[threadIdx.x] = tex1Dfetch(texRef, idx*n+x-1);
    __syncthreads();

    a[(idx)*n+idy] = tex1Dfetch(texRef, idx*n+idy) - srow[threadIdx.y]*scol[threadIdx.x];
  }
}


void lud_cuda_4(float *a, int n)
{
  int i;

  for (i=0; i < n-BLOCK_SIZE-1; i ++) {
    ludcmp_peri_col<<<(n-i)/BLOCK_SIZE, BLOCK_SIZE>>>(a, i, (BLOCK_SIZE-(n-i-1)%BLOCK_SIZE)%BLOCK_SIZE, n);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n-i)/BLOCK_SIZE, (n-i)/BLOCK_SIZE);
    ludcmp_internal<<<dimGrid, dimBlock>>>(a, i+1, (BLOCK_SIZE-(n-i-1)%BLOCK_SIZE)%BLOCK_SIZE, n);
  }
  for (;i < n-1; i ++){
    ludcmp_peri_col<<<1, BLOCK_SIZE>>>(a, i, (BLOCK_SIZE-(n-i-1)%BLOCK_SIZE)%BLOCK_SIZE, n);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    ludcmp_internal<<<1,dimBlock>>>(a,i+1,(BLOCK_SIZE-(n-i-1)%BLOCK_SIZE)%BLOCK_SIZE, n);
  }
}



/** Implementation Description
 *  Perform LUD based on the same algorithm used in baseline and omp, with
 *  several iterations of two kernels
 *
 *  Use texture memory
 **/

texture<float, 1, cudaReadModeElementType> texRef;

__global__ static void 
lu_dcmp_phase1(float *a, int i, int n){
  int idx = threadIdx.x;
  int j = idx + i;
  float sum;

  while (j < n){
    sum = tex1Dfetch(texRef, i*n+j);
    for (int k=0; k < i; k++) 
      sum -= tex1Dfetch(texRef, i*n+k) * tex1Dfetch(texRef, k*n+j);
    *(a+i*n+j) = sum;
    j += blockDim.x;
  }
  __syncthreads();
}

__global__ static void 
lu_dcmp_phase2(float *a, int i, int n){
  int idx = threadIdx.x;
  int j = idx + i + 1;
  float sum;
  while (j < n){
    sum = tex1Dfetch(texRef, j*n+i);
    for (int k = 0; k < i; k++) 
      sum -= tex1Dfetch(texRef, j*n+k) * tex1Dfetch(texRef, k*n+i);
    *(a+j*n+i)= sum/tex1Dfetch(texRef, i*n+i);
    j += blockDim.x;
  }
  __syncthreads();
}

void lud_cuda_2(float *a, int n)
{
  int i;
  for (i=0; i <n; i++){
    lu_dcmp_phase1<<<1, 512>>>(a,i,n);
    lu_dcmp_phase2<<<1, 512>>>(a,i,n);
  }
}



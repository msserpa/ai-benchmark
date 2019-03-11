/** Implementation Description
 *  Perform LUD based on the same algorithm used in baseline and omp, with
 *  several iterations of two kernels
 **/



__global__ static void 
lu_dcmp_phase1(float *a, int i, int n){
    int idx = threadIdx.x;
    int j = idx + i;
    float sum;
    
    while (j < n){
        sum = *(a+i*n+j);
        for (int k=0; k < i; k++) sum -= (*(a+i*n+k)) * (*(a+k*n+j));
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
        sum = *(a+j*n+i);
        for (int k = 0; k < i; k++) sum -=(*(a+j*n+k)) * (*(a+k*n+i));
        *(a+j*n+i)= sum/(*(a+i*n+i));
        j += blockDim.x;
    }
    __syncthreads();
}

void lud_cuda_3(float *a, int n)
{
     int i;
     for (i=0; i <n; i++){
         lu_dcmp_phase1<<<1, 512>>>(a,i,n);
         lu_dcmp_phase2<<<1, 512>>>(a,i,n);
     }
}



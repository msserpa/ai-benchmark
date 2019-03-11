#include "track_ellipse_kernel.h"
#include <cutil.h>

#define ONE_OVER_PI 1.0 / PI
#define MU 0.5
#define LAMBDA (8.0 * MU + 1.0)


float *host_I, *host_IMGVF_in, *host_IMGVF_out;
float *device_I, *device_IMGVF_in, *device_IMGVF_out;



__global__ void IMGVF_kernel(float *IMGVF_in, float *IMGVF_out, float *I, float vx, float vy, float e, int m, int n) {
	// Determine the thread's coordinates
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = thread_id / n;
	int j = thread_id % n;
	
	if (i >= m) return;
	
	// Compute neighboring pixel indices
	int rowU = (i == 0) ? 0 : i - 1;
	int rowD = (i == m - 1) ? m - 1 : i + 1;
	int colL = (j == 0) ? 0 : j - 1;
	int colR = (j == n - 1) ? n - 1 : j + 1;
	
	// Compute the difference between the pixel and its eight neighbors
	float old_val = IMGVF_in[(i * n) + j];
	float U  = IMGVF_in[(rowU * n) + j   ] - old_val;
	float D  = IMGVF_in[(rowD * n) + j   ] - old_val;
	float L  = IMGVF_in[(i    * n) + colL] - old_val;
	float R  = IMGVF_in[(i    * n) + colR] - old_val;
	float UR = IMGVF_in[(rowU * n) + colR] - old_val;
	float DR = IMGVF_in[(rowD * n) + colR] - old_val;
	float UL = IMGVF_in[(rowU * n) + colL] - old_val;
	float DL = IMGVF_in[(rowD * n) + colL] - old_val;
	
	// Compute the regularized heaviside value for these differences
	float one_over_e = 1.0 / e;
	float UHe  = ONE_OVER_PI * atan((U  *       -vy)  * one_over_e) + 0.5;
	float DHe  = ONE_OVER_PI * atan((D  *        vy)  * one_over_e) + 0.5;
	float LHe  = ONE_OVER_PI * atan((L  *  -vx     )  * one_over_e) + 0.5;
	float RHe  = ONE_OVER_PI * atan((R  *   vx     )  * one_over_e) + 0.5;
	float URHe = ONE_OVER_PI * atan((UR * ( vx - vy)) * one_over_e) + 0.5;
	float DRHe = ONE_OVER_PI * atan((DR * ( vx + vy)) * one_over_e) + 0.5;
	float ULHe = ONE_OVER_PI * atan((UL * (-vx - vy)) * one_over_e) + 0.5;
	float DLHe = ONE_OVER_PI * atan((DL * (-vx + vy)) * one_over_e) + 0.5;
	
	// Update the IMGVF value
	// Compute IMGVF += (mu / lambda)(UHe .*U  + DHe .*D  + LHe .*L  + RHe .*R +
	//                                URHe.*UR + DRHe.*DR + ULHe.*UL + DLHe.*DL);
	float new_val = old_val + (MU / LAMBDA) * (UHe  * U  + DHe  * D  + LHe  * L  + RHe  * R +
											   URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL);
	// Compute IMGVF -= (1 / lambda)(I .* (IMGVF - I))
	float vI = I[(i * n) + j];
	IMGVF_out[(i * n) + j] = new_val - ((1.0 / LAMBDA) * vI * (new_val - vI));
}



extern "C" void IMGVF_cuda(MAT *IMGVF_in, MAT *IMGVF_out, double vx, double vy, double e);

void IMGVF_cuda(MAT *IMGVF_in, MAT *IMGVF_out, double vx, double vy, double e) {
	// Compute the amount of memory required
	int m = IMGVF_in->m, n = IMGVF_in->n, i, j;
	unsigned int mem_size = sizeof(float) * m * n;

	// Copy host memory to device
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			host_IMGVF_in[(i * n) + j] = (float) m_get_val(IMGVF_in, i, j);
	cudaMemcpy(device_IMGVF_in, host_IMGVF_in, mem_size, cudaMemcpyHostToDevice);

	// Determine thread block size
	int num_threads = m * n;
	int threads_per_block = 64;
	int num_blocks = (int) (((float) num_threads / (float) threads_per_block) + 0.5);
	
	// Setup execution parameters
	dim3 grid(num_blocks, 1, 1);
	dim3 threads(threads_per_block, 1, 1);
    
	// Execute the kernel
	IMGVF_kernel <<< grid, threads >>> (device_IMGVF_in, device_IMGVF_out, device_I, (float) vx, (float) vy, (float) e, m, n);
	
	// Check if kernel execution caused an error
	CUT_CHECK_ERROR("Kernel execution failed");

	// Copy result from device to host
	cudaMemcpy(host_IMGVF_out, device_IMGVF_out, mem_size, cudaMemcpyDeviceToHost);
	// Pack the result into the matrix
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			m_set_val(IMGVF_out, i, j, (double) host_IMGVF_out[(i * n) + j]);
}


extern "C" void IMGVF_cuda_init(MAT *I);

void IMGVF_cuda_init(MAT *I) {
	// Compute the amount of memory required
	int m = I->m, n = I->n, i, j;
	unsigned int mem_size = sizeof(float) * m * n;
	
	//Allocate device memory
	cudaMalloc( (void**) &device_IMGVF_in, mem_size);
	cudaMalloc( (void**) &device_IMGVF_out, mem_size);
	cudaMalloc( (void**) &device_I, mem_size);
	
	// Allocate host memory
	host_IMGVF_in = (float *) malloc(mem_size);
	host_IMGVF_out = (float *) malloc(mem_size);
	host_I = (float *) malloc(mem_size);
	
	// Copy matrix I to device
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			host_I[(i * n) + j] = (float) m_get_val(I, i, j);
	CUDA_SAFE_CALL( cudaMemcpy( device_I, host_I, mem_size, cudaMemcpyHostToDevice) );
}


extern "C" void IMGVF_cuda_cleanup();

void IMGVF_cuda_cleanup() {
	// Free memory
	free(host_IMGVF_in);
	free(host_IMGVF_out);
	free(host_I);
	cudaFree(device_IMGVF_in);
	cudaFree(device_IMGVF_out);
	cudaFree(device_I);
}

#include "track_ellipse_kernel.h"
#include <cutil.h>

#define ONE_OVER_PI 1.0 / PI
#define MU 0.5
#define LAMBDA (8.0 * MU + 1.0)


__global__ void IMGVF_kernel(float *IMGVF_in, float *IMGVF_out, float vx, float vy, float e, int m, int n) {
	// Determine the thread's coordinates
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = thread_id / n;
	int j = thread_id % n;
	
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
	IMGVF_out[(i * n) + j] = old_val + (MU / LAMBDA) * (UHe  * U  + DHe  * D  + LHe  * L  + RHe  * R +
	                                                    URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL);

	// Compute IMGVF -= (1 / lambda)(I .* (IMGVF - I))
	// double vI = m_get_val(I, i, j);
	// double new_val = vHe - (one_over_lambda * vI * (vHe - vI));
}



extern "C" void IMGVF_cuda(MAT *IMGVF_in, MAT *IMGVF_out, double vx, double vy, double e);

void IMGVF_cuda(MAT *IMGVF_in, MAT *IMGVF_out, double vx, double vy, double e) {
	int m = IMGVF_in->m, n = IMGVF_in->n, i, j;
	
	// Allocate device memory
	unsigned int mem_size = sizeof(float) * m * n;
	float *device_IMGVF_in;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_IMGVF_in, mem_size));

	// Copy host memory to device
	float *host_IMGVF_in = (float *) malloc(mem_size);
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			host_IMGVF_in[(i * n) + j] = (float) m_get_val(IMGVF_in, i, j);
	CUDA_SAFE_CALL( cudaMemcpy( device_IMGVF_in, host_IMGVF_in, mem_size, cudaMemcpyHostToDevice) );

	// Allocate device memory for result
	float *device_IMGVF_out;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_IMGVF_out, mem_size));

	// Determine thread block size
	int num_threads = m * n;
	int threads_per_block = 256;
	int num_blocks = (int) (((float) num_threads / (float) threads_per_block) + 0.5);
	
	// Setup execution parameters
	dim3 grid(num_blocks, 1, 1);
	dim3 threads(threads_per_block, 1, 1);
    
	// Execute the kernel
	IMGVF_kernel <<< grid, threads >>> (device_IMGVF_in, device_IMGVF_out, (float) vx, (float) vy, (float) e, m, n);

	// Check if kernel execution caused an error
	CUT_CHECK_ERROR("Kernel execution failed");

	// Allocate mem for the result on host side
	float *host_IMGVF_out = (float *) malloc(mem_size);
	// Copy result from device to host
	CUDA_SAFE_CALL( cudaMemcpy( host_IMGVF_out, device_IMGVF_out, mem_size, cudaMemcpyDeviceToHost) );
	// Pack the result into the matrix
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			m_set_val(IMGVF_out, i, j, (double) host_IMGVF_out[(i * n) + j]);
	
	// Free memory
	free(host_IMGVF_in);
	free(host_IMGVF_out);
	CUDA_SAFE_CALL(cudaFree(device_IMGVF_in));
	CUDA_SAFE_CALL(cudaFree(device_IMGVF_out));
}




__global__ void heaviside_kernel(float *H, float *z, float v, float e, int m, int n, float inv_n) {
	
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = thread_id * inv_n;
	int j = thread_id - n*i;
	//int j = thread_id % n;

	if (i < m) H[(i * n) + j] = ONE_OVER_PI * atan((z[(i * n) + j] * v) * e) + 0.5;
}





float *device_z;
float *host_z;
float *device_H;
float *host_H;


extern "C" void heavyside_init(int n, int m);

void heavyside_init(int n, int m) {
	//Allocate device memory
	unsigned int mem_size = sizeof(float) * m * n;
	// Copy host memory to device
	cudaMallocHost((void**)&host_z,mem_size);

	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_z, mem_size));
	// Allocate device memory for result
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_H, mem_size));
	// Allocate mem for the result on host side
	cudaMallocHost((void**)&host_H,mem_size);	
}


extern "C" void heavyside_cleanup();

void heavyside_cleanup() {
	// Free memory
	cudaFree(host_z);
	cudaFree(host_H);
	CUDA_SAFE_CALL(cudaFree(device_z));
	CUDA_SAFE_CALL(cudaFree(device_H));
}


extern "C" void heaviside_cuda(MAT *H, MAT *z, double v, double e);

void heaviside_cuda(MAT *H, MAT *z, double v, double e) {
	int m = z->m, n = z->n, i, j;
	
	unsigned int mem_size = sizeof(float) * m * n;

	// Copy host memory to device
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			host_z[(i * n) + j] = (float) m_get_val(z, i, j);
	CUDA_SAFE_CALL( cudaMemcpy( device_z, host_z, mem_size, cudaMemcpyHostToDevice) );

	// Determine thread block size
	int num_threads = m * n;
	int threads_per_block = 256;
	int num_blocks = (int) (((float) num_threads / (float) threads_per_block) + 0.5);
	
	// Setup execution parameters
	dim3 grid(num_blocks, 1, 1);
	dim3 threads(threads_per_block, 1, 1);
    
	// precompute 1/n
	float inv_n = 1.0f / (float)n;
	// Execute the kernel
	heaviside_kernel <<< grid, threads >>> (device_H, device_z, (float) v, (float) e, m, n, inv_n);

	// Check if kernel execution caused an error
	CUT_CHECK_ERROR("Kernel execution failed");

	// Copy result from device to host
	CUDA_SAFE_CALL( cudaMemcpy( host_H, device_H, mem_size, cudaMemcpyDeviceToHost) );
	// Pack the result into the matrix
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			m_set_val(H, i, j, (double) host_H[(i * n) + j]);
	
}

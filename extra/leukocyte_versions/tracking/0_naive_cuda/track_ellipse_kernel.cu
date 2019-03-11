#include "track_ellipse_kernel.h"
#include <cutil.h>

//__constant__ float c_sin_angle[NPOINTS];
//texture<float, 1, cudaReadModeElementType> t_grad_x;

#define ONE_OVER_PI 1.0 / PI


__global__ void heaviside_kernel(float *H, float *z, float v, float e, int m, int n) {
    
	// i = blockIdx.x + MAX_RAD + 2;
	// j = threadIdx.x + MAX_RAD + 2;

	//float p =   tex1Dfetch(t_grad_x,addr) * c_cos_angle[n] + 
	//			tex1Dfetch(t_grad_y,addr) * c_sin_angle[n];
	
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = thread_id / n;
	int j = thread_id % n;

	if (i < m)		
		H[(i * n) + j] = ONE_OVER_PI * atan((z[(i * n) + j] * v) / e) + 0.5;
}



extern "C" void heaviside_cuda(MAT *H, MAT *z, double v, double e);

void heaviside_cuda(MAT *H, MAT *z, double v, double e) {
	int m = z->m, n = z->n, i, j;
	
	// Allocate device memory
	unsigned int mem_size = sizeof(float) * m * n;
	float *device_z;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_z, mem_size));

	// Copy host memory to device
	float *host_z = (float *) malloc(mem_size);
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			host_z[(i * n) + j] = m_get_val(z, i, j);
	CUDA_SAFE_CALL( cudaMemcpy( device_z, host_z, mem_size, cudaMemcpyHostToDevice) );
    
	// Bind texture
    //CUDA_SAFE_CALL( cudaBindTexture(0, t_grad_y, device_grad_y, grad_mem_size));

	// Allocate device memory for result
	float *device_H;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_H, mem_size));
	// Initialize memory
	//CUDA_SAFE_CALL( cudaMemset( device_H, 0, mem_size) );

	// Determine thread block size
	int num_threads = m * n;
	int threads_per_block = 256;
	int num_blocks = (int) (((float) num_threads / (float) threads_per_block) + 0.5);
	
	// Setup execution parameters
	dim3 grid(num_blocks, 1, 1);
	dim3 threads(threads_per_block, 1, 1);
    
	// Execute the kernel
	heaviside_kernel <<< grid, threads >>> (device_H, device_z, (float) v, (float) e, m, n);

	// Check if kernel execution caused an error
	CUT_CHECK_ERROR("Kernel execution failed");
	
	//printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));

	// Allocate mem for the result on host side
	float *host_H = (float *) malloc(mem_size);
	// Copy result from device to host
	CUDA_SAFE_CALL( cudaMemcpy( host_H, device_H, mem_size, cudaMemcpyDeviceToHost) );
	// Pack the result into the matrix
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			m_set_val(H, i, j, host_H[(i * n) + j]);
	
	// Free memory
	free(host_z);
	free(host_H);
	CUDA_SAFE_CALL(cudaFree(device_z));
	CUDA_SAFE_CALL(cudaFree(device_H));
}

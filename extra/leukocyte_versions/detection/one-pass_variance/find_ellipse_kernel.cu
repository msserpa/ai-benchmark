#include <cutil.h>
#include "find_ellipse_kernel.h"

#include <stdio.h>
#include <sys/param.h>
#include <sys/times.h>

#define NPOINTS 150
#define MIN_RAD 10
#define MAX_RAD 20
#define NCIRCLES 10
#define NCIRC 7

#define STREL_SIZE (12 * 2 + 1)

float *device_gicov;

__constant__ float c_sin_angle[NPOINTS];
__constant__ float c_cos_angle[NPOINTS];
__constant__ int c_tX[NCIRCLES * NPOINTS];
__constant__ int c_tY[NCIRCLES * NPOINTS];

__constant__ float c_strel[STREL_SIZE * STREL_SIZE];

// Texture references
texture<float, 1, cudaReadModeElementType> t_grad_x;
texture<float, 1, cudaReadModeElementType> t_grad_y;
texture<float, 1, cudaReadModeElementType> t_img;

//Find matrix of GICOV values at all given pixels given x and y gradients of the image
__global__ void ellipsematching_kernel(float * grad_x, float * grad_y, int grad_m, int grad_n, float * gicov) {
	float sum, ave, var, sGicov;
	int i, j, k, n, x, y;
    
	i = blockIdx.x + MAX_RAD + 2;
	j = threadIdx.x + MAX_RAD + 2;

	sGicov = 0;

	for (k = 0; k < NCIRC; k++) {
		sum = 0.0;
		float M2 = 0.f;
		float mean = 0.f;
	
		for (n = 0; n < NPOINTS; n++) {
			y = j + c_tY[(k * NPOINTS) + n];
			x = i + c_tX[(k * NPOINTS) + n];
			int addr = x * grad_m + y;
			float p =   tex1Dfetch(t_grad_x,addr) * c_cos_angle[n] + 
						tex1Dfetch(t_grad_y,addr) * c_sin_angle[n];                    
			sum += p;
			float delta = p - mean;
			mean = mean + (delta / (float) (n + 1));
			M2 = M2 + (delta * (p - mean));
		}
		
		ave = sum / ((float) NPOINTS);
		
		var = M2 / ((float) (NPOINTS - 1));
		
		if (((ave * ave) / var) > sGicov) {
			gicov[(i * grad_m) + j] = ave / sqrt(var);
			sGicov = (ave * ave) / var;
		}
	}
	
	// Square the gicov value
	gicov[(i * grad_m) + j] = gicov[(i * grad_m) + j] * gicov[(i * grad_m) + j];
}



extern "C" float *ellipsematching_setup(int grad_m, int grad_n, float *host_grad_x, float *host_grad_y);

float *ellipsematching_setup(int grad_m, int grad_n, float *host_grad_x, float *host_grad_y) {
	int MaxR = MAX_RAD + 2;

	unsigned int grad_mem_size = sizeof(float) * grad_m * grad_n;

	// allocate device memory
	float *device_grad_x, *device_grad_y;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_grad_x, grad_mem_size));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_grad_y, grad_mem_size));

	// copy host memory to device
	CUDA_SAFE_CALL( cudaMemcpy( device_grad_x, host_grad_x, grad_mem_size, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy( device_grad_y, host_grad_y, grad_mem_size, cudaMemcpyHostToDevice) );
    
	// Bind input matrices to textures
    CUDA_SAFE_CALL( cudaBindTexture(0, t_grad_x, device_grad_x, grad_mem_size));
    CUDA_SAFE_CALL( cudaBindTexture(0, t_grad_y, device_grad_y, grad_mem_size));

	// allocate device memory for result
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_gicov, grad_mem_size));
	// initialize memory (some of the cells are never assigned a value in the kernel)
	CUDA_SAFE_CALL( cudaMemset( device_gicov, 0, grad_mem_size) );

	// setup execution parameters
	dim3 grid( grad_n - (2 * MaxR), 1, 1);
	dim3 threads( grad_m - (2 * MaxR), 1, 1);
    
	// execute the kernel
	ellipsematching_kernel <<< grid, threads, 0 >>> (device_grad_x, device_grad_y, grad_m, grad_n, device_gicov);

	// check if kernel execution caused an error
	CUT_CHECK_ERROR("Kernel execution failed");

	// allocate mem for the result on host side
	float *host_gicov = (float *) malloc(grad_mem_size);
	// copy result from device to host
	CUDA_SAFE_CALL( cudaMemcpy( host_gicov, device_gicov, grad_mem_size, cudaMemcpyDeviceToHost) );

	// cleanup memory
	free(host_grad_x);
	free(host_grad_y);
	cudaUnbindTexture(t_grad_x);
	cudaUnbindTexture(t_grad_y);
	CUDA_SAFE_CALL(cudaFree(device_grad_x));
	CUDA_SAFE_CALL(cudaFree(device_grad_y));

	return host_gicov;
}




__global__ void dilate_f_kernel(float *img, int img_m, int img_n, int strel_m, int strel_n, float *dilated) {
	int i, j, el_i, el_j, x, y;
	int el_center_i = strel_m / 2, el_center_j = strel_n / 2;
	float max, temp;

	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	i = thread_id % img_m;
	j = thread_id / img_m;

	max = 0.0;

	for(el_i = 0; el_i < strel_m; el_i++) {
		y = i - el_center_i + el_i;
		if((y >= 0) && (y < img_m)) {
			for(el_j = 0; el_j < strel_n; el_j++) {
				x = j - el_center_j + el_j;
				if ((x >= 0) &&	(x < img_n) && (c_strel[(el_i * strel_n) + el_j] != 0)) {
					int addr = (x * img_m) + y;
					temp = tex1Dfetch(t_img, addr);
					if (temp > max) max = temp;
				}
			}
		}
	}

	dilated[(i * img_n) + j] = max;
}



extern "C" float *dilate_f_setup(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n);

float *dilate_f_setup(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n) {

	// Compute memory sizes
	unsigned int max_gicov_mem_size = sizeof(float) * max_gicov_m * max_gicov_n;

	// allocate device memory for result
	float* device_img_dilated;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_img_dilated, max_gicov_mem_size) );
	
	CUDA_SAFE_CALL( cudaBindTexture(0, t_img, device_gicov, max_gicov_mem_size));
    
	int num_threads = max_gicov_m * max_gicov_n;
	int threads_per_block = 176;
	int num_blocks = (int) (((float) num_threads / (float) threads_per_block) + 0.5);
    
	// setup execution parameters
	dim3 grid(num_blocks, 1, 1);
	dim3 threads(threads_per_block, 1, 1);

	// execute the kernel
	dilate_f_kernel <<< grid, threads, 0 >>> (device_gicov, max_gicov_m, max_gicov_n, strel_m, strel_n, device_img_dilated);

	// check if kernel execution generated an error
	CUT_CHECK_ERROR("Kernel execution failed");

	// allocate mem for the result on host side
	float *host_img_dilated = (float*) malloc(max_gicov_mem_size);
	// copy result from device to host
	CUDA_SAFE_CALL(cudaMemcpy(host_img_dilated, device_img_dilated, max_gicov_mem_size, cudaMemcpyDeviceToHost));

	// cleanup memory
	cudaUnbindTexture(t_img);
	CUDA_SAFE_CALL(cudaFree(device_gicov));
	CUDA_SAFE_CALL(cudaFree(device_img_dilated));

	return host_img_dilated;
}


// Chooses the most appropriate GPU on which to execute
void select_device() {
	// Figure out how many devices exist
	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	
	// Choose the device with the largest number of multiprocessors
	if (num_devices > 0) {
		int max_multiprocessors = 0, max_device = -1;
		for (device = 0; device < num_devices; device++) {
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, device);
			if (max_multiprocessors < properties.multiProcessorCount) {
				max_multiprocessors = properties.multiProcessorCount;
				max_device = device;
			}
		}
		cudaSetDevice(max_device);
	}
	
	// The following is to remove the API initialization overhead from the runtime measurements
	cudaFree(0);
}


void transfer_constants(float *host_sin_angle, float *host_cos_angle, int *host_tX, int *host_tY, int strel_m, int strel_n, float *host_strel) {
	// Compute the sizes of the matrices
	unsigned int angle_mem_size = sizeof(float) * NPOINTS;
	unsigned int t_mem_size = sizeof(int) * NCIRCLES * NPOINTS;
	unsigned int strel_mem_size = sizeof(float) * strel_m * strel_n;

	// Copy the matrices from host memory to device constant memory
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("c_sin_angle", host_sin_angle, angle_mem_size, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("c_cos_angle", host_cos_angle, angle_mem_size, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("c_tX", host_tX, t_mem_size, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("c_tY", host_tY, t_mem_size, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("c_strel", host_strel, strel_mem_size, 0, cudaMemcpyHostToDevice));
}


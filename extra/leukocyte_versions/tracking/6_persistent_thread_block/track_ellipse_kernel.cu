#include "track_ellipse_kernel.h"
#include <cutil.h>
#include <sys/time.h>
#include <time.h>

#define ONE_OVER_PI 1.0 / PI
#define MU 0.5
#define LAMBDA (8.0 * MU + 1.0)


float **host_I_array, **host_IMGVF_array;
float **device_I_array, **device_IMGVF_array;
int *host_m_array, *host_n_array;
int *device_m_array, *device_n_array;

float *host_I_all;
int total_mem_size;

const int threads_per_block = 352;
const int next_lowest_power_of_two = 256;

__device__ float fast_heaviside(float in) {
	float out = 0.f;
	if(in > -0.0001) out = 0.5;
	if(in >  0.0001) out = 1.0;
	
	return out;
}

__global__ void IMGVF_kernel(float **IMGVF_array, float **I_array,
							 int *m_array, int *n_array,
							 float vx, float vy, float e, float cutoff) {
	
	__shared__ int cell_converged;
	__shared__ float buffer[threads_per_block];
	__shared__ float IMGVF[41 * 81];   // FIXME: These should not be hardcoded
	
	float partial_sums;
	
	// Set the converged flag to false
	if (threadIdx.x == 0) cell_converged = 0;
	__syncthreads();
	
	// Get pointers to current cell's memory
	int cell_num = blockIdx.x;
	float *IMGVF_global = IMGVF_array[cell_num];
	float *I = I_array[cell_num];
	
	// Get current cell's memory dimensions
	int m = m_array[cell_num];
	int n = n_array[cell_num];
	
	// Compute number of virtual thread blocks
	int max = (int) ((float) (m * n) / (float) threads_per_block);
	if ((max * threads_per_block) < (m * n)) max++;
	
	// Load the initial IMGVF matrix
	int thread_id = threadIdx.x, thread_block, i, j;
	for (thread_block = 0; thread_block < max; thread_block++) {
		int offset = thread_block * threads_per_block;
		i = (thread_id + offset) / n;
		j = (thread_id + offset) % n;
		if (i < m) IMGVF[(i * n) + j] = IMGVF_global[(i * n) + j];
	}
	__syncthreads();
	
	
	const float one_nth = 1.f / (float)n;
	
	const int tid_mod = thread_id % n;
	const int tbsize_mod = threads_per_block % n;
	
	// Iteratively compute the IMGVF matrix
	int iter = 0;
	while ((! cell_converged) && (iter < 500)) {
	
		partial_sums = 0.0f;
		
		int old_i=0, old_j=0;
		
		j = tid_mod - tbsize_mod;
		
		// Iterate over virtual thread blocks
		for (thread_block = 0; thread_block < max; thread_block++) {
			int offset = thread_block * threads_per_block;
			
			//i = (thread_id + offset) / n;
			//j = (thread_id + offset) % n;
			
			// replace expensive integer divide and modulo with cheaper operations
			old_i = i;
			old_j = j;
			
			i = (thread_id + offset) * one_nth;
			j += tbsize_mod;
			if(j >= n) j -= n;
			
			float new_val = 0.0, old_val = 0.0;
			
			if (i < m) {
				// Compute neighboring pixel indices
				int rowU = (i == 0) ? 0 : i - 1;
				int rowD = (i == m - 1) ? m - 1 : i + 1;
				int colL = (j == 0) ? 0 : j - 1;
				int colR = (j == n - 1) ? n - 1 : j + 1;
				
				// Compute the difference between the pixel and its eight neighbors
				old_val = IMGVF[(i * n) + j];
				float U  = IMGVF[(rowU * n) + j   ] - old_val;
				float D  = IMGVF[(rowD * n) + j   ] - old_val;
				float L  = IMGVF[(i    * n) + colL] - old_val;
				float R  = IMGVF[(i    * n) + colR] - old_val;
				float UR = IMGVF[(rowU * n) + colR] - old_val;
				float DR = IMGVF[(rowD * n) + colR] - old_val;
				float UL = IMGVF[(rowU * n) + colL] - old_val;
				float DL = IMGVF[(rowD * n) + colL] - old_val;
				
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
				/*
				float UHe  = fast_heaviside((U  *       -vy)  * one_over_e);
				float DHe  = fast_heaviside((D  *        vy)  * one_over_e);
				float LHe  = fast_heaviside((L  *  -vx     )  * one_over_e);
				float RHe  = fast_heaviside((R  *   vx     )  * one_over_e);
				float URHe = fast_heaviside((UR * ( vx - vy)) * one_over_e);
				float DRHe = fast_heaviside((DR * ( vx + vy)) * one_over_e);
				float ULHe = fast_heaviside((UL * (-vx - vy)) * one_over_e);
				float DLHe = fast_heaviside((DL * (-vx + vy)) * one_over_e);
				*/
				
				// Update the IMGVF value
				// Compute IMGVF += (mu / lambda)(UHe .*U  + DHe .*D  + LHe .*L  + RHe .*R +
				//                                URHe.*UR + DRHe.*DR + ULHe.*UL + DLHe.*DL);
				new_val = old_val + (MU / LAMBDA) * (UHe  * U  + DHe  * D  + LHe  * L  + RHe  * R +
													 URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL);
				// Compute IMGVF -= (1 / lambda)(I .* (IMGVF - I))
				float vI = I[(i * n) + j];
				new_val -= ((1.0 / LAMBDA) * vI * (new_val - vI));
			}
			
			// Save the previous virtual thread block's value (if it exists)
			if (thread_block > 0) {
				offset = (thread_block - 1) * threads_per_block;
				if (old_i < m) IMGVF[(old_i * n) + old_j] = buffer[thread_id];
			}
			if (thread_block < max - 1) {
				// Write the new value to the buffer
				buffer[thread_id] = new_val;
			} else {
				// We've reached the final virtual thread block,
				//  so write directly to the matrix
				if (i < m) IMGVF[(i * n) + j] = new_val;
			}
			
			// do per thread sum of absolute value of the change
			partial_sums += fabs(new_val - old_val);
			
			__syncthreads();
		}
			
		// do final tree reduction across whole threadblock
		buffer[thread_id] = partial_sums;
		__syncthreads();
		
		// Account for thread block sizes that are not a power of 2
		if (thread_id >= next_lowest_power_of_two) {
			buffer[thread_id - next_lowest_power_of_two] += buffer[thread_id];
		}
		__syncthreads();
		
		// Perform the reduction
		int th;
		for (th = next_lowest_power_of_two / 2; th > 0; th /= 2) {
			if (thread_id < th) {
				buffer[thread_id] += buffer[thread_id + th];
			}
			__syncthreads();
		}
		
		// Figure out if we have converged, for early exit
		if(thread_id == 0) {
			float mean = buffer[thread_id] / (float) (m * n);
			if (mean < cutoff) cell_converged = 1;
		}
		__syncthreads();
		
		iter++;
	}
	
	// Output the final IMGVF matrix
	for (thread_block = 0; thread_block < max; thread_block++) {
		int offset = thread_block * threads_per_block;
		i = (thread_id + offset) / n;
		j = (thread_id + offset) % n;
		if (i < m) IMGVF_global[(i * n) + j] = IMGVF[(i * n) + j];
	}
}


extern "C" void IMGVF_cuda(MAT **I, MAT **IMGVF, double vx, double vy, double e, int iterations, double cutoff, int Nc);

void IMGVF_cuda(MAT **I, MAT **IMGVF, double vx, double vy, double e, int iterations, double cutoff, int Nc) {
	
	// Initialize the data on the GPU
	IMGVF_cuda_init(I, Nc);
	
	// Compute the MGVF
	IMGVF_kernel <<< Nc, threads_per_block >>> (device_IMGVF_array, device_I_array,
												device_m_array, device_n_array, 
												(float) vx, (float) vy, (float) e, cutoff);
	
	// cudaThreadSynchronize();
	// printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
	
	// Copy back the final results from the GPU
	IMGVF_cuda_cleanup(IMGVF, Nc);
}


extern "C" void IMGVF_cuda_init(MAT **IE, int Nc);

void IMGVF_cuda_init(MAT **IE, int Nc) {
	// Allocate arrays of pointers to device memory
	host_I_array = (float **) malloc(sizeof(float *) * Nc);
	host_IMGVF_array = (float **) malloc(sizeof(float *) * Nc);
	cudaMalloc( (void**) &device_I_array, Nc * sizeof(float *));
	cudaMalloc( (void**) &device_IMGVF_array, Nc * sizeof(float *));
	
	// Allocate arrays of memory dimensions
	host_m_array = (int *) malloc(sizeof(int) * Nc);
	host_n_array = (int *) malloc(sizeof(int) * Nc);
	cudaMalloc( (void**) &device_m_array, Nc * sizeof(int));
	cudaMalloc( (void**) &device_n_array, Nc * sizeof(int));
	
	// Figure out the size of all of the matrices combined
	int i, j, cell_num;
	int total_size = 0;
	for (cell_num = 0; cell_num < Nc; cell_num++) {
		MAT *I = IE[cell_num];
		int size = I->m * I->n;
		total_size += size;
	}
	total_mem_size = total_size * sizeof(float);
	
	// Allocate host memory just once for all cells
	host_I_all = (float *) malloc(total_mem_size);
	
	//Allocate device memory just once for all cells
	float *device_I_all, *device_IMGVF_all;
	cudaMalloc( (void**) &device_I_all, total_mem_size);
	cudaMalloc( (void**) &device_IMGVF_all, total_mem_size);
	
	int offset = 0;
	for (cell_num = 0; cell_num < Nc; cell_num++) {
		MAT *I = IE[cell_num];
		
		// Determine the size of the matrix
		int m = I->m, n = I->n;
		int size = m * n;
		
		// Store memory dimensions
		host_m_array[cell_num] = m;
		host_n_array[cell_num] = n;
		
		// Store pointers to allocated memory
		float *device_I = &(device_I_all[offset]);
		float *device_IMGVF = &(device_IMGVF_all[offset]);
		host_I_array[cell_num] = device_I;
		host_IMGVF_array[cell_num] = device_IMGVF;
		
		// Copy matrix I (which is also the initial IMGVF matrix) into the overall array
		for (i = 0; i < m; i++)
			for (j = 0; j < n; j++)
				host_I_all[offset + (i * n) + j] = (float) m_get_val(I, i, j);
		
		offset += size;
	}
	
	// Copy matrix I (which is also the initial IMGVF matrix) to device
	cudaMemcpy(device_I_all, host_I_all, total_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_IMGVF_all, host_I_all, total_mem_size, cudaMemcpyHostToDevice);
	
	// Copy pointer arrays to device
	cudaMemcpy(device_I_array, host_I_array, Nc * sizeof(float *), cudaMemcpyHostToDevice);
	cudaMemcpy(device_IMGVF_array, host_IMGVF_array, Nc * sizeof(float *), cudaMemcpyHostToDevice);
	
	// Copy memory dimensions to device
	cudaMemcpy(device_m_array, host_m_array, Nc * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_n_array, host_n_array, Nc * sizeof(int), cudaMemcpyHostToDevice);
}


extern "C" void IMGVF_cuda_cleanup(MAT **IMGVF_out_array, int Nc);

void IMGVF_cuda_cleanup(MAT **IMGVF_out_array, int Nc) {
	// Copy result from device to host
	cudaMemcpy(host_I_all, host_IMGVF_array[0], total_mem_size, cudaMemcpyDeviceToHost);
	
	int cell_num, offset = 0;	
	for (cell_num = 0; cell_num < Nc; cell_num++) {
		MAT *IMGVF_out = IMGVF_out_array[cell_num];
		
		// Determine the size of the matrix
		int m = IMGVF_out->m, n = IMGVF_out->n, i, j;
		// Pack the result into the matrix
		for (i = 0; i < m; i++)
			for (j = 0; j < n; j++)
				m_set_val(IMGVF_out, i, j, (double) host_I_all[offset + (i * n) + j]);
		
		offset += (m * n);
	}
	
	// Free memory
	free(host_m_array);
	free(host_n_array);
	cudaFree(device_m_array);
	cudaFree(device_n_array);
	
	cudaFree(device_IMGVF_array);
	cudaFree(device_I_array);
	cudaFree(host_IMGVF_array[0]);
	cudaFree(host_I_array[0]);
	free(host_IMGVF_array);
	free(host_I_array);
	
	free(host_I_all);
}

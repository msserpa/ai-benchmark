#include "track_ellipse_kernel.h"
#include <cutil.h>
#include <sys/time.h>
#include <time.h>

#define ONE_OVER_PI 1.0 / PI
#define MU 0.5
#define LAMBDA (8.0 * MU + 1.0)


float *host_I, *host_IMGVF;
float *device_I, *device_IMGVF_in, *device_IMGVF_out;
float *device_partial_sums, *host_partial_sums;
int *device_in_out;
unsigned int *flags;

const int threads_per_block = 128;

__global__ void IMGVF_kernel(float *IMGVF_in, float *IMGVF_out, float *I, float *partial_sums,
                             float vx, float vy, float e, int m, int n, int num_blocks,
							 float converge, int max_iterations, int *in_out, unsigned int *flags) {
	
	__shared__ float val[threads_per_block];
	__shared__ int converged;
	
	// Keep track of which matrix is more recent
	if (threadIdx.x == 0 && blockIdx.x == 0) *in_out = 1 - *in_out;
	
	// Determine the thread's coordinates
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = thread_id / n;
	int j = thread_id % n;
	
	// Start computing the iterative approximation
	int iterations = 0;
	do {		
		// --------------------------------------------
		// Compute the new value of this matrix element
		// --------------------------------------------
		
		float new_val = 0.0, old_val = 0.0;
		
		if (i < m) {
			// Compute neighboring pixel indices
			int rowU = (i == 0) ? 0 : i - 1;
			int rowD = (i == m - 1) ? m - 1 : i + 1;
			int colL = (j == 0) ? 0 : j - 1;
			int colR = (j == n - 1) ? n - 1 : j + 1;
			
			// Compute the difference between the pixel and its eight neighbors
			old_val = IMGVF_in[(i * n) + j];
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
			new_val = old_val + (MU / LAMBDA) * (UHe  * U  + DHe  * D  + LHe  * L  + RHe  * R +
												 URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL);
			// Compute IMGVF -= (1 / lambda)(I .* (IMGVF - I))
			float vI = I[(i * n) + j];
			new_val -= ((1.0 / LAMBDA) * vI * (new_val - vI));
			IMGVF_out[(i * n) + j] = new_val;
		}
		
		// Swap the input and output matrix pointers
		float *temp = IMGVF_in;
		IMGVF_in = IMGVF_out;
		IMGVF_out = temp;
		
		
		// ---------------------------------------------------------------------
		// Compute this block's contribution to the global convergence criterion
		// ---------------------------------------------------------------------
		
		// Sum the absolute values of the differences
		//  across the entire thread block
		val[threadIdx.x] = fabs(new_val - old_val);
		__syncthreads();
		
		// Perform the reduction
		int th;
		for (th = threads_per_block / 2; th > 0; th /= 2) {
			if (threadIdx.x < th) val[threadIdx.x] += val[threadIdx.x + th];
			__syncthreads();
		}
		
		if (threadIdx.x == 0) {
			// Save the final value to global memory
			partial_sums[blockIdx.x] = val[0];
		}
		
		// Global memory fence to ensure that this thread's update to the MGVF matrix is visible to other blocks
		__threadfence();
		
		if (threadIdx.x == 0) {		
			// Set a flag to indicate that this block is done
			atomicInc(&(flags[iterations]), num_blocks + 1);
			
			// Wait until all blocks are done
			while (flags[iterations] < num_blocks) {}
		}		
		
		__syncthreads();
		
		
		// -----------------------------------------------
		// Determine whether the computation has converged
		// -----------------------------------------------

		// Note: if the number of blocks is greater than the number of threads
		//  per block, then this will not work (so if threads_per_block < 64)
		// Load all of the partial_sums into shared memory
		if (threadIdx.x < num_blocks) val[threadIdx.x] = partial_sums[threadIdx.x];
		__syncthreads();
		
		// Thread zero figure out if we have converged
		if (threadIdx.x == 0) {
			int k;
			float sum = 0.0;
			for (k = 0; k < num_blocks; k++) sum += val[k];
			float mean = sum / (float) (m * n);
			if (mean < converge) converged = 1;
			else                 converged = 0;
		}
		__syncthreads();
	
	// If we just figured out that we have converged then return
	} while ((! converged) && (iterations < max_iterations));
}


extern "C" void IMGVF_cuda(MAT *I, MAT *IMGVF, double vx, double vy, double e, int max_iterations, double cutoff);

void IMGVF_cuda(MAT *I, MAT *IMGVF, double vx, double vy, double e, int max_iterations, double cutoff) {
	
	// Initialize the data on the GPU
	IMGVF_cuda_init(I, max_iterations);

	// Determine thread block size
	int m = IMGVF->m, n = IMGVF->n;
	int num_threads = m * n;
	int num_blocks = (int) (num_threads + threads_per_block - 1) / threads_per_block;
	
	// Compute the MGVF
	IMGVF_kernel <<< num_blocks, threads_per_block >>> (device_IMGVF_in, device_IMGVF_out, device_I,
														device_partial_sums, (float) vx, (float) vy, (float) e, m, n,
														num_blocks, cutoff, max_iterations, device_in_out, flags);
	
	// Copy back the final results from the GPU
	IMGVF_cuda_cleanup(IMGVF);
}


extern "C" void IMGVF_cuda_init(MAT *I, int max_iterations);

void IMGVF_cuda_init(MAT *I, int max_iterations) {
	// Compute the amount of memory required and the number of thread blocks
	int m = I->m, n = I->n, i, j;
	int mem_size = sizeof(float) * m * n;
	int num_threads = m * n;
	int num_blocks = (int) (((float) num_threads / (float) threads_per_block) + 0.5);
	
	//Allocate device memory
	cudaMalloc( (void**) &device_IMGVF_in, mem_size);
	cudaMalloc( (void**) &device_IMGVF_out, mem_size);
	cudaMalloc( (void**) &device_I, mem_size);
	cudaMalloc( (void**) &device_partial_sums, num_blocks*sizeof(float));
	cudaMalloc( (void**) &device_in_out, sizeof(int));
	cudaMalloc( (void**) &flags, sizeof(int) * max_iterations);
	
	// Allocate host memory
	host_IMGVF = (float *) malloc(mem_size);
	host_I = (float *) malloc(mem_size);
	host_partial_sums = (float *) malloc(num_blocks*sizeof(float));
	
	// Copy matrix I (which is also the initial IMGVF matrix) to device
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			host_I[(i * n) + j] = (float) m_get_val(I, i, j);
	cudaMemcpy(device_I, host_I, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_IMGVF_in, host_I, mem_size, cudaMemcpyHostToDevice);
	
	// Set up convergence variables
	cudaMemset(device_in_out, 0, sizeof(int));
	cudaMemset(flags, 0, sizeof(int) * max_iterations);
	
	// Initialize partial sums so that we do not converge on the first kernel call
	for (i = 0; i < num_blocks; i++) host_partial_sums[i] = 1.0;
	cudaMemcpy(device_partial_sums, host_partial_sums, num_blocks * sizeof(float), cudaMemcpyHostToDevice);
}


extern "C" void IMGVF_cuda_cleanup(MAT *IMGVF_out);

void IMGVF_cuda_cleanup(MAT *IMGVF_out) {
	// Compute the amount of memory required
	int m = IMGVF_out->m, n = IMGVF_out->n, i, j;
	int mem_size = sizeof(float) * m * n;
	
	// Determine which array to copy from
	int host_in_out;
	cudaMemcpy(&host_in_out, device_in_out, sizeof(int), cudaMemcpyDeviceToHost);
	float *IMGVF;
	if (host_in_out) IMGVF = device_IMGVF_out;
	else             IMGVF = device_IMGVF_in;
	
	// Copy result from device to host
	cudaMemcpy(host_IMGVF, IMGVF, mem_size, cudaMemcpyDeviceToHost);
	// Pack the result into the matrix
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			m_set_val(IMGVF_out, i, j, (double) host_IMGVF[(i * n) + j]);

	// Free memory
	free(host_IMGVF);
	free(host_I);
	free(host_partial_sums);
	cudaFree(device_IMGVF_in);
	cudaFree(device_IMGVF_out);
	cudaFree(device_I);
	cudaFree(device_partial_sums);
	cudaFree(device_in_out);
	cudaFree(flags);
}

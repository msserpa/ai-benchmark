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
int *device_converged, *device_in_out;

const int threads_per_block = 64;

__global__ void IMGVF_kernel(float *IMGVF_in, float *IMGVF_out, float *I, float *device_partial_sums, int *converged, float vx, float vy, float e, int m, int n) {
	if (*converged == 1) return;

	// Determine the thread's coordinates
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = thread_id / n;
	int j = thread_id % n;
	
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
	
	// Sum the absolute values of the differences
	//  across the entire thread block
	__shared__ float val[threads_per_block];
	val[threadIdx.x] = fabs(new_val - old_val);
	__syncthreads();
	
	// Perform the reduction
	int th;
	for (th = threads_per_block / 2; th > 0; th /= 2) {
		if (threadIdx.x < th) val[threadIdx.x] += val[threadIdx.x + th];
		__syncthreads();
	}
	
	// Save the final value
	if (threadIdx.x == 0) device_partial_sums[blockIdx.x] = val[0];
}


__global__ void reduce_kernel(float *partial_sums, int num_blocks, int num_threads, float converge, int *converged, int *in_out) {
	if (*converged == 1) return;

	__shared__ float val[256];
	val[threadIdx.x] = partial_sums[threadIdx.x];
	
	if (threadIdx.x == 0) {
		*in_out = 1 - *in_out;
		int i;
		float sum = 0.0;
		for (i = 0; i < num_blocks; i++) sum += val[i];
		float mean = sum / (float) num_threads;
		if (mean < converge) *converged = 1;
	}
}


extern "C" void IMGVF_cuda(MAT *I, MAT *IMGVF, double vx, double vy, double e, int iterations, double cutoff);

// Note: about 2/3 of the execution time is spent in the kernel
//       the other 1/3 is spent copying memory back and forth
void IMGVF_cuda(MAT *I, MAT *IMGVF, double vx, double vy, double e, int iterations, double cutoff) {
	
	// Initialize the data on the GPU
	IMGVF_cuda_init(I);

	// Determine thread block size
	int m = IMGVF->m, n = IMGVF->n;
	int num_threads = m * n;
	int num_blocks = (int) (((float) num_threads / (float) threads_per_block) + 0.5);
    
	// struct timeval tv;
    // gettimeofday(&tv, NULL); 
    // long long loop_start_time = tv.tv_sec*1000000 + tv.tv_usec;
	// long long copy_time = 0;
	
	// Compute the MGVF
	int host_converged = 0, iter = 0;
	while ((iter < iterations) && (! host_converged)) {
		// Execute the kernel multiple times
		int i, unroll = 16;
		for (i = 0; i < unroll; i++) {
			IMGVF_kernel <<< num_blocks, threads_per_block >>> (device_IMGVF_in, device_IMGVF_out, device_I, device_partial_sums, device_converged, (float) vx, (float) vy, (float) e, m, n);
			reduce_kernel <<< 1, num_blocks >>> (device_partial_sums, num_blocks, num_threads, (float) cutoff, device_converged, device_in_out);
			
			IMGVF_kernel <<< num_blocks, threads_per_block >>> (device_IMGVF_out, device_IMGVF_in, device_I, device_partial_sums, device_converged, (float) vx, (float) vy, (float) e, m, n);
			reduce_kernel <<< 1, num_blocks >>> (device_partial_sums, num_blocks, num_threads, (float) cutoff, device_converged, device_in_out);
		}
		
		// cudaThreadSynchronize();
		// gettimeofday(&tv, NULL); long long copy_start_time = tv.tv_sec*1000000 + tv.tv_usec;
		cudaMemcpy(&host_converged, device_converged, sizeof(int), cudaMemcpyDeviceToHost);
		// gettimeofday(&tv, NULL); long long copy_end_time = tv.tv_sec*1000000 + tv.tv_usec;
		// copy_time += copy_end_time - copy_start_time;
		
		iter += (unroll * 2);
	}
	
	// gettimeofday(&tv, NULL); 
    // long long loop_end_time = tv.tv_sec*1000000 + tv.tv_usec;
	// printf("kernel,copy:");
	// printf(" %.8f,", ((float) (loop_end_time - loop_start_time - copy_time)) / (1000*1000));
	// printf(" %.8f\n", ((float) (copy_time)) / (1000*1000));
	
	// Copy back the final results from the GPU
	IMGVF_cuda_cleanup(IMGVF);
}


extern "C" void IMGVF_cuda_init(MAT *I);

void IMGVF_cuda_init(MAT *I) {
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
	cudaMalloc( (void**) &device_converged, sizeof(int));
	cudaMalloc( (void**) &device_in_out, sizeof(int));
	
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
	int boolean = 0;
	cudaMemcpy(device_converged, &boolean, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_in_out, &boolean, sizeof(int), cudaMemcpyHostToDevice);
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
	cudaFree(device_converged);
	cudaFree(device_in_out);
}

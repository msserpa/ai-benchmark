#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cutil.h>
#include <iostream>
#include <bucketsort.cuh>
#include <mergesort.cuh>
using namespace std; 

////////////////////////////////////////////////////////////////////////////////
// Size of the testset 
////////////////////////////////////////////////////////////////////////////////
#define SIZE	(1 << 22)
////////////////////////////////////////////////////////////////////////////////
// Number of tests to average over
////////////////////////////////////////////////////////////////////////////////
#define TEST	4
////////////////////////////////////////////////////////////////////////////////
// The timers for the different parts of the algo
////////////////////////////////////////////////////////////////////////////////
unsigned int uploadTimer, downloadTimer, bucketTimer, 
			 mergeTimer, totalTimer, cpuTimer; 
////////////////////////////////////////////////////////////////////////////////
// Compare method for CPU sort
////////////////////////////////////////////////////////////////////////////////
inline int compare(const void *a, const void *b) {
	if(*((float *)a) < *((float *)b)) return -1; 
	else if(*((float *)a) > *((float *)b)) return 1; 
	else return 0; 
}
////////////////////////////////////////////////////////////////////////////////
// Forward declaration
////////////////////////////////////////////////////////////////////////////////
void cudaSort(float *origList, float minimum, float maximum,
			  float *resultList, int numElements);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{
	CUT_DEVICE_INIT(argc, argv); 
    CUT_SAFE_CALL(cutCreateTimer(&uploadTimer));
    CUT_SAFE_CALL(cutCreateTimer(&downloadTimer));
    CUT_SAFE_CALL(cutCreateTimer(&bucketTimer));
    CUT_SAFE_CALL(cutCreateTimer(&mergeTimer));
    CUT_SAFE_CALL(cutCreateTimer(&totalTimer));
    CUT_SAFE_CALL(cutCreateTimer(&cpuTimer));

	int numElements = SIZE; 
	cout << "Sorting list of " << numElements << " floats\n";

	// Generate random data
	int mem_size = numElements * sizeof(float); 
	float *cpu_idata = (float *)malloc(mem_size);
	float *cpu_odata = (float *)malloc(mem_size);
	float *gpu_odata = (float *)malloc(mem_size);

	float datamin = FLT_MAX; 
	float datamax = -FLT_MAX; 
	for (int i = 0; i < numElements; i++) {
		cpu_idata[i] = ((float) rand() / RAND_MAX); 
		datamin = min(cpu_idata[i], datamin);
		datamax = max(cpu_idata[i], datamax);
	}

	cout << "Sorting on GPU..." << flush; 
	// GPU Sort
	for (int i = 0; i < TEST; i++) 
		cudaSort(cpu_idata, datamin, datamax, gpu_odata, numElements);		
	cout << "done.\n";

	cout << "Sorting on CPU..." << flush; 
	// CPU Sort
	memcpy(cpu_odata, cpu_idata, mem_size); 		
	cutStartTimer(cpuTimer); 
		qsort(cpu_odata, numElements, sizeof(float), compare);
	cutStopTimer(cpuTimer); 
	cout << "done.\n";


	cout << "Checking result..." << flush; 
	// Result checking
	int count = 0; 
	for(int i = 0; i < numElements; i++)
		if(cpu_odata[i] != gpu_odata[i])
		{
			printf("Sort missmatch on element %d: \n", i); 
			printf("CPU = %f : GPU = %f\n", cpu_odata[i], gpu_odata[i]); 
			count++; 
			break; 
		}
	if(count == 0) cout << "PASSED.\n";
	else cout << "FAILED.\n";

	// Timer report
	printf("GPU iterations: %d\n", TEST); 
	printf("Average CPU execution time: %f ms\n", cutGetTimerValue(cpuTimer));
	printf("Average GPU execution time: %f ms\n", cutGetTimerValue(totalTimer) / TEST);
	printf("    - Upload		: %f ms\n", cutGetTimerValue(uploadTimer) / TEST);
	printf("    - Download		: %f ms\n", cutGetTimerValue(downloadTimer) / TEST);
	printf("    - Bucket sort	: %f ms\n", cutGetTimerValue(bucketTimer) / TEST);
	printf("    - Merge sort	: %f ms\n", cutGetTimerValue(mergeTimer) / TEST);
	
	// Release memory
    cutDeleteTimer(uploadTimer);
    cutDeleteTimer(downloadTimer);
    cutDeleteTimer(bucketTimer);
    cutDeleteTimer(mergeTimer);
    cutDeleteTimer(totalTimer);
    cutDeleteTimer(cpuTimer);
	free(cpu_idata); free(cpu_odata); free(gpu_odata); 
	CUT_EXIT(argc, argv);
}

void cudaSort(float *origList, float minimum, float maximum,
			  float *resultList, int numElements)
{
	// Initialization and upload data
	float *d_input  = NULL; 
	float *d_output = NULL; 
	int mem_size = (numElements + DIVISIONS * 4) * sizeof(float); 
	cutStartTimer(uploadTimer);
	{
		cudaMalloc((void**) &d_input, mem_size);
		cudaMalloc((void**) &d_output, mem_size);
		cudaMemcpy((void *) d_input, (void *)origList, numElements * sizeof(float),
				   cudaMemcpyHostToDevice);
		init_bucketsort(numElements); 
	}
	cutStopTimer(uploadTimer); 
	CUT_CHECK_ERROR("Upload data"); 

	cutStartTimer(totalTimer); 

	// Bucketsort the list
	cutStartTimer(bucketTimer); 
		int *sizes = (int*) malloc(DIVISIONS * sizeof(int)); 
		int *nullElements = (int*) malloc(DIVISIONS * sizeof(int));  
		unsigned int *origOffsets = (unsigned int *) malloc((DIVISIONS + 1) * sizeof(int)); 
		bucketSort(d_input, d_output, numElements, sizes, nullElements, 
				   minimum, maximum, origOffsets); 
	cutStopTimer(bucketTimer); 

	// Mergesort the result
	cutStartTimer(mergeTimer); 
		float4 *d_origList = (float4*) d_output, 
		*d_resultList = (float4*) d_input;
		int newlistsize = 0; 
	
		for(int i = 0; i < DIVISIONS; i++)
			newlistsize += sizes[i] * 4;

		float4 *mergeresult = runMergeSort(	newlistsize, DIVISIONS, d_origList, d_resultList, 
			sizes, nullElements, origOffsets); //d_origList; 
		cudaThreadSynchronize(); 
	cutStopTimer(mergeTimer); 
	cutStopTimer(totalTimer); 

	// Download result
	cutStartTimer(downloadTimer); 
		CUDA_SAFE_CALL(	cudaMemcpy((void *) resultList, 
				(void *)mergeresult, numElements * sizeof(float), cudaMemcpyDeviceToHost) );
	cutStopTimer(downloadTimer); 

	// Clean up
	finish_bucketsort(); 
	cudaFree(d_input); cudaFree(d_output); 
	free(nullElements); free(sizes); 
}

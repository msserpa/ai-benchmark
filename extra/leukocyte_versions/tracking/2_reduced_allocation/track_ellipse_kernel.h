#ifndef _TRACK_ELLIPSE_KERNEL_H_
#define _TRACK_ELLIPSE_KERNEL_H_

#include "matrix.h"
#include "misc_math.h"


// We don't want the "C" when compiling track_ellipse.c
#ifdef COMPILING_TRACK_ELLIPSE_C
extern void IMGVF_cuda_init(MAT *I);
extern void IMGVF_cuda_cleanup();
extern void IMGVF_cuda(MAT *IMGVF_in, MAT *IMGVF_out, double vx, double vy, double e);
# else
extern "C" void IMGVF_cuda_init(MAT *I);
extern "C" void IMGVF_cuda_cleanup();
extern "C" void IMGVF_cuda(MAT *IMGVF_in, MAT *IMGVF_out, double vx, double vy, double e);
#endif

extern void IMGVF_kernel(float *IMGVF_in, float *IMGVF_out, float *I, float vx, float vy, float e, int m, int n);


#endif

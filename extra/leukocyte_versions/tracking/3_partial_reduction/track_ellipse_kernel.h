#ifndef _TRACK_ELLIPSE_KERNEL_H_
#define _TRACK_ELLIPSE_KERNEL_H_

#include "matrix.h"
#include "misc_math.h"


// We don't want the "C" when compiling track_ellipse.c
#ifdef COMPILING_TRACK_ELLIPSE_C
extern void IMGVF_cuda_init(MAT *I, MAT *IMGVF_in);
extern void IMGVF_cuda_cleanup(MAT *IMGVF_out);
extern float IMGVF_cuda(MAT *IMGVF, double vx, double vy, double e);
# else
extern "C" void IMGVF_cuda_init(MAT *I, MAT *IMGVF_in);
extern "C" void IMGVF_cuda_cleanup(MAT *IMGVF_out);
extern "C" float IMGVF_cuda(MAT *IMGVF, double vx, double vy, double e);
#endif

extern void IMGVF_kernel(float *IMGVF_in, float *IMGVF_out, float *I, float vx, float vy, float e, int m, int n);


#endif

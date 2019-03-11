#ifndef _TRACK_ELLIPSE_KERNEL_H_
#define _TRACK_ELLIPSE_KERNEL_H_

#include "matrix.h"
#include "misc_math.h"


// We don't want the "C" when compiling track_ellipse.c
#ifdef COMPILING_TRACK_ELLIPSE_C
extern void heaviside_cuda(MAT *H, MAT *z, double v, double e);
extern void heavyside_init(int n, int m);
extern void heavyside_cleanup();
extern void IMGVF_cuda(MAT *IMGVF_in, MAT *IMGVF_out, double vx, double vy, double e);
# else
extern "C" void heaviside_cuda(MAT *H, MAT *z, double v, double e);
extern "C" void heavyside_init(int n, int m);
extern "C" void heavyside_cleanup();
extern "C" void IMGVF_cuda(MAT *IMGVF_in, MAT *IMGVF_out, double vx, double vy, double e);
#endif

extern void IMGVF_kernel(float *IMGVF_in, float *IMGVF_out, float vx, float vy, float e, int m, int n);
extern void heaviside_kernel(float *H, float *z, float v, float e, int m, int n);


#endif

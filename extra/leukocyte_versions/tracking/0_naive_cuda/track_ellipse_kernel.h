#ifndef _TRACK_ELLIPSE_KERNEL_H_
#define _TRACK_ELLIPSE_KERNEL_H_

#include "matrix.h"
#include "misc_math.h"


// We don't want the "C" when compiling track_ellipse.c
#ifdef COMPILING_TRACK_ELLIPSE_C
extern void heaviside_cuda(MAT *H, MAT *z, double v, double e);
# else
extern "C" void heaviside_cuda(MAT *H, MAT *z, double v, double e);
#endif

extern void heaviside_kernel(float *H, float *z, float v, float e, int m, int n);


#endif

#ifndef _TRACK_ELLIPSE_KERNEL_H_
#define _TRACK_ELLIPSE_KERNEL_H_

#include "matrix.h"
#include "misc_math.h"


// We don't want the "C" when compiling track_ellipse.c
#ifdef COMPILING_TRACK_ELLIPSE_C
extern void IMGVF_cuda_init(MAT **I, int Nc);
extern void IMGVF_cuda_cleanup(MAT **IMGVF_out, int Nc);
extern void IMGVF_cuda(MAT **I, MAT **IMGVF, double vx, double vy, double e, int iterations, double cutoff, int Nc);
# else
extern "C" void IMGVF_cuda_init(MAT **I, int Nc);
extern "C" void IMGVF_cuda_cleanup(MAT **IMGVF_out, int Nc);
extern "C" void IMGVF_cuda(MAT **I, MAT **IMGVF, double vx, double vy, double e, int iterations, double cutoff, int Nc);
#endif


#endif

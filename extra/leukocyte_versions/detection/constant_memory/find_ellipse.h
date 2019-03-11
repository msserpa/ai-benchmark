#ifndef FIND_ELLIPSE_H
#define FIND_ELLIPSE_H

#include "matrix.h"

#define CUDA

extern MAT * chop_image(const MAT * image, int top, int bottom, int left, int right);
extern MAT * ellipsematching(MAT * grad_x, MAT * grad_y);
extern void choose_GPU();
extern void compute_constants();
#ifdef CUDA
extern float * structuring_element(int radius);
#else
extern MAT * structuring_element(int radius);
#endif
extern MAT * dilate_f(MAT * img_in);
extern MAT * TMatrix(unsigned int N, unsigned int M);
extern void uniformseg(VEC * cellx_row, VEC * celly_row, MAT * x, MAT * y);
extern double m_min(MAT * m);
extern double m_max(MAT * m);
extern VEC * getsampling(MAT * m, int ns);
extern VEC * getfdriv(MAT * m, int ns);
extern MAT * linear_interp2(MAT * m, VEC * X, VEC * Y);
extern void splineenergyform01(MAT * Cx, MAT * Cy, MAT * Ix, MAT * Iy, int ns, double delta, double dt, int typeofcell);

#endif

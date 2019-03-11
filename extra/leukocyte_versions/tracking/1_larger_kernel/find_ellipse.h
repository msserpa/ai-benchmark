#ifndef FIND_ELLIPSE_H
#define FIND_ELLIPSE_H

#include "avilib.h"
#include "matrix.h"
#include "misc_math.h"
#include <math.h>
#include <stdlib.h>

// Defines the region in the video frame containing the blood vessel
#define TOP 110
#define BOTTOM 328

extern MAT * chop_image(const MAT * image, int top, int bottom, int left, int right);
extern MAT * get_frame(avi_t *cell_file, int frame_num, int cropped, int scaled);
extern MAT * chop_flip_image(unsigned char *image, int height, int width, int top, int bottom, int left, int right, int scaled);
extern void choose_GPU();
extern void compute_constants();
extern MAT * ellipsematching(MAT * grad_x, MAT * grad_y);
extern float * structuring_element(int radius);
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

#ifndef _FIND_ELLIPSE_KERNEL_H_
#define _FIND_ELLIPSE_KERNEL_H_


// We don't want the "C" when compling find_ellipse.c
#ifdef COMPILING_FIND_ELLIPSE_C
extern float *ellipsematching_setup(int grad_m, int grad_n, float *host_grad_x,float *host_grad_y, float *host_sin_angle, float *host_cos_angle, int *host_tX, int *host_tY);
extern float *dilate_f_setup(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n, float *host_max_gicov, float *host_strel);
extern void select_device();
# else
extern "C" float *ellipsematching_setup(int grad_m, int grad_n, float *host_grad_x, float *host_grad_y, float *host_sin_angle, float *host_cos_angle, int *host_tX, int *host_tY);
extern "C" float *dilate_f_setup(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n, float *host_max_gicov, float *host_strel);
extern "C" void select_device();
#endif

extern void ellipsematching_kernel(float * grad_x, float * grad_y, int grad_n, float * sin_angle, float * cos_angle, int * tX, int * tY, float * gicov);
extern void dilate_f_kernel(float *img, int img_m, int img_n, float *strel, int strel_m, int strel_n, float *dilated);



#endif

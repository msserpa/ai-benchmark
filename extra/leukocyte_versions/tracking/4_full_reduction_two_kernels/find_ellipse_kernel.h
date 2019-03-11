#ifndef _FIND_ELLIPSE_KERNEL_H_
#define _FIND_ELLIPSE_KERNEL_H_


// We don't want the "C" when compling find_ellipse.c
#ifdef COMPILING_FIND_ELLIPSE_C
extern float *ellipsematching_setup(int grad_m, int grad_n, float *host_grad_x, float *host_grad_y);
extern float *dilate_f_setup(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n);
extern void select_device();
extern void transfer_constants(float *host_sin_angle, float *host_cos_angle, int *host_tX, int *host_tY, int strel_m, int strel_n, float *host_strel);
# else
extern "C" float *ellipsematching_setup(int grad_m, int grad_n, float *host_grad_x, float *host_grad_y);
extern "C" float *dilate_f_setup(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n);
extern "C" void select_device();
extern "C" void transfer_constants(float *host_sin_angle, float *host_cos_angle, int *host_tX, int *host_tY, int strel_m, int strel_n, float *host_strel);
#endif


extern void ellipsematching_kernel(float * grad_x, float * grad_y, int grad_m, int grad_n, float * gicov);
extern void dilate_f_kernel(float *img, int img_m, int img_n, int strel_m, int strel_n, float *dilated);


#endif

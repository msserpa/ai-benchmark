#include "find_ellipse.h"
#include "misc_math.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>

#define COMPILING_FIND_ELLIPSE_C
#include "find_ellipse_kernel.h"

#define NPOINTS 150
#define RADIUS 10
#define MIN_RAD RADIUS - 2
#define MAX_RAD RADIUS * 2
#define NCIRCLES 10


extern MAT * m_inverse(MAT * A, MAT * out);

void swap(unsigned int *x, unsigned int *y)
{
    unsigned int temp;

    temp = *x;
    *x = *y;
    *y = temp;
}

MAT * chop_image(const MAT * image, int top, int bottom, int left, int right)
{
	MAT * result = m_get(bottom-top+1, right-left+1);
	int i, j;

	for(i = top; i <= bottom; i++)
	{
		for(j = left; j <= right; j++)
		{
			m_set_val(result, i-top, j-left, m_get_val(image, i, j));
		}
	}

	return result;
}

// Choose the best GPU on the current machine
void choose_GPU() {
	select_device();
}

#ifdef CUDA

MAT * ellipsematching(MAT * grad_x, MAT * grad_y) {
  int grad_m = grad_x->m;
  int grad_n = grad_y->n;

  int k, m, n;
  int MaxR = MAX_RAD + 2;
  
  unsigned int grad_mem_size = sizeof(float) * grad_m * grad_n;
  unsigned int angle_mem_size = sizeof(float) * NPOINTS;
  unsigned int t_mem_size = sizeof(int) * NCIRCLES * NPOINTS;
  
  
  // allocate host memory for grad_x and grad_y
  float* host_grad_x = (float*) malloc(grad_mem_size);
  float* host_grad_y = (float*) malloc(grad_mem_size);
  // initalize float versions of grad_x and grad_y
  for (m = 0; m < grad_m; m++) {
    for (n = 0; n < grad_n; n++) {
      host_grad_x[(m * grad_n) + n] = (float) m_get_val(grad_x, m, n);
      host_grad_y[(m * grad_n) + n] = (float) m_get_val(grad_y, m, n);
	  /*host_grad_x[(n * grad_m) + m] = (float) m_get_val(grad_x, m, n);
      host_grad_y[(n * grad_m) + m] = (float) m_get_val(grad_y, m, n);*/
    }
  }
  
  // precompute sin_angle and cos_angle
  float host_sin_angle[NPOINTS], host_cos_angle[NPOINTS], theta[NPOINTS];
  for(n = 0; n < NPOINTS; n++) {
    theta[n] = (((float) n) * 2.0 * PI) / ((float) NPOINTS);
    host_sin_angle[n] = sin(theta[n]);
    host_cos_angle[n] = cos(theta[n]);
  }
  
  // precompute tX and tY
  int host_tX[NCIRCLES * NPOINTS], host_tY[NCIRCLES * NPOINTS];
  //int ncircle = (MAX_RAD - MIN_RAD + 2) / 2;
  for (k = 0; k < NCIRCLES; k++) {
    float rad = (float) (MIN_RAD + (2 * k)); 
    for (n = 0; n < NPOINTS; n++) {
      host_tX[(k * NPOINTS) + n] = (int)(cos(theta[n]) * rad);
      host_tY[(k * NPOINTS) + n] = (int)(sin(theta[n]) * rad);
    }
  }

  
  // compute result
  float *host_gicov = ellipsematching_setup(grad_m, grad_n, host_grad_x, host_grad_y, host_sin_angle, host_cos_angle, host_tX, host_tY);
  
  // copy result into a new matrix
  MAT * gicov = m_get(grad_m, grad_n);
  for (m = 0; m < grad_m; m++)
    for (n = 0; n < grad_n; n++)
      m_set_val(gicov, m, n, host_gicov[(m * grad_n) + n]);
	  //m_set_val(gicov, m, n, host_gicov[(n * grad_m) + m]);
  
  
  // cleanup memory
  free(host_gicov);
	
  return gicov;
}

#else

//Find matrix of GICOV values at all given pixels given x and y gradients of the image
MAT * ellipsematching(MAT * grad_x, MAT * grad_y) {
	//number of points in ellipse
	int npoints = NPOINTS;

	MAT * gicov = m_get(grad_x->m, grad_y->n);
	float rad, Grad[NPOINTS];
	float sum, ep, ave, var, sGicov;
	float sin_angle[NPOINTS], cos_angle[NPOINTS], theta[NPOINTS];
	int i, j, k, n, x, y, iIndex, ncircle, MaxR, minRad = MIN_RAD, maxRad = MAX_RAD, height = grad_x->m, width = grad_x->n;
	int tX[NCIRCLES][NPOINTS], tY[NCIRCLES][NPOINTS]; 

	for(n = 0; n < npoints; n++) {
		theta[n] = (float)n * 2.0 * PI / (float)npoints;
		sin_angle[n] = sin(theta[n]);
		cos_angle[n] = cos(theta[n]);
	}


	ncircle = (int)(maxRad-minRad+2)/2;


	for (k=0; k<ncircle; k++)
	{
		rad = (float)(minRad + 2*k); 
		//printf("the number is %f\n", rad);
		for (n=0; n<npoints; n++)
		{
			//theta[n] = (float)n * 2 * PI / npoints;
			tX[k][n] = (int)(cos(theta[n])*rad);
			tY[k][n] = (int)(sin(theta[n])*rad);

		}

	}

	MaxR = maxRad + 2;

	//Scan from left to right, top to bottom, getting GICOV values
	for(i = MaxR; i < width-MaxR; i++)
	{
		for(j = MaxR; j < height - MaxR; j++)
		{
			sGicov = 0;
			
			for(k = 0; k < ncircle; k++)
			{
				for(n = 0; n< npoints; n++)
				{
					y = j + tY[k][n];
					x = i + tX[k][n];

					Grad[n] = m_get_val(grad_x, y, x) * cos_angle[n] + m_get_val(grad_y, y, x) * sin_angle[n];
				}
				sum = 0.0;
				ep = 0.0;

				for(iIndex = 0; iIndex < npoints; iIndex++)	sum+=Grad[iIndex];

				ave = sum/(float)npoints;
				var = 0.0;

				for(iIndex = 0; iIndex < npoints; iIndex++)
				{
					sum = Grad[iIndex] - ave;
					var += sum*sum;
					ep+=sum;
				}

				var = (var - ep*ep/(float)npoints) / (float)(npoints-1);

				if(ave*ave/var > sGicov)
				{
					m_set_val(gicov, j, i, ave/sqrt(var));
					sGicov = ave*ave/var;
				}
			}
		}
	}
	return gicov;
}

#endif


#ifdef CUDA

float * structuring_element(int radius) {
	int m = radius*2+1;
	int n = radius*2+1;
	float *result = (float *) malloc(sizeof(float) * m * n);
	
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (sqrt((float)((i-radius)*(i-radius)+(j-radius)*(j-radius))) <= radius)
				//m_set_val_cuda(result, i, j, 1.0, m);
				result[(i * m) + j] = 1.0;
			else
				//m_set_val_cuda(result, i, j, 0.0, m);
				result[(i * m) + j] = 0.0;
		}
	}

	return result;
}

#else

//Return a circular structuring element of radius "radius"
MAT * structuring_element(int radius)
{
	MAT * result = m_get(radius*2+1, radius*2+1);
	int i, j;

	for(i = 0; i < result->m; i++)
	{
		for(j = 0; j < result->n; j++)
		{
			if(sqrt((i-radius)*(i-radius)+(j-radius)*(j-radius)) <= radius)
			{
				m_set_val(result, i, j, 1.0);
			}
			else	m_set_val(result, i, j, 0.0);
		}
	}

	return result;

}

#endif


#ifdef CUDA

MAT * dilate_f(MAT * img_in) {
  int max_gicov_m = img_in->m;
  int max_gicov_n = img_in->n;
  
  int strel_m = 12 * 2 + 1;  //strel->m;
  int strel_n = 12 * 2 + 1;  //strel->n;
  
  unsigned int max_gicov_mem_size = sizeof(float) * max_gicov_m * max_gicov_n;
  unsigned int strel_mem_size = sizeof(float) * strel_m * strel_n;
    
  // allocate host memory for max_gicov
  float *host_max_gicov = (float*) malloc(max_gicov_mem_size);
  // initalize the memory
  int m, n;
  for (m = 0; m < max_gicov_m; m++)
    for (n = 0; n < max_gicov_n; n++)
      host_max_gicov[(m*max_gicov_n) + n] = (float) m_get_val(img_in, m, n);
  
  // allocate host memory for strel
  float *host_strel = structuring_element(12);
  
  float *host_img_dilated = dilate_f_setup(max_gicov_m, max_gicov_n, strel_m, strel_n, host_max_gicov, host_strel);
  
  // write result to matrix
  MAT * dilated = m_get(max_gicov_m, max_gicov_n);
  for (m = 0; m < max_gicov_m; m++)
    for (n = 0; n < max_gicov_n; n++)
      m_set_val(dilated, m, n, host_img_dilated[(m * max_gicov_n) + n]);
  
  // cleanup memory
  free(host_img_dilated);
  
  return dilated;
}

#else

//Perform grayscale dilation on img_in using the provided sturcturing element
MAT * dilate_f(MAT * img_in)
{
  MAT *strel = structuring_element(12);

	int i, j, el_i, el_j, x, y, el_center_i = strel->m/2, el_center_j = strel->n/2;
	float max, temp;
	MAT * dilated = m_get(img_in->m, img_in->n);

	for(i = 0; i < img_in->m; i++)
	{
		for(j = 0; j < img_in->n; j++)
		{
			max = 0.0;

			for(el_i = 0; el_i < strel->m; el_i++)
			{
				for(el_j = 0; el_j < strel->n; el_j++)
				{
					y = i - el_center_i + el_i;
					x = j - el_center_j + el_j;
					if(y >=0 && x >= 0 && y < img_in->m && x < img_in->n && m_get_val(strel, el_i, el_j)!=0)
					{
						temp = m_get_val(img_in, y, x);
						if(temp > max)	max = temp;
					}
				}
			}
			m_set_val(dilated, i, j, max);
		}
	}

	return dilated;
}

#endif


//M = # of sampling points in each segment
//N = number of segment of curve
//Get special TMatrix
MAT * TMatrix(unsigned int N, unsigned int M)
{
	MAT * B = NULL, * LB = NULL, * B_TEMP = NULL, * B_TEMP_INV = NULL, * B_RET = NULL;
	int * aindex, * bindex, * cindex, * dindex;
	int i, j;

	aindex = (int*) malloc(N*sizeof(int));
	bindex = (int*) malloc(N*sizeof(int));
	cindex = (int*) malloc(N*sizeof(int));
	dindex = (int*) malloc(N*sizeof(int));

	for(i = 1; i < N; i++)
		aindex[i] = i-1;
	aindex[0] = N-1;

	for(i = 0; i < N; i++)
		bindex[i] = i;
	
	for(i = 0; i < N-1; i++)
		cindex[i] = i+1;
	cindex[N-1] = 0;

	for(i = 0; i < N-2; i++)
		dindex[i] = i+2;
	dindex[N-2] = 0;
	dindex[N-1] = 1;


	B = m_get(N*M, N);
	LB = m_get(M, N);

	for(i = 0; i < N; i++)
	{
		m_zero(LB);
		
		for(j = 0; j < M; j++)
		{
			double s = (double)j / (double)M;
			double a, b, c, d;

			a = (-1.0*s*s*s + 3.0*s*s - 3.0*s + 1.0) / 6.0;
			b = (3.0*s*s*s - 6.0*s*s + 4.0) / 6.0;
			c = (-3.0*s*s*s + 3.0*s*s + 3.0*s + 1.0) / 6.0;
			d = s*s*s / 6.0;

			m_set_val(LB, j, aindex[i], a);
			m_set_val(LB, j, bindex[i], b);
			m_set_val(LB, j, cindex[i], c);
			m_set_val(LB, j, dindex[i], d);
		}
		int m, n;

		for(m = i*M; m < (i+1)*M; m++)
			for(n = 0; n < N; n++)
				m_set_val(B, m, n, m_get_val(LB, m%M, n));
	}

	B_TEMP = mtrm_mlt(B, B, B_TEMP);
	B_TEMP_INV = m_inverse(B_TEMP, B_TEMP_INV);
	B_RET = mmtr_mlt(B_TEMP_INV, B, B_RET);

	free(dindex);
	free(cindex);
	free(bindex);
	free(aindex);

	return B_RET;
}

void uniformseg(VEC * cellx_row, VEC * celly_row, MAT * x, MAT * y)
{
	double dx[36], dy[36], dist[36], dsum[36], perm = 0.0, uperm;
	int i, j, index[36];

	for(i = 1; i <= 36; i++)
	{
		dx[i%36] = v_get_val(cellx_row, i%36) - v_get_val(cellx_row, (i-1)%36);
		dy[i%36] = v_get_val(celly_row, i%36) - v_get_val(celly_row, (i-1)%36);
		dist[i%36] = sqrt(dx[i%36]*dx[i%36] + dy[i%36]*dy[i%36]);
		perm+= dist[i%36];
	}
	uperm = perm / 36.0;
	dsum[0] = dist[0];
	for(i = 1; i < 36; i++)
		dsum[i] = dsum[i-1]+dist[i];

	for(i = 0; i < 36; i++)
	{
		double minimum=DBL_MAX, temp;
		int min_index = 0;
		for(j = 0; j < 36; j++)
		{
			temp = fabs(dsum[j]- (double)i*uperm);
			if (temp < minimum)
			{
				minimum = temp;
				min_index = j;
			}
		}
		index[i] = min_index;
	}

	for(i = 0; i < 36; i++)
	{
		m_set_val(x, 0, i, v_get_val(cellx_row, index[i]));
		m_set_val(y, 0, i, v_get_val(celly_row, index[i]));
	}
}

//Get minimum element in a matrix
double m_min(MAT * m)
{
	int i, j;
	double minimum = DBL_MAX, temp;
	for(i = 0; i < m->m; i++)
	{
		for(j = 0; j < m->n; j++)
		{
			temp = m_get_val(m, i, j);
			if(temp < minimum)
				minimum = temp;
		}
	}
	return minimum;
}

//Get maximum element in a matrix
double m_max(MAT * m)
{
	int i, j;
	double maximum = DBL_MIN, temp;
	for(i = 0; i < m->m; i++)
	{
		for(j = 0; j < m->n; j++)
		{
			temp = m_get_val(m, i, j);
			if(temp > maximum)
				maximum = temp;
		}
	}
	return maximum;
}

VEC * getsampling(MAT * m, int ns)
{
	int N = m->n > m->m ? m-> n:m->m, M = ns;
	int * aindex, * bindex, * cindex, * dindex;
	int i, j;
	VEC * retval = v_get(N*M);

	aindex = (int*) malloc(N*sizeof(int));
	bindex = (int*) malloc(N*sizeof(int));
	cindex = (int*) malloc(N*sizeof(int));
	dindex = (int*) malloc(N*sizeof(int));

	for(i = 1; i < N; i++)
		aindex[i] = i-1;
	aindex[0] = N-1;

	for(i = 0; i < N; i++)
		bindex[i] = i;
	
	for(i = 0; i < N-1; i++)
		cindex[i] = i+1;
	cindex[N-1] = 0;

	for(i = 0; i < N-2; i++)
		dindex[i] = i+2;
	dindex[N-2] = 0;
	dindex[N-1] = 1;

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < M; j++)
		{
			double s = (double)j / (double)M;
			double a, b, c, d;

			a = m_get_val(m, 0, aindex[i]) * (-1.0*s*s*s + 3.0*s*s - 3.0*s + 1.0);
			b = m_get_val(m, 0, bindex[i]) * (3.0*s*s*s - 6.0*s*s + 4.0);
			c = m_get_val(m, 0, cindex[i]) * (-3.0*s*s*s + 3.0*s*s + 3.0*s + 1.0);
			d = m_get_val(m, 0, dindex[i]) * s*s*s;
			v_set_val(retval, i*M+j,(a+b+c+d)/6.0);

		}
	}

	free(dindex);
	free(cindex);
	free(bindex);
	free(aindex);

	return retval;
}

VEC * getfdriv(MAT * m, int ns)
{
	int N = m->n > m->m ? m-> n:m->m, M = ns;
	int * aindex, * bindex, * cindex, * dindex;
	int i, j;
	VEC * retval = v_get(N*M);

	aindex = (int*) malloc(N*sizeof(int));
	bindex = (int*) malloc(N*sizeof(int));
	cindex = (int*) malloc(N*sizeof(int));
	dindex = (int*) malloc(N*sizeof(int));

	for(i = 1; i < N; i++)
		aindex[i] = i-1;
	aindex[0] = N-1;

	for(i = 0; i < N; i++)
		bindex[i] = i;
	
	for(i = 0; i < N-1; i++)
		cindex[i] = i+1;
	cindex[N-1] = 0;

	for(i = 0; i < N-2; i++)
		dindex[i] = i+2;
	dindex[N-2] = 0;
	dindex[N-1] = 1;

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < M; j++)
		{
			double s = (double)j / (double)M;
			double a, b, c, d;

			a = m_get_val(m, 0, aindex[i]) * (-3.0*s*s + 6.0*s - 3.0);
			b = m_get_val(m, 0, bindex[i]) * (9.0*s*s - 12.0*s);
			c = m_get_val(m, 0, cindex[i]) * (-9.0*s*s + 6.0*s + 3.0);
			d = m_get_val(m, 0, dindex[i]) * (3.0 *s*s);
			v_set_val(retval, i*M+j, (a+b+c+d)/6.0);

		}
	}

	free(dindex);
	free(cindex);
	free(bindex);
	free(aindex);

	return retval;
}

//Performs bilinear interpolation, getting the values of m specified in the vectors X and Y
MAT * linear_interp2(MAT * m, VEC * X, VEC * Y)
{
	//Kind of assumes X and Y have same len!

	MAT * retval = m_get(1, X->dim);
	double x_coord, y_coord, new_val, a, b;
	int l, k, i;

	for(i = 0; i < X->dim; i++)
	{
		x_coord = v_get_val(X, i);
		y_coord = v_get_val(Y, i);

		l = (int)x_coord;
		k = (int)y_coord;

		a = x_coord - (double)l;
		b = y_coord - (double)k;

		//printf("xc: %f \t yc: %f \t i: %d \t l: %d \t k: %d \t a: %f \t b: %f\n", x_coord, y_coord, i, l, k, a, b);

		new_val = (1.0-a)*(1.0-b)*m_get_val(m, k, l) +
				  a*(1.0-b)*m_get_val(m, k, l+1) +
				  (1.0-a)*b*m_get_val(m, k+1, l) +
				  a*b*m_get_val(m, k+1, l+1);

		m_set_val(retval, 0, i, new_val);
	}

	return retval;
}

void splineenergyform01(MAT * Cx, MAT * Cy, MAT * Ix, MAT * Iy, int ns, double delta, double dt, int typeofcell)
{
	VEC * X, * Y, * Xs, * Ys, * Nx, * Ny, * X1, * Y1, * X2, * Y2, *	XY, * XX, * YY, * dCx, * dCy, * Ix1, * Ix2, *Iy1, *Iy2;
	MAT * Ix1_mat, * Ix2_mat, * Iy1_mat, * Iy2_mat;
	int i,j, N, * aindex, * bindex, * cindex, * dindex;

	X = getsampling(Cx, ns);
	Y = getsampling(Cy, ns);
	Xs = getfdriv(Cx, ns);
	Ys = getfdriv(Cy, ns);

	Nx = v_get(Ys->dim);
	for(i = 0; i < Nx->dim; i++)
		v_set_val(Nx, i, v_get_val(Ys, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));

	Ny = v_get(Xs->dim);
	for(i = 0; i < Ny->dim; i++)
		v_set_val(Ny, i, -1.0 * v_get_val(Xs, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));
	
	X1 = v_get(Nx->dim);
	for(i = 0; i < X1->dim; i++)
		v_set_val(X1, i, v_get_val(X, i) + delta*v_get_val(Nx, i));

	Y1 = v_get(Ny->dim);
	for(i = 0; i < Y1->dim; i++)
		v_set_val(Y1, i, v_get_val(Y, i) + delta*v_get_val(Ny, i));

	X2 = v_get(Nx->dim);
	for(i = 0; i < X2->dim; i++)
		v_set_val(X2, i, v_get_val(X, i) - delta*v_get_val(Nx, i));

	Y2 = v_get(Ny->dim);
	for(i = 0; i < Y2->dim; i++)
		v_set_val(Y2, i, v_get_val(Y, i) + delta*v_get_val(Ny, i));

	Ix1_mat = linear_interp2(Ix, X1, Y1);
	Iy1_mat = linear_interp2(Iy, X1, Y1);
	Ix2_mat = linear_interp2(Ix, X2, Y2);
	Iy2_mat = linear_interp2(Iy, X2, Y2);

	Ix1 = v_get(Ix1_mat->n);
	Iy1 = v_get(Iy1_mat->n);
	Ix2 = v_get(Ix2_mat->n);
	Iy2 = v_get(Iy2_mat->n);

	Ix1 = get_row(Ix1_mat, 0, Ix1);
	Iy1 = get_row(Iy1_mat, 0, Iy1);
	Ix2 = get_row(Ix2_mat, 0, Ix2);
	Iy2 = get_row(Iy2_mat, 0, Iy2);

	N = Cx->m;

	//VEC * retval = v_get(N*ns);

	aindex = (int*) malloc(N*sizeof(int));
	bindex = (int*) malloc(N*sizeof(int));
	cindex = (int*) malloc(N*sizeof(int));
	dindex = (int*) malloc(N*sizeof(int));

	for(i = 1; i < N; i++)
		aindex[i] = i-1;
	aindex[0] = N-1;

	for(i = 0; i < N; i++)
		bindex[i] = i;
	
	for(i = 0; i < N-1; i++)
		cindex[i] = i+1;
	cindex[N-1] = 0;

	for(i = 0; i < N-2; i++)
		dindex[i] = i+2;
	dindex[N-2] = 0;
	dindex[N-1] = 1;

	XY = v_get(Xs->dim);
	for(i = 0; i < Xs->dim; i++)
		v_set_val(XY, i, v_get_val(Xs, i) * v_get_val(Ys, i));

	XX = v_get(Xs->dim);
	for(i = 0; i < Xs->dim; i++)
		v_set_val(XX, i, v_get_val(Xs, i) * v_get_val(Xs, i));

	YY = v_get(Ys->dim);
	for(i = 0; i < Xs->dim; i++)
		v_set_val(YY, i, v_get_val(Ys, i) * v_get_val(Ys, i));

	dCx = v_get(Cx->m);
	dCy = v_get(Cy->m);

	//get control points for splines
	for(i = 0; i < Cx->m; i++)
	{
		for(j = 0; j < ns; j++)
		{
			double s = (double)j / (double)ns;
			double A1, A2, A3, A4, B1, B2, B3, B4, D, D_3, Tx1, Tx2, Tx3, Tx4, Ty1, Ty2, Ty3, Ty4;
			int k;

			A1 = (-1.0*(s-1.0)*(s-1.0)*(s-1.0)) / 6.0;
			A2 = (3.0*s*s*s - 6.0*s*s + 4.0) / 6.0;
			A3 = (-3.0*s*s*s + 3.0*s*s + 3.0*s + 1.0) / 6.0;
			A4 = s*s*s / 6.0;

			B1 = (-3.0*s*s + 6.0*s - 3.0) / 6.0;
			B2 = (9.0*s*s - 12.0*s) / 6.0;
			B3 = (-9.0*s*s + 6.0*s + 3.0) / 6.0;
			B4 = 3.0*s*s / 6.0;

			k = i*ns+j;
			D = sqrt(v_get_val(Xs, k)*v_get_val(Xs, k) + v_get_val(Ys, k)*v_get_val(Ys, k));
			D_3 = D*D*D;
			
			//1st control point
			
			Tx1 = A1 - delta * v_get_val(XY, k) * B1 / D_3;
			Tx2 = -1.0 * delta*(B1/D - v_get_val(XX, k)*B1/D_3);
			Tx3 = A1 + delta * v_get_val(XY, k) * B1 / D_3;
			Tx4 = delta*(B1/D - v_get_val(XX, k)*B1/D_3);

			Ty1 = delta*(B1/D - v_get_val(YY, k)*B1/D_3);
			Ty2 = A1 + delta * v_get_val(XY, k) * B1 / D_3;
			Ty3 = -1.0 * delta*(B1/D - v_get_val(YY, k)*B1/D_3);
			Ty4 = A1 - delta * v_get_val(XY, k) * B1 / D_3;

			v_set_val(dCx, aindex[i], v_get_val(dCx, aindex[i]) + Tx1*v_get_val(Ix1, k) + Tx2*v_get_val(Iy1,k) - Tx3*v_get_val(Ix2, k) - Tx4*v_get_val(Iy2, k));
			v_set_val(dCy, aindex[i], v_get_val(dCy, aindex[i]) + Ty1*v_get_val(Ix1, k) + Ty2*v_get_val(Iy1,k) - Ty3*v_get_val(Ix2, k) - Ty4*v_get_val(Iy2, k));
		
			//2nd control point

			Tx1 = A2 - delta * v_get_val(XY, k) * B2 / D_3;
			Tx2 = -1.0 * delta*(B2/D - v_get_val(XX, k)*B2/D_3);
			Tx3 = A2 + delta * v_get_val(XY, k) * B2 / D_3;
			Tx4 = delta*(B2/D - v_get_val(XX, k)*B2/D_3);

			Ty1 = delta*(B2/D - v_get_val(YY, k)*B2/D_3);
			Ty2 = A2 + delta * v_get_val(XY, k) * B2 / D_3;
			Ty3 = -1.0 * delta*(B2/D - v_get_val(YY, k)*B2/D_3);
			Ty4 = A2 - delta * v_get_val(XY, k) * B2 / D_3;

			v_set_val(dCx, bindex[i], v_get_val(dCx, bindex[i]) + Tx1*v_get_val(Ix1, k) + Tx2*v_get_val(Iy1,k) - Tx3*v_get_val(Ix2, k) - Tx4*v_get_val(Iy2, k));
			v_set_val(dCy, bindex[i], v_get_val(dCy, bindex[i]) + Ty1*v_get_val(Ix1, k) + Ty2*v_get_val(Iy1,k) - Ty3*v_get_val(Ix2, k) - Ty4*v_get_val(Iy2, k));

			//3nd control point

			Tx1 = A3 - delta * v_get_val(XY, k) * B3 / D_3;
			Tx2 = -1.0 * delta*(B3/D - v_get_val(XX, k)*B3/D_3);
			Tx3 = A3 + delta * v_get_val(XY, k) * B3 / D_3;
			Tx4 = delta*(B3/D - v_get_val(XX, k)*B3/D_3);

			Ty1 = delta*(B3/D - v_get_val(YY, k)*B3/D_3);
			Ty2 = A3 + delta * v_get_val(XY, k) * B3 / D_3;
			Ty3 = -1.0 * delta*(B3/D - v_get_val(YY, k)*B3/D_3);
			Ty4 = A3 - delta * v_get_val(XY, k) * B3 / D_3;

			v_set_val(dCx, cindex[i], v_get_val(dCx, cindex[i]) + Tx1*v_get_val(Ix1, k) + Tx2*v_get_val(Iy1,k) - Tx3*v_get_val(Ix2, k) - Tx4*v_get_val(Iy2, k));
			v_set_val(dCy, cindex[i], v_get_val(dCy, cindex[i]) + Ty1*v_get_val(Ix1, k) + Ty2*v_get_val(Iy1,k) - Ty3*v_get_val(Ix2, k) - Ty4*v_get_val(Iy2, k));
	
			//4nd control point

			Tx1 = A4 - delta * v_get_val(XY, k) * B4 / D_3;
			Tx2 = -1.0 * delta*(B4/D - v_get_val(XX, k)*B4/D_3);
			Tx3 = A4 + delta * v_get_val(XY, k) * B4 / D_3;
			Tx4 = delta*(B4/D - v_get_val(XX, k)*B4/D_3);

			Ty1 = delta*(B4/D - v_get_val(YY, k)*B4/D_3);
			Ty2 = A4 + delta * v_get_val(XY, k) * B4 / D_3;
			Ty3 = -1.0 * delta*(B4/D - v_get_val(YY, k)*B4/D_3);
			Ty4 = A4 - delta * v_get_val(XY, k) * B4 / D_3;

			v_set_val(dCx, dindex[i], v_get_val(dCx, dindex[i]) + Tx1*v_get_val(Ix1, k) + Tx2*v_get_val(Iy1,k) - Tx3*v_get_val(Ix2, k) - Tx4*v_get_val(Iy2, k));
			v_set_val(dCy, dindex[i], v_get_val(dCy, dindex[i]) + Ty1*v_get_val(Ix1, k) + Ty2*v_get_val(Iy1,k) - Ty3*v_get_val(Ix2, k) - Ty4*v_get_val(Iy2, k));		
		}
	}

	if(typeofcell==1)
	{
		for(i = 0; i < Cx->n; i++)
			m_set_val(Cx, 0, i, m_get_val(Cx, 1, i) - dt*v_get_val(dCx, i));

		for(i = 0; i < Cy->n; i++)
			m_set_val(Cy, 0, i, m_get_val(Cy, 1, i) - dt*v_get_val(dCy, i));
	}
	else
	{
		for(i = 0; i < Cx->n; i++)
			m_set_val(Cx, 0, i, m_get_val(Cx, 1, i) + dt*v_get_val(dCx, i));

		for(i = 0; i < Cy->n; i++)
			m_set_val(Cy, 0, i, m_get_val(Cy, 1, i) + dt*v_get_val(dCy, i));
	}

	v_free(dCy); v_free(dCx); v_free(YY); v_free(XX); v_free(XY);

	free(dindex); free(cindex); free(bindex); free(aindex); 

	v_free(Iy2); v_free(Ix2); v_free(Iy1); v_free(Ix1); 

	m_free(Iy2_mat); m_free(Ix2_mat); m_free(Iy1_mat); m_free(Ix1_mat); 

	v_free(Y2); v_free(X2); v_free(Y1); v_free(X1); v_free(Ny); v_free(Nx); v_free(Ys); v_free(Xs); v_free(Y); v_free(X); 
}

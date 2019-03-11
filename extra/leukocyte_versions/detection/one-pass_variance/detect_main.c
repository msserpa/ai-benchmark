#include "find_ellipse.h"
#include "misc_math.h"
#include "avilib.h"
#include <stdlib.h>
#include <math.h>
#include <sys/param.h>
#include <sys/times.h>
#include "matrix.h"
#include <cuda.h>

#include <sys/time.h>
#include <time.h>

//#include <vsip.h>

// Defines the region in the video frame containing the blood vessel
#define TOP 110
#define BOTTOM 328


MAT * get_frame(avi_t *cell_file, int frame_num);

int main(int argc, char ** argv) {
	// Choose the best GPU
	choose_GPU();

	// Keep track of the start time of the program
    struct timeval tv;
    gettimeofday(&tv, NULL); 
    long long program_start_time = tv.tv_sec*1000000 + tv.tv_usec;
	
	int num_correct = 0;
	
	long long ellipse_time = 0, dilate_time = 0;

	int i, j, *crow, *ccol, pair_counter = 0, x_result_len = 0, Iter = 20, ns = 4, k_count = 0, n, frame_num;
	unsigned char * image_buf;
	MAT **grad_x, **grad_y, * cellx, * celly, **gicov, **img_dilated, * A;
	double * GICOV_spots, * t, * G, * x_result, * y_result, *V, *QAX_CENTERS, *QAY_CENTERS;
	double threshold = 1.8, r = 10.0, delta = 3.0, dt = 0.01, b = 5.0;
	avi_t * cell_file;

	// Let the user specify the number of frames to process
	int max_frame_num = 0;
	if (argc > 1) max_frame_num = atoi(argv[1]) - 1;
	
	// Open video file
	cell_file = AVI_open_input_file("testfiles/testfile.avi", 1);
	if(cell_file == NULL)	{
		AVI_print_error("Error with AVI_open_input_file");
		return -1;
	}
	
	// Transfer precomputed constants to the GPU
	compute_constants();
	
	for (frame_num = 0; frame_num <= max_frame_num; frame_num++) {
		// Extract the cropped frame from the video file
		MAT *image_chopped = get_frame(cell_file, frame_num);
		
		//Get gradient matrices in x and y directions
		MAT *grad_x = gradient_x(image_chopped);
		MAT *grad_y = gradient_y(image_chopped);
		
		m_free(image_chopped);
		
		//Get GICOV matrices corresponding to image gradients
        gettimeofday(&tv, NULL); 
        long long start_time = tv.tv_sec*1000000 + tv.tv_usec;
		MAT *gicov = ellipsematching(grad_x, grad_y);
		gettimeofday(&tv, NULL); 
        long long end_time = tv.tv_sec*1000000 + tv.tv_usec;
		ellipse_time += end_time - start_time;

		//Dilate the GICOV matrices
        gettimeofday(&tv, NULL); 
        start_time = tv.tv_sec*1000000 + tv.tv_usec;
		MAT *img_dilated = dilate_f(gicov);
        gettimeofday(&tv, NULL); 
        end_time = tv.tv_sec*1000000 + tv.tv_usec;
		dilate_time += end_time - start_time;
		
		//Find possible matches for cell centers based on GICOV, record the rows/columns in which they are found
		pair_counter = 0;
		crow = (int *) malloc(gicov->m * gicov->n * sizeof(int));
		ccol = (int *) malloc(gicov->m * gicov->n * sizeof(int));
		for(i = 0; i < gicov->m; i++) {
			for(j = 0; j < gicov->n; j++) {
				if(!double_eq(m_get_val(gicov,i,j), 0.0) && double_eq(m_get_val(img_dilated,i,j), m_get_val(gicov,i,j)))
				{
					crow[pair_counter]=i;
					ccol[pair_counter]=j;
					pair_counter++;
				}
			}
		}

		GICOV_spots = (double *) malloc(sizeof(double) * pair_counter);
		for(i = 0; i < pair_counter; i++)
			GICOV_spots[i] = sqrt(m_get_val(gicov, crow[i], ccol[i]));

		G = (double *) calloc(pair_counter, sizeof(double));
		x_result = (double *) calloc(pair_counter, sizeof(double));
		y_result = (double *) calloc(pair_counter, sizeof(double));

		x_result_len = 0;
		for(i = 0; i < pair_counter; i++) {
			if((crow[i] > 29) && (crow[i] < BOTTOM - TOP + 40)) {
				x_result[x_result_len] = ccol[i];
				y_result[x_result_len] = crow[i]-40;
				G[x_result_len] = GICOV_spots[i];
				x_result_len++;
			}
		}

		//Make an array t which holds each "time step" for the possible cells
		t = (double *) malloc(sizeof(double) * 36);
		for(i = 0; i < 36; i++) {
			t[i] = (double)i * 2.0 * PI / 36.0;
		}

		//Store cell boundaries (as simple circles) for all cells
		cellx = m_get(x_result_len, 36);
		celly = m_get(x_result_len, 36);
		for(i = 0; i < x_result_len; i++) {
			for(j = 0; j < 36; j++) {
				m_set_val(cellx, i, j, x_result[i] + r*cos(t[j]));
				m_set_val(celly, i, j, y_result[i] + r*sin(t[j]));
			}
		}
		
		A = TMatrix(9,4);

		
		V = (double *) malloc(sizeof(double) * pair_counter);
		QAX_CENTERS = (double *) malloc(sizeof(double) * pair_counter);
		QAY_CENTERS = (double *) malloc(sizeof(double) * pair_counter);
		memset(V, 0, sizeof(double) * pair_counter);
		memset(QAX_CENTERS, 0, sizeof(double) * pair_counter);
		memset(QAY_CENTERS, 0, sizeof(double) * pair_counter);

		//For all possible results, find the ones that are feasibly leukocytes, store their centers
		k_count = 0;
		for(n = 0; n < x_result_len; n++) {
			if((G[n] < -1 * threshold) || G[n] > threshold) {
				MAT * x, *y;
				VEC * x_row, * y_row;
				x = m_get(1, 36);
				y = m_get(1, 36);

				x_row = v_get(36);
				y_row = v_get(36);

				//Get current values of possible cells from cellx/celly matrices
				x_row = get_row(cellx, n, x_row);
				y_row = get_row(celly, n, y_row);
				uniformseg(x_row, y_row, x, y);

				//Make sure that the possible leukocytes are within a certain size range before moving on
				if((m_min(x) > b) && (m_min(y) > b) && (m_max(x) < cell_file->width - b) && (m_max(y) < cell_file->height - b)) {
					MAT * Cx, * Cy, *Cy_temp, * Ix1, * Iy1;
					VEC  *Xs, *Ys, *W, *Nx, *Ny, *X, *Y;
					Cx = m_get(1, 36);
					Cy = m_get(1, 36);
					Cx = mmtr_mlt(A, x, Cx);
					Cy = mmtr_mlt(A, y, Cy);

					Cy_temp = m_get(Cy->m,Cy->n);

					for(i = 0; i < 9; i++)
						m_set_val(Cy, i, 0, m_get_val(Cy, i, 0) + 40.0);

					//Do Iter iterations of spline refinement
					for(i = 0; i < Iter; i++) {
						int typeofcell;

						if(G[n] > 0.0)
							typeofcell = 0;
						else typeofcell = 1;

						splineenergyform01(Cx, Cy, grad_x, grad_y, ns, delta, 2.0*dt, typeofcell);
					}

					X=getsampling(Cx,ns);
					for(i = 0; i < Cy->m; i++)
						m_set_val(Cy_temp, i, 0, m_get_val(Cy, i, 0) - 40.0);
					Y=getsampling(Cy_temp,ns);
					
					Ix1 = linear_interp2(grad_x,X,Y);
					Iy1 = linear_interp2(grad_x,X,Y);
					Xs = getfdriv(Cx,ns);
					Ys = getfdriv(Cy,ns);

					Nx = v_get(Ys->dim);
					for(i = 0; i < Ys->dim; i++) {
						v_set_val(Nx, i, v_get_val(Ys, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));
					}

					Ny = v_get(Xs->dim);
					for(i = 0; i < Xs->dim; i++) {
						v_set_val(Ny, i, -1.0 * v_get_val(Xs, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));
					}


					W = v_get(Nx->dim);
					for(i = 0; i < Nx->dim; i++) {
						v_set_val(W, i, m_get_val(Ix1, 0, i) * v_get_val(Nx, i) + m_get_val(Iy1, 0, i) * v_get_val(Ny, i));
					}

					

					V[n] = mean(W)/std_dev(W);
					

					//get means of X and Y values for all "snaxels" of the spline contour, thus finding the cell centers
					QAX_CENTERS[k_count]=mean(X);
					QAY_CENTERS[k_count]=mean(Y) + TOP+40.0;
					
					k_count++;
					
					v_free(W);
					v_free(Ny);
					v_free(Nx);
					v_free(Ys);
					v_free(Xs);
					m_free(Iy1);
					m_free(Ix1);
					v_free(Y);
					v_free(X);
					m_free(Cy_temp);
					m_free(Cy);
					m_free(Cx);				
				}

				v_free(y_row);
				v_free(x_row);
				m_free(y);
				m_free(x);
			}
		}


		printf("K (frame %d): %d\n", frame_num, k_count);
		/*printf("QAX_CENTERS: ");
		for(i = 0; i < k_count; i++)
			printf("%f\t", QAX_CENTERS[i]);
		printf("\n");
		printf("QAY_CENTERS: ");
		for(i = 0; i < k_count; i++)
			printf("%f\t", QAY_CENTERS[i]);
		printf("\n"); // */
		//printf("V VALUES: ");
		//for(i = 0; i < x_result_len; i++)
		//	printf("%f\t", V[i]);
		//printf("\n");

		free(QAY_CENTERS);
		free(QAX_CENTERS);
		free(V);
		free(ccol);
		free(crow);
		free(GICOV_spots);
		free(t);
		free(G);
		free(x_result);
		free(y_result);

		m_free(A);
		m_free(celly);
		m_free(cellx);
		m_free(gicov);
		m_free(grad_y);
		m_free(grad_x);
		m_free(img_dilated);
	}
	
	free(cell_file);
	
    gettimeofday(&tv, NULL); 
    long long program_end_time = tv.tv_sec*1000000 + tv.tv_usec;
    printf("\nTotal run time: %.5f seconds\n", ((float) (program_end_time - program_start_time)) / (1000*1000));
    printf("\nAverage time per frame:\n");
	printf("\tellipse_matching: %.5f seconds\n", ((float) ellipse_time) / ((float) (max_frame_num + 1)) / (1000*1000));
	printf("\tdilate_f: %.5f seconds\n", ((float) dilate_time) / ((float) (max_frame_num + 1)) / (1000*1000));
	printf("\tother: %.5f seconds\n", ((float) (program_end_time - program_start_time - ellipse_time - dilate_time)) / ((float) (max_frame_num + 1)) / (1000*1000));
	printf("\tTotal: %.5f seconds\n", (((float) (program_end_time - program_start_time)) / (1000*1000)) / ((float) (max_frame_num + 1)));
	

	return 0;
}


MAT * get_frame(avi_t *cell_file, int frame_num) {
	int dummy;
	int width = AVI_video_width(cell_file);
	int height = AVI_video_height(cell_file);
	unsigned char *image_buf = (unsigned char *) malloc(width * height);

	// There are 600 frames in this file (i.e. frame_num = 600 causes an error)
	AVI_set_video_position(cell_file, frame_num);

	//Read in the frame from the AVI
	if(AVI_read_frame(cell_file, (char *)image_buf, &dummy) == -1) {
		AVI_print_error("Error with AVI_read_frame");
		exit(-1);
	}

	//Chop and flip image so we deal only with the interior of the vein
	MAT * image_chopped = chop_flip_image(image_buf, height, width, TOP, BOTTOM, 0, width-1);
	
	free(image_buf);
	
	return image_chopped;
}

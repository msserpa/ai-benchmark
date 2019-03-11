#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>


#define GET_RAND_FP ( (float)rand() /   \
    ((float)(RAND_MAX)+(float)(1)) )

#define MIN(i,j) ((i)<(j) ? (i) : (j))

#define DEFAULT_N 512
#define DEFAULT_B 16
#define BLOCK_SIZE 16

int n = DEFAULT_N;
int b = DEFAULT_B;
char *input_file=NULL;
int do_timing=1;
int do_verify=0;

extern void
lud_cuda_1(float *d_m, int matrix_dim);
extern void
lud_cuda_2(float *d_m, int matrix_dim);
extern void
lud_cuda_3(float *d_m, int matrix_dim);
extern void
lud_cuda_4(float *d_m, int matrix_dim);
extern void
lud_cuda_5(float *d_m, int matrix_dim);
extern void
lud_cuda_6(float *d_m, int matrix_dim);



int read_matrix(float **mp, const char* filename){
  int i, j, size;
  float *m;

  FILE *fp = fopen(filename, "rb");
  if ( fp == NULL) {
    printf("read_matrix: can not open input file %s\n", filename);
    return -1;
  }

  fscanf(fp, "%d\n", &size);

  m = (float*) malloc(sizeof(float)*size*size);
  if ( m == NULL) {
    printf("read_matrix: can not allocate memory\n");  
    fclose(fp);
    return -1;
  }

  for (i=0; i < size; i++) {
    for (j=0; j < size; j++) {
      fscanf(fp, "%f ", m+i*size+j);
    }
  }

  fclose(fp);

  *mp = m;

  return size;
}

int create_matrix(float **mp, int size) {
  float *l, *u, *m;
  int i,j,k;


  l = (float*)malloc(size*size*sizeof(float));
  if ( l == NULL) {
    printf("create_matrix: can not allocate memory\n");
    return -1;
  }

  u = (float*)malloc(size*size*sizeof(float));
  if ( u == NULL) {
    free(l);
    printf("create_matrix: can not allocate memory\n");
    return -1;
  }

  m = (float*) malloc(size*size*sizeof(float));
  if ( m == NULL) {
    printf("create_matrix: can not allocate memory\n");  
    free(l);
    free(u);
    return -1;
  }

  srand(time(NULL));
  for (i = 0; i < size; i++) {
    for (j=0; j < size; j++) {
      if (i>j) {
        l[i*size+j] = GET_RAND_FP;
      } else if (i == j) {
        l[i*size+j] = 1;
      } else {
        l[i*size+j] = 0;
      }
    }
  }

  for (j=0; j < size; j++) {
    for (i=0; i < size; i++) {
      if (i>j) {
        u[j*size+i] = 0;
      }else {
        u[j*size+i] = GET_RAND_FP; 
      }
    }
  }

  for (i=0; i < size; i++) {
    for (j=0; j < size; j++) {
      for (k=0; k <= MIN(i,j); k++)
        m[i*size+j] = l[i*size+k] * u[j*size+k];
    }
  }

  free(l);
  free(u);

  *mp = m;

  return size;
}

void print_matrix(float *a, int n){
  int i,j;
  for (i=0; i < n; i ++){
    for (j=0; j < n; j++)
      printf("%f ", a[i*n+j]);
    printf("\n");
  }
}


  int
main ( int argc, char *argv[] )
{
  int opt;
  int i,j,size;
  struct timeval start, end;
  double time;

  float *m, *d_m;

  while ((opt=getopt(argc,argv, "n:b:i:tvh")) != -1) {
    switch(opt) {
      case 'n': n=atoi(optarg); break;
      case 'b': b=atoi(optarg); break;
      case 'i': input_file=optarg;break;
      case 't': do_timing=1; break;
      case 'v': do_verify=1; break;
      case 'h': printf("Usage: %s <options>\n\n", argv[0]);
                printf("Options:\n");
                printf("  -nN : Decompse NxN matrix.\n");
                printf("  -bB : Use a block size of B.(Not supported in this version)\n");
                printf("  -i/path/to/file : Specify input file.\n");
                printf("  -t  : Count execution time.\n");
                printf("  -v  : Verify result.\n");
                printf("  -h  : print this help.\n\n");
                printf("Default: %s -n%ld -b%ld -t\n", argv[0], DEFAULT_N, DEFAULT_B);
                exit(0);
                break;
    }
  }


  if (input_file) {
    printf("Input file is presented, ignore size option if possible\n");
    printf("Reading matrix from file %s\n", input_file);
    if ( (n=read_matrix(&m, input_file)) == -1) {
      exit(1);
    }
  } else {
    assert (n > 0);
    if ( (n=create_matrix(&m, n)) == -1) {
      exit(1);
    } 
  } 

  if (do_verify)
    print_matrix(m, n);

  size = n*n*sizeof(float);


  cudaMalloc((void **)&d_m, size);    

  /** start timer. */
  if (do_timing) {
    bzero(&start, sizeof(struct timeval));
    bzero(&end, sizeof(struct timeval));

    gettimeofday(&start, NULL);
  }

  cudaMemcpy(d_m, m, size, cudaMemcpyHostToDevice);

//  lud_cuda_1(d_m, n);
//  lud_cuda_2(d_m, n);
//  lud_cuda_3(d_m, n);
//  lud_cuda_4(d_m, n);
  lud_cuda_5(d_m, n);
//  lud_cuda_6(d_m, n);

  cudaMemcpy(m, d_m, size, cudaMemcpyDeviceToHost);

  if (do_timing){
    gettimeofday(&end, NULL);

    /** time is counted by seconds. */
    time = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    time = time / 1000000;
  }

  cudaFree(d_m);

  printf("Time consumed by kernel: %lfs\n", time);

  if (do_verify) {
    printf(">>>>>>>>>>>>>>After LU decomposition<<<<<<<<<<<<<<\n");
    for (i=0; i < n; i++){
      for(j=0; j < n; j++){
        printf("%f ", *(m+i*n+j));
      }
      printf("\n");
    }
  }

  if (do_verify) {
    float l,u,sum;
    int k;
    printf(">>>>>>>result<<<<<<<<<<\n");
    for (i=0; i < n; i ++){
      for (j=0; j < n; j ++){
        sum=0;
        for(k=0; k < n; k ++){
          if (i == k) {
            l = 1;
          }else if ( i < k){
            l = 0;
          }else {//i > k
            l = *(m+i*n+k);
          }
          if (k > j) {
            u = 0;
          }else { //k >= j
            u = *(m+k*n+j);
          }
          sum += l * u; 
        }
        printf("%f ", sum);
      }
      printf("\n");
    }
  }

  free(m);

  return 0;

}				/* ----------  end of function main  ---------- */

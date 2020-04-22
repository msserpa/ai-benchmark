/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*************************************************************************/
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/
/*************************************************************************/
/**   File:         cluster.c                                           **/
/**   Description:  Takes as input a file, containing 1 data point per  **/
/**                 per line, and performs a fuzzy c-means clustering   **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Brendan McCane                                           **/
/**            James Cook University of North Queensland.               **/
/**            Australia. email: mccane@cs.jcu.edu.au                   **/
/**                                                                     **/
/**   Edited by: Jay Pisharath, Wei-keng Liao                           **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**                                                                     **/
/**   Edited by: Sang-Ha  Lee                                           **/
/**              University of Virginia                                 **/
/**                                                                     **/
/**   Description:  No longer supports fuzzy c-means clustering;        **/
/**                 only regular k-means clustering.                    **/
/**                 Simplified for main functionality: regular k-means  **/
/**                 clustering.                                         **/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
float euclid_dist_2(float *pt1,
                    float *pt2,
                    int    numdims)
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

    return(ans);
}

int find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float **pts,         /* [npts][nfeatures] */
                       int     npts)
{
    int index = 0, i;
    float min_dist=FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i=0; i<npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures);  /* no need square root */
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}


/*----< kmeans_clustering() >---------------------------------------------*/
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          int    *membership) /* out: [npoints] */
{

    int      i, j, k, n=0, index=0, loop=0;
    int     *new_centers_len;           /* [nclusters]: no. of points in each cluster */
    float  **new_centers;               /* [nclusters][nfeatures] */
    float  **clusters;                  /* out: [nclusters][nfeatures] */
    float    delta;
    
    int      nthreads;
    int    **partial_new_centers_len;
    float ***partial_new_centers;

    #pragma omp parallel
    #pragma omp single
    nthreads = omp_get_num_threads();

    /* allocate space for returning variable clusters[] */
    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    for (i=1; i<nclusters; i++)
        clusters[i] = clusters[i-1] + nfeatures;

    /* randomly pick cluster centers */
    for (i=0; i<nclusters; i++) {
        //n = (int)rand() % npoints;
        for (j=0; j<nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    for (i=0; i<npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int*) calloc(nclusters, sizeof(int));

    new_centers    = (float**) malloc(nclusters *            sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;


    partial_new_centers_len    = (int**) malloc(nthreads * sizeof(int*));
    partial_new_centers_len[0] = (int*)  calloc(nthreads*nclusters, sizeof(int));
    for (i=1; i<nthreads; i++)
        partial_new_centers_len[i] = partial_new_centers_len[i-1]+nclusters;

    partial_new_centers    =(float***)malloc(nthreads * sizeof(float**));
    partial_new_centers[0] =(float**) malloc(nthreads*nclusters * sizeof(float*));
    for (i=1; i<nthreads; i++)
        partial_new_centers[i] = partial_new_centers[i-1] + nclusters;

    for (i=0; i<nthreads; i++)
    {
        for (j=0; j<nclusters; j++)
            partial_new_centers[i][j] = (float*)calloc(nfeatures, sizeof(float));
    }

    do {
        delta = 0.0;
        #pragma omp parallel \
                shared(feature,clusters,membership,partial_new_centers,partial_new_centers_len)
        {
            int tid = omp_get_thread_num();             
            #pragma omp for \
                        private(i,j,index) \
                        firstprivate(npoints,nclusters,nfeatures) \
                        schedule(static) \
                        reduction(+:delta)
            for (i=0; i<npoints; i++) {
            /* find the index of nestest cluster centers */                 
            index = find_nearest_point(feature[i],
                     nfeatures,
                     clusters,
                     nclusters);                
            /* if membership changes, increase delta by 1 */
            if (membership[i] != index) delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;
                
            /* update new cluster centers : sum of all objects located
               within */
            partial_new_centers_len[tid][index]++;              
            for (j=0; j<nfeatures; j++)
               partial_new_centers[tid][index][j] += feature[i][j];
            }
        } /* end of #pragma omp parallel */

        /* let the main thread perform the array reduction */
        for (i=0; i<nclusters; i++) {
            for (j=0; j<nthreads; j++) {
                new_centers_len[i] += partial_new_centers_len[j][i];
                partial_new_centers_len[j][i] = 0.0;
                for (k=0; k<nfeatures; k++) {
                    new_centers[i][k] += partial_new_centers[j][i][k];
                    partial_new_centers[j][i][k] = 0.0;
                }
            }
        }    

        /* replace old cluster centers with new_centers */
        for (i=0; i<nclusters; i++) {
            for (j=0; j<nfeatures; j++) {
                if (new_centers_len[i] > 0)
                    clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0.0;   /* set back to 0 */
            }
            new_centers_len[i] = 0;   /* set back to 0 */
        }
        
    } while (loop++ < 500);

    
    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}

/*---< cluster() >-----------------------------------------------------------*/
int cluster(int      numObjects,      /* number of input objects */
            int      numAttributes,   /* size of attribute of each object */
            float  **attributes,      /* [numObjects][numAttributes] */            
            int      nclusters,
            float ***cluster_centres /* out: [best_nclusters][numAttributes] */
    
            )
{
    int    *membership;
    float **tmp_cluster_centres;

    membership = (int*) malloc(numObjects * sizeof(int));
   
    srand(7);
    /* perform regular Kmeans */
    tmp_cluster_centres = kmeans_clustering(attributes,
                                            numAttributes,
                                            numObjects,
                                            nclusters,
                                            membership);      
    
    if (*cluster_centres) {
        free((*cluster_centres)[0]);
        free(*cluster_centres);
    }
    *cluster_centres = tmp_cluster_centres;

   
    free(membership);

    return 0;
}



extern double wtime(void);

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
    char *help =
        "Usage: %s -i ../data/kmeans/kdd_cup -k 10\n"
        "       -i filename     		: file containing data to be clustered\n"
		"       -k                 	: number of clusters (default is 5) \n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     nclusters=5;
           char   *filename = 0;           
           float  *buf;
           float **attributes;
           float **cluster_centres=NULL;
           int     i, j;
                
           int     numAttributes;
           int     numObjects;        
           char    line[1024];           
           int     isBinaryFile = 0;
           int     nloops = 1;
		   double  timing;		   

	while ( (opt=getopt(argc,argv,"i:k:t:b:n:?"))!= EOF) {
		switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'k': nclusters = atoi(optarg);
                      break;			
            case '?': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }


    if (filename == 0) usage(argv[0]);

    numAttributes = numObjects = 0;

    /* from the input file, get the numAttributes and numObjects ------------*/
   
    if (isBinaryFile) {
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        read(infile, &numObjects,    sizeof(int));
        read(infile, &numAttributes, sizeof(int));
   

        /* allocate space for attributes[] and read attributes of all objects */
        buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
        attributes    = (float**)malloc(numObjects*             sizeof(float*));
        attributes[0] = (float*) malloc(numObjects*numAttributes*sizeof(float));
        for (i=1; i<numObjects; i++)
            attributes[i] = attributes[i-1] + numAttributes;

        read(infile, buf, numObjects*numAttributes*sizeof(float));

        close(infile);
    }
    else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        while (fgets(line, 1024, infile) != NULL)
            if (strtok(line, " \t\n") != 0)
                numObjects++;
        rewind(infile);
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first attribute): numAttributes = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) numAttributes++;
                break;
            }
        }
     

        /* allocate space for attributes[] and read attributes of all objects */
        buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
        attributes    = (float**)malloc(numObjects*             sizeof(float*));
        attributes[0] = (float*) malloc(numObjects*numAttributes*sizeof(float));
        for (i=1; i<numObjects; i++)
            attributes[i] = attributes[i-1] + numAttributes;
        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue; 
            for (j=0; j<numAttributes; j++) {
                buf[i] = atof(strtok(NULL, " ,\t\n"));
                i++;
            }
        }
        fclose(infile);
    }     
	printf("I/O completed\n");	

	memcpy(attributes[0], buf, numObjects*numAttributes*sizeof(float));

	timing = omp_get_wtime();
    for (i=0; i<nloops; i++) {
        
        cluster_centres = NULL;
        cluster(numObjects,
                numAttributes,
                attributes,           /* [numObjects][numAttributes] */                
                nclusters,
                &cluster_centres   
               );
     
    }
    timing = omp_get_wtime() - timing;
	

	printf("number of Clusters %d\n",nclusters); 
	printf("number of Attributes %d\n\n",numAttributes); 
  /*  	printf("Cluster Centers Output\n"); 
	printf("The first number is cluster number and the following data is arribute value\n");
	printf("=============================================================================\n\n");
	
    for (i=0; i< nclusters; i++) {
		printf("%d: ", i);
        for (j=0; j<numAttributes; j++)
            printf("%.2f ", cluster_centres[i][j]);
        printf("\n\n");
    }
*/
	fprintf(stderr, "time:%.5lf\n", timing);

    free(attributes);
    free(cluster_centres[0]);
    free(cluster_centres);
    free(buf);
    return(0);
}


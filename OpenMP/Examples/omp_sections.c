#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N     1000

void main ()
{
    int i;
    float a[N], b[N], c[N], d[N];

    /* Some initializations */
    for (i=0; i < N; i++) {
        a[i] = i * 1.5;
        b[i] = i + 22.35;
    }

#pragma omp parallel shared(a,b,c,d) private(i)
    {
        int numberThreads = omp_get_num_threads();
        int threadNumber = omp_get_thread_num();
        printf("numberThreads = %d threadNumber = %d\n", numberThreads, threadNumber);

#pragma omp sections nowait
        {

#pragma omp section
            {
                threadNumber = omp_get_thread_num();
                printf("threadNumber = %d, running section code\n", threadNumber);

                for (i=0; i < N; i++)
                    c[i] = a[i] + b[i];
            }


#pragma omp section
            {
                threadNumber = omp_get_thread_num();
                printf("threadNumber = %d, running section code\n", threadNumber);

                for (i=0; i < N; i++)
                    d[i] = a[i] * b[i];
            }


        }  /* end of sections */



    }  /* end of parallel section */
}

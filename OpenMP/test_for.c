#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

int main(int argc, char *argv[]) {
    int i;
    int j;
    int n = 10;
    int t = 0;

    int a[] = {1, 2, 3, 4, 5};
    for (int k = 0; k < 5; ++k) {
        printf("a[%d] = %d\n", k, a[k]);
    }

    printf("a[-3] = %d\n", a[-3]);

    int *g;
    fill_n(g, 5, 1);
    //g[0] = 1;
    //g[1] = 1;
    //g[2] = 1;
    //g[3] = 1;
    //g[4] = 1;
    for (int k = 0; k < 5; ++k) {
        printf("g[%d] = %d\n", k, g[k]);
    }


    omp_set_num_threads(2);

    #pragma omp parallel private(j)
    {
        #pragma omp for
        for (i = 0; i < n; i++) {
            printf("a\n");
        }

        for (j = 0; j < n; j++) {
            printf("b\n");
        }
    }
}
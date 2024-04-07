#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

int main(int argc, char *argv[]) {

    omp_set_num_threads(3);

    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                printf("A \n");
            }

            #pragma omp section
            {
                printf("B \n");
            }

        }

    }

}
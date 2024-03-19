#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

void methodA() {

    //#pragma omp for nowait
    for (int i = 0; i < 2; i++) {
        #pragma omp task
        {
            char* result;
            char buffer[128];
            bool isContinue = true;

            printf("consumer %d: starting\n", i);
            sleep(10);
            printf("consumer %d: exiting\n", i);
        }
    }
}

void methodB() {

    //#pragma omp for
    for (int i = 0; i < 2; i++) {
        #pragma omp task
        {
            char* result;
            char buffer[128];
            bool isContinue = true;

            printf("producer %d: starting\n", i);

            while (isContinue) {
                //#pragma omp critical
                result = fgets(buffer, 128, stdin);
                if(result != NULL){
                    int number = atoi(buffer);
                    printf("methodB(), Input number = %d\n", number);
                    sleep(4);
                }else{
                    printf("methodB(), fgets() returns NULL.\n");
                    isContinue = false;
                }
            }

            printf("producer %d: exiting\n", i);
        }
    }
}

int main(int argc, char *argv[]) {

    omp_set_num_threads(4);

    char buffer[128];
    //int number = -1;

    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                methodA();
            }
            #pragma omp section
            {
                methodB();
            }
        }
    }

}
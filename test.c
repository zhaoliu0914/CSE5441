#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

int update_buffer(char operation, int number) {
    if (operation == 'i') {
        printf("operation = %c, Insert\n", operation);
        printf("number = %d\n", number);
    } else {
        printf("operation = %c, Extra\n", operation);
    }

}

int main(int argc, char *argv[]) {

    update_buffer('e', 0);

    printf("argc = %d\n", argc);
    for (int i = 0; i < argc; i++) {
        printf("argv[%d] = %s\n", i, argv[i]);
    }

    int buffer[10] = {0};
    for (int i = 0; i < 10; i++) {
        printf("buffer[%d] = %d\n", i, buffer[i]);
    }

    #pragma omp parallel master
    {
        printf("run OpenMP parallel single, Start to sleep 2 seconds.");
        fflush(stdout);

        sleep(3);

        printf("Finish OpenMP parallel single.");
        fflush(stdout);
    }

    char inputBuffer[128];
    int number = -1;


    omp_set_num_threads(6);

    printf("Test for loop input, Please enter a number: \n");
#pragma omp parallel firstprivate(inputBuffer)
    {
        #pragma omp for nowait
        for (int i = 0; i < 4; i++) {
            char* result;
            bool isContinue = true;

            while (isContinue) {
                //#pragma omp critical
                result = fgets(inputBuffer, 128, stdin);
                if(result != NULL){
                    int number = atoi(inputBuffer);
                    printf("Input number = %d\n", number);
                    sleep(4);
                }else{
                    printf("fgets() returns NULL.\n");
                    isContinue = false;
                }
            }
        }

        printf("Complete omp for loop input.\n");
    }

    printf("Complete omp parallel and for loop input.\n");

    //char inputBuffer[128];
    number = -1;
    printf("Test normal case, Please enter a number: \n");
    #pragma omp parallel num_threads(4)
    {
        printf("Running #pragma omp parallel num_threads(4).....\n");
        fflush(stdout);
        while (fgets(inputBuffer, 128, stdin) != NULL) {
            number = atoi(inputBuffer);
            printf("Input number = %d\n", number);
        }
    }


    //char temp[128];
    //while (fgets(temp, 128, stdin) != NULL) {
    //    int number = atoi(temp);
    //    printf("number = %d\n", number);
    //}


    int location = 0;

#pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        printf("running thread, tid = %d\n", tid);

        /* Only master does this */
        if (tid == 0) {
            while (location < 10) {
                usleep(1000000);
                printf("location = %d\n", location);
            }
        } else {
            int i = 10;
            while (i > 0) {
                i--;
                usleep(1000000);
                //printf("location = %d\n", location);
                location++;
            }
        }
    }

}
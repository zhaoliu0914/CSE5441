#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

void methodA() {

    #pragma omp parallel for schedule(static, 2)
    for (int i = 0; i < 2; i++) {
        char* result;
        char buffer[128];
        bool isContinue = true;

        printf("methodA %d: starting\n", i);
        sleep(10);
        printf("methodA %d: exiting\n", i);
    }
}

void methodB() {

    #pragma omp parallel for schedule(static, 2)
    for (int i = 0; i < 2; i++) {
        char* result;
        char buffer[128];
        bool isContinue = true;

        printf("methodB %d: starting\n", i);

        while (isContinue) {
            //#pragma omp critical
            result = fgets(buffer, 128, stdin);
            if(result != NULL){
                int number = atoi(buffer);
                printf("methodB() %d, Input number = %d\n", i, number);
                sleep(4);
            }else{
                printf("methodB() %d, fgets() returns NULL.\n", i);
                isContinue = false;
            }
        }

        printf("methodB %d: exiting\n", i);
    }
}

int main(int argc, char *argv[]) {

    omp_set_num_threads(4);

    char buffer[128];
    //int number = -1;

    #pragma omp parallel
    {
        methodA();
        methodB();
    }

}
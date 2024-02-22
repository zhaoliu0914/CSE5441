#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

int update_buffer(char operation, int number){
    if (operation == 'i'){
        printf("operation = %c, Insert\n", operation);
        printf("number = %d\n", number);
    } else {
        printf("operation = %c, Extra\n", operation);
    }

}

int main(int argc, char *argv[]) {

    update_buffer('e', 0);

    printf("argc = %d\n", argc);
    for(int i=0; i<argc; i++){
        printf("argv[%d] = %s\n", i, argv[i]);
    }

    int buffer[10] = {0};
    for(int i=0; i<10; i++){
        printf("buffer[%d] = %d\n", i, buffer[i]);
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
            while(location < 10){
                usleep(1000000);
                printf("location = %d\n", location);
            }
        } else {
            int i = 10;
            while(i > 0){
                i--;
                usleep(1000000);
                //printf("location = %d\n", location);
                location++;
            }
        }
    }

}
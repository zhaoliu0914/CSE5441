/* 
 * This problem has you solve the classic "bounded buffer" problem with
 * multiple producers and multiple consumers:
 *
 *  ------------                         ------------
 *  | producer |-\                    /->| consumer |
 *  ------------ |                    |  ------------
 *               |                    |
 *  ------------ |                    |  ------------
 *  | producer | ----> bounded buffer -->| consumer |
 *  ------------ |                    |  ------------
 *               |                    |
 *  ------------ |                    |  ------------
 *  | producer |-/                    \->| consumer |
 *  ------------                         ------------
 *
 *  The program below includes everything but the implementation of the
 *  bounded buffer itself.  main() should do the following
 *
 *  1. starts N producers as per the first argument (default 1)
 *  2. starts N consumers as per the second argument (default 1)
 *
 *  The producer reads positive integers from standard input and passes those
 *  into the buffer.  The consumers read those integers and "perform a
 *  command" based on them (all they really do is sleep for some period...)
 *
 *  on EOF of stdin, the first producer passes N copies of -1 into the buffer.
 *  The consumers interpret -1 as a signal to exit.
 */

#include <stdio.h>
#include <stdlib.h>             /* atoi() */
#include <unistd.h>             /* usleep() */
#include <assert.h>             /* assert() */
#include <signal.h>             /* signal() */
#include <alloca.h>             /* alloca() */
#include <omp.h>
#include <stdbool.h>

void print_insertion(int producerno, int number, int location) {
    fflush(stdout);
    printf("producer %d inserting %d at location %d\n", producerno, number, location);
    fflush(stdout);
}

void print_extraction(int consumerno, int number, int location) {
    fflush(stdout);
	printf("consumer %d extracting %d from location %d\n", consumerno, number, location);
    fflush(stdout);
}

/**************************************************************************\
 *                                                                        *
 * Bounded buffer.                                                        *
 * Your buffer should have space for up to 10 integers in it at a time.   *
 *                                                                        *
 * Add any data structures you need (globals are fine) and fill in        *
 * implementations for these procedures:                                  *
 *                                                                        *
 * void insert_data(int producerno, int number)                           *
 *                                                                        *
 *      insert_data() inserts a number into the next available slot in    *
 *      the buffer.  If no slots are available, the thread should wait    *
 *      for an empty slot to become available.                            *
 *      Note: multiple producer may call insert_data() simulaneously.     *
 *                                                                        *
 * int extract_data(int consumerno)                                       *
 *                                                                        *
 *      extract_data() removes and returns the number in the next         *
 *      available slot.  If no number is available, the thread should     *
 *      wait for a number to become available.                            *
 *      Note: multiple consumers may call extract_data() simulaneously.   *
 *                                                                        *
\**************************************************************************/

/* DO NOT change MAX_BUF_SIZE */
#define MAX_BUF_SIZE    10

/* The simplest way to implement this would be with a stack as below. But you
 * are free to choose any other data structure you prefer. */
int buffer[MAX_BUF_SIZE] = {0};
int location = 0;

void insert_data(int producerno, int number)
{
    int insertedLocation = -1;
    bool isContinue = true;

    /* Wait until consumers consumed something from the buffer and there is space */
    while(isContinue){
        #pragma omp critical
        {
            if(location < MAX_BUF_SIZE){
                buffer[location] = number;
                insertedLocation = location;
                if(number == -1){
                    int index = location;
                    for(int i=0; i<=location; i++){
                        int temp = buffer[i];
                        if(temp > 0){
                            index = i;
                            break;
                        }
                    }
                    insertedLocation = index;
                    int temp = buffer[index];
                    buffer[index] = buffer[location];
                    buffer[location] = temp;
                }
                location++;
                isContinue = false;
            }
        }
        if(isContinue){
            usleep(100);
        }
    }

    /* Put data in the buffer */

    print_insertion(producerno, number, insertedLocation);
}

int extract_data(int consumerno)
{
    int done = 0;
    int value = -1;
    int extractedLocation = -1;

    /* Wait until producers have put something in the buffer */
    bool isContinue = true;
    while(isContinue){
        #pragma omp critical
        {
            if(location > 0){
                location--;
                value = buffer[location];
                isContinue = false;
                extractedLocation = location;
            }
        }
        if(isContinue){
            usleep(100);
        }
    }

    print_extraction(consumerno, value, extractedLocation);
    return value;
}

/**************************************************************************\
 *                                                                        *
 * The consumer. Each consumer reads and "interprets"                     *
 * numbers from the bounded buffer.                                       *
 *                                                                        *
 * The interpretation is as follows:                                      *
 *                                                                        *
 * o  positive integer N: sleep for N * 100ms                             *
 * o  negative integer:  exit                                             *
 *                                                                        *
\**************************************************************************/

void consumer(int nproducers, int nconsumers)
{
    /* Do not move this declaration */
    int number = -1;
    int consumerno = -1;

    for (consumerno = 0; consumerno < nconsumers; consumerno++) {
        #pragma omp task firstprivate(number)
        {
            printf("consumer %d: starting\n", consumerno);
            fflush(stdout);
            while (1)
            {
                number = extract_data(consumerno);

                if (number < 0)
                    break;

                //usleep(10 * number);  /* "interpret" command for development */
                usleep(100000 * number);  /* "interpret" command for submission */
                fflush(stdout);
            }
            printf("consumer %d: exiting\n", consumerno);
            fflush(stdout);
        }
    }

    return;
}

/**************************************************************************\
 *                                                                        *
 * Each producer reads numbers from stdin, and inserts them into the      *
 * bounded buffer.  On EOF from stdin, it finished up by inserting a -1   *
 * for every consumer so that all the consumers exit cleanly              *
 *                                                                        *
\**************************************************************************/

#define MAXLINELEN 128

void producer(int nproducers, int nconsumers)
{
    int i = 0;
    /* Thread number */
    int producerno = -1;
    char buffer[MAXLINELEN];
    int number = -1;

    #pragma omp taskgroup
    {
        for (producerno = 0; producerno < nproducers; producerno++) {
            #pragma omp task firstprivate(buffer, number)
            {
                printf("producer %d: starting\n", producerno);
                fflush(stdout);

                while (fgets(buffer, MAXLINELEN, stdin) != NULL)
                {
                    number = atoi(buffer);
                    insert_data(producerno, number);
                }

                printf("producer %d: exiting\n", producerno);
                fflush(stdout);
            }
        }
    }

//#pragma omp taskwait

    /* For simplicity, you can make it so that only one producer inserts the
     * "-1" to all the consumers. However, if you are able to create a logic to
     * distribute this among the producer threads, that is also fine. */
    printf("producer: read EOF, sending %d '-1' numbers\n", nconsumers);
    fflush(stdout);

    for (i = 0; i < nconsumers; i++) {
        insert_data(-1, -1);
    }

    printf("producer %d: exiting\n", -1);
    fflush(stdout);
}

/*************************************************************************\
 *                                                                       *
 * main program.  Main calls does necessary initialization of OpenMP     *
 * threads. Calls the main consumer and producer functions which         *
 * extracts and inserts data in parallel.                                *
 *                                                                       *
\*************************************************************************/

int main(int argc, char *argv[])
{
    int tid = -1;
    int nproducers = 1;
    int nconsumers = 1;

    if (argc != 3) {
        fprintf(stderr, "Error: This program takes one inputs.\n");
        fprintf(stderr, "e.g. ./a.out nproducers nconsumers < <input_file>\n");
        exit (1);
    } else {
        nproducers = atoi(argv[1]);
        nconsumers = atoi(argv[2]);
        if (nproducers <= 0 || nconsumers <= 0) {
            fprintf(stderr, "Error: nproducers & nconsumers should be >= 1\n");
            exit (1);
        }
    }

    printf("main: nproducers = %d, nconsumers = %d\n", nproducers, nconsumers);

    omp_set_num_threads(nproducers + nconsumers);

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            /* Spawn N Consumer OpenMP Threads */
            consumer(nproducers, nconsumers);
        }
        //#pragma omp section
        #pragma omp single
        {
            /* Spawn N Producer OpenMP Threads */
            producer(nproducers, nconsumers);
        }
    }

    return(0);
}


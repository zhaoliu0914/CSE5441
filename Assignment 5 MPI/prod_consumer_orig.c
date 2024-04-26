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
#include <omp.h>                /* For OpenMP */
#include <mpi.h>                /* For MPI */

/**************************************************************************\
 *                                                                        *
 * Bounded buffer.  This is the only part you need to modify.  Your       *
 * buffer should have space for up to 10 integers in it at a time.        *
 *                                                                        *
 * Add any data structures you need (globals are fine) and fill in        *
 * implementations for these two procedures:                              *
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

/* DO NOT change MAX_BUF_SIZE or MAX_NUM_PROCS */
#define MAX_BUF_SIZE    10
#define MAX_NUM_PROCS   5
int buffer[MAX_BUF_SIZE] = {0};
int location = 0;
int num_procs = -1, myid = -1;
char hostname[MPI_MAX_PROCESSOR_NAME];

void print_insertion(int producerno, int number, int location) {
    printf("%s: producer %d on process %d inserting %d at location %d\n", hostname, producerno, myid, number,
            location);
}

void print_extraction(int consumerno, int number, int location) {
    printf("%s: consumer %d on process %d extracting %d from location %d\n", hostname, consumerno, myid, number,
            location);
}

void insert_data(int producerno, int number) {
    /* This print must be present in this function. Do not remove this print.
     * Used for data validation */
    print_insertion(myid, producerno, number, location);
}

int extract_data(int consumerno) {
    int value = -1;

    /* This print must be present in this function. Do not remove this print.
     * Used for data validation */
    print_extraction(myid, consumerno, value, location);

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

void consumer(int nproducers, int nconsumers) {
    /* Do not move this declaration */
    int number = -1;
    int consumerno = -1;

    {
        while (1) {
            number = extract_data(consumerno);

            /* Do not remove this print. Used for data validation */
            if (number < 0)
                break;

            usleep(10 * number);
            // usleep(100000 * number);
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

void producer(int nproducers, int nconsumers) {
    int number, producerno;
    char tmp_buffer[MAXLINELEN];

    {
        while (fgets(tmp_buffer, MAXLINELEN, stdin) != NULL) {
            number = atoi(tmp_buffer);
            insert_data(producerno, number);
        }
    }
}

/*************************************************************************\
 *                                                                       *
 * main program.  Main calls does necessary initialization.              *
 * Calls the main consumer and producer functions which extracts and     *
 * inserts data in parallel.                                             *
 *                                                                       *
\*************************************************************************/

int main(int argc, char *argv[]) {
    int tid = -1, len = 0;
    int nproducers = 1;
    int nconsumers = 1;

    if (argc != 3) {
        nproducers = 8;
        nconsumers = 8;
    } else {
        nproducers = atoi(argv[1]);
        nconsumers = atoi(argv[2]);
        if (nproducers <= 0 || nconsumers <= 0) {
            fprintf(stderr, "Error: nproducers & nconsumers should be >= 1\n");
            exit(1);
        }
    }

    /***** MPI Initializations - get rank, comm_size and hostame - refer to
     * bugs/examples for necessary code *****/
    if (num_procs > MAX_NUM_PROCS) {
        fprintf(stderr, "Error: Max num procs should <= 5\n");
        exit(1);
    }

    printf("main: nproducers = %d, nconsumers = %d\n", nproducers, nconsumers);

    /* Spawn N Consumer OpenMP Threads */
    consumer(nproducers, nconsumers);
    /* Spawn N Producer OpenMP Threads */
    producer(nproducers, nconsumers);

    /* Finalize and cleanup */
    return(0);
}

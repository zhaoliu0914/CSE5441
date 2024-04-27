/* Copyright (c) 1993-2015, CS Department of OSU. All rights reserved.*/

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
#include <stdbool.h>

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
    fprintf(stderr, "%s: producer %d on process %d inserting %d at location %d\n",
            hostname, producerno, myid, number, location);
}

void print_extraction(int consumerno, int number, int location) {
    fprintf(stderr, "%s: consumer %d on process %d extracting %d from location %d\n",
            hostname, consumerno, myid, number, location);
}

void insert_data(int producerno, int number) {
    int insertedLocation = -1;
    bool isContinue = true;

    /* Wait until consumers consumed something from the buffer and there is space */
    while (isContinue) {
#pragma omp critical
        {
            if (location < MAX_BUF_SIZE) {
                buffer[location] = number;
                insertedLocation = location;
                if (number == -1) {
                    int index = location;
                    for (int i = 0; i <= location; i++) {
                        int temp = buffer[i];
                        if (temp > 0) {
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
        if (isContinue) {
            usleep(100);
        }
    }

    /* Put data in the buffer */

    print_insertion(producerno, number, insertedLocation);
}

int extract_data(int consumerno) {
    int done = 0;
    int value = -1;
    int extractedLocation = -1;

    /* Wait until producers have put something in the buffer */
    bool isContinue = true;
    while (isContinue) {
#pragma omp critical
        {
            if (location > 0) {
                location--;
                value = buffer[location];
                isContinue = false;
                extractedLocation = location;
            }
        }
        if (isContinue) {
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

void consumer(int nproducers, int nconsumers) {
    /* Do not move this declaration */
    int number = -1;
    int consumerno = -1;

    consumerno = omp_get_thread_num();
    if (consumerno < nconsumers) {
        fprintf(stderr, "consumer %d: starting\n", consumerno);
        while (1) {
            number = extract_data(consumerno);

            if (number < 0)
                break;

            //usleep(10 * number);  /* "interpret" command for development */
            usleep(100000 * number);  /* "interpret" command for submission */
        }
        fprintf(stderr, "consumer %d: exiting\n", consumerno);
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
    /* Thread number */
    int producerno = -1;
    char buffer[MAXLINELEN];
    int number = -1;
    int source = 0;
    int tag = 100;
    int rc;
    MPI_Status Stat;

    producerno = omp_get_thread_num();
    if (producerno >= nconsumers) {
        fprintf(stderr, "producer %d: starting\n", producerno);

        if (myid == 0) {
            while (fgets(buffer, MAXLINELEN, stdin) != NULL) {
                number = atoi(buffer);

                int destination = rand() % nproducers;
                if (destination == 0) {
                    insert_data(producerno, number);
                } else {
                    //fprintf(stderr, "producer: Process %d Thread %d send [%d] to %d\n", myid,  producerno, number, destination);
                    rc = MPI_Send(&number, 1, MPI_INT, destination, tag, MPI_COMM_WORLD);
                }
            }
#pragma omp master
            {
                number = -1;
                for (int i = 0; i < nproducers; i++) {
                    for (int j = 0; j < num_procs; ++j) {
                        rc = MPI_Send(&number, 1, MPI_INT, j, tag, MPI_COMM_WORLD);
                    }
                    //fprintf(stderr, "producer: Process %d Thread %d send [%d] to %d\n", myid, producerno, number, i);
                }
            }

        } else {
            while (number != -1) {
                rc = MPI_Recv(&number, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &Stat);
                //fprintf(stderr, "producer: Process %d Thread %d receive [%d] from %d\n", myid, producerno, number, source);
                insert_data(producerno, number);
            }

        }

        fprintf(stderr, "producer %d: exiting\n", producerno);
    }

    /* For simplicity, you can make it so that only one producer inserts the
     * "-1" to all the consumers. However, if you are able to create a logic to
     * distribute this among the producer threads, that is also fine. */

    if (producerno == (nproducers + nconsumers - 1)) {
        fprintf(stderr, "producer: read EOF, sending %d '-1' numbers\n", nconsumers);

        for (int i = 0; i < nconsumers; i++) {
            insert_data(-1, -1);
        }

        fprintf(stderr, "producer %d: exiting\n", -1);
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
    fprintf(stderr, "Start MPI_Init().......\n");
    MPI_Init(&argc, &argv);
    fprintf(stderr, "Finish MPI_Init().......\n");
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (num_procs > MAX_NUM_PROCS) {
        fprintf(stderr, "Error: Max num procs should <= 5\n");
        exit(1);
    }

    fprintf(stderr, "main: nproducers = %d, nconsumers = %d\n", nproducers, nconsumers);

    omp_set_num_threads(nproducers + nconsumers);

#pragma omp parallel
    {
        /* Spawn N Consumer OpenMP Threads */
        consumer(nproducers, nconsumers);
        /* Spawn N Producer OpenMP Threads */
        producer(nproducers, nconsumers);
    }

    /* Finalize and cleanup */
    MPI_Finalize();
    return (0);
}

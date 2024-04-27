#include <stdio.h>
#include <stdlib.h>             /* atoi() */
#include <unistd.h>             /* usleep() */
#include <assert.h>             /* assert() */
#include <signal.h>             /* signal() */
#include <alloca.h>             /* alloca() */
#include <omp.h>                /* For OpenMP */
#include <mpi.h>                /* For MPI */
#include <stdbool.h>


#define MAXLINELEN 128

int main(int argc, char *argv[]) {
    int num_procs = -1;
    int myid = -1;
    int tag = 9;
    int rc;
    int numberThreads = 3;
    MPI_Status Stat;

    fprintf(stderr, "Start MPI_Init().......\n");
    MPI_Init(&argc, &argv);
    fprintf(stderr, "Finish MPI_Init().......\n");
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    omp_set_num_threads(numberThreads);

    #pragma omp parallel
    {
        int producerno = omp_get_thread_num();
        int number;

        if (myid == 0) {
            int destination = 1;
            int number = 123456;
            char buffer[MAXLINELEN];

            while (fgets(buffer, MAXLINELEN, stdin) != NULL) {
                number = atoi(buffer);

                fprintf(stderr, "producer: Process %d Thread %d send [%d] to %d\n", myid,  producerno, number, destination);
                rc = MPI_Send(&number, 1, MPI_INT, destination, tag, MPI_COMM_WORLD);
            }
        } else {
            int source = 0;

            rc = MPI_Recv(&number, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &Stat);
            fprintf(stderr, "producer: Process %d Thread %d receive [%d] from %d\n", myid, producerno, number, 0);
        }
    }

    fprintf(stderr, "Finish and Cleanup.......\n");
    /* Finalize and cleanup */
    MPI_Finalize();
    return(0);
}
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // Determine root's rank
    int root_rank = 0;

    // Get the size of the communicator
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        printf("This application is meant to be run with 4 MPI processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Get my rank
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int send_value = my_rank + 1;

    // Each MPI process sends its rank to reduction, root MPI process collects the result
    int reduction_result = 20;
    MPI_Reduce(&send_value, &reduction_result, 1, MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

    //if (my_rank == root_rank) {
        printf("The sum of all ranks is %d.\n", reduction_result);
    //}

    MPI_Finalize();

    return EXIT_SUCCESS;
}
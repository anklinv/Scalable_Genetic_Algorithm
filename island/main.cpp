
#include <mpi.h> /* requirement for MPI */

#include <stdio.h>
#include <stdlib.h>

#include "travelling_salesman_problem.hpp"
#include "island.hpp"

using namespace std;


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv); /* requirement for MPI */

    
    // execution of the GA
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank != 0) {
        
        // all ranks but 0 send the result to rank 0
        
    } else {
        
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // rank 0 assembles the final result
        
    }
    
    printf("process rank: %d\n", rank);
        
    
    MPI_Finalize(); /* requirement for MPI */
    
    return 0;
}


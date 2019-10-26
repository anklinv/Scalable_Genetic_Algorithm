
#include <mpi.h> /* requirement for MPI */

#include <stdio.h>
#include <stdlib.h>

#include "travelling_salesman_problem.hpp"


using namespace std;


/* global parameters for the GA:
 *
 * - frequency
 * - extent
 * - selection policy (source)
 * - replacement policy (target)
 * - topology
 */






int main(int argc, char** argv) {

    int rank;
    
    
    MPI_Init(&argc, &argv); /* requirement for MPI */

    
    // execution of the GA
    
    
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


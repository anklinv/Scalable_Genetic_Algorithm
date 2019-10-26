
#include "mpi.h" /* requirement for MPI */

#include "travelling_salesman_problem.hpp"

/**
 An Island wraps around a TravellingSalesmanProblem
 - Fitness evaluation (Ceval) and cross-over / mutation (Coper) is done
  by the underlying TSP
 - The Island does the communication (Ccomm)
 */

/**
 Idea: consider making Island a subclass of TravellingSalesmanProblem
 - change the access modifier of important fields to protected
 */

class Island {
    
public:
    
    /**
     - Get the parameters for the underlying TSP
     - Get frequency, extent
     */
    Island(TravellingSalesmanProblem* tsp, int migrationPeriod, int migrationAmount);
    
    enum Topology {
        FULLY_CONNECTED;
    }
    
    /**
     Run the GA
     */
    double solve();
    
    
private:
    
    /**
     Use a pointer to avoid dealing with object initialization
     */
    TravellingSalesmanProblem* tsp;
    
    int migrationPeriod;
    int migrationAmount;
    
    void send();
    void receiveAll();
    
}


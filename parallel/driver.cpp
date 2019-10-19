
#include <mpi.h> /* requirement for MPI */

#include "travelling_salesman_problem.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <iomanip>


using namespace std;


// constants
int DATA_TAG = 0x011;


int main(int argc, char** argv) {
    
    MPI_Init(&argc, &argv); /* requirement for MPI */
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // each process runs the GA and stores the result
    double final_distance = setupAndRunGA(rank);
    
    if (rank != 0) {
        
        // send result to rank 0
        MPI_Send(&final_distance, 1, MPI_DOUBLE, 0, DATA_TAG, MPI_COMM_WORLD);
        
    } else {
        
        int numProcesses;
        MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
        
        double res = final_distance;
        double buff;
        
        // receive results from all other ranks
        // combine results to find the optimum
        for(int i = 0; i < numProcesses; i++) {
            
            MPI_Recv(&buff, 1, MPI_DOUBLE, MPI_ANY_SOURCE, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            res = min(res, buff);
        }
        
        // output optimum
        cout << "Final result: " << res << endl;
    }
    
    
    MPI_Finalize(); /* requirement for MPI */
    
    return 0;
}


int parseGraphHeader(string& graph) {
    
    ifstream input(graph);
    string num_cities;
    
    for (int i = 6; i > 0; i--) {
        
        string line_buffer;
        getline(line_buffer);
        
        if(i == 3) num_cities =
            line_buffer.substr(12, line_buffer.length());
    }
    
    return stoi(num_cities);
}


double setupAndRunGA(int rank) {
    
    string graph = "../data/att48.tsp";
    int num_cities = parseGraphHeader(graph);

    TravellingSalesmanProblem problem(num_cities, 100, 10, 0.05);

#ifdef debug
    cout << "Reading " << number_cities << " cities ... (rank " << rank << ")" << endl;
#endif
    
    for (int i = 0; i < num_cities; ++i) {
        
        int index;
        double x, y;
        
        input >> index >> x >> y;
        problem.cities.push_back({x, y});
    }
    
    input.close();
    
#ifdef debug
    cout << "Done!  (rank " << rank << ")" << endl;

    cout << "Start running the GA ...  (rank " << rank << ")" << endl;
#endif
    
    double final_distance = problem.solve(1000);

#ifdef debug
    cout << "Done!  (rank " << rank << ")" << endl;
#endif
    
    return final_distance;
}


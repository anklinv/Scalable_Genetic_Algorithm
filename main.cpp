
#include <mpi.h> /* requirement for MPI */

#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>

#include "sequential/travelling_salesman_problem.hpp"


using namespace std;


// typedefs
typedef vector<double> vec_d;

// constants
const int DATA_TAG = 0x011;


double setupAndRunGA(int rank);

double computeStdDev(vec_d data);

double computeMean(vec_d data);


int main(int argc, char** argv) {
    
    MPI_Init(&argc, &argv); /* requirement for MPI */
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    
    // reads the graph from a file and runs the GA
    double final_distance = setupAndRunGA(rank);
    
    if (rank != 0) {
        
        // send result to rank 0
        MPI_Send(&final_distance, 1, MPI_DOUBLE, 0, DATA_TAG, MPI_COMM_WORLD);
        
    } else {
        
        int numProcesses;
        MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
        
        double buff;
        
        vec_d all_dists;
        all_dists.push_back(final_distance);
        
        // receive results from all other ranks
        for(int i = 0; i < numProcesses - 1; i++) {
            
            MPI_Recv(&buff, 1, MPI_DOUBLE, MPI_ANY_SOURCE, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            all_dists.push_back(buff);
            final_distance = min(buff, final_distance);
        }
        
        double mean = computeMean(all_dists);
        double stddev = computeStdDev(all_dists);
        
        cout << "Best final distance overall is " << final_distance << endl;
        cout << "(mean is " << mean << ", std dev is " << stddev << ")" << endl;
    }
    
    
    MPI_Finalize(); /* requirement for MPI */
    
    return 0;
}


double setupAndRunGA(int rank) {
    
    // TODO: Make this nicer, the files are not as consistent as I hoped.
    //       The files can be found at http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html
    ifstream input("data/att48.tsp");
    string name, comment, type, dimension, edge_weight_type, node;
    
    getline(input, name);
    name = name.substr(7, name.length());
    
    getline(input, comment);
    comment = comment.substr(10, comment.length());
    
    getline(input, type);
    type = type.substr(7, type.length());
    
    getline(input, dimension);
    dimension = dimension.substr(12, dimension.length());
    
    getline(input, edge_weight_type);
    edge_weight_type = edge_weight_type.substr(18, edge_weight_type.length());
    
    getline(input, node);
    
    // Read cities
    int number_cities = stoi(dimension);
    TravellingSalesmanProblem problem(number_cities, 100, 10, 0.05);
    cout << "Reading " << dimension << " cities of problem " << name << "... (rank " << rank << ")" << endl;
    // Read city coordinates
    for (int i = 0; i < number_cities; ++i) {
        int index;
        double x, y;
        input >> index >> x >> y;
        problem.cities.push_back({x,y});
    }
    input.close();
    cout << "Done! (rank " << rank << ")" << endl;
    
    // Solve problem
    double final_distance;
    final_distance = problem.solve(1000);
    cout << "Final distance is " << final_distance << " (rank " << rank << ")" << endl;
    
    // TODO: Graph, maybe visualization
    
    return final_distance;
}


double computeStdDev(vec_d data) {
    
    double mean = accumulate(data.begin(), data.end(), 0.0) / data.size();
    
    double sum_squares = 0.0;
    
    for(auto it = data.begin(); it != data.end(); it++) {
        sum_squares += (*it - mean) * (*it - mean);
    }
    
    sum_squares = sum_squares / data.size();
    
    return sqrt(sum_squares);
}


double computeMean(vec_d data) {
    
    double sum = accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}


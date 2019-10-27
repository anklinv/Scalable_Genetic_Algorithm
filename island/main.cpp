
#include <mpi.h> /* requirement for MPI */

#include <stdio.h>
#include <stdlib.h>

#include <iostream> // needed to read files
#include <string>
#include <fstream>
#include <sstream>

#include "travelling_salesman_problem.hpp"
#include "island.hpp"


using namespace std;


TravellingSalesmanProblem makeTSP(int rank);


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv); /* requirement for MPI */

    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    TravellingSalesmanProblem tsp = makeTSP(rank);
    
    // 1000 epochs is def
    Island island(tsp, 100, 5, 10); // period, amount, numPeriods
    
    
    MPI_Finalize(); /* requirement for MPI */
    
    return 0;
}


/**
 Factory function.
 */
TravellingSalesmanProblem makeTSP(int rank) {
    
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
    int node_edge_mat[number_cities * number_cities];
    cout << "Reading " << dimension << " cities of problem " << name << "... (rank " << rank << ")" << endl;
    // Instead of reading city coordinates here, load distance matrix from a preprocessed file
    input.close();

    // Read city coordinates
    ifstream input2("data/att48.csv");
    
    for (int i = 0; i < number_cities; ++i) {
        string line;
        getline(input2, line);
        if(!input2.good()){
            break;
        }
        stringstream iss(line);
        
        for (int j = 0; j < number_cities; ++j) {
            string val;
            getline(iss, val, ';');
            if(!iss.good()){
                break;
            }
            stringstream converter(val);
            converter >> node_edge_mat[i + number_cities * j];
        }
    }
    input2.close();
    cout << "Done!" << endl;

    TravellingSalesmanProblem problem(number_cities, 100, 10, 16);
    problem.set_logger(new Logger(rank));
    problem.cities = node_edge_mat;
    
    return problem;
}


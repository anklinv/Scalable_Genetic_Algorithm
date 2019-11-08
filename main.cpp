
#include <mpi.h> /* requirement for MPI */

#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <sstream>
#include <chrono>
#include "sequential/travelling_salesman_problem.hpp"
#include "island/island.hpp"
#include <assert.h>

using namespace std;

// Global variables (parameters)
int nr_epochs = 1000;
int nr_individuals = 100;
bool runSequential = true;
bool runIsland = false;
string data_dir = "data";
string data_file = "att48.csv";
string log_dir = "logs/";
int migration_period = 200;
int migration_amount = 5;
int num_migrations = 5;


// typedefs
typedef vector<double> vec_d;

// constants
const int DATA_TAG = 0x011;
const int THREADS_PER_ISLAND = 2;

inline bool file_exists(const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

void parse_args(int argc, char** argv, bool verbose=true) {
    if (verbose) {
        cout << "Found " << argc - 1 << " arguments." << endl;
    }
    for (int i = 1; i < argc; ++i) {
        // Single arguments
        if (argv[i] == (string) "sequential") {
            runSequential = true;
            runIsland = false;
            if (verbose) {
                cout << "Mode " << argv[i] << endl;
            }
        } else if (argv[i] == (string) "island") {
            runIsland = true;
            runSequential = false;
            if (verbose) {
                cout << "Mode " << argv[i] << endl;
            }
        }

        // Dual arguments
        else if (argv[i] == (string) "--epochs") {
            assert(i + 1 < argc);
            try {
                nr_epochs = stoi(argv[i+1]);
            } catch (const std::invalid_argument &e) {
                cerr << "Invalid integer for " << argv[i] << endl;
                exit(1);
            }
            if (verbose) {
                cout << "Number of epochs:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--data") {
            assert(i + 1 < argc);
            if (file_exists(data_dir + "/" + argv[i+1])) {
                data_file = argv[i+1];
            } else {
                cerr << "Invalid data file" << endl;
                exit(1);
            }
            if (verbose) {
                cout << "Problem name:\t\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--population") {
            assert(i + 1 < argc);
            try {
                nr_individuals = stoi(argv[i+1]);
            } catch (const std::invalid_argument &e) {
                cerr << "Invalid integer for " << argv[i] << endl;
                exit(1);
            }
            if (verbose) {
                cout << "Number of individuals:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--log_dir") {
            assert(i + 1 < argc);
            log_dir = argv[i + 1];
            if (verbose) {
                cout << "Logging location:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--migration_period") {
            assert(i + 1 < argc);
            try {
                migration_period = stoi(argv[i+1]);
            } catch (const std::invalid_argument &e) {
                cerr << "Invalid integer for " << argv[i] << endl;
                exit(1);
            }
            if (verbose) {
                cout << "Migration Period:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--migration_amount") {
            assert(i + 1 < argc);
            try {
                migration_amount = stoi(argv[i+1]);
            } catch (const std::invalid_argument &e) {
                cerr << "Invalid integer for " << argv[i] << endl;
                exit(1);
            }
            if (verbose) {
                cout << "Migration Amount:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--num_migrations") {
            assert(i + 1 < argc);
            try {
                num_migrations = stoi(argv[i+1]);
            } catch (const std::invalid_argument &e) {
                cerr << "Invalid integer for " << argv[i] << endl;
                exit(1);
            }
            if (verbose) {
                cout << "Number of Migrations:\t" << argv[i+1] << endl;
            }
        }
    }
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

    TravellingSalesmanProblem problem(number_cities, nr_individuals, 10, 16);
    problem.set_logger(new Logger(log_dir, rank));
    problem.cities = node_edge_mat;

    // Solve problem
    double final_distance;
    final_distance = problem.solve(nr_epochs, rank);
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

// Data file
// --data att48.csv
// Number of individuals (per island if island model)
// --population 100
// Logging direction
// --log_dir folder_name_in_logs
//
// Run sequential:
// sequential --epochs 1000
//
// Run island
// island
int main(int argc, char** argv) {

    // Parse arguments and save in global variables
    parse_args(argc, argv);

    MPI_Init(&argc, &argv); /* requirement for MPI */

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (runSequential) {

        // reads the graph from a file and runs the GA
        auto start = chrono::high_resolution_clock::now();
        double final_distance = setupAndRunGA(rank);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << duration.count() << " ms total runtime (rank " << rank << ")" << endl;

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

    } // end runSequential


    if (runIsland) {
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

        TravellingSalesmanProblem problem(number_cities, nr_individuals, 10, 16);

        // TODO: Pass log_dir to the logger
        problem.set_logger(new Logger(log_dir, rank));
        problem.cities = node_edge_mat;

        // 1000 epochs is def
        Island island(&problem, migration_period, migration_amount, num_migrations); // period, amount, numPeriods
        double bestDistance = island.solve();

        if(rank == 0) {
            cout << "Best final distance overall is " << bestDistance << endl;
        }
    } // end runIsland


    MPI_Finalize(); /* requirement for MPI */

    return 0;
}

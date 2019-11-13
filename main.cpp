
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
#include <cassert>

using namespace std;

// Global variables (parameters)

// --epochs
int nr_epochs = 50000;

// --population
int nr_individuals = 1000;

// island or sequential
bool runIsland = false;

string data_dir = "data";

// --data
string data_file = "a280.csv";

// --log_dir
string log_dir = "logs/";

// --elite_size
int elite_size = 16;

// --mutation
int mutation = 16;

// --migration_period
int migration_period = 200;

// --migration_amount
int migration_amount = 5;

// --migration_topology {isolated, ring, fully_connected}
Island::MigrationTopology migration_topology = Island::MigrationTopology::FULLY_CONNECTED;

// --selection_policy {pure_random, truncation, fitness_proportionate_selection, stochastic_universal_sampling, tournament_selection}
Island::SelectionPolicy selection_policy = Island::SelectionPolicy::TOURNAMENT_SELECTION;

// --replacement_policy {pure_random, truncation, dejong_crowding}
Island::ReplacementPolicy replacement_policy = Island::ReplacementPolicy::TRUNCATION;

// --verbose
int verbose = 1;


// typedefs
typedef vector<double> vec_d;

// constants
const int DATA_TAG = 0x011;
const int THREADS_PER_ISLAND = 2;

inline bool file_exists(const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

void parse_args(int argc, char** argv, bool verbose_args=false) {
    if (verbose_args) {
        cout << "Found " << argc - 1 << " arguments." << endl;
    }
    for (int i = 1; i < argc; ++i) {
        // Single arguments
        if (argv[i] == (string) "sequential") {
            runIsland = false;
            if (verbose_args) {
                cout << "Mode " << argv[i] << endl;
            }
        } else if (argv[i] == (string) "island") {
            runIsland = true;
            if (verbose_args) {
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
            if (verbose_args) {
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
            if (verbose_args) {
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
            if (verbose_args) {
                cout << "Number of individuals:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--log_dir") {
            assert(i + 1 < argc);
            log_dir = argv[i + 1];
            if (verbose_args) {
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
            if (verbose_args) {
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
            if (verbose_args) {
                cout << "Migration Amount:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--elite_size") {
            assert(i + 1 < argc);
            try {
                elite_size = stoi(argv[i+1]);
            } catch (const std::invalid_argument &e) {
                cerr << "Invalid integer for " << argv[i] << endl;
                exit(1);
            }
            if (verbose_args) {
                cout << "Elite size:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--mutation") {
            assert(i + 1 < argc);
            try {
                mutation = stoi(argv[i+1]);
            } catch (const std::invalid_argument &e) {
                cerr << "Invalid integer for " << argv[i] << endl;
                exit(1);
            }
            if (verbose_args) {
                cout << "Mutation:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--verbose") {
            assert(i + 1 < argc);
            try {
                verbose_args = stoi(argv[i + 1]);
            } catch (const std::invalid_argument &e) {
                cerr << "Invalid integer for " << argv[i] << endl;
                exit(1);
            }
            if (verbose_args) {
                cout << "Verbose:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--migration_topology") {
            assert(i + 1 < argc);
            if (argv[i+1] == (string) "isolated") {
                migration_topology = Island::MigrationTopology::ISOLATED;
            } else if (argv[i+1] == (string) "ring") {
                migration_topology = Island::MigrationTopology::RING;
            } else if (argv[i+1] == (string) "fully_connected") {
                migration_topology = Island::MigrationTopology::FULLY_CONNECTED;
            } else {
                cerr << "Invalid choice (" << argv[i+1] << ") for " << argv[i] << endl;
                exit(1);
            }
            if (verbose_args) {
                cout << "Migration Topology:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--selection_policy") {
            assert(i + 1 < argc);
            if (argv[i+1] == (string) "pure_random") {
                selection_policy = Island::SelectionPolicy::PURE_RANDOM;
            } else if (argv[i+1] == (string) "truncation") {
                selection_policy = Island::SelectionPolicy::TRUNCATION;
            } else if (argv[i+1] == (string) "fitness_proportionate_selection") {
                selection_policy = Island::SelectionPolicy::FITNESS_PROPORTIONATE_SELECTION;
            } else if (argv[i+1] == (string) "stochastic_universal_sampling") {
                selection_policy = Island::SelectionPolicy::STOCHASTIC_UNIVERSAL_SAMPLING;
            } else if (argv[i+1] == (string) "tournament_selection") {
                selection_policy = Island::SelectionPolicy::TOURNAMENT_SELECTION;
            } else {
                cerr << "Invalid choice (" << argv[i+1] << ") for " << argv[i] << endl;
                exit(1);
            }
            if (verbose_args) {
                cout << "Selection Policy:\t" << argv[i+1] << endl;
            }
        } else if (argv[i] == (string) "--replacement_policy") {
            assert(i + 1 < argc);
            if (argv[i+1] == (string) "pure_random") {
                replacement_policy = Island::ReplacementPolicy::PURE_RANDOM;
            } else if (argv[i+1] == (string) "truncation") {
                replacement_policy = Island::ReplacementPolicy::TRUNCATION;
            } else if (argv[i+1] == (string) "dejong_crowding") {
                replacement_policy = Island::ReplacementPolicy::DEJONG_CROWDING;
            } else {
                cerr << "Invalid choice (" << argv[i+1] << ") for " << argv[i] << endl;
                exit(1);
            }
            if (verbose_args) {
                cout << "Replacement Policy:\t" << argv[i+1] << endl;
            }
        }
    }
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

void read_input(int &num_cities, float*& cities_matrix) {
    // READ INPUT
    ifstream input(data_dir + "/" + data_file);
    // Read number of cities
    string dim;
    getline(input, dim);
    num_cities = stoi(dim);
    cities_matrix = new float[num_cities * num_cities];
    // Read values
    for (int i = 0; i < num_cities; ++i) {
        string line;
        getline(input, line);
        if(!input.good()){
            break;
        }
        stringstream iss(line);

        for (int j = 0; j < num_cities; ++j) {
            string val;
            getline(iss, val, ';');
            if(!iss.good()){
                break;
            }
            stringstream converter(val);
            converter >> cities_matrix[i + num_cities * j];
        }
    }
    input.close();
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
    parse_args(argc, argv, verbose);

    MPI_Init(&argc, &argv); /* requirement for MPI */

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Read input
    int number_cities = -1;
    float* node_edge_mat;
    read_input(number_cities, node_edge_mat);
    assert(number_cities != -1);

    // Create problem
    TravellingSalesmanProblem problem(number_cities, node_edge_mat, nr_individuals, elite_size, mutation, verbose);
    problem.set_logger(new Logger(log_dir, rank));

    // NAIVE PARALLEL MODEL
    if (not runIsland) {
        auto start = chrono::high_resolution_clock::now();
        double final_distance = problem.solve(nr_epochs, rank);
        if (verbose > 0) {
            cout << "Final distance is " << final_distance << " (rank " << rank << ")" << endl;
        }

        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        if (verbose > 0) {
            cout << duration.count() << " ms total runtime (rank " << rank << ")" << endl;
        }

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
            for (int i = 0; i < numProcesses - 1; i++) {

                MPI_Recv(&buff, 1, MPI_DOUBLE, MPI_ANY_SOURCE, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                all_dists.push_back(buff);
                final_distance = min(buff, final_distance);
            }

            double mean = computeMean(all_dists);
            double stddev = computeStdDev(all_dists);

            if (verbose > 0) {
                cout << "Best final distance overall is " << final_distance << endl;
                cout << "(mean is " << mean << ", std dev is " << stddev << ")" << endl;
            }
        }

    // ISLAND MODEL
    } else if (runIsland) {
        
        Island island(problem, migration_topology, migration_amount, migration_period, // immigrants RECEIVED per migration, migration period
                      selection_policy,
                      replacement_policy);
        
        double bestDistance = island.solve(nr_epochs); // number of evolution steps
        if (verbose > 0) {
            cout << bestDistance << endl;
        }
      
    } // end runIsland

    // Delete cities matrix
    delete(node_edge_mat);

    MPI_Finalize(); /* requirement for MPI */

    return 0;
}

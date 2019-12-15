#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "gaul.h"
#include "logging.h"

#if HAVE_MPI != 1
int main(int argc, char **argv) {
    printf("GAUL was not compiled with MPI support.\n");
    exit(EXIT_FAILURE);
}
#else
// #define MAX_TSP(x, y) (((x) > (y)) ? (x) : (y))
// #define MIN(x, y) (((x) < (y)) ? (x) : (y))
// int mutcounter;
// int crossovercounter;
float* cities;
int problem_size;
boolean cevaluate_fitness(population *pop, entity *entity);
void cmutate(population *pop, entity *mother, entity *daughter);
void cbreed(population *pop, entity *mother, entity *father, entity *daughter, entity *son);
boolean cinitialize(population *pop, entity *adam);
// utility functions
int rand_range(const int a, const int b);
void shuffle(int *array, size_t n);
boolean clog_generation(const int generation, population *pop);
int rank;
double best_fitness = 1e9;
int evolution_counter = 0;
// parameters
int num_islands; // initialized from number of processes
int population_size = 1000; // divided evenly among islands
int num_epochs = 1000;
char *log_dir = "logs/";
/*
 * MAKE SURE the log_dir is there (otherwise segfault)
 */

void parse_args(int argc, char** argv) {
    //printf("argc is %d\n", argc);
    for (int i = 1; i < argc; ++i) {
        // Dual arguments
        if (strcmp(argv[i], "--epochs")==0) { // ==0 means equal
            //printf("argc is %d\n", argc);
            //printf("i+1 is %d\n", (i+1));
            assert(i + 1 < argc);
            sscanf(argv[i+1], "%d", &num_epochs);
            printf("number of epochs is %d\n", num_epochs);
        } else if (strcmp(argv[i], "--population")==0) {
            //printf("argc is %d\n", argc);
            //printf("i+1 is %d\n", (i+1));
            assert(i + 1 < argc);
            sscanf(argv[i+1], "%d", &population_size);
            printf("population size is %d\n", population_size);
        } else if (strcmp(argv[i], "--log_dir")==0) {
            //printf("argc is %d\n", argc);
            //printf("i+1 is %d\n", (i+1));
            //printf("strlen is %d\n", strlen(argv[i + 1]));
            assert(i + 1 < argc);
            log_dir = (char*)malloc((strlen(argv[i + 1])+1)
                                    *sizeof(log_dir));
            strcpy(log_dir, argv[i + 1]); // cpy
            printf("log directory is %s\n", log_dir);
        }
    }
}

int main(int argc, char **argv) {
    // parse arguments
    parse_args(argc, argv);
    // read problem
    FILE* fp = fopen("data/d1291.csv", "r"); //d1291.csv
    if(fp == NULL) {
        printf("could not open the csv file\n");
        exit(EXIT_FAILURE);
    }
    char line[8000];
    int number_cities;
    if(fgets(line, sizeof(line), fp)) {
        sscanf(line, "%d", &number_cities);
    }
    problem_size = number_cities;
    cities = (float*)malloc(number_cities*number_cities*sizeof(cities));;
    for(int i = 0; i < number_cities; i++) {
        if(fgets(line, sizeof(line), fp) == 0) {
            printf("error while reading the csv file\n");
            exit(EXIT_FAILURE);
        }
        int line_idx = 0;
        for(int j = 0; j < number_cities; j++) {
            char buff[10];
            int buff_idx = 0;
            while((buff[buff_idx] = line[line_idx]) != ';') {
                buff_idx++;
                line_idx++;
            }
            buff[buff_idx] = '\0';
            line_idx++;
            float cf;
            sscanf(buff, "%f", &cf);
            cities[i + j*number_cities] = cf;
        }
    }
    if(fp != NULL) {
        fclose(fp);
    }
    // start of the actual program
    // int population_size = 100;
    //population *pop = NULL;
    population *slavepop;
    entity *best_individual = NULL;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_islands);
    if(num_islands == 2) {
        //
    } else {
        num_islands = num_islands - 1; // eliminate master
    }
    population *pop[num_islands];
    printf("there are %d islands\n", num_islands);
    printf("Process %d initialised (rank %d)\n", getpid(), rank);
    printf("Problem size is %d\n (rank %d)", problem_size, rank);
    // set up logging
    if(rank == 0) {
        //char *log_dir = "logs/";
        setup_logger(log_dir, rank);
        open_logger();
    }
    if(rank != 0) {
       /* A population is created so that the callbacks are defined.  Evolution doesn't
         * occur with this population, so population_size can be zero.  In such a case,
         * no entities are ever seeded, so there is no significant overhead.
         * Strictly, several of these callbacks are not needed on the slave processes, but
         * their definition doesn't have any adverse effects.
         */
        // pop =
        slavepop = ga_genesis_integer(
                              0, /* const int population_size */
                              1, /* const int num_chromo */
                              problem_size, /* const int len_chromo */
                              NULL, /* GAgeneration_hook generation_hook */
                              NULL, /* GAiteration_hook iteration_hook */
                              NULL, /* GAdata_destructor data_destructor */
                              NULL, /* GAdata_ref_incrementor data_ref_incrementor */
                              cevaluate_fitness, /* GAevaluate evaluate */
                              cinitialize, /* GAseed seed */
                              NULL, /* GAadapt adapt */
                              ga_select_one_random, /* GAselect_one select_one */
                              ga_select_two_random, /* GAselect_two select_two */
                              cmutate, /* GAmutate mutate */
                              cbreed, /* GAcrossover crossover */
                              ga_replace_by_fitness, /* GAreplace replace */
                              NULL /* vpointer User data */
                              );
        ga_population_set_parameters(
                                     slavepop, /* population *pop */
                                     GA_SCHEME_DARWIN, /* const ga_scheme_type scheme */
                                     GA_ELITISM_PARENTS_SURVIVE, /* const ga_elitism_type elitism */
                                     0.5, /* double crossover */
                                     0.1, /* double mutation */ // can SEGFAULT if prob > 0
                                     0.02 /* double migration */
                                     );
        printf("Attaching process %d\n", rank);
        ga_attach_mpi_slave(slavepop); /* The slaves halt here until ga_detach_mpi_slaves(), below, is called. */
    } else {
       /*
         * This is the master process.  Other than calling ga_evolution_mpi() instead
         * of ga_evolution(), there are no differences between this code and the usual
         * GAUL invocation.
         */
        int seed_i = 0;
        random_seed(seed_i);
        printf("Population size is %d\n (rank %d)", population_size, rank);
        for(int i = 0; i < num_islands; i++) {
            pop[i] = ga_genesis_integer(
                             population_size/num_islands, /* const int population_size */
                             1, /* const int num_chromo */
                             problem_size, /* const int len_chromo */
                             clog_generation, /* GAgeneration_hook generation_hook */
                             NULL, /* GAiteration_hook iteration_hook */
                             NULL, /* GAdata_destructor data_destructor */
                             NULL, /* GAdata_ref_incrementor data_ref_incrementor */
                             cevaluate_fitness, /* GAevaluate evaluate */
                             cinitialize, /* GAseed seed */
                             NULL, /* GAadapt adapt */
                             ga_select_one_random, /* GAselect_one select_one */
                             ga_select_two_random, /* GAselect_two select_two */
                             cmutate, /* GAmutate mutate */
                             cbreed, /* GAcrossover crossover */
                             ga_replace_by_fitness, /* GAreplace replace */
                             NULL /* vpointer User data */
                             );
            ga_population_set_parameters(
                                     pop[i], /* population *pop */
                                     GA_SCHEME_DARWIN, /* const ga_scheme_type scheme */
                                     GA_ELITISM_PARENTS_SURVIVE, /* const ga_elitism_type elitism */
                                     0.5, /* double crossover */
                                     0.1, /* double mutation */ // can SEGFAULT if prob > 0
                                     0.02 /* double migration */
                                     );
        }
        printf("Starting evolution (rank 0)\n");
        //ga_evolution_archipelago_mpi(const int num_pops, population **pops, const int max_generations);
        /*ga_evolution_mpi(
                         pop, // population *pop
                         1000 // const int max_generations
                         );*/
        ga_evolution_archipelago_mpi(num_islands, pop, num_epochs);
        
        for (int i=0; i<num_islands; i++)
        {
            double tmp = ga_get_entity_from_rank(pop[i],0)->fitness;
            tmp = sqrt(((double)1)/tmp);
            printf( "The best solution on island %d with score %f was:\n", i, tmp);
            ga_extinction(pop[i]);
            
            //LOG(BEST_FITNESS, tmp);
            //LOG_WC(EPOCH_END);
            // add evolution counter
            // add iter frequency
        }
        
        /*printf("The final solution was (seed %d):\n", seed_i);
        best_individual = ga_get_entity_from_rank(pop, 0);
        double best_fitness = best_individual->fitness;
        best_fitness = sqrt(((double)1)/best_fitness);
        printf("With score = %f\n", best_fitness);
        ga_extinction(pop);*/
        ga_detach_mpi_slaves(); /* Allow all slave processes to continue. */
    }
    // clean up logging
    if(rank == 0) {
        close_logger();
    }
    MPI_Finalize();
    exit(EXIT_SUCCESS);
}

boolean cevaluate_fitness(population *pop, entity *entity) {
    //printf("calling evaluate fitness (rank %d)\n", rank);
    int* chromosome = (int *)(entity->chromosome[0]);
    //assert(sanity_check(chrom));
    double route_distance = 0;
    for(int j = 0; j < (pop->len_chromosomes) - 1; j++) {
        route_distance += cities[chromosome[j] + chromosome[j+1]*(pop->len_chromosomes)];
    }
    route_distance += cities[chromosome[(pop->len_chromosomes)-1] + chromosome[0]*(pop->len_chromosomes)];
    //printf("\n%f\n", route_distance);
    //printf("\n\n");
    // return route_distance;
    //if (route_distance < bestfitness) {
    //    bestfitness = route_distance;
    //    printf("new best fitness! %f", bestfitness);
    //}
    double corrected_fitness = ((double)1) / pow(route_distance, 2);
    // DON'T CHANGE THIS BECAUSE OF PRINTING
    entity->fitness = corrected_fitness;
    //printf("ending evaluate fitness %f (rank %d)\n", route_distance, rank);
    return TRUE;
}
void cmutate(population *pop, entity *mother, entity *daughter) {
    //printf("calling mutate (rank %d)\n", rank);
    int* chromosomeM = (int *)(mother->chromosome[0]);
    int* chromosomeD = (int *)(daughter->chromosome[0]);
    int swap = rand_range(0, (pop->len_chromosomes) - 1);
    int swap_with = rand_range(0, (pop->len_chromosomes) - 1);
    for(int i = 0; i < (pop->len_chromosomes); i++) {
        if(i == swap) {
            chromosomeD[swap] = chromosomeM[swap_with];
        } else if (i == swap_with) {
            chromosomeD[swap_with] = chromosomeM[swap];
        } else {
            chromosomeD[i] = chromosomeM[i];
        }
    }
    //printf("ending mutate (rank %d)\n", rank);
}
void cbreed(population *pop, entity *mother, entity *father, entity *daughter, entity *son) {
    //printf("calling crossover (rank %d)\n", rank);
    int* parent1 = (int *)(mother->chromosome[0]);
    int* parent2 = (int *)(father->chromosome[0]);
    int* child1 = (int *)(daughter->chromosome[0]);
    int* child2 = (int *)(son->chromosome[0]);
    // child1
    int geneA = rand_range(0, (pop->len_chromosomes) - 1);
    int geneB = rand_range(0, (pop->len_chromosomes) - 1);
    int startGene = MIN(geneA, geneB);
    int endGene = MAX(geneA, geneB);
    int mask[(pop->len_chromosomes)];
    for(int idx = 0; idx < (pop->len_chromosomes); idx++) {
        mask[idx] = 1;
    }
    for (int i = startGene; i <= endGene; ++i) {
        child1[i] = parent1[i];
        mask[parent1[i]] = 0; // no comparisons
    }
    int parent2idx = 0;
    for(int childIdx = 0; childIdx < startGene;) {
        child1[childIdx] = parent2[parent2idx];
        childIdx = childIdx + (1 & mask[parent2[parent2idx]]);
        parent2idx++;
    }
    for(int childIdx = endGene+1; childIdx < (pop->len_chromosomes);) {
        child1[childIdx] = parent2[parent2idx];
        childIdx = childIdx + (1 & mask[parent2[parent2idx]]);
        parent2idx++;
    }
    // child2
    geneA = rand_range(0, (pop->len_chromosomes) - 1);
    geneB = rand_range(0, (pop->len_chromosomes) - 1);
    startGene = MIN(geneA, geneB);
    endGene =MAX(geneA, geneB);
    for(int idx = 0; idx < (pop->len_chromosomes); idx++) {
        mask[idx] = 1;
    }
    for (int i = startGene; i <= endGene; ++i) {
        child2[i] = parent1[i];
        mask[parent1[i]] = 0; // no comparisons
    }
    parent2idx = 0;
    for(int childIdx = 0; childIdx < startGene;) {
        child2[childIdx] = parent2[parent2idx];
        childIdx = childIdx + (1 & mask[parent2[parent2idx]]);
        parent2idx++;
    }
    for(int childIdx = endGene+1; childIdx < (pop->len_chromosomes);) {
        child2[childIdx] = parent2[parent2idx];
        childIdx = childIdx + (1 & mask[parent2[parent2idx]]);
        parent2idx++;
    }
    //crossovercounter = crossovercounter+1;
    //printf("co counter %d\n", crossovercounter);
    //for(int i = 0; i < problem_size; i++) {
    //    printf("| p1:%ld p2:%ld c1:%ld c2:%ld |", parent1[i], parent2[i], child1[i], child2[i]);
    //} printf("\n\n\n");
    //printf("ending crossover (rank %d)\n", rank);
}
boolean cinitialize(population *pop, entity *adam) {
    //printf("calling inizialize (rank %d)\n", rank);
    int* chromosome = (int *)(adam->chromosome[0]);
    for(int i = 0; i < (pop->len_chromosomes); i++) {
        chromosome[i] = i;
    }
    shuffle(chromosome, (pop->len_chromosomes));
    //printf("ending inizialize (rank %d)\n", rank);
}
// utility functions
int rand_range(const int a, const int b) {
    //printf("calling rand range (rank %d)\n", rank);
    return (rand() % (b - a + 1) + a);
}
// see https://stackoverflow.com/questions/6127503/shuffle-array-in-c
void shuffle(int *array, size_t n) {
    //printf("calling shuffle (rank %d)\n", rank);
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
    //printf("ending shuffle (rank %d)\n", rank);
}
boolean clog_generation(const int generation, population *pop) {
    double tmp = ga_get_entity_from_rank(pop,0)->fitness;
    tmp = sqrt(((double)1)/tmp);
    if(tmp < best_fitness) {
        best_fitness = tmp;
    }
    evolution_counter++;
    if(evolution_counter % num_islands == 0) {
        LOG(BEST_FITNESS, best_fitness);
        LOG_WC(EPOCH_END);
        //printf( "The best solution in generation %d was scored %f \n", generation, best_fitness);
    }
    return true;
}
#endif

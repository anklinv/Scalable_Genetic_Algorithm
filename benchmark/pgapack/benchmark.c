#include <pgapack.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "logging.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

//double bestfitness = 1e8;
//int mutcounter = 0;
//int crossovercounter = 0;
// data for fitness evaluation
float* cities;
int problem_size;
#define DIST(i,j) cities[i * problem_size + j]
// fitness evaluation, mutation and crossover
double evaluate_fitness(PGAContext *ctx, int p, int pop);
int mutate(PGAContext *ctx, int p, int pop, double mr);
void breed(PGAContext *ctx, int p1, int p2, int p_pop,
           int c1, int c2, int c_pop);
void MyEndOfGen(PGAContext *ctx);
// utility functions
int rand_range(const int a, const int b);
// function for sanity check
int sanity_check(PGAInteger* chrom);
void print_dists();
int rank;
// parameters
int nr_epochs = 1000;
int nr_individuals = 1000;
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
            sscanf(argv[i+1], "%d", &nr_epochs);
            printf("number of epochs is %d\n", nr_epochs);
        } else if (strcmp(argv[i], "--population")==0) {
            //printf("argc is %d\n", argc);
            //printf("i+1 is %d\n", (i+1));
            assert(i + 1 < argc);
            sscanf(argv[i+1], "%d", &nr_individuals);
            printf("population size is %d\n", nr_individuals);
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

int main( int argc, char **argv ) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // parse arguments
    parse_args(argc, argv);
    // read problem
    FILE* fp = fopen("data/d1291.csv", "r");
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
    // set up logging
    if(rank == 0) {
        //char *log_dir = "logs/";
        setup_logger(log_dir, rank);
        open_logger();
    }
    //print_dists();
    // PGA setup
    PGAContext *ctx;
    // set chromosomes
    ctx = PGACreate(&argc, argv, PGA_DATATYPE_INTEGER, number_cities, PGA_MAXIMIZE);
    PGASetSelectType(ctx, PGA_SELECT_PROPORTIONAL);
    // initialize chromosomes
    PGASetIntegerInitPermute(ctx, 0, problem_size-1);
    // set population size
    PGASetPopSize(ctx, nr_individuals);
    // i don't know exactly what the subsequent ones do
    PGASetMutationProb(ctx, 0.10);
    //PGASetRandomSeed(ctx, 50);
    PGASetMutationAndCrossoverFlag(ctx, PGA_TRUE);
    PGASetCrossoverProb(ctx, 1.0);
    int elite_size = nr_individuals / 2;
    PGASetNumReplaceValue(ctx, nr_individuals-elite_size);
    PGASetPopReplaceType(ctx, PGA_POPREPL_BEST);
    // set frequency of printing statistics
    PGASetPrintFrequencyValue(ctx, 400); // 1e3
    PGASetRestartFlag(ctx, PGA_FALSE);
    PGASetNoDuplicatesFlag(ctx, PGA_FALSE);
    // for debugging
    PGASetPrintOptions(ctx, PGA_REPORT_WORST);
    // set number of iterations
    PGASetStoppingRuleType(ctx, PGA_STOP_MAXITER);
    PGASetMaxGAIterValue(ctx, nr_epochs);
    // set mutation and crossover
    PGASetUserFunction(ctx, PGA_USERFUNCTION_MUTATION, mutate);
    PGASetUserFunction(ctx, PGA_USERFUNCTION_CROSSOVER, breed);
    // logging
    PGASetUserFunction(ctx, PGA_USERFUNCTION_ENDOFGEN, MyEndOfGen);
    // island model, neighborhood model
    //PGASetNumDemes(ctx, 4);
    //PGASetNumIslands(ctx, 1);
    // set app other parameters to their default value
    PGASetUp(ctx);
    // set fitness evaluation
    PGARun(ctx, evaluate_fitness);
    PGADestroy(ctx);
    // clean up logging
    if(rank == 0) {
        close_logger();
    }
    return 0;
}

int sanity_check(PGAInteger* chrom) {
    int sane = 1;
    for(int i = 0; i < problem_size; i++) {
	int contained = 0;
	for(int j = 0; j < problem_size; j++) {
	    if(chrom[j] == i) {
		contained = 1;
	    }
	}
	if(!contained) {
	    sane = 0;
	}
    }
    return sane;
}
void print_dists() {
    for(int i = 0; i < problem_size; i++) {
	for(int j = 0; j < problem_size; j++) {
	    printf(" %f ", cities[i + j*problem_size]);
	} printf("\n\n\n");
    }
}

// fitness evaluation, mutation and crossover
double evaluate_fitness(PGAContext *ctx, int p, int pop) {
    PGAInteger* chrom = PGAGetIndividual(ctx, p, pop)->chrom;
    //assert(sanity_check(chrom));
    double route_distance = 0;
    for(int j = 0; j < problem_size - 1; j++) {
        //printf("%ld | ", chrom[j]);
	route_distance += cities[chrom[j] + chrom[j+1]*problem_size];
    }
    route_distance += cities[chrom[problem_size-1] + chrom[0]*problem_size];
    //printf("\n%f\n", route_distance);
    //printf("\n\n");
    // return route_distance;
    //if (route_distance < bestfitness) {
    //	bestfitness = route_distance;
    //	printf("new best fitness! %f", bestfitness);
    //}
    double corrected_fitness = ((double)1) / pow(route_distance, 2);
    // DON'T CHANGE THIS BECAUSE OF PRINTING
    return corrected_fitness;
}
int mutate(PGAContext *ctx, int p, int pop, double mr) {
    PGAInteger* chrom = (PGAInteger *)PGAGetIndividual(ctx, p, pop)->chrom;
    //printf("mu prob is %f\n", mr);
    int mutation_rate = (int)(mr*((double)100));
    //printf("mu rate is %d\n", mutation_rate);
    if(rand() % mutation_rate == 0) {
        int swap = rand_range(0, problem_size - 1);
        int swap_with = rand_range(0, problem_size - 1);
        int city1 = chrom[swap];
        int city2 = chrom[swap_with];
        chrom[swap] = city2;
        chrom[swap_with] = city1;
	//mutcounter = mutcounter+1;
	//printf("mutcounter is %d\n", mutcounter);
        return 1; // number of mutations
    }
    return 0; // number of mutations
}
void breed(PGAContext *ctx, int p1, int p2, int p_pop,
                 int c1, int c2, int c_pop) {
    PGAInteger* parent1 = (PGAInteger *)PGAGetIndividual(ctx, p1, p_pop)->chrom;
    PGAInteger* parent2 = (PGAInteger *)PGAGetIndividual(ctx, p2, p_pop)->chrom;
    PGAInteger* child1 = (PGAInteger *)PGAGetIndividual(ctx, c1, c_pop)->chrom;
    PGAInteger* child2 = (PGAInteger *)PGAGetIndividual(ctx, c2, c_pop)->chrom;
    // child1
    int geneA = rand_range(0, problem_size - 1);
    int geneB = rand_range(0, problem_size - 1);
    int startGene = MIN(geneA, geneB);
    int endGene = MAX(geneA, geneB);
    int mask[problem_size];
    for(int idx = 0; idx < problem_size; idx++) {
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
    for(int childIdx = endGene+1; childIdx < problem_size;) {
        child1[childIdx] = parent2[parent2idx];
        childIdx = childIdx + (1 & mask[parent2[parent2idx]]);
        parent2idx++;
    }
    // child2
    geneA = rand_range(0, problem_size - 1);
    geneB = rand_range(0, problem_size - 1);
    startGene = MIN(geneA, geneB);
    endGene =MAX(geneA, geneB);
    for(int idx = 0; idx < problem_size; idx++) {
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
    for(int childIdx = endGene+1; childIdx < problem_size;) {
        child2[childIdx] = parent2[parent2idx];
        childIdx = childIdx + (1 & mask[parent2[parent2idx]]);
        parent2idx++;
    }

    //crossovercounter = crossovercounter+1;
    //printf("co counter %d\n", crossovercounter);

    //for(int i = 0; i < problem_size; i++) {
    //	printf("| p1:%ld p2:%ld c1:%ld c2:%ld |", parent1[i], parent2[i], child1[i], child2[i]);
    //} printf("\n\n\n");
}

/*  After each generation, this funciton will get called.  */
void MyEndOfGen(PGAContext *ctx) {
    
    int p, best_p;
    double e, best_e;
    
    best_p = PGAGetBestIndex(ctx, PGA_NEWPOP); // see utility.c
    best_e = PGAGetEvaluation(ctx, best_p, PGA_NEWPOP);
    best_e = sqrt(((double)1)/best_e);
    
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //printf("end of gen - rank is %d\n", rank);
    // rank is always 0
    LOG(BEST_FITNESS, best_e);
    LOG_WC(EPOCH_END);
    
    //best_p = PGAGetBestIndex(ctx, pop);
    //best_e = PGAGetEvaluation(ctx, best_p, pop);
    //best_e = sqrt(((double)1)/best_e);

    /*  Do something useful; display the population on a graphics output,
     *  let the user adjust the population, etc.
     */
}

// utility functions
int rand_range(const int a, const int b) {
    return (rand() % (b - a + 1) + a);
}

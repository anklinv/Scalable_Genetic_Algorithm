
#include "island.hpp"


using namespace std;


void Island::overwriteGene(int* newGene, int* oldGene, int geneSize) {
    
    for(int geneIdx = 0; geneIdx < geneSize; geneIdx++) {
        oldGene[geneIdx] = newGene[geneIdx];
    }
}


int Island::computeHammingDistance(int* firstGene, int* scndGene, int geneSize) {
    
    int hammingDistance = 0;
    
    int idxFirst = 0;
    while(firstGene[idxFirst] != 1) { // firstGene[idxFirst] == 1
        idxFirst++;
    }
    
    int idxScnd = 0;
    while(scndGene[idxScnd] != 1) { // scndGene[idxScnd] == 1
        idxScnd++;
    }
    
    for(int geneIdx = 0; geneIdx < geneSize; geneIdx++) {
        
        if (firstGene[idxFirst] != scndGene[idxScnd]) {
            hammingDistance++;
        }
        
        idxFirst = (idxFirst + 1) % geneSize;
        idxScnd = (idxScnd + 1) % geneSize;
    }
    
    return hammingDistance;
}


int* Island::tournamentSelection(double* fitness, int numIndividuals, // TODO: change access pattern to these variables
                                 int tournamentSize, int numIndividualsToSample) {
    
    int* sampledIndividuals[numIndividualsToSample];
    
    for(int sampleIdx = 0; sampleIdx < numIndividualsToSample; sampleIdx++) {
        
        int bestIdx = -1;
        double bestFitness = numeric_limits<double>::max(); // smaller fitness value is better
        
        for(int t = 0; t < tournamentSize; t++) {
            
            // RAND_MAX is at least 32767
            // yields slightly skewed distribution
            int indivIdx = rand() % numIndividuals;
            
            if (fitness[indivIdx] < bestFitness) { // smaller fitness value is better
                bestIdx = indivIdx;
                bestFitness = fitness[indivIdx];
            }
        }
        
        sampledIndividuals[sampleIdx] = indivIdx;
        
    } // end sample
    
    return sampledIndividuals;
}


int* Island::fitnessProportionateSelection(double* fitness, int numIndividuals, // TODO: change access pattern to these variables
                                           int numIndividualsToSample) {
    
    int* sampledIndividuals[numIndividualsToSample];
    
    // total fitness for weighting individuals
    double totalFitness = 0;
    
    for(int indivIdx = 0; indivIdx < numIndividuals; indivIdx++) {
        totalFitness = totalFitness + fitness[indivIdx];
    }
    
    for(int sampleIdx = 0; sampleIdx < numIndividualsToSample; sampleIdx++) {
    
        double rnd = (double)rand() / (double)RAND_MAX; // [0, 1]
        double rndScaledToFitness = rnd * totalFitness; // [0, totalFitness]
        
        double tmpFitness = 0;
        
        
        if (rndScaledToFitness == 0) { // edge case rndScaledToFitness == 0
            
            if (totalFitness == 0) {
                sampledIndividuals[sampleIdx] = numIndividuals - 1;
            } else {
                sampledIndividuals[sampleIdx] = 0;
            }
            
            continue; // sampleIdx = sampleIdx + 1
        }
        
        if (rndScaledToFitness == totalFitness) { // edge case rndScaledToFitness == totalFitness
            
            sampledIndividuals[sampleIdx] = numIndividuals - 1;
            
            continue; // sampleIdx = sampleIdx + 1
        }
        
        for(int indivIdx = 0; indivIdx < numIndividuals; indivIdx++) {
            
            tmpFitness = tmpFitness + fitness[indivIdx]; // f_0, f_1, ..., f_n-1
            
            // ]0, f_0[, [f_0, f_1[, [f_1, f_2[, ..., [f_n-2, f_n-1[
            // 0, f_n-1 edge cases covered above
            if (tmpFitness <= rndScaledToFitness) {
                
                // do nothing
                // (rndScaledToFitness lies in a subsequent interval)
                
            } else {
                // first interval hit
                sampledIndividuals[sampleIdx] = indivIdx;
                
                break;
            }
            
        } // end fitness accumulation
        
    } // end sample
    
    return sampledIndividuals;
}


int* Island::stochasticUniversalSampling(double* fitness, int numIndividuals, // TODO: change access pattern to these variables
                                         int numIndividualsToSample) {
    
    int* sampledIndividuals[numIndividualsToSample];
    
    double totalFitness = 0;
    
    for(int indivIdx = 0; indivIdx < numIndividuals; indivIdx++) {
        totalFitness = totalFitness + fitness[indivIdx];
    }
    
    double delta = totalFitness / numIndividualsToSample;
    
    double rnd = (double)rand() / (double)RAND_MAX; // [0, 1]
    double currScaledFitness = rnd * delta; // current selection threshold
    
    double tmpFitness = fitness[0]; // accumulated fitness
    
    
    int indivIdx = 0;
    
    for(int sampleIdx = 0; sampleIdx < numIndividualsToSample; sampleIdx++) {
        
        if (currScaledFitness == 0) { // edge case 0 (definition)
            
            sampledIndividuals[sampleIdx] = 0;
            
            currScaledFitness = currScaledFitness + delta;
            
            continue; // sampleIdx = sampleIdx + 1
        }
        
        if (currScaledFitness == totalFitness) { // edge case f_n-1 (definition)
            
            sampledIndividuals[sampleIdx] = numIndividuals - 1;
            
            currScaledFitness = currScaledFitness + delta;
            
            continue; // sampleIdx = sampleIdx + 1
        }
        
        // ]0, f_0[, [f_0, f_1[, ..., [f_n-2, f_n-1[
        // 0, f_n-1 edge cases covered above
        while(tmpFitness <= currScaledFitness) {
            
            tmpFitness = tmpFitness + fitness[indivIdx];
            indivIdx = indivIdx + 1;
        }
        
        sampledIndividuals[sampleIdx] = indivIdx;
        // sampleIdx = sampleIdx + 1
        
    } // end sample
    
    
    return sampledIndividuals;
}


// TODO: needs access to the ranks
int* Island::truncationSelection(int numIndividualsToSample) {
    
    int* sampledIndividuals[numIndividualsToSample];
    
    for(int indivIdx = 0; indivIdx < numIndividualsToSample; indivIdx++) {
     
        sampledIndividuals[indivIdx] = indivIdx;
    }
    
    return sampledIndividuals;
}


int* Island::pureRandomSelection(int numIndividuals, // TODO: change access to this
                                 int numIndividualsToSample) {
    
    int* sampledIndividuals[numIndividualsToSample];
    
    for(int sampleIdx = 0; sampleIdx < numIndividualsToSample; sampleIdx++) {
        
        sampledIndividuals[sampleIdx] = rand() % numIndividuals;
    }
    
    return sampledIndividuals;
}


void Island::truncationReplacement(const TravellingSalesmanProblem* tsp, int geneSize, int islandSize, // TODO: change access to this. Use the tsp pointer.
                                   int numImmigrants, int** immigrantGenes, double* immigrantFitnesses) {
    
    // TODO: change access to ranks
    int* ranks = (this->tsp)->getRanks();
    
    // Use (idx, fitness) pairs to facilitate sorting
    Individual immigrants[numImmigrants];
    
    for(int immigrantIdx = 0; immigrantIdx < numImmigrants; immigrantIdx++) {
        
        (immigrants[immigrantIdx]).idx = immigrantIdx;
        (immigrants[immigrantIdx]).fitness = immigrantFitnesses[immigrantIdx];
    }
    
    // Sort incoming data in ascending order
    // (the data of the current island is already sorted)
    sort(immigrants, immigrants + numImmigrants);
    
    
    // Determine how many individuals of the island are going to be replaced
    int numToReplace = 0;
    
    int immigrantIdx = 0;
    int islandIndivIdx = 0;
    
    int offsetIsland = numIndivsIsland - numImmigrants;
    
    while(immigrantIdx < numImmigrants
          && islandIndivIdx < numImmigrants) {
        
        double currFitnessImmigrant = (immigrants[immigrantIdx]).fitness;
        double currFitnessIslandIndiv = (this->tsp)->getFitness(ranks[offsetIsland + islandIndivIdx]);
        
        if(currFitnessImmigrant < currFitnessIslandIndiv) { // immigrant is better
            
            numToReplace++;
            
            immigrantIdx++;
            islandIndivIdx++;
            
        } else { // island individual is better
            
            islandIndivIdx++;
        }
    }
    
    // Just throw out the numToReplace weakest individuals of the island by replacing their gene
    // - the fitness is updated too
    // - the ranks stored in the TSP object are no longer correct after this step
    for(int indivIdx = 0; indivIdx < numToReplace; indivIdx++) {
        
        int* currGene = (this->tsp)->getGene(ranks[(islandSize - numImmigrants) + indivIdx]);
        int* newGene = immigrantGenes[immigrants[indivIdx].idx];
        
        overwriteGene(newGene, currGene, geneSize);
        
        (this->tsp)->setFitness(ranks[(islandSize - numImmigrants) + indivIdx],
                                immigrants[indivIdx].fitness);
    }
    
}


void Island::pureRandomReplacement(int islandSize, int geneSize, // TODO: change access to these variables
                                   int numImmigrants, int** immigrantGenes, double* immigrantFitnesses) {
    
    for(int immigrantIdx = 0; immigrantIdx < numImmigrants; immigrantIdx++) {
        
        int indivToReplace = rand() % islandSize;
        
        int* currGene = (this->tsp)->getGene(indivToReplace);
        int* newGene = immigrantGenes[immigrantIdx];
        
        overwriteGene(newGene, currGene, geneSize);
        
        (this->tsp)->setFitness(indivToReplace, immigrantFitnesses[immigrantIdx]);
    }
    
}


void Island::crowdingReplacement(int geneSize, int islandSize, // TODO: change access to these variables
                                 int crowdSize,
                                 int numImmigrants, double* immigrantFitnesses, int** immigrantGenes) {
    
    for(int immigrantIdx = 0; immigrantIdx < numImmigrants; immigrantIdx++) {
        
        int idxClosest = -1;
        int minHammingDist = geneSize + 1;
        
        for(int crowdIdx = 0; crowdIdx < crowdSize; crowdIdx++) {
            
            int randIndivIdx = rand() % islandSize;
            int currHammingDist = computeHammingDistance(immigrantGenes[immigrantIdx],
                                                         (this->tsp)->getGene(randIndivIdx));
            
            if (currHammingDist < minHammingDist) {
                
                minHammingDist = currHammingDist;
                idxClosest = randIndivIdx;
            }
            
        }
        
        // replace individual
        overwriteGene(immigrantGenes[immigrantIdx], (this->tsp)->getGene(idxClosest), geneSize);
        (this->tsp)->setFitness(idxClosest, immigrantFitnesses[immigrantIdx]);
    }
    
}


double Island::solve() {
    
    // For MPI_Allgather and to compute how much data is received
    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    
    // Local helper variables for convenience
    int numNodes = (this->tsp)->problem_size;
    int numIntsGene = numNodes;
    int migrationAmount = this->migrationAmount;
    int numIndivsIsland = (this->tsp)->population_count;
        
    
    for(int currPeriod = 0; currPeriod < this->numPeriods; currPeriod++) {
        
        // Run the GA for migrationPeriod iterations
        (this->tsp)->solve(this->migrationPeriod, rank);
        
        
        // Set up buffers for sending data
        double sendBufferFitness[migrationAmount];
        int sendBufferGenes[migrationAmount * numIntsGene];
        
        
        int* ranks = (this->tsp)->getRanks();
        
        for(int indivIdx = 0; indivIdx < migrationAmount; indivIdx++) {
            
            sendBufferFitness[indivIdx] = (this->tsp)->getFitness(ranks[indivIdx]);
            
            
            int* gene = (this->tsp)->getGene(ranks[indivIdx]);
            
            for(int nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
                sendBufferGenes[(indivIdx * numNodes) + nodeIdx] = gene[nodeIdx];
            }
            
        }
        
        
        // Set up buffers for receiving data
        double receiveBufferFitness[migrationAmount * numProcesses];
        int receiveBufferGenes[migrationAmount * numNodes * numProcesses];
        
        
        // "Gathers data from all tasks and distribute the combined data to all tasks"
        // - I suppose this is synchronized
        // - amount of data sent == amount of data received from any process
        MPI_Allgather(sendBufferFitness, migrationAmount, MPI_DOUBLE,
                      receiveBufferFitness, migrationAmount, MPI_DOUBLE, MPI_COMM_WORLD);
        
        MPI_Allgather(sendBufferGenes, migrationAmount * numNodes, MPI_INT,
                      receiveBufferGenes, migrationAmount * numNodes, MPI_INT, MPI_COMM_WORLD);
        
        
        // Assemble result
        // - there is some redundancy as each process is receiving its own data
        // - this is an attempt to solve this more or less efficiently
        
        // Use (idx, fitness) pairs to facilitate sorting
        Individual incomingIndividuals[migrationAmount * (numProcesses - 1)];
        
        int helperIdx = 0;
        
        for(int indivIdx = 0; indivIdx < migrationAmount * numProcesses; indivIdx++) {
            
            if(indivIdx < rank * migrationAmount || (rank + 1) * migrationAmount <= indivIdx) {
            
                (incomingIndividuals[helperIdx]).idx = indivIdx;
                (incomingIndividuals[helperIdx]).fitness = receiveBufferFitness[indivIdx];
            
                helperIdx++;
            }
            
        }
        
        // Sort the incoming data in ascending order
        // (the data at the current island is already sorted)
        sort(incomingIndividuals, incomingIndividuals + (migrationAmount * (numProcesses - 1)));
        
        // Determine how many individuals are going to be replaced
        int numReplaced = 0;
        
        int idxIncoming = 0;
        
        int offsetIsland = numIndivsIsland - (migrationAmount * (numProcesses - 1));
        int idxIsland = 0;
        
        while(idxIncoming < migrationAmount * (numProcesses - 1)
              && idxIsland < migrationAmount * (numProcesses - 1)) {
            
            double currFitnessIncoming = (incomingIndividuals[idxIncoming]).fitness;
            double currFitnessIsland = (this->tsp)->getFitness(ranks[offsetIsland + idxIsland]);
            
            if(currFitnessIncoming < currFitnessIsland) { // incoming individual is better
                
                numReplaced++;
                
                idxIncoming++;
                idxIsland++;
                
            } else { // island individual is better
                
                idxIsland++;
            }
        }
        
        // Just throw out the numReplaced weakest individuals of the island by replacing their gene
        // - the fitness is updated too
        // - the ranks stored in the TSP object are no longer correct after this step
        for(int indivIdx = 0; indivIdx < numReplaced; indivIdx++) {
            
            int* currGene = (this->tsp)->getGene(ranks[(numIndivsIsland - 1) - indivIdx]); // TODO: indexing for ranks seems to be wrong
            int* newGene = &receiveBufferGenes[incomingIndividuals[indivIdx].idx * numNodes];
            
            copyArray(newGene, currGene, numNodes);
            
            (this->tsp)->setFitness(ranks[(numIndivsIsland - 1) - indivIdx],
                                   incomingIndividuals[indivIdx].fitness);
        }
        
    } // end numPeriods
        
    double bestLocalFitness;
    bestLocalFitness = (this->tsp)->getMinFitness();
    
    return bestLocalFitness;
}



#include "island.hpp"


using namespace std;


Island::Island(TravellingSalesmanProblem TSP,
               const MigrationTopology mt, const int numIndivsReceivedPerMigration, const int migrationPeriod,
               const SelectionPolicy sp, const ReplacementPolicy rp):
TSP(TSP),
MIGRATION_TOPOLOGY(mt),
NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION(numIndivsReceivedPerMigration),
MIGRATION_PERIOD(migrationPeriod),
SELECTION_POLICY(sp),
REPLACEMENT_POLICY(rp) { // initializer list
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rankID);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeCommWorld);
    
    numIntegersGene = TSP.problem_size;
    numIndividualsIsland = TSP.population_count;
    
    numIndividualsSendBuffer = computeSendBufferSize();
    numIndividualsReceiveBuffer = computeReceiveBufferSize(numIndividualsSendBuffer);
    
    
    
};


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

// double* fitness, int numIndividuals, // TODO: change access pattern to these variables
void Island::tournamentSelection(int tournamentSize, int* sampledIndividuals, int numIndividualsToSample) {
        
    for(int sampleIdx = 0; sampleIdx < numIndividualsToSample; sampleIdx++) {
        
        int bestIdx = -1;
        double bestFitness = numeric_limits<double>::max(); // smaller fitness value is better
        
        for(int t = 0; t < tournamentSize; t++) {
            
            // RAND_MAX is at least 32767
            // yields slightly skewed distribution
            int indivIdx = rand() % numIndividualsIsland;
            
            if (TSP.getFitness(indivIdx) < bestFitness) { // smaller fitness value is better
                bestIdx = indivIdx;
                bestFitness = TSP.getFitness(indivIdx);
            }
        }
        
        sampledIndividuals[sampleIdx] = bestIdx;
        
    } // end sample
    
}


//double* fitness, int numIndividuals, // TODO: change access pattern to these variables
void Island::fitnessProportionateSelection(int* sampledIndividuals, int numIndividualsToSample) {
        
    // total fitness for weighting individuals
    double totalFitness = 0;
    
    for(int indivIdx = 0; indivIdx < numIndividualsIsland; indivIdx++) {
        totalFitness = totalFitness + TSP.getFitness(indivIdx);
    }
    
    for(int sampleIdx = 0; sampleIdx < numIndividualsToSample; sampleIdx++) {
    
        double rnd = (double)rand() / (double)RAND_MAX; // [0, 1]
        double rndScaledToFitness = rnd * totalFitness; // [0, totalFitness]
        
        double tmpFitness = 0;
        
        
        if (rndScaledToFitness == 0) { // edge case rndScaledToFitness == 0
            
            if (totalFitness == 0) {
                sampledIndividuals[sampleIdx] = numIndividualsIsland - 1;
            } else {
                sampledIndividuals[sampleIdx] = 0;
            }
            
            continue; // sampleIdx = sampleIdx + 1
        }
        
        if (rndScaledToFitness == totalFitness) { // edge case rndScaledToFitness == totalFitness
            
            sampledIndividuals[sampleIdx] = numIndividualsIsland - 1;
            
            continue; // sampleIdx = sampleIdx + 1
        }
        
        for(int indivIdx = 0; indivIdx < numIndividualsIsland; indivIdx++) {
            
            tmpFitness = tmpFitness + TSP.getFitness(indivIdx); // f_0, f_1, ..., f_n-1
            
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
    
}

//double* fitness, int numIndividuals, // TODO: change access pattern to these variables
void Island::stochasticUniversalSampling(int* sampledIndividuals, int numIndividualsToSample) {
        
    double totalFitness = 0;
    
    for(int indivIdx = 0; indivIdx < numIndividualsIsland; indivIdx++) {
        totalFitness = totalFitness + TSP.getFitness(indivIdx);
    }
    
    double delta = totalFitness / numIndividualsToSample;
    
    double rnd = (double)rand() / (double)RAND_MAX; // [0, 1]
    double currScaledFitness = rnd * delta; // current selection threshold
    
    double tmpFitness = TSP.getFitness(0); // accumulated fitness
    
    
    int indivIdx = 0;
    
    for(int sampleIdx = 0; sampleIdx < numIndividualsToSample; sampleIdx++) {
        
        if (currScaledFitness == 0) { // edge case 0 (definition)
            
            sampledIndividuals[sampleIdx] = 0;
            
            currScaledFitness = currScaledFitness + delta;
            
            continue; // sampleIdx = sampleIdx + 1
        }
        
        if (currScaledFitness == totalFitness) { // edge case f_n-1 (definition)
            
            sampledIndividuals[sampleIdx] = numIndividualsIsland - 1;
            
            currScaledFitness = currScaledFitness + delta;
            
            continue; // sampleIdx = sampleIdx + 1
        }
        
        // ]0, f_0[, [f_0, f_1[, ..., [f_n-2, f_n-1[
        // 0, f_n-1 edge cases covered above
        while(tmpFitness <= currScaledFitness) {
            
            tmpFitness = tmpFitness + TSP.getFitness(indivIdx);
            indivIdx = indivIdx + 1;
        }
        
        sampledIndividuals[sampleIdx] = indivIdx;
        // sampleIdx = sampleIdx + 1
        
    } // end sample
        
}


// TODO: needs access to the ranks
void Island::truncationSelection(int* sampledIndividuals, int numIndividualsToSample) {
        
    for(int indivIdx = 0; indivIdx < numIndividualsToSample; indivIdx++) {
     
        sampledIndividuals[indivIdx] = indivIdx;
    }
    
}


//int numIndividuals, // TODO: change access to this
void Island::pureRandomSelection(int* sampledIndividuals, int numIndividualsToSample) {
        
    for(int sampleIdx = 0; sampleIdx < numIndividualsToSample; sampleIdx++) {
        
        sampledIndividuals[sampleIdx] = rand() % numIndividualsIsland;
    }
    
}

//const TravellingSalesmanProblem* tsp, int geneSize, int islandSize, // TODO: change access to this. Use the tsp pointer.
void Island::truncationReplacement(int numImmigrants, int* immigrantGenes, double* immigrantFitnesses) {
    
    // TODO: change access to ranks
    
    // TODO: IGNORE OWN DATA HERE
    
    int* ranks = TSP.getRanks();
    
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
    
    int offsetIsland = numIndividualsIsland - numImmigrants;
    
    while(immigrantIdx < numImmigrants
          && islandIndivIdx < numImmigrants) {
        
        double currFitnessImmigrant = (immigrants[immigrantIdx]).fitness;
        double currFitnessIslandIndiv = TSP.getFitness(ranks[offsetIsland + islandIndivIdx]);
        
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
        
        int* currGene = TSP.getGene(ranks[(numIndividualsIsland - numImmigrants) + indivIdx]);
        int* newGene = &immigrantGenes[immigrants[indivIdx].idx * numIntegersGene];
        
        overwriteGene(newGene, currGene, numIntegersGene);
        
        TSP.setFitness(ranks[(numIndividualsIsland - numImmigrants) + indivIdx],
                       immigrants[indivIdx].fitness);
    }
    
}

//int islandSize, int geneSize, // TODO: change access to these variables
void Island::pureRandomReplacement(int numImmigrants, int* immigrantGenes, double* immigrantFitnesses) {
    
    for(int immigrantIdx = 0; immigrantIdx < numImmigrants; immigrantIdx++) {
        
        int indivToReplace = rand() % numIndividualsIsland;
        
        int* currGene = TSP.getGene(indivToReplace);
        int* newGene = &immigrantGenes[immigrantIdx * numIntegersGene];
        
        overwriteGene(newGene, currGene, numIntegersGene);
        
        TSP.setFitness(indivToReplace, immigrantFitnesses[immigrantIdx]);
    }
    
}


//int geneSize, int islandSize, // TODO: change access to these variables
void Island::crowdingReplacement(int crowdSize,
                                 int numImmigrants, int* immigrantGenes, double* immigrantFitnesses) {
    
    for(int immigrantIdx = 0; immigrantIdx < numImmigrants; immigrantIdx++) {
        
        int idxClosest = -1;
        int minHammingDist = numIntegersGene + 1;
        
        for(int crowdIdx = 0; crowdIdx < crowdSize; crowdIdx++) {
            
            int randIndivIdx = rand() % numIndividualsIsland;
            int currHammingDist = computeHammingDistance(&immigrantGenes[immigrantIdx * numIntegersGene],
                                                         TSP.getGene(randIndivIdx), numIntegersGene);
            
            if (currHammingDist < minHammingDist) {
                
                minHammingDist = currHammingDist;
                idxClosest = randIndivIdx;
            }
            
        }
        
        // replace individual
        overwriteGene(&immigrantGenes[immigrantIdx * numIntegersGene],
                      TSP.getGene(idxClosest), numIntegersGene);
        TSP.setFitness(idxClosest, immigrantFitnesses[immigrantIdx]);
    }
    
}


int Island::computeSendBufferSize() { // TODO: could be implemented in a more functional way
    
    switch(MIGRATION_TOPOLOGY) {
            
        case MigrationTopology::FULLY_CONNECTED: {
            
            int numSenders = sizeCommWorld - 1;
            
            if (NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION % numSenders == 0) {
                return max(1, NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION / numSenders);
            } else {
                return max(1, NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION / numSenders + 1);
            }
            break;
        }
            
        case MigrationTopology::ISOLATED: {
            
            return 0;
            break;
        }
            
        case MigrationTopology::RING: {
            
            return NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION;
            break;
        }
            
    } // end switch case
    
}


// TODO: could be implemented in a more functional way
int Island::computeReceiveBufferSize(int sendBufferSize) { // TODO: currently adjusted to MPI_Allgather fix this
    
    switch (MIGRATION_TOPOLOGY) {
            
        case MigrationTopology::FULLY_CONNECTED: {
            
            // number of senders MPI_Allgather
            return sendBufferSize * numIndividualsIsland;
            break;
        }
            
        case MigrationTopology::ISOLATED: {
            
            return 0;
            break;
        }
            
        case MigrationTopology::RING: {
            
            return sendBufferSize;
            break;
        }
            
    } // end switch case
    
}


void Island::fillSendBuffers() { // TODO: could be implemented in a more functional way
    
    int migrants[numIndividualsSendBuffer]; // store indices of individuals
    
    
    switch(SELECTION_POLICY) {
        
        case SelectionPolicy::TRUNCATION: {
            
            truncationSelection(migrants, numIndividualsSendBuffer);
            break;
        }
        
        case SelectionPolicy::FITNESS_PROPORTIONATE_SELECTION: {
            
            fitnessProportionateSelection(migrants, numIndividualsSendBuffer);
            break;
        }
        
        case SelectionPolicy::STOCHASTIC_UNIVERSAL_SAMPLING: {
            
            stochasticUniversalSampling(migrants, numIndividualsSendBuffer);
            break;
        }
            
        case SelectionPolicy::TOURNAMENT_SELECTION: {
            
            tournamentSelection(3, migrants, numIndividualsSendBuffer); // TODO: allow for tournament size as a parameter
            break;
        }
            
        case SelectionPolicy::PURE_RANDOM: {
            
            pureRandomSelection(migrants, numIndividualsSendBuffer);
            break;
        }
            
    } // end switch case
    
    
    for(int migrantIdx = 0; migrantIdx < numIndividualsSendBuffer; migrantIdx++) {
        
        // simply copy array
        overwriteGene(TSP.getGene(migrants[migrantIdx]),
                      &sendBufferGenes[migrantIdx * numIntegersGene], numIntegersGene); // TODO: fix access
    }
    
    for(int migrantIdx = 0; migrantIdx < numIndividualsSendBuffer; migrantIdx++) {
        
        sendBufferFitness[migrantIdx] = TSP.getFitness(migrants[migrantIdx]);
    }
    
}


void Island::doSynchronousBlockingCommunication() { // TODO: could be implemented in a more functional way
    
    switch(MIGRATION_TOPOLOGY) {
        
        case MigrationTopology::FULLY_CONNECTED: {
            
            // "Gathers data from all tasks and distribute the combined data to all tasks"
            // - I suppose this is synchronized
            // - amount of data sent == amount of data received from any process
            MPI_Allgather(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE,
                          receiveBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE, MPI_COMM_WORLD);
            
            MPI_Allgather(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT,
                          receiveBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT, MPI_COMM_WORLD);
            break;
        }
        
        case MigrationTopology::ISOLATED: {
            
            int status = MPI_Barrier(MPI_COMM_WORLD); // Synchronize islands
            // could check status == MPI_SUCCESS
            break;
        }
        
        case MigrationTopology::RING: {
            
            // A cyclic list seems to be a good use case for MPI_Sendrecv
            
            int destRankID = (rankID + 1) % sizeCommWorld;
            
            const int FITNESS_TAG = 0x01011;
            const int GENE_TAG = 0x01100;
            
            
            int status = MPI_Sendrecv(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE,
                                      destRankID, FITNESS_TAG,
                                      receiveBufferFitness, numIndividualsReceiveBuffer, MPI_DOUBLE,
                                      rankID, FITNESS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // could check status == MPI_SUCCESS
            
            status = MPI_Sendrecv(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT,
                                  destRankID, GENE_TAG,
                                  receiveBufferGenes, numIndividualsReceiveBuffer * numIntegersGene, MPI_INT,
                                  rankID, GENE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // could check status == MPI_SUCCESS
            
            break;
        }
        
    } // end switch case
    
}


void Island::emptyReceiveBuffers() {
    
    switch(REPLACEMENT_POLICY) { // TODO: remove some parameters and access these values from object variables instead
            // TODO: fix crowding replacement parameter
            // TODO: maybe use a object variable function pointer which is set inside the constructor
            
        case ReplacementPolicy::TRUNCATION:
            
            truncationReplacement(numIndividualsReceiveBuffer, receiveBufferGenes, receiveBufferFitness);
            break;
        
        case ReplacementPolicy::PURE_RANDOM:
            
            pureRandomReplacement(numIndividualsReceiveBuffer, receiveBufferGenes, receiveBufferFitness);
            break;
            
        case ReplacementPolicy::DEJONG_CROWDING:
            
            crowdingReplacement(3, numIndividualsReceiveBuffer, receiveBufferGenes, receiveBufferFitness);
            break;
            
    } // end switch case
    
}


double Island::solve(const int numEvolutions) {
    
    /*// For MPI_Allgather and to compute how much data is received
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
    
    return bestLocalFitness;*/
    
    return 0;
}


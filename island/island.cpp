
#include "island.hpp"


using namespace std;


Island::Island(TravellingSalesmanProblem& TSP,
               const MigrationTopology mt, const int numIndivsReceivedPerMigration, const int migrationPeriod,
               const SelectionPolicy sp, const ReplacementPolicy rp,
               const UnderlyingCommunication communication):
TSP(TSP),
MIGRATION_TOPOLOGY(mt),
NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION(numIndivsReceivedPerMigration),
MIGRATION_PERIOD(migrationPeriod),
SELECTION_POLICY(sp),
REPLACEMENT_POLICY(rp),
COMMUNICATION(communication) { // initializer list
        
    int status;
    
    status = MPI_Comm_rank(MPI_COMM_WORLD, &rankID);
    //assert(status == MPI_SUCCESS);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeCommWorld);
    //assert(status == MPI_SUCCESS);
        
    
    // convenience variables
    numIntegersGene = TSP.problem_size;
    numIndividualsIsland = TSP.population_count;
    
    ranks = TSP.getRanks();
    genes = TSP.getGenes();
    
    
    // allocate buffers
    numIndividualsSendBuffer = computeSendBufferSize();
    numIndividualsReceiveBuffer = computeReceiveBufferSize(numIndividualsSendBuffer);
    
    sendBufferGenes = new int[numIndividualsSendBuffer * numIntegersGene];
    sendBufferFitness = new double[numIndividualsSendBuffer];
    
    receiveBufferGenes = new int[numIndividualsReceiveBuffer * numIntegersGene];
    receiveBufferFitness = new double[numIndividualsReceiveBuffer];
    
    
    // initialize communication request handles to null such that they do not block
    // during the first migration
    sendBufferFitnessRequest = MPI_REQUEST_NULL;
    sendBufferGenesRequest = MPI_REQUEST_NULL;
    MPI_Request receiveBufferFitnessRequest = MPI_REQUEST_NULL;
    MPI_Request receiveBufferGenesRequest = MPI_REQUEST_NULL;
    
}


Island::~Island() {
    
    // free buffers
    delete sendBufferGenes;
    delete sendBufferFitness;
    
    delete receiveBufferGenes;
    delete receiveBufferFitness;
    
}


void Island::overwriteGene(int* newGene, int* oldGene, int geneSize) {
    
    for(int geneIdx = 0; geneIdx < geneSize; geneIdx++) {
        oldGene[geneIdx] = newGene[geneIdx];
    }
    
    //for(int geneIdx = 0; geneIdx < geneSize; geneIdx++)
    //    assert(oldGene[geneIdx] == newGene[geneIdx]);
    
    //assert(newGene + geneSize - 1 < oldGene || oldGene + geneSize - 1 < newGene);
}


int Island::computeHammingDistance(int* firstGene, int* scndGene, int geneSize) {
    
    int hammingDistance = 0;
    
    int idxFirst = 0;
    while(firstGene[idxFirst] != 1) { // firstGene[idxFirst] == 1
        idxFirst++;
    }
    //assert(0 <= idxFirst && idxFirst < geneSize && firstGene[idxFirst] == 1);
    
    int idxScnd = 0;
    while(scndGene[idxScnd] != 1) { // scndGene[idxScnd] == 1
        idxScnd++;
    }
    //assert(0 <= idxScnd && idxScnd < geneSize && scndGene[idxScnd] == 1);
    
    for(int geneIdx = 0; geneIdx < geneSize; geneIdx++) {
        
        if (firstGene[idxFirst] != scndGene[idxScnd]) {
            hammingDistance++;
        }
        
        idxFirst = (idxFirst + 1) % geneSize;
        idxScnd = (idxScnd + 1) % geneSize;
    }
    
    return hammingDistance;
}


void Island::tournamentSelection(int* sampledIndividuals, int numIndividualsToSample) {
    
    int tournamentSize = 3;
        
    for(int sampleIdx = 0; sampleIdx < numIndividualsToSample; sampleIdx++) {
        
        int bestIdx = -1;
        double bestFitness = numeric_limits<double>::max(); // smaller fitness value is better
        
        for(int t = 0; t < tournamentSize; t++) {
            
            // RAND_MAX is at least 32767
            // yields slightly skewed distribution
            int indivIdx = rand() % numIndividualsIsland;
            
            if (TSP.getFitness(indivIdx) < bestFitness) { // smaller fitness value is better
                bestIdx = indivIdx;
                bestFitness = TSP.getFitness(indivIdx); // TODO: fix access
            }
        }
        
        sampledIndividuals[sampleIdx] = bestIdx;
        
    } // end sample
    
}


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


void Island::truncationSelection(int* sampledIndividuals, int numIndividualsToSample) {
    
    //double lastFitness = -1;
    //double currFitness = -1;
    
    for(int indivIdx = 0; indivIdx < numIndividualsToSample; indivIdx++) {
     
        sampledIndividuals[indivIdx] = ranks[indivIdx];
        
        //currFitness = TSP.getFitness(ranks[indivIdx]);
        //if (indivIdx != 0) {
        //    assert(lastFitness <= currFitness);
        //}
        //lastFitness = currFitness;
    }
    
}


void Island::pureRandomSelection(int* sampledIndividuals, int numIndividualsToSample) {
        
    for(int sampleIdx = 0; sampleIdx < numIndividualsToSample; sampleIdx++) {
        
        sampledIndividuals[sampleIdx] = rand() % numIndividualsIsland;
    }
    
}


void Island::truncationReplacement(int numImmigrants, int* immigrantGenes, double* immigrantFitnesses) {
    
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
        
        int* currGene = &genes[ranks[(numIndividualsIsland - numImmigrants) + indivIdx] * numIntegersGene];
        int* newGene = &immigrantGenes[immigrants[indivIdx].idx * numIntegersGene];
        
        overwriteGene(newGene, currGene, numIntegersGene);
        
        TSP.setFitness(ranks[(numIndividualsIsland - numImmigrants) + indivIdx],
                       immigrants[indivIdx].fitness);
    }
    
}


void Island::pureRandomReplacement(int numImmigrants, int* immigrantGenes, double* immigrantFitnesses) {
    
    for(int immigrantIdx = 0; immigrantIdx < numImmigrants; immigrantIdx++) {
        
        int indivToReplace = rand() % numIndividualsIsland;
        
        int* currGene = &genes[indivToReplace * numIntegersGene];
        int* newGene = &immigrantGenes[immigrantIdx * numIntegersGene];
        
        overwriteGene(newGene, currGene, numIntegersGene);
        
        TSP.setFitness(indivToReplace, immigrantFitnesses[immigrantIdx]);
    }
    
}


void Island::crowdingReplacement(int numImmigrants, int* immigrantGenes, double* immigrantFitnesses) {
    
    int crowdSize = 3;
    
    for(int immigrantIdx = 0; immigrantIdx < numImmigrants; immigrantIdx++) {
        
        int idxClosest = -1;
        int minHammingDist = numIntegersGene + 1;
        
        for(int crowdIdx = 0; crowdIdx < crowdSize; crowdIdx++) {
            
            int randIndivIdx = rand() % numIndividualsIsland;
            int currHammingDist = computeHammingDistance(&immigrantGenes[immigrantIdx * numIntegersGene],
                                                         &genes[randIndivIdx * numIntegersGene], numIntegersGene);
            
            if (currHammingDist < minHammingDist) {
                
                minHammingDist = currHammingDist;
                idxClosest = randIndivIdx;
            }
            
        }
        
        // replace individual
        overwriteGene(&immigrantGenes[immigrantIdx * numIntegersGene],
                      &genes[idxClosest * numIntegersGene], numIntegersGene);
        TSP.setFitness(idxClosest, immigrantFitnesses[immigrantIdx]);
    }
    
}


int Island::computeSendBufferSize() { // TODO: could be implemented in a more functional way
    
    switch(MIGRATION_TOPOLOGY) {
            
        case MigrationTopology::FULLY_CONNECTED: {
            
            int numSenders = sizeCommWorld - 1;
            
            if (numSenders == 0) { // bug fix for mpiexec -np 1
                return 0;
            }
            
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
            
        default: {
            return -1; // error
        }
            
    } // end switch case
    
}


int Island::computeReceiveBufferSize(int sendBufferSize) { // TODO: could be implemented in a more functional way
    
    switch (MIGRATION_TOPOLOGY) {
            
        case MigrationTopology::FULLY_CONNECTED: {
            
            // sizeCommWorld is total number of ranks using MPI_Allgather
            return sendBufferSize * sizeCommWorld;
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
            
        default: {
            return -1; // error
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
            
            tournamentSelection(migrants, numIndividualsSendBuffer);
            break;
        }
            
        case SelectionPolicy::PURE_RANDOM: {
            
            pureRandomSelection(migrants, numIndividualsSendBuffer);
            break;
        }
            
    } // end switch case
    
    
    for(int migrantIdx = 0; migrantIdx < numIndividualsSendBuffer; migrantIdx++) {
        
        // simply copy array
        overwriteGene(&genes[migrants[migrantIdx] * numIntegersGene],
                      &sendBufferGenes[migrantIdx * numIntegersGene], numIntegersGene);
    }
    
    for(int migrantIdx = 0; migrantIdx < numIndividualsSendBuffer; migrantIdx++) {
        
        sendBufferFitness[migrantIdx] = TSP.getFitness(migrants[migrantIdx]);
    }
    
}


void Island::doSynchronousBlockingCommunication() { // TODO: could be implemented in a more functional way
    
    switch(MIGRATION_TOPOLOGY) {
        
        case MigrationTopology::FULLY_CONNECTED: {
            
            fillSendBuffers();
            
            // "Gathers data from all tasks and distribute the combined data to all tasks"
            // - I suppose this is synchronized
            // - amount of data sent == amount of data received from any process
            MPI_Allgather(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE,
                          receiveBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE, MPI_COMM_WORLD);
            
            MPI_Allgather(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT,
                          receiveBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT, MPI_COMM_WORLD);
            
            emptyReceiveBuffers();
            
            break;
        }
        
        case MigrationTopology::ISOLATED: {
            
            int status = MPI_Barrier(MPI_COMM_WORLD); // Synchronize islands
            //assert(status == MPI_SUCCESS);
            break;
        }
        
        case MigrationTopology::RING: {
            
            // A cyclic list seems to be a good use case for MPI_Sendrecv
            
            int destRankID = (rankID + 1) % sizeCommWorld;
            
            int srcRankID = rankID - 1;
            if(srcRankID < 0) {
                srcRankID = sizeCommWorld - 1;
            }
            
            const int FITNESS_TAG = 0x01011;
            const int GENE_TAG = 0x01100;
            
            
            fillSendBuffers();
            
            int status = MPI_Sendrecv(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE,
                                      destRankID, FITNESS_TAG,
                                      receiveBufferFitness, numIndividualsReceiveBuffer, MPI_DOUBLE,
                                      srcRankID, FITNESS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //assert(status == MPI_SUCCESS);
            
            status = MPI_Sendrecv(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT,
                                  destRankID, GENE_TAG,
                                  receiveBufferGenes, numIndividualsReceiveBuffer * numIntegersGene, MPI_INT,
                                  srcRankID, GENE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //assert(status == MPI_SUCCESS);
            
            emptyReceiveBuffers();
                        
            break;
        }
        
    } // end switch case
    
}


void Island::doNonblockingCommunicationSetup() {
    
    switch(MIGRATION_TOPOLOGY) {
            
        case MigrationTopology::FULLY_CONNECTED: {
                
            // initiate Iallgather
            fillSendBuffers();
                
            MPI_Iallgather(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE,
                           receiveBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE,
                           MPI_COMM_WORLD, &receiveBufferFitnessRequest);
                
            MPI_Iallgather(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT,
                           receiveBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT,
                           MPI_COMM_WORLD, &receiveBufferGenesRequest);
            
            break;
        }
            
        case MigrationTopology::ISOLATED: {
            
            // do nothing (as opposed to Barrier in synchronous blocking communication)
            break;
        }
        
        case MigrationTopology::RING: {
            
            const int FITNESS_TAG = 0x01011;
            const int GENE_TAG = 0x01100;
            
            
            // initiate Isend
            int destRankID = (rankID + 1) % sizeCommWorld;
            
            fillSendBuffers();
            
            MPI_Isend(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE, destRankID,
                      FITNESS_TAG, MPI_COMM_WORLD, &sendBufferFitnessRequest);
            
            MPI_Isend(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT, destRankID,
                      GENE_TAG, MPI_COMM_WORLD, &sendBufferGenesRequest);
            
            
            // initiate Ireceive
            int srcRankID = rankID - 1;
            if(srcRankID < 0) {
                srcRankID = sizeCommWorld - 1;
            }
            
            MPI_Irecv(receiveBufferFitness, numIndividualsReceiveBuffer, MPI_DOUBLE, srcRankID,
                      FITNESS_TAG, MPI_COMM_WORLD, &receiveBufferFitnessRequest);
            
            MPI_Irecv(receiveBufferGenes, numIndividualsReceiveBuffer * numIntegersGene, MPI_INT, srcRankID,
                      GENE_TAG, MPI_COMM_WORLD, &receiveBufferGenesRequest);
            
            break;
        }
    
    } // end switch case
    
}


void Island::doNonblockingCommunication() {
    
    switch(MIGRATION_TOPOLOGY) {
            
        case MigrationTopology::FULLY_CONNECTED: {
            
            // wait on last Iallgather
            MPI_Wait(&receiveBufferFitnessRequest, MPI_STATUS_IGNORE);
            MPI_Wait(&receiveBufferGenesRequest, MPI_STATUS_IGNORE);
            
            emptyReceiveBuffers();
            
            
            // initiate Iallgather
            fillSendBuffers();
            
            MPI_Iallgather(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE,
                           receiveBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE,
                           MPI_COMM_WORLD, &receiveBufferFitnessRequest);
            
            MPI_Iallgather(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT,
                           receiveBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT,
                           MPI_COMM_WORLD, &receiveBufferGenesRequest);
            
            break;
        }
            
        case MigrationTopology::ISOLATED: {
            
            // do nothing (as opposed to Barrier in synchronous blocking communication)
            break;
        }
            
        case MigrationTopology::RING: {
            
            const int FITNESS_TAG = 0x01011;
            const int GENE_TAG = 0x01100;
            
            
            // do alternating (send, recv), (recv, send) to avoid a deadlock
            // for a ring it only works if the largest rankID is odd!
            if(rankID % 2 == 0) {
                
                // wait on last Isend, initiate Isend
                MPI_Wait(&sendBufferFitnessRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&sendBufferGenesRequest, MPI_STATUS_IGNORE);
                
                int destRankID = (rankID + 1) % sizeCommWorld;
                
                fillSendBuffers();
                
                MPI_Isend(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE, destRankID,
                          FITNESS_TAG, MPI_COMM_WORLD, &sendBufferFitnessRequest);
                
                MPI_Isend(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT, destRankID,
                          GENE_TAG, MPI_COMM_WORLD, &sendBufferGenesRequest);
                
                
                // wait on last Irecv, initiate Irecv
                MPI_Wait(&receiveBufferFitnessRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&receiveBufferGenesRequest, MPI_STATUS_IGNORE);
                
                emptyReceiveBuffers();
                
                int srcRankID = rankID - 1;
                if(srcRankID < 0) {
                    srcRankID = sizeCommWorld - 1;
                }
                
                MPI_Irecv(receiveBufferFitness, numIndividualsReceiveBuffer, MPI_DOUBLE, srcRankID,
                          FITNESS_TAG, MPI_COMM_WORLD, &receiveBufferFitnessRequest);
                
                MPI_Irecv(receiveBufferGenes, numIndividualsReceiveBuffer * numIntegersGene, MPI_INT, srcRankID,
                          GENE_TAG, MPI_COMM_WORLD, &receiveBufferGenesRequest);
                
                
            } else { // rankID % 2 == 1
                
                // wait on last Irecv, initiate Irecv
                MPI_Wait(&receiveBufferFitnessRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&receiveBufferGenesRequest, MPI_STATUS_IGNORE);
                
                emptyReceiveBuffers();
                                
                int srcRankID = rankID - 1;
                if(srcRankID < 0) {
                    srcRankID = sizeCommWorld - 1;
                }
                
                MPI_Irecv(receiveBufferFitness, numIndividualsReceiveBuffer, MPI_DOUBLE, srcRankID,
                          FITNESS_TAG, MPI_COMM_WORLD, &receiveBufferFitnessRequest);
                
                MPI_Irecv(receiveBufferGenes, numIndividualsReceiveBuffer * numIntegersGene, MPI_INT, srcRankID,
                          GENE_TAG, MPI_COMM_WORLD, &receiveBufferGenesRequest);
                
                
                // wait on last Isend, initiate Isend
                MPI_Wait(&sendBufferFitnessRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&sendBufferGenesRequest, MPI_STATUS_IGNORE);
                                
                fillSendBuffers();
                
                int destRankID = (rankID + 1) % sizeCommWorld;
                
                MPI_Isend(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE, destRankID,
                          FITNESS_TAG, MPI_COMM_WORLD, &sendBufferFitnessRequest);
                
                MPI_Isend(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, MPI_INT, destRankID,
                          GENE_TAG, MPI_COMM_WORLD, &sendBufferGenesRequest);

            }
            
            break;
        }
            
    } // end switch case
    
}


void Island::doNonblockingCommunicationCleanup() {
    
    switch(MIGRATION_TOPOLOGY) {
        
        case MigrationTopology::FULLY_CONNECTED: {
            
            // wait on Iallgather
            MPI_Wait(&receiveBufferFitnessRequest, MPI_STATUS_IGNORE);
            MPI_Wait(&receiveBufferGenesRequest, MPI_STATUS_IGNORE);
            
            emptyReceiveBuffers();
            
            break;
        }
            
        case MigrationTopology::ISOLATED: {
                
            // do nothing (as opposed to Barrier in synchronous blocking communication)
            break;
        }
            
        case MigrationTopology::RING: {
            
            if(rankID % 2 == 0) {
                
                // wait on Isend
                MPI_Wait(&sendBufferFitnessRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&sendBufferGenesRequest, MPI_STATUS_IGNORE);
                
                // wait on Ireceive
                MPI_Wait(&receiveBufferFitnessRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&receiveBufferGenesRequest, MPI_STATUS_IGNORE);
                
                emptyReceiveBuffers();
                
            } else {
                
                // wait on Ireceive
                MPI_Wait(&receiveBufferFitnessRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&receiveBufferGenesRequest, MPI_STATUS_IGNORE);
                
                emptyReceiveBuffers();
                
                // wait on Isend
                MPI_Wait(&sendBufferFitnessRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&sendBufferGenesRequest, MPI_STATUS_IGNORE);
                
            }
            
            break;
        }
            
    } // end switch case
    
}


void Island::emptyReceiveBuffers() { // TODO: maybe use a function pointer which is set inside the constructor
    
    switch(REPLACEMENT_POLICY) {
            
        case ReplacementPolicy::TRUNCATION:
            
            truncationReplacement(numIndividualsReceiveBuffer, receiveBufferGenes, receiveBufferFitness);
            break;
        
        case ReplacementPolicy::PURE_RANDOM:
            
            pureRandomReplacement(numIndividualsReceiveBuffer, receiveBufferGenes, receiveBufferFitness);
            break;
            
        case ReplacementPolicy::DEJONG_CROWDING:
            
            crowdingReplacement(numIndividualsReceiveBuffer, receiveBufferGenes, receiveBufferFitness);
            break;
            
    } // end switch case
    
}


double Island::solve(const int numEvolutions) {
        
    int numPeriods = numEvolutions / MIGRATION_PERIOD;
        
    switch(COMMUNICATION) {
            
        case UnderlyingCommunication::BLOCKING: {
            break;
        }
            
        case UnderlyingCommunication::NON_BLOCKING: {
            if (numPeriods >= 1) {
                // initiate first nonblocking send, recv
                doNonblockingCommunicationSetup();
            }
            // needs doNonblockingCommunication() or doNonblockingCommunicationCleanup()
            // to be called after this
            break;
        }
            
        case UnderlyingCommunication::RMA: {
            break;
        }
    } // end switch case
    
        
    for(int currPeriod = 0; currPeriod < numPeriods; currPeriod++) {
        
        // Run the GA for MIGRATION_PERIOD iterations
        TSP.solve(MIGRATION_PERIOD, rankID);
        
        
        // MPI part (Migration part)
        switch(COMMUNICATION) {
                
            case UnderlyingCommunication::BLOCKING: {
                
                doSynchronousBlockingCommunication();
                break;
            }
                
            case UnderlyingCommunication::NON_BLOCKING: {
                
                if (currPeriod < numPeriods - 1) {
                    doNonblockingCommunication();
                    // needs doNonblockingCommunication() or doNonblockingCommunicationCleanup()
                    // to be called after this
                } else {
                    // don't initiate any new nonblocking send, recv
                    doNonblockingCommunicationCleanup();
                }
                
                break;
            }
                
            case UnderlyingCommunication::RMA: {
                break;
            }
        } // end switch case
        
    }
    
    // deal with excess iterations that no more fit in a migration period
    if (numEvolutions % MIGRATION_PERIOD != 0) {
        
        TSP.solve(numEvolutions % MIGRATION_PERIOD, rankID);
    }
    
    
    return TSP.getMinFitness();
}


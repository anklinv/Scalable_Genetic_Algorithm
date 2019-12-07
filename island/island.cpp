
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

    // Initialize clock
    this->clock = std::chrono::high_resolution_clock();
    
    // convenience variables
    numIntegersGene = TSP.problem_size;
    numIndividualsIsland = TSP.population_count;
    
    ranks = TSP.getRanks();
    genes = TSP.getGenes();
    
    
    // compute sizes of send buffers and receive buffers
    numIndividualsSendBuffer = computeSendBufferSize();
    numIndividualsReceiveBuffer = computeReceiveBufferSize(numIndividualsSendBuffer);
    // allocate send buffers
    sendBufferGenes = new Int[numIndividualsSendBuffer * numIntegersGene];
    sendBufferFitness = new double[numIndividualsSendBuffer];
    // allocate receive buffers
    receiveBufferGenes = new Int[numIndividualsReceiveBuffer * numIntegersGene];
    receiveBufferFitness = new double[numIndividualsReceiveBuffer];
    
    
    // create a custom MPI datatype in order to transfer gene data
    MPI_Type_contiguous(sizeof(Int), MPI_BYTE, &CUSTOM_MPI_INT);
    MPI_Type_commit(&CUSTOM_MPI_INT);
    
    
    // initialize communication request handles to null
    sendBufferFitnessRequest = MPI_REQUEST_NULL;
    sendBufferGenesRequest = MPI_REQUEST_NULL;
    MPI_Request receiveBufferFitnessRequest = MPI_REQUEST_NULL;
    MPI_Request receiveBufferGenesRequest = MPI_REQUEST_NULL;
    
}


Island::~Island() {
    
    // free custom MPI datatype
    MPI_Type_free(&CUSTOM_MPI_INT);
    
    // free buffers
    delete sendBufferGenes;
    delete sendBufferFitness;
    
    delete receiveBufferGenes;
    delete receiveBufferFitness;
    
}


void Island::overwriteGene(Int* newGene, Int* oldGene, int geneSize) {
    
    for(int geneIdx = 0; geneIdx < geneSize; geneIdx++) {
        oldGene[geneIdx] = newGene[geneIdx];
    }
    
    //for(int geneIdx = 0; geneIdx < geneSize; geneIdx++)
    //    assert(oldGene[geneIdx] == newGene[geneIdx]);
    
    //assert(newGene + geneSize - 1 < oldGene || oldGene + geneSize - 1 < newGene);
}


int Island::computeHammingDistance(Int* firstGene, Int* scndGene, int geneSize) {
    
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


void Island::truncationReplacement(int numImmigrants, Int* immigrantGenes, double* immigrantFitnesses) {
    // Use (idx, fitness) pairs to facilitate sorting
    Individual immigrants[numImmigrants - numIndividualsSendBuffer];

    int helperIdx = 0;
    for(int immigrantIdx = 0; immigrantIdx < numImmigrants; immigrantIdx++) {
        if(immigrantIdx < rankID * numIndividualsSendBuffer || (rankID + 1) * numIndividualsSendBuffer <= immigrantIdx) {
            (immigrants[helperIdx]).idx = immigrantIdx;
            (immigrants[helperIdx]).fitness = immigrantFitnesses[immigrantIdx];
            helperIdx++;
        }
    }
    assert(helperIdx == numImmigrants - numIndividualsSendBuffer);
    
    // Sort incoming data in ascending order
    // (the data of the current island is already sorted)
    sort(immigrants, immigrants + (numImmigrants - numIndividualsSendBuffer));
    
    
    // Determine how many individuals of the island are going to be replaced
    int numToReplace = 0;
    
    int immigrantIdx = 0;
    int islandIndivIdx = 0;
    
    int offsetIsland = numIndividualsIsland - (numImmigrants - numIndividualsSendBuffer);
    
    while(immigrantIdx < (numImmigrants - numIndividualsSendBuffer)
          && islandIndivIdx < (numImmigrants - numIndividualsSendBuffer)) {
        
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
        
        Int* currGene = &genes[ranks[(numIndividualsIsland - 1) - indivIdx] * numIntegersGene];
        Int* newGene = &immigrantGenes[immigrants[indivIdx].idx * numIntegersGene];
        
        overwriteGene(newGene, currGene, numIntegersGene);
        
        // don't update fitness
    }
    
}


void Island::pureRandomReplacement(int numImmigrants, Int* immigrantGenes, double* immigrantFitnesses) {
    
    for(int immigrantIdx = 0; immigrantIdx < numImmigrants; immigrantIdx++) {
        
        int indivToReplace = rand() % numIndividualsIsland;
        
        Int* currGene = &genes[indivToReplace * numIntegersGene];
        Int* newGene = &immigrantGenes[immigrantIdx * numIntegersGene];
        
        overwriteGene(newGene, currGene, numIntegersGene);
        
        TSP.setFitness(indivToReplace, immigrantFitnesses[immigrantIdx]);
    }
    
}


void Island::crowdingReplacement(int numImmigrants, Int* immigrantGenes, double* immigrantFitnesses) {
    
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
            int numExternalSenders = sizeCommWorld - 1;
            if (numExternalSenders == 0) { // bug fix for mpiexec -np 1
                // this would cause a division by zero
                return 0;
            }
            if (NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION == 0) {
                // treat this case separatly
                return 0;
            }
            if (NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION % numExternalSenders == 0) {
                return NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION / numExternalSenders;
            } else {
                return (NUM_INDIVIDUALS_RECEIVED_PER_MIGRATION / numExternalSenders) + 1; // ceil
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
            // see semantics of MPI_Allgather
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
    // used as out parameter
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
                      &sendBufferGenes[migrantIdx * numIntegersGene],
                      numIntegersGene);
    }
    for(int migrantIdx = 0; migrantIdx < numIndividualsSendBuffer; migrantIdx++) {
        sendBufferFitness[migrantIdx] = TSP.getFitness(migrants[migrantIdx]);
    }
}


void Island::doSynchronousBlockingCommunication() { // TODO: could be implemented in a more functional way
    switch(MIGRATION_TOPOLOGY) {
        case MigrationTopology::FULLY_CONNECTED: {
            fillSendBuffers(); // fill send buffers
            // "Gathers data from all tasks and distribute the combined data to all tasks"
            // - I suppose this is synchronized
            // - amount of data sent == amount of data received from any process
            MPI_Allgather(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE,
                          receiveBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE,
                          MPI_COMM_WORLD);
            MPI_Allgather(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT,
                          receiveBufferGenes, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT,
                          MPI_COMM_WORLD);
            emptyReceiveBuffers(); // empty receive buffers
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
            
            status = MPI_Sendrecv(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT,
                                  destRankID, GENE_TAG,
                                  receiveBufferGenes, numIndividualsReceiveBuffer * numIntegersGene, CUSTOM_MPI_INT,
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
                
            MPI_Iallgather(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT,
                           receiveBufferGenes, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT,
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
            
            MPI_Isend(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT, destRankID,
                      GENE_TAG, MPI_COMM_WORLD, &sendBufferGenesRequest);
            
            
            // initiate Ireceive
            int srcRankID = rankID - 1;
            if(srcRankID < 0) {
                srcRankID = sizeCommWorld - 1;
            }
            
            MPI_Irecv(receiveBufferFitness, numIndividualsReceiveBuffer, MPI_DOUBLE, srcRankID,
                      FITNESS_TAG, MPI_COMM_WORLD, &receiveBufferFitnessRequest);
            
            MPI_Irecv(receiveBufferGenes, numIndividualsReceiveBuffer * numIntegersGene, CUSTOM_MPI_INT, srcRankID,
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
            
            MPI_Iallgather(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT,
                           receiveBufferGenes, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT,
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
                
                MPI_Isend(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT, destRankID,
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
                
                MPI_Irecv(receiveBufferGenes, numIndividualsReceiveBuffer * numIntegersGene, CUSTOM_MPI_INT, srcRankID,
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
                
                MPI_Irecv(receiveBufferGenes, numIndividualsReceiveBuffer * numIntegersGene, CUSTOM_MPI_INT, srcRankID,
                          GENE_TAG, MPI_COMM_WORLD, &receiveBufferGenesRequest);
                
                
                // wait on last Isend, initiate Isend
                MPI_Wait(&sendBufferFitnessRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&sendBufferGenesRequest, MPI_STATUS_IGNORE);
                                
                fillSendBuffers();
                
                int destRankID = (rankID + 1) % sizeCommWorld;
                
                MPI_Isend(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE, destRankID,
                          FITNESS_TAG, MPI_COMM_WORLD, &sendBufferFitnessRequest);
                
                MPI_Isend(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT, destRankID,
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


void Island::setupRMACommunication() {
    
    int destRankID = (rankID + 1) % sizeCommWorld;
    
    
    MPI_Win_allocate(numIndividualsReceiveBuffer * sizeof(double),
                     sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD,
                     &fitnessWindowBaseAddress, &fitnessWindow);
    
    MPI_Win_allocate(numIndividualsReceiveBuffer * numIntegersGene * sizeof(Int),
                     sizeof(Int), MPI_INFO_NULL, MPI_COMM_WORLD,
                     &geneWindowBaseAddress, &geneWindow);
        
    // allocate the memory containing the lock
    // size is basically the amount of incoming edges
    MPI_Alloc_mem(1 * sizeof(int),
                  MPI_INFO_NULL, &lockWindowBaseAddress);
    // initialize the memory containing the lock
    *lockWindowBaseAddress = RECV_BUFFER_EMPTY;
    // put the memory containing the lock in a window
    MPI_Win_create(lockWindowBaseAddress, 1 * sizeof(int), sizeof(int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &lockWindow);
    
    
    // we need access to the window of the destination rank
    MPI_Win_lock(MPI_LOCK_SHARED, destRankID, 0, fitnessWindow);
    MPI_Win_lock(MPI_LOCK_SHARED, destRankID, 0, geneWindow);
    MPI_Win_lock(MPI_LOCK_SHARED, destRankID, 0, lockWindow);
    
    // we need need acces to our own window
    MPI_Win_lock(MPI_LOCK_SHARED, rankID, 0, fitnessWindow);
    MPI_Win_lock(MPI_LOCK_SHARED, rankID, 0, geneWindow);
    MPI_Win_lock(MPI_LOCK_SHARED, rankID, 0, lockWindow);
    
    
    // assert that the lock of our own window is correctly initialized
    MPI_Fetch_and_op(&ignoreOrigin, &testBuffer,
                     MPI_INT, rankID, 0, // no offset
                     MPI_NO_OP, lockWindow);
    MPI_Win_flush(rankID, lockWindow); // block until fetch and op is done
    assert(testBuffer == RECV_BUFFER_EMPTY);
    // end assert
}


void Island::emptyReceiveBuffersRMA() {
    
    // separate method because the data is read directly from the window memory
    
    switch(REPLACEMENT_POLICY) {
            
        case ReplacementPolicy::TRUNCATION:
            
            truncationReplacement(numIndividualsReceiveBuffer, geneWindowBaseAddress, fitnessWindowBaseAddress);
            break;
        
        case ReplacementPolicy::PURE_RANDOM:
            
            pureRandomReplacement(numIndividualsReceiveBuffer, geneWindowBaseAddress, fitnessWindowBaseAddress);
            break;
            
        case ReplacementPolicy::DEJONG_CROWDING:
            
            crowdingReplacement(numIndividualsReceiveBuffer, geneWindowBaseAddress, fitnessWindowBaseAddress);
            break;
            
    } // end switch case
    
}


bool Island::doRMASend() {
    
    int destRankID = (rankID + 1) % sizeCommWorld;
    
    MPI_Fetch_and_op(&ignoreOrigin, &testBuffer,
                     MPI_INT, destRankID, 0, // no offset
                     MPI_NO_OP, lockWindow);
    MPI_Win_flush(destRankID, lockWindow); // block until fetch and op is done
    
    
    if(testBuffer == RECV_BUFFER_FULL) {
        // data can't be sent yet
        return false;
    } else { // testBuffer == RECV_BUFFER_EMPTY
        // send
        fillSendBuffers();
        
        // atomic nonblocking puts
        MPI_Accumulate(sendBufferFitness, numIndividualsSendBuffer, MPI_DOUBLE,
                       destRankID,
                       0, numIndividualsSendBuffer, MPI_DOUBLE, MPI_REPLACE, fitnessWindow);
        
        MPI_Accumulate(sendBufferGenes, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT,
                       destRankID,
                       0, numIndividualsSendBuffer * numIntegersGene, CUSTOM_MPI_INT, MPI_REPLACE, geneWindow);
        
        // block until RMA communication is done
        MPI_Win_flush(destRankID, fitnessWindow);
        MPI_Win_flush(destRankID, geneWindow);
        
        
        // set "result is there" flag at target
        // one could use only one of these using clever logical operations
        // there are a bunch of constants defined in the docs (MPI_NO_OP, MPI_REPLACE, MPI_LXOR, ...)
        // maybe CAS is more suited for this case
        // CAS / fetch and op: atomic, work on one variable only
        // I think they're meant to do locks etc. with
        MPI_Fetch_and_op(&RECV_BUFFER_FULL, &testBuffer,
                         MPI_INT, destRankID, 0, // no offset
                         MPI_NO_OP, lockWindow);
        MPI_Win_flush(destRankID, lockWindow); // block until fetch and op is done
        
        // assert the buffer of the target was empty
        assert(testBuffer == RECV_BUFFER_EMPTY);
        // end assert
        
        return true;
    } // end else
    
}


bool Island::doRMAPoll() {
    
    MPI_Fetch_and_op(&ignoreOrigin, &testBuffer,
                     MPI_INT, rankID, 0, // no offset
                     MPI_NO_OP, lockWindow);
    MPI_Win_flush(rankID, lockWindow); // block until fetch and op is done
    
    
    if(testBuffer == RECV_BUFFER_EMPTY) {
        // no new data arrived
        return false;
    } else { // testBuffer == RECV_BUFFER_FULL
        // receive
        
        // potential problem: the window data is directly accessed & it is not protected
        emptyReceiveBuffersRMA();
        
        
        MPI_Fetch_and_op(&RECV_BUFFER_EMPTY, &testBuffer,
                         MPI_INT, rankID, 0, // no offset
                         MPI_NO_OP, lockWindow);
        MPI_Win_flush(rankID, lockWindow); // block until fetch and op is done
        
        // assert the buffer was full
        assert(testBuffer == RECV_BUFFER_FULL);
        // end assert
        
        return true;
    } // end else
    
}


bool Island::doRMACommunication(bool startMigration) {
    
    // always do this
    doRMAPoll();
    
    if(startMigration == true) {
        // indicate whether send was successful
        return doRMASend();
    } else {
        // success as nothing had to be done
        return true;
    }
}


void Island::cleanupRMACommunication() {
    
    int destRankID = (rankID + 1) % sizeCommWorld;
    
    
    // unlock window of destination rank
    MPI_Win_unlock(destRankID, fitnessWindow);
    MPI_Win_unlock(destRankID, geneWindow);
    MPI_Win_unlock(destRankID, lockWindow);
    
    // unlock our own window
    MPI_Win_unlock(rankID, fitnessWindow);
    MPI_Win_unlock(rankID, geneWindow);
    MPI_Win_unlock(rankID, lockWindow);
    
    
    // free window
    MPI_Win_free(&fitnessWindow);
    MPI_Win_free(&geneWindow);
    // free memory allocated with MPI_Alloc_mem
    MPI_Win_free(&lockWindow); // free window first
    MPI_Free_mem(lockWindowBaseAddress);
}


void Island::solveBlocking(const int numEvolutions) {
    int numPeriods = numEvolutions / MIGRATION_PERIOD;
    comp_duration = 0;
    comm_duration = 0;
    int logging_frequency = TSP.log_iter_freq / MIGRATION_PERIOD + 1;
    for (int currPeriod = 0; currPeriod < numPeriods; currPeriod++) {
        // Run the GA for MIGRATION_PERIOD iterations
        comp_start = clock.now();
        TSP.solve(MIGRATION_PERIOD, rankID);
        comp_end = clock.now();
        comp_duration += std::chrono::duration_cast<std::chrono::microseconds>(comp_end - comp_start).count();

        // MPI part (Migration part)
        comm_start = clock.now();
        doSynchronousBlockingCommunication();
        comm_end = clock.now();
        comm_duration += std::chrono::duration_cast<std::chrono::microseconds>(comm_end - comm_start).count();

        // Log if necessary
        if (currPeriod % logging_frequency == 0) {
            TSP.logger->LOG(COMPUTATION, comp_duration / logging_frequency);
            TSP.logger->LOG(COMMUNICATION, comm_duration / logging_frequency);
            comp_duration = 0;
            comm_duration = 0;
        }
    }
    // deal with excess iterations that no more fit in a migration period
    if (numEvolutions % MIGRATION_PERIOD != 0) {
        TSP.solve(numEvolutions % MIGRATION_PERIOD, rankID);
    }
}


void Island::solveNonblocking(const int numEvolutions) {
    
    int numPeriods = numEvolutions / MIGRATION_PERIOD;
    
    if (numPeriods >= 1) {
        // initiate first nonblocking send, recv
        doNonblockingCommunicationSetup();
    }
    // needs doNonblockingCommunication() or doNonblockingCommunicationCleanup()
    // to be called after this

    comp_duration = 0;
    comm_duration = 0;
    int logging_frequency = min(1, TSP.log_iter_freq / MIGRATION_PERIOD);
    for (int currPeriod = 0; currPeriod < numPeriods; currPeriod++) {

        // Run the GA for MIGRATION_PERIOD iterations
        comp_start = clock.now();
        TSP.solve(MIGRATION_PERIOD, rankID);
        comp_end = clock.now();
        comp_duration += std::chrono::duration_cast<std::chrono::microseconds>(comp_end - comp_start).count();

        // MPI part (Migration part)
        comm_start = clock.now();
        if(currPeriod < numPeriods - 1) {
            doNonblockingCommunication();
            // needs doNonblockingCommunication() or doNonblockingCommunicationCleanup()
            // to be called after this
        } else {
            // don't initiate any new nonblocking send, recv
            doNonblockingCommunicationCleanup();
        }
        comm_end = clock.now();
        comm_duration += std::chrono::duration_cast<std::chrono::microseconds>(comm_end - comm_start).count();

        // Log if necessary
        if (currPeriod % logging_frequency == 0) {
            TSP.logger->LOG(COMPUTATION, comp_duration);
            TSP.logger->LOG(COMMUNICATION, comm_duration);
            comp_duration = 0;
            comm_duration = 0;
        }
    }
    
    // deal with excess iterations that no more fit in a migration period
    if (numEvolutions % MIGRATION_PERIOD != 0) {
        TSP.solve(numEvolutions % MIGRATION_PERIOD, rankID);
    }
    
}


void Island::solveRMA(const int numEvolutions) {
    
    int POLLING_PERIOD = max(1, MIGRATION_PERIOD / 8);
    // polling period is at most 100% of migration period (MIGRATION_PERIOD < 8)
    // polling eriod is at most 12.5% of migration period (else)
    
    int numPeriods = numEvolutions / MIGRATION_PERIOD;
    int numPolls = MIGRATION_PERIOD / POLLING_PERIOD;
    // numPolls >= 1 (MIGRATION_PERIOD < 8)
    // numPolls >= 8 (else)
    // (important at it must be guaranteed to be >= 1 for the communication to happen)
    assert(numPolls >= 1);
    
    setupRMACommunication();
    
    bool startMigration = false;
    
    for(int currPeriod = 0; currPeriod < numPeriods; currPeriod++) {
        
        for(int currPoll = 0; currPoll < numPolls; currPoll++) {
            if(doRMACommunication(startMigration)) {
                // send succeeded or startMigration was false
                // continue polling as we also check for data which can arrive any time
                startMigration = false;
            }

            // POLLING_PERIOD
            TSP.solve(MIGRATION_PERIOD, rankID);
        }
        
        // deal with excess iterations that no more fit in a polling period
        if (MIGRATION_PERIOD % POLLING_PERIOD != 0) {
            TSP.solve(MIGRATION_PERIOD % POLLING_PERIOD, rankID);
        }
        
        startMigration = true;
        
    }
    
    // there currently is no final migration after the last migration period
    
    
    // deal with excess iterations that no more fit in a migration period
    if (numEvolutions % MIGRATION_PERIOD != 0) {
        TSP.solve(numEvolutions % MIGRATION_PERIOD, rankID);
    }
    
    cleanupRMACommunication();
    
}


double Island::solve(const int numEvolutions) {
    int numReceived = numIndividualsReceiveBuffer - numIndividualsSendBuffer;
    assert(numReceived <= numIndividualsIsland);
    switch(COMMUNICATION) {
        case UnderlyingCommunication::BLOCKING: {
            solveBlocking(numEvolutions);
            break;
        }
        case UnderlyingCommunication::NON_BLOCKING: {
            solveNonblocking(numEvolutions);
            break;
        }
        case UnderlyingCommunication::RMA: {
            solveRMA(numEvolutions);
            break;
        }
    } // end switch case
    return TSP.getMinFitness();
}


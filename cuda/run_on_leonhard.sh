#!/bin/sh

if [ "$1" == "" ]; then
    echo "Usage: $0 <username> [<remote_directory>] [<local_directory>]"
    exit
fi

if [ "$2" == "" ]; then
    REMOTE_DIR="sga/cuda"
else
    REMOTE_DIR=$2
fi

if [ "$3" == "" ]; then
    LOCAL_DIR="./"
else
    LOCAL_DIR=$3
fi

echo "copying files..."
# add --delete to rsync to delete files on the remote which don't exist locally
rsync -r $LOCAL_DIR $1@login.leonhard.ethz.ch:$REMOTE_DIR && \
ssh $1@login.leonhard.ethz.ch "module load cuda && cd $REMOTE_DIR && make && bsub -I -R \"rusage[ngpus_excl_p=1]\" ./Test"

# Scalable_Genetic_Algorithm
 Scalable implementation for the course project of Design of Parallel and High Performance Computing HS19

## How to run
You need mpicxx, mpiexec (any MPI implementation) and also cmake
(install this using `sudo apt install cmake g++`)

Then you first run cmake:
`cmake .`
Compile the program:
`make`
Run the program:
`mpiexec -np <number of processes> ./Distributed_Genetic_Algorithm`

## How to setup Leonhard
#### Login to Leonhard
Note: you need to be in the ETH network! (use VPN)
```ssh <netz>@login.leonhard.ethz.ch```

To copy your SSH key, in your command line use:

`ssh-copy-id <NETHZ>@login.leonhard.ethz.ch` (Linux and OSX)

`cat ~/.ssh/id_rsa.pub | ssh <NETHZ>@login.leonhard.ethz.ch "cat >> ~/.ssh/authorized_keys"` (Windows)

#### Load Modules
```module load gcc/8.2.0```

```module load openmpi/4.0.1```

To avoid typing this in at every login, you can also add it to your `.bashrc` file using the following commands (you only need to do this once):
```
cd ~
echo "module load gcc/8.2.0" >> .bashrc
echo "module load openmpi/4.0.1" >> .bashrc
```

(if you ever need to remove it again, just open `.bashrc` and delete the last two lines again.

#### Compile
```
cmake .
make
```

#### Test if it works
Do not do this exessively as it runs on the login node. Just use for sanity check.

```mpirun -np 2 ./Distributed_Genetic_Algorithm```

## Running a single job on Leonhard
#### Submit job
Use the following command to submit a job on Leonhard:

```bsub -n 4 -W 00:10 -o log_test -R "rusage[mem=1024]" mpirun ./Distributed_Genetic_Algorithm```

`-n` specifies the number of cores to use. Note: in the mpirun call you do not need to actually specify `-np`.

`-W 00:10` specifies that we want to run it for at most 10 minutes (after which the process gets killed). As we should not submit jobs that run longer than 30 minutes, keep this at most at `00:30`, but if the value is lower it might get scheduled faster.

`-o log_test` specifies the name of the logfile. If unspecified then the cluster will create a cryptic name.

`-R "rusage[mem=1024]"` specifies how much memory is used PER CORE.

#### Look at all jobs
`bjobs -a` lists all jobs for that user. You can also use `bbjobs -a` to get more information.
#### Peek at the output (possible as soon as the job is running)
`bpeek -f` shows the console output of the job (if only one is active you do not need to specify the job-id)
#### Read the logfile
`vi log_test`

## Running experiments on Leonhard
#### Experiment specification
Write a .json file that adheres to the specification below to describe what kind of jobs the cluster
#### Validate the experiment specification
To avoid frustration that the jobs did not run correctly, there is an easy way to test what kind of jobs will be scheduled consecutively:
```
python run_experiments.py --dry_run -e experiment.json
```
where after `-e` follows a list of experiment files to run. The script simply prints all the jobs that will be submitted. If you are happy with the results you can proceed to actually scheduling the jobs
#### Run the experiment
As this python script will run for a long time, we recommend that you use `nohup` to avoid any interruptions.
```
nohup python run_experiments.py -e experiment.json >> output.log &
```
This script will also create a folder in `logs` for each experiment and put any outputs in `output.log`.
When executing the command it outputs the PID of the job. You will need this if you ever need to kill the script.
#### Kill the experiment script
If you ever need to kill the experiment script, you can use the command `kill <PID>` where PID is the PID that was outputted when running the experiment. In case you don't know it, use `htop` and search for `python` to find a process that runs under your own username and use that PID.

## Evaluating experiments that were run on Leonhard
#### Copying files to a local directory
After running an experiment to completion, all the logfiles were automatically written to `logs/<EXPERIMENT-DIRECTORY>`. You should try to find the right folder first (using `ls logs`), then go to that directory (using `cd logs/<EXPERIMENT-DIRECTORY>`).

Find the current working directory (using `pwd`) and copy it to your clipboard.

Log out from SSH (using `Ctrl-D`/`CMD-D`) and navigate to the directory where you want to copy the logfiles to. Usually this should be `/path/to/Scalable_Genetic_Algorithm/logs`.

Using SCP, copy all the files to the current directory: `scp -r <NETHZ>@login.leonhard.ethz.ch:<WORKING-DIR> .` where <WORKING-DIR> is the directory that you have saved to your clipboard before. This might take a while.
 
#### Plot fitness values
To plot the results on your local machine navigate to the `logs` directory and start a jupyter notebook (using `jupyter notebook`, make sure you have it installed with `pip install jupyter`).

Open the file `plot_results.ipynb` and install the necessary libraries if necessary (by running the first cell). 

You can extract the fitness values using the following line:

```
experiments, dataframes = extract_all_run_values(<EXPERIMENT-DIRECTORY>)
```

This returns into `experiments` a list of names of the experiments that were run. `dataframes` is also a list of dataframes of the corresponding experiments. Each element of `dataframes` is a dataframe with the following columns:

- epoch: epoch when it was recorded
- fitness: minimum fitness across all ranks
- run: which repetition it the values are from

Plot the values using multiple lines:

```
ax = sns.lineplot(x="epoch", y="fitness", hue="run", data=dataframes[0])
ax.set_title(experiments[0])
ax
```

Plot the values using confidence intervals:

```
ax = sns.lineplot(x="epoch", y="fitness", data=dataframes[0])
ax.set_title(experiments[0])
ax
```

You can save plots as you usually would using matplotlib (using `plt.savefig("fig.png")`)

## JSON specification for experiments
When writing an experiment specification you need to follow this standard:
```
{
  "name" : <EXPERIMENT_NAME>,
  "repetitions" : <NR_REPETITIONS>,
  "fixed_params" : {
    <FIXED_PARAM_1_NAME> : <FIXED_PARAM_1_VALUE>,
    ...
    <FIXED_PARAM_N_NAME> : <FIXED_PARAM_N_VALUE>
  },
  "variable_params" : {
    <VAR_PARAM_1_NAME> : <VAR_PARAM_1_VALUE>,
    ...
    <VAR_PARAM_M_NAME> : <VAR_PARAM_M_VALUE>
  }
}
```
- <EXPERIMENT_NAME> is any identifier for the experiment. White spaces will be turned into underscores for the folder name
- <NR_REPETITIONS> is the number of repetitions of the experiment
- <FIXED_PARAM_i_NAME> is the string of the argument identifier that should be fixed in the experiment.
- <FIXED_PARAM_i_VALUE> is the integer or string of the value.
- <VAR_PARAM_j_NAME> is the string of the argument identifier that should be varied in the experiment.
- <VAR_PARAM_j_VALUE> is a dict of the elements that should be varied.
  - It can be a range from <MIN> to <MAX> with stride <STRIDE> (if stride is not specified the value 1 is assumed):
    ```
    {
      "type" : "range",
      "min" : <MIN>,
      "max" : <MAX>,
      "stride" : <STRIDE>
    }
    ```
   - It can be a list of values <VAL_1>, ..., <VAL_K>:
     ```
     {
       "type" : "list",
       "list" : [
         <VAL_1>,
         ...
         <VAL_K>
       ]
     }
     ```

## Example JSON for experiment
In this example we try different number of islands of different sizes. We try every combination of using an island of size k where `1 <= k < 4` and the population size p where `p ∈ {100, 200, 400, 800}` for 10 repetitions.
```
{
  "name" : "try scaling and population size",
  "repetitions" : 10,
  "fixed_params" : {
    "mode" : "island"
  },
  "variable_params" : {
    "-n": {
      "type": "range",
      "min": 1,
      "max": 4,
      "stride": 1
    },
    "--population": {
      "type": "list",
      "list": [
        100,
        200,
        400,
        800
      ]
    }
  }
}
```

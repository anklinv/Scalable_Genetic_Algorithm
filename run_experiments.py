import time
import subprocess
import os
import json
import argparse

dry_run = True


def no_job_running():
    if dry_run:
        return True
    proc = subprocess.Popen(["bjobs"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout.read()
    return output == "No unfinished job found\n"


def submit_job(log_name, filename="Distributed_Genetic_Algorithm", n=1, W="00:30", mem=256):
    command = "bsub -n {} -W {} -o {}.log -R \"rusage[mem={}]\" mpirun ./{}".format(
        n, W, log_name, mem, filename
    )
    if dry_run:
        print(command)
    else:
        os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running some experiments")
    parser.add_argument('-e', type=str, default="experiment.json", dest="experiment",
                        help="Which experiment file to run")
    args = parser.parse_args()
    experiment_file = open(args.experiment)
    experiments = json.load(experiment_file)
    experiment_file.close()
    experiment_name=experiments["name"]
    print("Running Experiment {} from file {}".format(experiment_name, args.experiment))

    for i, experiment in enumerate(experiments["jobs"]):
        success = False
        while not success:
            if no_job_running():
                success = True
                job_name = "_".join([experiment_name, str(i)])
                n = experiment["n"]
                submit_job(log_name=job_name, n=n)
                # Wait to make sure submitted job is visible
                time.sleep(1)
            else:
                # Do not retry too much
                time.sleep(60)


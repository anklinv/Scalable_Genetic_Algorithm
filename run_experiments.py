import time
import subprocess
import os
import json
import argparse
import itertools
import datetime
from shutil import copyfile

default_params = {
    "mode": "sequential",
    "-n": 1,
}


def no_job_running(dry_run=False):
    if dry_run:
        return True
    proc = subprocess.Popen(["bjobs"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout.read()
    return output == "No unfinished job found\n"


def submit_job(log_name, filename="Distributed_Genetic_Algorithm", n=1, W="00:30", mem=256, program_params="", dry_run=False):
    command = "bsub -n {} -W {} -o {}.log -R \"rusage[mem={}]\" mpirun ./{} {}".format(
        n, W, log_name, mem, filename, program_params
    )
    if dry_run:
        print(command)
    else:
        os.system(command)


def param_to_list(param):
    if param["type"] == "range":
        start = param["min"]
        end = param["max"]
        step = 1 if "stride" not in param else param["stride"]
        return list(range(start, end, step))
    elif param["type"] == "list":
        return param["list"]


def find_param(name, var_names, var_vals, fixed_params, default_params):
    # Try in variable params
    if name in var_names:
        i = var_names.index(name)
        return var_vals[i]

    # Try fixed params
    if name in fixed_params:
        return fixed_params[name]

    # Try default params
    if name in default_params:
        return default_params[name]

    print("Could not find parameter {} anywhere!".format(name))
    exit(1)


def create_filename(name):
    # Replace spaces by underscores
    file_name = name.replace(" ", "_")

    # Append data and time
    timestamp = datetime.datetime.now().strftime("%b_%d_%H%M%S")

    return file_name + "_" + timestamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running some experiments")
    parser.add_argument('--dry_run', default=False, dest="dry_run", action="store_true",
                        help="Whether to just print the jobs for debugging")
    parser.add_argument('-e', type=str, default=['experiment.json'], nargs="+", dest="experiment",
                        help="Which experiment file to run")
    args = parser.parse_args()

    print("Running {} consecutive experiment(s)".format(len(args.experiment)))
    for e, experiment in enumerate(args.experiment):
        print("*"*50)
        try:
            experiment_file = open(experiment, mode="r")
        except OSError:
            print("Failed to open file {}".format(experiment))
            continue

        experiments = json.load(experiment_file)
        experiment_file.close()
        experiment_name = experiments["name"]
        repetitions = experiments["repetitions"]

        print("Running Experiment {} from file {} with {} repetitions".format(experiment_name, experiment, repetitions))

        # Create folder
        experiment_name = create_filename(experiment_name)
        experiment_dir = os.path.join("logs", experiment_name, "")
        if args.dry_run:
            print("This will create the folder {}".format(experiment_dir))
        else:
            if os.path.isdir(experiment_dir):
                print("Folder {} already exists... exiting".format(experiment_dir))
                exit(1)
            else:
                print("Logging into folder {}".format(experiment_dir))
                os.mkdir(experiment_dir)
                copyfile(experiment, experiment_dir + experiment)

        # Finished setup
        print("*"*50)

        # Variable parameters
        grid = list()
        grid_param = list()
        for param_name, param in experiments["variable_params"].items():
            param_list = param_to_list(param)
            grid.append(param_list)
            grid_param.append(param_name)

        for job_num, job in enumerate(itertools.product(*grid)):
            n = find_param("-n", grid_param, job, experiments["fixed_params"], default_params)
            mode = find_param("mode", grid_param, job, experiments["fixed_params"], default_params)

            program_params = mode + " "
            # Use all variable params
            for i, el in enumerate(grid_param):
                # Skip already handeled parameters
                if el == "-n" or el == "mode":
                    continue
                program_params += el
                program_params += " "
                program_params += str(job[i])
                program_params += " "

            # Use all fixed params
            for name, param in experiments["fixed_params"].items():
                if name == "-n" or name == "mode":
                    continue
                program_params += name
                program_params += " "
                program_params += str(param)
                program_params += " "

            for repetition in range(repetitions):
                job_string = "_".join(job).replace(".", "").replace("-", "")
                job_string += "_" + str(repetition)

                repetition_logging_location = os.path.join(experiment_dir, job_string, "")
                if not args.dry_run:
                    if os.path.isdir(repetition_logging_location):
                        print("Folder {} already exists... exiting".format(experiment_dir))
                        exit(1)
                    else:
                        os.mkdir(repetition_logging_location)

                rep_program_params = program_params
                rep_program_params += "--log_dir"
                rep_program_params += " "
                rep_program_params += repetition_logging_location

                success = False
                while not success:
                    if no_job_running(args.dry_run):
                        success = True
                        job_name = "leonhard"
                        log_dir = os.path.join(experiment_dir, repetition_logging_location)
                        submit_job(log_name=log_dir, n=n, program_params=rep_program_params, dry_run=args.dry_run)

                        # Wait to make sure submitted job is visible
                        if not args.dry_run:
                            time.sleep(1)
                    else:
                        # Do not retry too much
                        time.sleep(15)


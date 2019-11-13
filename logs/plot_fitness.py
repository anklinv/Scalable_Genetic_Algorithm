import argparse
import csv
import matplotlib.pyplot as plt
import glob
import os
import json
import pandas as pd


def plot_fitness_from_file(filename: str):
    epoch = []
    fitness = []
    with open(filename) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            epoch.append(int(row[0]))
            fitness.append([float(f) for f in row[1:]])
    plt.plot(epoch, fitness)
    plt.show()


def extract_all_rank_values(directory_name: str):
    assert os.path.isdir(directory_name), f"{directory_name} is not a valid directory"

    relevant_files = list(filter(lambda x: "fitness" in x, os.listdir(directory_name)))

    dataframe = None
    for index, filename in enumerate(relevant_files):
        full_filename = os.path.join(directory_name, filename)
        new_csv = pd.read_csv(full_filename, names=["epoch", "fitness"])
        new_csv["rank"] = index
        if dataframe is None:
            dataframe = new_csv
        else:
            dataframe = dataframe.append(new_csv, ignore_index=True)
    return dataframe


def extract_best_rank_values(directory_name: str):
    assert os.path.isdir(directory_name), f"{directory_name} is not a valid directory"

    dataframe = extract_all_rank_values(directory_name)
    dataframe = dataframe.groupby(["epoch"], as_index=False).agg({"fitness" : "min"})
    return dataframe


def extract_all_run_values(directory_name: str):
    all_names = os.listdir(directory_name)

    # Validate JSON
    json_file = list(filter(lambda x: ".json" in x, all_names))
    if len(json_file) == 0:
        print("Could not find JSON file in directory {}".format(directory_name))
        exit(1)
    if len(json_file) > 1:
        print("Found multiple JSON files ({}) in the directory {}".format(json_file, directory_name))
        exit(1)
    json_file = json_file[0]
    with open(os.path.join(directory_name, json_file)) as file:
        json_file = json.load(file)
        repetitions = json_file["repetitions"]

    # Find unique runs
    all_names = list(filter(lambda x: os.path.isdir(os.path.join(directory_name, x)), all_names))
    unique_names = list(set(map(lambda x: "_".join(x.split("_")[:-1]), all_names)))

    dataframes = list()
    for run_name in unique_names:
        dataframe = None
        for repetition in range(repetitions):
            folder_name = run_name + "_" + str(repetition)
            best_repetition_values = extract_best_rank_values(os.path.join(directory_name, folder_name))
            best_repetition_values["run"] = repetition
            if dataframe is None:
                dataframe = best_repetition_values
            else:
                dataframe = dataframe.append(best_repetition_values, ignore_index=True)
        dataframes.append(dataframe)

    return unique_names, dataframes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot fitness per epoch')
    parser.add_argument("-f", dest="filename", default="")
    args = parser.parse_args()

    if args.filename is "":
        list_of_files = glob.glob('*.csv')
        latest_file = max(list_of_files, key=os.path.getctime)
        plot_fitness_from_file(latest_file)
    else:
        plot_fitness_from_file(args.filename)

import argparse
import csv
import matplotlib.pyplot as plt
import glob
import os


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

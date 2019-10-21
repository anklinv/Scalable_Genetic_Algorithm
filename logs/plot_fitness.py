import argparse
import csv
import matplotlib.pyplot as plt

def plot_fitness_from_file(filename: str):
    epoch = []
    fitness = []
    with open(args.filename) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            epoch.append(int(row[0]))
            fitness.append(float(row[1]))
    plt.plot(epoch, fitness)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot fitness per epoch')
    parser.add_argument('filename')
    args = parser.parse_args()

    plot_fitness_from_file(args.filename)
    
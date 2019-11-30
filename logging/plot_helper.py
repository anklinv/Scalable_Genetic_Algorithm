import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from process_log import Tags, Log, Epochs, generate_fitness_wc_dataframe


def create_barplot(df, ax, nr_bars=10, rnd=-1, ylog=False, thresholds=None):
    assert "fitness" in df, "Fitness not in dataframe"
    assert "wall clock time" in df, "Wall clock time not in dataframe"

    if "rank" in df:
        to_keep = ["epoch"]
        if "n" in df:
            to_keep.append("n")
        if "rep" in df:
            to_keep.append("rep")
        df = df.groupby(to_keep, as_index=False).agg({"fitness" : "min", "wall clock time" : "max"}).drop(columns=["epoch"])

    # Calculate start and end of thresholds
    if "n" in df and "rep" in df:
        max_fitness = df.groupby(["n", "rep"]).agg({"fitness" : "max"}).fitness.min()
        min_fitness = df.groupby(["n", "rep"]).agg({"fitness" : "min"}).fitness.max()
    elif "n" in df:
        max_fitness = df.groupby(["n"]).agg({"fitness" : "max"}).fitness.min()
        min_fitness = df.groupby(["n"]).agg({"fitness" : "min"}).fitness.max()
    else:
        max_fitness = df.fitness.max()
        min_fitness = df.fitness.min()

    # Calculate thresholds
    if thresholds is None:
        thresholds = np.round(np.linspace(min_fitness, max_fitness, num=nr_bars), rnd).astype(int)

    # Check what to group for
    if "n" in df:
        to_keep = ["rep", "n"]
    else:
        to_keep = ["rep"]

    # Generate dataframe for plotting
    plot_df = None
    for threshold in thresholds:
        tmp_df = df[df.fitness <= threshold].groupby(to_keep, as_index=False).agg({"wall clock time" : "min"})
        tmp_df["threshold"] = threshold

        if plot_df is None:
            plot_df = tmp_df
        else:
            plot_df = plot_df.append(tmp_df, ignore_index=True)

    # Generate descending order
    order = list(set(plot_df.threshold))
    order.sort()
    order = list(reversed(order))

    chart = sns.barplot(ax=ax, x="threshold", y="wall clock time", hue="n", data=plot_df, order=order, palette="autumn")
    if ylog:
        chart.set_yscale("log")
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


def create_violinplot(df, ax):
    # Preprocess dataframe if necessary
    if "communication time" in df and "computation time" in df and not "time" in df and not "type" in df:
        comp_df = df.drop(columns="communication time")
        comp_df = comp_df.rename(columns={"computation time" : "time"})
        comp_df["type"] = "computation"
        comm_df = df.drop(columns="computation time")
        comm_df = comm_df.rename(columns={"communication time" : "time"})
        comm_df["type"] = "communication"
        new_df = comp_df.append(comm_df, ignore_index=True)
    else:
        new_df = df

    sns.violinplot(x="time", y="type", palette="muted", data=new_df, ax=ax)


def create_lineplot(df, ax, resolution=50, rnd=0):
    # Aggregate rank information
    if "rank" in df:
        new_df = df.groupby(["n", "rep", "epoch"], as_index=False).agg({"fitness": "min", "wall clock time": "max"})
    else:
        new_df = df

    # Find wall clock time thresholds
    max_wct = new_df.groupby(["n", "rep"]).agg({"wall clock time": "max"})["wall clock time"].min()
    min_wct = new_df.groupby(["n", "rep"]).agg({"wall clock time": "min"})["wall clock time"].max()
    thresholds = np.round(np.linspace(min_wct, max_wct, num=resolution), rnd).astype(int)

    # Create dataframe to plot
    line_df = None
    for threshold in thresholds:
        tmp_df = new_df[new_df["wall clock time"] >= threshold].groupby(["n", "rep"]).agg({"fitness": "max"})
        tmp_df["wall clock time"] = threshold
        tmp_df = tmp_df.reset_index()

        if line_df is None:
            line_df = tmp_df
        else:
            line_df = line_df.append(tmp_df, ignore_index=True)

    # Actually plot it
    sns.lineplot(ax=ax, y="fitness", x="wall clock time", hue="n", legend="full", data=line_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot GA')
    parser.add_argument('--extract', default=False, dest="extract", action="store_true",
                        help="Whether extract the data frames from the log directory")
    parser.add_argument('--bar', default=False, dest="bar", action="store_true",
                        help="Create a bar plot instead of a line plot")
    parser.add_argument('--dir', dest="dir", required=True, help='path to directory with log files or a data frame')
    parser.add_argument('-n', dest="n", type=int, help="Select the number of islands from the data frame")
    parser.add_argument('--data', dest="data", type=str, help="Select the data from the data frame")
    parser.add_argument('--save', dest="save", type=str, help="Save the data to a file name instead of showing")
    args = parser.parse_args()

    if args.extract:
        print("Creating dataframe... ", end="", flush=True)
        name = os.path.split(args.dir)[1]
        df = generate_fitness_wc_dataframe(args.dir, name)
        print("Done!")
        print(f"Saved to {name}.gz")
    else:
        print("Loading dataframe... ", end="", flush=True)
        df = pd.read_csv(args.dir)
        print("Done!")

        # Select data
        if args.n:
            df = df[df.n == args.n]
        if args.data:
            df = df[df.data == args.data]

        fig, ax = plt.subplots()
        if args.dir.endswith("_periods.gz"):
            create_violinplot(df, ax)
        elif args.bar:
            create_barplot(df, ax, rnd=-1)
        else:
            create_lineplot(df, ax)

        # show the plot
        if args.save:
            fig.savefig(args.save, dpi=300)
        else:
            plt.show()

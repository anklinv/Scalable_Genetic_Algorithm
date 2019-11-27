import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from process_log import Tags, Log, Epochs


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
    if "n" in df:
        max_fitness = df.groupby("n").agg({"fitness" : "max"}).fitness.min()
        min_fitness = df.groupby("n").agg({"fitness" : "min"}).fitness.max()
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


if __name__ == "__main__":
    logging_directory = "../logs/island_scaling_Nov_15_003228"
    df = pd.read_csv("island_scaling_fitness_time.gz")
    df = df.drop(columns="Unnamed: 0")

    fig, ax = plt.subplots()
    create_barplot(df[df.data == "a280csv"], ax, rnd=-1)
    plt.show()

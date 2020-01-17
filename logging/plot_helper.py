import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from process_log import Tags, Log, Epochs, generate_fitness_wc_dataframe, generate_periods_dataframe
from tqdm.auto import tqdm


def create_barplot(df, ax, hue="n", nr_bars=10, rnd=-1, ylog=False, thresholds=None):
    assert "fitness" in df, "Fitness not in dataframe"
    assert "wall clock time" in df, "Wall clock time not in dataframe"
    assert hue in df, "Hue not in dataframe"

    if "rank" in df:  # do aggregation over ranks

        to_keep = ["epoch"]
        if "n" in df:
            to_keep.append("n")
        if "rep" in df:
            to_keep.append("rep")  # used for confidence interval

        if hue in df:
            to_keep.append(hue)

        df = df.groupby(to_keep, as_index=False)

        # at max wall clock time min fitness was reached for sure
        # it is at max wall clock time when the result of an epoch is known
        df = df.agg({"fitness": "min", "wall clock time": "max"})

    # Calculate start and end of thresholds
    if "n" in df and "rep" in df:
        max_fitness = df.groupby(["n", "rep"]).agg({"fitness": "max"}).fitness.min()
        min_fitness = df.groupby(["n", "rep"]).agg({"fitness": "min"}).fitness.max()
    elif "n" in df:
        max_fitness = df.groupby(["n"]).agg({"fitness": "max"}).fitness.min()
        min_fitness = df.groupby(["n"]).agg({"fitness": "min"}).fitness.max()
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
        to_keep = ["rep", hue]

    # Generate dataframe for plotting
    plot_df = None
    for threshold in thresholds:
        tmp_df = df[df.fitness <= threshold].groupby(to_keep, as_index=False).agg({"wall clock time": "min"})
        tmp_df["threshold"] = threshold

        if plot_df is None:
            plot_df = tmp_df
        else:
            plot_df = plot_df.append(tmp_df, ignore_index=True)

    # Generate descending order
    order = list(set(plot_df.threshold))
    order.sort()
    order = list(reversed(order))

    chart = sns.barplot(ax=ax, x="threshold", y="wall clock time", hue=hue, data=plot_df, order=order, palette="autumn")
    if ylog:
        chart.set_yscale("log")
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    return plot_df


def create_violinplot(df, ax):
    # Preprocess dataframe if necessary
    if "communication time" in df and "computation time" in df and not "time" in df and not "type" in df:
        comp_df = df.drop(columns="communication time")
        comp_df = comp_df.rename(columns={"computation time": "time"})
        comp_df["type"] = "computation"
        comm_df = df.drop(columns="computation time")
        comm_df = comm_df.rename(columns={"communication time": "time"})
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


def create_speedup_plot(df, ax, threshold, hue=None):
    to_keep = ["n", "rep", "epoch"]
    if hue:
        to_keep.append(hue)
    df_n = df.groupby(to_keep, as_index=False).agg({"fitness": "min", "wall clock time": "max"}).drop(columns=["epoch"])
    to_keep = ["n", "rep"]
    if hue:
        to_keep.append(hue)
    df_reached = df_n[df_n.fitness <= threshold].groupby(to_keep).agg({"wall clock time": "min"})
    tmp = df_reached.reset_index()
    baseline = tmp[tmp.n == 1]["wall clock time"].mean()
    df_speedup = 1 / df_reached.divide(baseline, 1)
    df_speedup = df_speedup.reset_index()
    if hue:
        sns.lineplot(ax=ax, x="n", y="wall clock time", hue=hue, data=df_speedup, legend="full")
    else:
        sns.lineplot(ax=ax, x="n", y="wall clock time", data=df_speedup, legend="full")
    ax.set(xlabel='n', ylabel='speedup')


'''
Given a dataframe, remove the rank column by taking the best fitness values among all ranks and the maximal wall clock time
'''
def remove_rank(df):
    if "rank" in df:
        assert "fitness" in df
        assert "wall clock time" in df
        to_keep = list(df.columns)
        to_keep.remove("rank")
        to_keep.remove("fitness")
        to_keep.remove("wall clock time")
        df = df.groupby(to_keep, as_index=False).agg({"fitness" : "min", "wall clock time" : "max"})
    return df


'''
Prints the best parallel and sequential length scores that were ever achieved
'''
def print_best_length(df):
    best_fitness = df[df['fitness'] == df['fitness'].min()]['fitness'].min()
    time_needed = np.round(df[df['fitness'] == df['fitness'].min()]['wall clock time'].max() / 1000, 1)
    needed_cores = df[df['fitness'] == df['fitness'].min()]['n'].min()
    print(f"Best ever parallel length {best_fitness} needed time {time_needed} with {needed_cores} cores")
    df_sequential = df[df['n'] == 1]
    best_fitness = df_sequential[df_sequential['fitness'] == df_sequential['fitness'].min()]['fitness'].min()
    time_needed = np.round(df_sequential[df_sequential['fitness'] == df_sequential['fitness'].min()]['wall clock time'].max() / 1000, 1)
    print(f"Best ever sequential fitness {best_fitness} needed time {time_needed} with 1 core")


'''
Calculates the wall clock thresholds that were achieved by all runs
'''
def calculate_wall_clock_time_thresholds(df, num=100):
    assert "wall clock time" in df
    assert "fitness" in df
    assert "epoch" in df
    to_keep = list(df.columns)
    to_keep.remove("wall clock time")
    to_keep.remove("fitness")
    to_keep.remove("epoch")
    max_wct = df.groupby(to_keep, as_index=False).agg({"wall clock time": "max"})["wall clock time"].min()
    min_wct = df.groupby(to_keep, as_index=False).agg({"wall clock time": "min"})["wall clock time"].max()
    thresholds = np.round(np.linspace(min_wct, max_wct, num=num), 0).astype(int)
    thresholds.sort()
    return thresholds


'''
Given a dataframe a thresholds, extract the best length values that were achieved after the specified threshold time
'''
def calculate_lengths_achieved(df, thresholds):
    assert "fitness" in df
    assert "wall clock time" in df
    line_df = None
    for threshold in tqdm(thresholds):
        to_keep = list(df.columns)
        to_keep.remove("fitness")
        to_keep.remove("wall clock time")
        if "epoch" in df:
            to_keep.remove("epoch")
        tmp_df = df[df["wall clock time"] >= threshold].groupby(to_keep).agg({"fitness": "max"})
        tmp_df["wall clock time"] = threshold
        tmp_df = tmp_df.reset_index()

        if line_df is None:
            line_df = tmp_df
        else:
            line_df = line_df.append(tmp_df, ignore_index=True)
    return line_df


'''
Generate scaling lineplots for a given dataframe and problem name.
Uses y_scaling_limits to determine the limits and legend_n to determine the legend
'''
def create_scaling_lineplot(line_df, problem, palette="plasma", limits=True, ylow=None, yhigh=None, manual_legend=True, legend_n=None, y_scaling_limits=None, hue="n", legend_col=1, figsize=(8,6)):
    assert not limits or ((ylow is not None and yhigh is not None) or y_scaling_limits is not None)
    assert not manual_legend or (legend_n is not None)

    plot_df = line_df.copy()

    # Turn wall clock time into seconds
    plot_df["wall clock time"] = plot_df["wall clock time"] / 1000

    # Fix the color map linear scaling
    plot_df[hue] = np.log2(plot_df[hue])

    for mode in line_df["mode"].unique():
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(ax=ax, y="fitness", x="wall clock time", hue=hue, data=plot_df[plot_df["mode"] == mode], palette=palette)
        #ax.set_title(f"scaling of the {mode} model")

        # Figure out limits
        if limits:
            if ylow is not None and yhigh is not None:
                ax.set_ylim(ylow, yhigh)
            else:
                low, high = y_scaling_limits[problem]
                ax.set_ylim(low, high)
            zoom_level = "zoom"
        else:
            # No scaling
            zoom_level = "full"

        # Figure out legend
        if manual_legend:
            all_n = list(line_df[hue].unique())
            all_n.sort()
            index = [all_n.index(n) for n in legend_n]
            from matplotlib.lines import Line2D
            if isinstance(palette, str):
                custom_lines = [Line2D([0], [0], color=plt.get_cmap(palette)(c / (len(all_n) - 1)), lw=2) for c in index]
            else:
                custom_lines = [Line2D([0], [0], color=palette[i], lw=2) for i in index]
            ax.legend(handles=custom_lines, title=hue, loc="upper right", labels=list(map(str, legend_n)), ncol=legend_col)

        ax.get_yaxis().set_label_text("length")
        ax.get_xaxis().set_label_text("wall clock time [s]")
        fig.savefig(f"scaling_{mode}_{problem}_{zoom_level}.pdf", pad_inches=0, bbox_inches="tight")


'''
Generate compact scaling lineplots for a given dataframe and problem name.
Uses y_scaling_limits to determine the limits and legend_n to determine the legend
'''
def create_compact_scaling_lineplot(line_df, problem, palette="plasma", limits=True, ylow=None, yhigh=None, manual_legend=True, legend_n=None, y_scaling_limits=None, hue="n", legend_col=1, xhigh=100):
    assert not limits or ((ylow is not None and yhigh is not None) or y_scaling_limits is not None)
    assert not manual_legend or (legend_n is not None)

    plot_df = line_df.copy()

    # Turn wall clock time into seconds
    plot_df["wall clock time"] = plot_df["wall clock time"] / 1000

    # Fix the color map linear scaling
    plot_df[hue] = np.log2(plot_df[hue])

    low, high = y_scaling_limits[problem]

    ax = sns.FacetGrid(plot_df, col="mode", hue=hue, palette=palette, legend_out=True, aspect=1.5, height=5,
                       ylim=(low, high), xlim=(0, xhigh), despine=False)
    plt.rc_context({'lines.linewidth': 1})
    ax = ax.map(sns.lineplot, "wall clock time", "fitness")

    if manual_legend:
        all_n = list(line_df[hue].unique())
        all_n.sort()
        index = [all_n.index(n) for n in legend_n]
        from matplotlib.lines import Line2D
        if isinstance(palette, str):
            custom_lines = [Line2D([0], [0], color=plt.get_cmap(palette)(c / (len(all_n) - 1)), lw=1) for c in index]
        else:
            custom_lines = [Line2D([0], [0], color=palette[i], lw=1) for i in index]
        ax.add_legend(handles=custom_lines, title=hue, labels=list(map(str, legend_n)), ncol=legend_col, bbox_to_anchor=(1.04, 0.95))

    zoom_level = "zoom" if limits else "full"

    ax.set_ylabels("length")
    ax.set_xlabels("wall clock time [s]")
    ax.set_titles(row_template = '{row_name}', col_template = '{col_name}')
    plt.savefig(f"scaling_compact_{problem}_{zoom_level}.svg", pad_inches=0, bbox_inches="tight")



'''
Create speedup plot from dataframe and problem name
Can specify thresholds directly, otherwise scaling_thresholds is used
'''
def create_scaling_speedup_plot(df, problem, xticks_n, scaling_thresholds=None, thresholds=None, baseline_df=None, static_ylim=True, y_scaling_limits_speedup=None, yticks_speedup=None, ymin=0):
    assert (thresholds is not None) or (scaling_thresholds is not None)
    assert not static_ylim or (y_scaling_limits_speedup is not None and yticks_speedup is not None)

    # Figure out thresholds
    if thresholds is None:
        threshold_list = scaling_thresholds[problem]
    else:
        threshold_list = thresholds

    for threshold in tqdm(threshold_list):
        to_keep = ["rep", "mode", "n", "epoch"]
        df_n = df.groupby(to_keep, as_index=False).agg({"fitness": "min", "wall clock time": "max"}).drop(columns=["epoch"])
        to_keep = ["rep", "mode", "n"]
        df_reached = df_n[df_n.fitness <= threshold].groupby(to_keep).agg({"wall clock time": "min"})
        tmp = df_reached.reset_index()

        # Figure out baseline
        if baseline_df is not None:
            baseline_n_rep = baseline_df[baseline_df["fitness"] <= threshold].groupby(["rep", "population"]).agg({"wall clock time" : "min"})
            baseline = baseline_n_rep.groupby(["population"]).agg({"wall clock time" : "median"})["wall clock time"].min()
        else :
            baseline = tmp[tmp["n"] == 1]["wall clock time"].mean()

        df_speedup = 1 / df_reached.divide(baseline, 1)
        df_speedup = df_speedup.reset_index()
        max_n = int(df_speedup.groupby(["mode"]).agg({"n" : "max"}).min())
        df_speedup = df_speedup[df_speedup["n"] <= max_n]

        if static_ylim:
            ylim = y_scaling_limits_speedup[problem]
            xlim = 65
        else:
            ylim = int(np.ceil(df_speedup["wall clock time"].max()))
            xlim = int(df_speedup["n"].max()) + 1

        # Plot the speedup
        fig, ax = plt.subplots(figsize=(8, 4))
        legend = ["island", "naive"]
        sns.lineplot(ax=ax, x="n", y="wall clock time", hue="mode", hue_order=["island", "naive"], data=df_speedup,
                     legend="full", err_style="bars", err_kws={"ecolor" : "k", "fmt" : "o"}, estimator=np.median, ci=95)
        plt.plot([1,ylim],[1, ylim], "--", color='#888888')
        ax.set(xlabel='n', ylabel='speedup')
        ax.xaxis.set_major_locator(plt.FixedLocator(xticks_n))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        if static_ylim:
            ax.yaxis.set_major_locator(plt.FixedLocator(yticks_speedup[problem]))
            #ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.set_xlim(0, xlim)
        ax.set_ylim(ymin, ylim)
        ax.set_autoscale_on(False)
        plt.legend(legend)
        # ax.set_title(f"speedup island model vs naive model (threshold {threshold})")
        fig.savefig(f"speedup_{problem}_{threshold}.pdf", pad_inches=0, bbox_inches="tight")


'''
Create a population scaling barplot from dataframe with predefined wall clock time thresholds
'''
def create_population_barplot(line_df, problem, thresholds, palette="plasma", figsize=(8,6)):

    threshold_list = [thresholds[i] for i in [len(thresholds) // 50, len(thresholds) // 10, len(thresholds) // 2, -1]]

    for threshold in tqdm(threshold_list):
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(ax=ax, y="fitness", x="population", data=line_df[line_df["wall clock time"] == threshold], palette=palette)

        # Turn thresholds into seconds
        threshold_name = int(np.round(threshold / 1000, 0))

        # Make sure axis labels fit
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

        ax.set_title(f"{problem} population comparison after {threshold_name}s")
        ax.set_ylabel("length")
        fig.savefig(f"population_scaling_{problem}_{threshold_name}s.pdf", pad_inches=0, bbox_inches="tight")


'''
Create a plot of the communication percentage for a periods dataframe
'''
def create_communication_boxplot(df, problem, subset_n=None):
    assert "period" in df
    if subset_n is not None:
        df = df[df["n"].isin(subset_n)]
    to_keep = list(df.columns)
    to_keep.remove("time")
    to_keep.remove("type")
    summed_df = df.groupby(to_keep, as_index=False).agg({"time" : "sum"})
    summed_df["type"] = "sum"
    df_bar = df.copy()
    df_bar = df_bar.append(summed_df, ignore_index=True)
    df_sum = df_bar[df_bar["type"] == "sum"].rename(columns={"time" : "sum"}).drop(columns=["type"])
    df_comm = df_bar[df_bar["type"] == "communication"].rename(columns={"time" : "communication"}).drop(columns=["type"])
    df_comp = df_bar[df_bar["type"] == "computation"].rename(columns={"time" : "computation"}).drop(columns=["type"])
    df_mix = pd.merge(df_sum, df_comm,  how='left', on=["rep", "rank", "n", "mode", "elite_size", "epochs", "log_freq", "migration_amount", "period", "population"])
    df_mixed = pd.merge(df_mix, df_comp, how="left", on=["rep", "rank", "n", "mode", "elite_size", "epochs", "log_freq", "migration_amount", "period", "population"])
    df_mixed["communication"] = (df_mixed["communication"] / df_mixed["sum"]) * 100
    df_mixed["computation"] = (df_mixed["computation"] / df_mixed["sum"]) * 100
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(x="n", y="communication", data=df_mixed[df_mixed["period"] != 0], notch=True)
    #ax.set_title(f"Communication Percentages for {problem}")
    ax.set_ylabel("communication fraction [%]")
    fig.savefig(f"communication_boxplot_{problem}.pdf", pad_inches=0, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot GA')
    parser.add_argument('--extract', default=False, dest="extract", action="store_true",
                        help="Whether extract the data frames from the log directory")
    parser.add_argument('--periods', default=False, dest="periods", action="store_true",
                        help="Whether to extract the periods instead")
    parser.add_argument('--bar', default=False, dest="bar", action="store_true",
                        help="Create a bar plot instead of a line plot")
    parser.add_argument('--speedup', dest="speedup", type=str, default="",
                        help="Create a speedup plot from the data with given hue parameter")
    parser.add_argument('--threshold', dest="threshold", type=int, default=0, help='Threshold to use for speedup plot')
    parser.add_argument('--dir', dest="dir", required=True, help='path to directory with log files or a data frame')
    parser.add_argument('--dark', default=False, dest="dark", action="store_true",
                        help="Take all pngs in the directory and turn them into a dark mode version")
    parser.add_argument('-n', dest="n", type=int, help="Select the number of islands from the data frame")
    parser.add_argument('--data', dest="data", type=str, help="Select the data from the data frame")
    parser.add_argument('--save', dest="save", type=str, help="Save the data to a file name instead of showing")
    args = parser.parse_args()

    if args.dark:
        from PIL import Image
        import PIL.ImageOps
        from tqdm import tqdm

        for image_file in tqdm(os.listdir(args.dir)):
            if ".png" in image_file:
                name, _ = image_file.split(".")

                # Skip already processed
                if name.endswith("_inv1") or name.endswith("_inv2"):
                    continue

                image = Image.open(os.path.join(args.dir, image_file))
                if image.mode == 'RGBA':
                    r, g, b, a = image.split()
                    rgb_image = Image.merge('RGB', (r, g, b))

                    inverted_image = PIL.ImageOps.invert(rgb_image)

                    r2, g2, b2 = inverted_image.split()

                    final_transparent_image = Image.merge('RGBA', (r2, g2, b2, a))
                    final_transparent_image.save(os.path.join(args.dir, f'{name}_inv1.png'))
                    final_transparent_image = Image.merge('RGBA', (b2, g2, r2, a))
                    final_transparent_image.save(os.path.join(args.dir, f'{name}_inv2.png'))
                else:
                    inverted_image = PIL.ImageOps.invert(image)
                    inverted_image.save(os.path.join(args.dir, f'{name}_inv1.png'))
                    r, g, b = inverted_image.split()
                    inverted_image = Image.merge('RGB', (r, g, b))
                    inverted_image.save(os.path.join(args.dir, f'{name}_inv2.png'))

    elif args.extract:
        if args.periods:
            print("Creating periods dataframe... ", end="", flush=True)
            name = os.path.split(args.dir.strip("/").strip("\\"))[-1]
            df = generate_periods_dataframe(args.dir, name)
            print("Done!")
            print(f"Saved to {name}.gz")
        else:
            print("Creating dataframe... ", end="", flush=True)
            name = os.path.split(args.dir.strip("/").strip("\\"))[-1]
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
        elif args.speedup != "":
            if args.threshold >= 0:
                threshold = args.threshold
            else:
                # Extract suiting threshold
                df_n = df.groupby(["n", "rep", "epoch", "migration_period"], as_index=False).agg(
                    {"fitness": "min", "wall clock time": "max"}).drop(columns=["epoch"])
                max_fitness = df_n.groupby("n").agg({"fitness": "max"}).fitness.min()
                min_fitness = df_n.groupby("n").agg({"fitness": "min"}).fitness.max()
                thresholds = np.round(np.linspace(min_fitness, max_fitness, num=10), -1).astype(int)
                threshold = thresholds[-2]

            # Create plot
            create_speedup_plot(df, ax, threshold=threshold, hue=args.speedup)
        else:
            create_lineplot(df, ax)

        # show the plot
        if args.save:
            fig.savefig(args.save, dpi=300)
        else:
            plt.show()

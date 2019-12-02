# How to use plotting scripts

Once the runs are completed, you should have a log folder with the structure as specified [below](https://github.com/anklinv/Scalable_Genetic_Algorithm/tree/master/logging#logging-folder-structure). There are two steps that are now required to create plots. First you need to parse the `.bin` files and create a pandas data frame. Second you need to create the plots directly from these pandas data frames. You can directly use both by running the `plot_helper.py` script:

#### Extract data frame from log directory

```
python plot_helper.py --dir <LOG-DIR> --extract
```

Note: this might take a while...

#### Create plots

```
python plot_helper.py --dir <DATA-FRAME.gz>
```

To create a line plot from the given stored data frame. You can also use the arguments `--bar` to create a line plot instead or `--speedup` to create a speedup plot.

If you want to use the functionality in your own scripts, then you can look at the explainations below to help you use these functions:

## Creating Pandas Data Frames

The `process_log.py` file contains the classes `Tags`, `Log` and `Epochs` that help to parse the logs. For convenience there are two functions that help to parse an entire log directory. These functions are explained below and can automatically be used by calling the `plot_helper.py` script explained below.

#### `generate_fitness_wc_dataframe(log_dir, name, tag_loc)`

This function takes a log directory and generates a suitable data frame. It directly extracts all the variable parameters of the experiment based on the JSON [specification](https://github.com/anklinv/Scalable_Genetic_Algorithm#json-specification-for-experiments) of the experiment. You need to specify the `log_dir`, a relative path to the directory where the log files are. As it generates the dataframe and saves that to disk (in compressed format), you need to specify a file name for the saved dataframe in `name`. Optionally you can also optionally specify the path to the `tags.hpp` file in `tag_log`.

#### `generate_periods_dataframe(log_dir, name, tag_loc)`

The arguments are the same as for the `generate_fitness_wc_dataframe` function above. Its purpose is to extract a data frame from the log directory that contains the computation time and communication time for each migration period. It does so by looking at the `LOGGING_TAG_WC_EPOCH_BEGIN` and `LOGGING_TAG_WC_EPOCH_END` tags in steps of the `migration_period`. This parameter is also extracted from the JSON [specification](https://github.com/anklinv/Scalable_Genetic_Algorithm#json-specification-for-experiments). Note: this assumes that it is logged every epoch, so if you log less you need to divide the `migration_period` by your logging frequency.

## Creating Plots

The `plot_helper.py` file there are four functions that create different plots:

### Common arguments

All of the functions take the argument `df`. It refers to the pandas data frame that is required to create the plots. The columns it required are specified in each function seperately.

All of the functions take the argument `ax`. It refers to the `axes.Axes` from `matplotlib` where the plot should be placed. Basically you can just call `fig, ax = plt.subplots()` before the call for the plot and give `ax` as the argument for the plotting function.

#### `create_barplot(df, ax, nr_bars, rnd, ylog, thresholds)`

This function is used to create plots of different fitness thresholds on the x-axis and the wall clock time needed to reach them on the y-axis. You need to specify how many bars are placed next to each other on the x-axis in `nr_bars`. The thresholds are extracted automatically by taking the minimum and maximum values recorded among all ranks and runs and then taking `nr_bars` many linearly spaced thresholds rounded to the next `rnd` digit (a negative value means a digit before the decimal point, e.g. use -1 to round to the nearest multiple of 10). These thresholds can also be chosen manually by passing a list of integers in the `thresholds` parameter. The plot is a lin-lin plot, unless `ylog` is set to true, in which case it is a lin-log plot.

The data frame needs the following columns to work properly:
`fitness` is the fitness value achieved at time point `wall clock time`.
If you add the `n` column, it automatically creates several parameters next to each other to compare.
If you add the `rep` column, confidence intervals will automatically be computed and added to the graph.
If you add the `rank` column, it will automatically select the minimum fitness per epoch among all ranks and the maximum wall clock time per epoch.

#### `create_violinplot(df, ax)`

This function plots the communication time vs the computation time of a run. To use it, you must create the data frame first using the `generate_periods_dataframe()` function which is currently not possible directly from the command line. 

The data frame needs to contain the columns `time` and `type` or alternatively the columns `communication time` and `computation time` with the used times. From all other columns should be selected first.

#### `create_lineplot(df, ax, resolution, rnd)`

This function creates a line plot of best fitness values achieved after a certain wall clock time. The number of intervals it takes into account can be controlled using the `resolution` parameter and can be rounded to the nearest number using the `rnd` parameter (a negative value means a digit before the decimal point, e.g. use -1 to round to the nearest multiple of 10).

The data frame needs to have the columns `fitness`, `wall clock time` and `n`.

#### `create_speedup_plot(df, ax, threshold, hue)`

This function creates a speedup plot of a given data frame using a fixed fitness threshold `threshold`. Which column it uses for the hue can also be controlled using the `hue` parameter. If the `rep` column is specified it automatically computes the 95% confidence intervals.

The data frame needs to have the columns `fitness`, `wall clock time` and optionally `rep` for confidence interval and some other column for the different hues.

# Logging folder structure

The folder structure and file naming scheme is somewhat arbitrary, but needs to be consistent for all the scripts to work. The whole structure depends on the declared experiment JSON file. The specification for that can be found [here](https://github.com/anklinv/Scalable_Genetic_Algorithm#json-specification-for-experiments).

The important part is that an experiments specifies a set `K` variable parameters that vary in the different runs. It also specifies the number of repetitions `P` that are run. Each chosen configuration of the `K` parameters and for each repetiton a different folder inside the top-level log folder is created as follows:

```
<p1>_<p2>_ ... _<pK>_1
<p1>_<p2>_ ... _<pK>_2
...
<p1>_<p2>_ ... _<pK>_P
```

Note: The order of the parameters p1 to pK is the same as specified in the JSON file and the scripts use the JSON file to infer their meaning. The JSON file itself needs to follow the [specification](https://github.com/anklinv/Scalable_Genetic_Algorithm#json-specification-for-experiments) and be placed directly in the top-level log folder.

Each of these runs contains the files: `leonhard.log` is the output that the Leonhard system produces. It is not used by the scripts, but it can be used for debugging. The other files are as follows:

```
<date>_<time>_0000_tags.bin
<date>_<time>_0001_tags.bin
...
<date>_<time>_<maxRank>_tags.bin
```

Note: The `date` and `time` strings should not include a `_`, but they are otherwise ignored. The rank numbers need to be a 4-digit number. In case there is just one rank (e.g. just a single log file for the GPU), make sure to name it `<data>_<time>_0000_tags.bin` and set the parameter `n` to `1` in the JSON file or do not specify it at all.

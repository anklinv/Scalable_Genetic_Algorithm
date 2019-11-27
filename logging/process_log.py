import argparse
from collections import namedtuple
import glob
import numpy as np
import os
import re
import struct
import json
import pandas as pd
from collections import OrderedDict


class Tags(object):
    """
    Represents tag name to id mapping from a header file.
    Preprocessor macros are held in the structure:
        tag_names: Dict<str, int>
    We also reverse index for faster subsequent parsing:
        tag_ids: Dict<int, str>
    """
    def __init__(self, fn: str):
        define = re.compile(r'^\s*#define\s+LOGGING_TAG_(\w+)\s+\(?(\w*?)\)?\s*$')
        self.tag_names = dict()
        with open(fn, 'r') as f:
            for line in f:
                match = define.match(line)
                if match is None:
                    continue
                try:
                    name, value = match.groups()
                    int_value = int(value, 0)
                    self.tag_names[name.lower()] = int_value
                except:
                    raise SyntaxError(line)
        self.tag_ids = {v: k for k, v in self.tag_names.items()}

    def __str__(self):
        return f'{{tag_names: {self.tag_names}, tag_ids: {self.tag_ids}}}'


class Log(object):
    """
    Represents a single log file.
        log: List<Tuple<int, int>>
    Also creates a human-readable version with tag names instead of IDs.
    This could hit performance issues for large log files.
    Preferably use IDs instead.
        log_h: List<Tuple<string, int>>
    """
    def __init__(self, fn: str, tags: Tags):
        self.fn = fn
        self.tags = tags

        print(f'Parsing {self.fn}...', end='')
        self.parse_log()
        print(' done')
        # print('log', self.log)

        self.make_human_readable()
        self.check_version()
        self.extract_constants()

    def make_human_readable(self):
        ''' Substitute tags for tag IDs to create a human-readable log '''
        assert all([tag_id in self.tags.tag_ids for (tag_id, value) in self.log])
        self.log_h = [(self.tags.tag_ids[tag_id], value) for (tag_id, value) in self.log]
        # for tag, value in self.log_h:
        #     print(f'{tag}: {value}')

    def check_version(self):
        """
        Assert logged version matches version from header file
        """
        try:
            self.version = next(value for (tag, value) in self.log_h if tag == 'version')
        except StopIteration:
            raise ValueError('no version tag found')
        if self.version is not self.tags.tag_names['version']:
            raise ValueError(f"expected version {self.tags.tag_names['version']}, got {self.version}")

    def extract_constants(self):
        self.clocks_per_sec = next(value for (tag, value) in self.log_h if tag == 'clocks_per_sec')

    def parse_log(self):
        with open(self.fn, 'rb') as f:
            d = f.read()
        self.log = list()
        offset = 0
        while (offset < len(d)):
            fmt = 'II'
            tag, value = struct.unpack_from(fmt, d, offset)
            offset += struct.calcsize(fmt)
            self.log.append((tag, value))

    def get_values(self, name: str):
        full_name = f'val_{name}'
        val_tag_id = self.tags.tag_names[full_name]
        values = [value for (tag_id, value) in self.log if tag_id == val_tag_id]
        return values

    def get_wall_clock_durations(self, name: str):
        ''' Automatically appends the wc_ prefix and _begin or _end suffixes'''
        begin_full_name = f'wc_{name}_begin'
        end_full_name = f'wc_{name}_end'

        begin_tag_id = self.tags.tag_names[begin_full_name]
        end_tag_id = self.tags.tag_names[end_full_name]

        begins = [value for (tag_id, value) in self.log if tag_id == begin_tag_id]
        ends = [value for (tag_id, value) in self.log if tag_id == end_tag_id]

        assert len(begins) == len(ends)
        # Some durations may wrap around, but check whether all fit in half-range
        diffs = [(ends[i] - begins[i]) % (1 << 32) for i in range(len(begins))]
        assert all([d < (1 << 31) for d in diffs])

        durations = [float(d) / 1e6 * 1e3 for d in diffs]
        return durations
        
    def get_cpu_clock_durations(self, name: str):
        ''' Automatically appends the cc_ prefix and _begin or _end suffixes'''
        begin_full_name = f'cc_{name}_begin'
        end_full_name = f'cc_{name}_end'

        begin_tag_id = self.tags.tag_names[begin_full_name]
        end_tag_id = self.tags.tag_names[end_full_name]

        begins = [value for (tag_id, value) in self.log if tag_id == begin_tag_id]
        ends = [value for (tag_id, value) in self.log if tag_id == end_tag_id]

        assert len(begins) == len(ends)
        # Some durations may wrap around, but check whether all fit in half-range
        diffs = [(ends[i] - begins[i]) % (1 << 32) for i in range(len(begins))]
        assert all([d < (1 << 31) for d in diffs])

        durations = [float(d) / self.clocks_per_sec * 1e3 for d in diffs]
        return durations


class Epochs(object):
    """
    Represents epochs across one or more logs.
    """
    def __init__(self, log: Log, tags: Tags):
        self.log = log
        self.tags = tags
        self.process_epochs_in_log()

    # class Epoch(object):
    #     self.begin = None
    #     self.end = None
    #     self.best_fitness = None

    def process_epochs_in_log(self) -> None:
        Epoch = namedtuple('Epoch', ['begin', 'end', 'fitness'])

        # get relevant tag IDs for fast lookups
        begin_tag_id = self.tags.tag_names['wc_epoch_begin']
        end_tag_id = self.tags.tag_names['wc_epoch_end']
        fitness_tag_id = self.tags.tag_names['val_best_fitness']
        tag_ids = (begin_tag_id, end_tag_id, fitness_tag_id)

        it = iter(self.log.log)
        self.epochs: [Epoch] = []
        try:
            while True:
                begin = next(value for (tag_id, value) in it if tag_id == begin_tag_id)
                fitness = next(value for (tag_id, value) in it if tag_id == fitness_tag_id)
                end = next(value for (tag_id, value) in it if tag_id == end_tag_id)
                self.epochs.append(Epoch(
                    begin=float(begin)/1e3,
                    end=float(end)/1e3,
                    fitness=fitness
                ))
        except StopIteration:
            pass
    
    def get_fitness_vs_time(self) -> ([int], [float]):
        return (
            [e.fitness for e in self.epochs],
            [e.end for e in self.epochs]
        )

    def get_fitness_vs_time_dataframe(self):
        return ((e.fitness, e.end, epoch) for epoch, e in enumerate(self.epochs))


def last_log():
    log_fns = glob.glob(os.path.join('logs', '*_tags.bin'))
    if len(log_fns) == 0:
        raise FileNotFoundError('No log files found')
    #print(log_fns)
    last_log_fn = max(log_fns)
    #print(last_log_fn)
    return last_log_fn


def logs_in_dir(path):
    log_fns = glob.glob(os.path.join(path, '*_tags.bin'))
    return log_fns


# Given a log_dir (generated with a leonhard run) and a name, saves a dataframe to name.gz
# That dataframe contains the epochs, wall clock times, fitness, rep, rank and all the variable parameters
def generate_dataframe(log_dir, name, tag_loc="tags.hpp"):
    assert isinstance(name, str), "name must be a string"
    assert os.path.isdir(log_dir), "log_dir must be a directory"
    assert os.path.isfile(tag_loc), f"Could not find tag_loc {tag_loc}"

    # Extract tags
    tags = Tags("tags.hpp")

    all_names = os.listdir(log_dir)

    # Validate JSON
    json_file = list(filter(lambda x: ".json" in x, all_names))
    assert len(json_file) > 0, "Found no JSON file in the directory"
    assert len(json_file) <= 1, "Found multiple JSON files in the directory"
    json_file = json_file[0]

    # Extract repetitions
    with open(os.path.join(log_dir, json_file)) as file:
        json_file = json.load(file, object_pairs_hook=OrderedDict)
        repetitions = json_file["repetitions"]

    # Find unique runs (without repetitions)
    all_names = list(filter(lambda x: os.path.isdir(os.path.join(log_dir, x)), all_names))
    unique_names = list(set(map(lambda x: "_".join(x.split("_")[:-1]), all_names)))

    df = None
    for run_name in unique_names:
        params = run_name.split("_")
        param_names = list(map(lambda x: x.replace("-", ""), json_file["variable_params"].keys()))
        for repetition in range(repetitions):
            folder_name = run_name + "_" + str(repetition)
            folder_contents = os.listdir(os.path.join(log_dir, folder_name))
            folder_contents = list(filter(lambda x: ".bin" in x, folder_contents))
            for filename in folder_contents:
                log = Log(os.path.join(log_dir, folder_name, filename), tags)
                rank = int(filename.split("_")[-2])
                epochs = Epochs(log, tags)
                if df is None:
                    df = pd.DataFrame(epochs.get_fitness_vs_time_dataframe(), columns=["fitness", "wall clock time", "epoch"])
                    df["rank"] = rank
                    df["rep"] = repetition
                    for param, param_name in zip(params, param_names):
                        df[param_name] = param
                else:
                    df2 = pd.DataFrame(epochs.get_fitness_vs_time_dataframe(), columns=["fitness", "wall clock time", "epoch"])
                    df2["rank"] = rank
                    df2["rep"] = repetition
                    for param, param_name in zip(params, param_names):
                        df2[param_name] = param
                    df = df.append(df2, ignore_index=True)

    # Figure out correct file ending
    if name.endswith(".gz"):
        file_name = name
    else:
        file_name = name + ".gz"

    # Save to disk
    df.to_csv(file_name, compression="gzip")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process GA logs')
    parser.add_argument('--tags', help='path to C header file with log defines')
    parser.add_argument('--log', help='path to log file')
    parser.add_argument('--dir', help='path to directory with log files')
    args = parser.parse_args()

    # Parse header file with tag definitions
    tags_fn = os.path.join('logging', 'tags.hpp') if not args.tags else args.tags
    tags = Tags(tags_fn)
    # print(tags)

    # parse binary log
    log_fns = (
        [args.log] if args.log else
        logs_in_dir(args.dir) if args.dir else
        [last_log()]
    )
    logs = [Log(log_fn, tags) for log_fn in log_fns]

    epochss = [Epochs(log, tags) for log in logs]
    from matplotlib import pyplot as plt
    for epochs in epochss:
        fitness, time = epochs.get_fitness_vs_time()
        plt.plot([t/1e3 for t in time], fitness)
    plt.xlabel('time [s]')
    plt.ylabel('distance')
    plt.show()
    
    for log in logs:
        print(log.fn)
        print('    total:',
            f"wall clock {log.get_wall_clock_durations('logging')[0]:.3f}ms",
            f"CPU clock {log.get_cpu_clock_durations('logging')[0]:.3f}ms"
        )
        def print_stats(name):
            ri_arr = log.get_wall_clock_durations(name)
            print(f'    {name}:',
                f"length {len(ri_arr)},",
                f"mean {np.mean(ri_arr):.3f}ms,",
                f"std {np.std(ri_arr):.3f}ms"
        )
        print_stats('epoch')
        print_stats('rank_individuals')
        print_stats('breed_population')
        print_stats('mutate_population')

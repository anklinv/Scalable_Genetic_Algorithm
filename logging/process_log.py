import argparse
import glob
import numpy as np
import os
import re
import struct

class Tags(object):
    '''
    Represents tag name to id mapping from a header file.
    Preprocessor macros are held in the structure:
        tag_names: Dict<str, int>
    We also reverse index for faster subsequent parsing:
        tag_ids: Dict<int, str>
    '''
    def __init__(self, fn):
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
    '''
    Represents a single log file.
        log: List<Tuple<int, int>>
    Also creates a human-readable version with tag names instead of IDs.
    This could hit performance issues for large log files.
    Preferably use IDs instead.
        log_h: List<Tuple<string, int>>
    '''
    def __init__(self, fn, tags):
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
        ''' Assert logged version matches version from header file '''
        try:
            self.version = next(value for (tag, value) in self.log_h if tag == 'version')
            #print('version', self.version)
        except StopIteration:
            raise ValueError('no version tag found')
        if self.version is not self.tags.tag_names['version']:
            raise ValueError(f"expected version {self.tags.tag_names['version']}, got {version}")

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

    def get_wall_clock_durations(self, name):
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
        
    def get_cpu_clock_durations(self, name):
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
        
def last_log():
    log_fns = glob.glob(os.path.join('logs', '*_tags.bin'))
    #print(log_fns)
    last_log_fn = max(log_fns)
    #print(last_log_fn)
    return last_log_fn

def logs_in_dir(path):
    log_fns = glob.glob(os.path.join(path, '*_tags.bin'))
    return log_fns

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
    
    for log in logs:
        print('total:',
            f"wall clock {log.get_wall_clock_durations('logging')[0]:.3f}ms",
            f"CPU clock {log.get_cpu_clock_durations('logging')[0]:.3f}ms"
        )
        ri_arr = log.get_wall_clock_durations('rank_individuals')
        print('rank_individuals:',
            f"length {len(ri_arr)},",
            f"mean {np.mean(ri_arr):.3f}ms,",
            f"std {np.std(ri_arr):.3f}ms"
        )

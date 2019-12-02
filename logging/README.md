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

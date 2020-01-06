import json
import math

repetitions = 10

base_n = 4
populations = [2048, 4096, 8192, 16384]

elite_size = lambda x: x // 2
log_freq = lambda x: max(1, x // 1000)

base_times = {
    "u2319": 78.710
}

# in minutes
target_runtimes = {
    "u2319": 3
}

base_population = 64
base_epochs = 100000

total_runtime = 0

for data in base_times.keys():
    experiment = dict()
    experiment["name"] = f"population test {data}"
    experiment["repetitions"] = repetitions
    fixed_params = dict()
    fixed_params["--data"] = data + ".csv"
    fixed_params["--mutation"] = 10
    fixed_params["--migration_topology"] = "fully_connected"
    fixed_params["--selection_policy"] = "truncation"
    fixed_params["--replacement_policy"] = "truncation"
    fixed_params["mode"] = "sequential"
    fixed_params["-n"] = 1
    experiment["fixed_params"] = fixed_params
    variable_params = dict()
    pop = dict()
    pop["type"] = "tuple"
    pop["names"] = ["--population", "--elite_size", "--epochs", "--log_freq"]
    values = list()
    estimated_runtime = target_runtimes[data]
    for population in populations:
        multiplier = base_population / population
        epochs = int(round((estimated_runtime * 60 / base_times[data]) * base_epochs * multiplier, 0))
        val = [
            population,
            elite_size(population),
            epochs,
            log_freq(epochs)
        ]
        value = dict()
        value["value"] = val
        values.append(value)
    pop["values"] = values
    variable_params["pop"] = pop
    experiment["variable_params"] = variable_params
    with open(f"population_test__remaining_{data}.json", mode="w") as file:
        json.dump(experiment, file, indent=2)

    data_runtime = estimated_runtime * len(populations) * repetitions / 60
    print(f"Estimated run time for {data}: {data_runtime} h per problem")
    total_runtime += data_runtime

print(f"Estimated total time: {total_runtime} h per problem")

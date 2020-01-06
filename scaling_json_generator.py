import json
import math

repetitions = 30

# Result of grid search
best_population = {
    "u2319": 32,
    "d1291": 64,
    "a280": 2048
}
base_n = 4
nr_islands = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
population_size = lambda islands, data: max(islands, int(best_population[data] / (islands / base_n)))

migration_period = 25
migration_amount = lambda x: x // 8
elite_size = lambda x: x // 2
log_freq = lambda x: max(1, x // 1000)

base_times = {
    "u2319": 78.710,
    "d1291": 16.946,
    "a280": 4.491
}

# in minutes
target_runtimes = {
    "a280": 1,
    "d1291": 2,
    "u2319": 4
}

base_population = 64
base_epochs = 100000

total_runtime = 0
for data in base_times.keys():
    experiment = dict()
    experiment["name"] = f"scaling test {data}"
    experiment["repetitions"] = repetitions
    fixed_params = dict()
    fixed_params["--data"] = data + ".csv"
    fixed_params["--mutation"] = 10
    fixed_params["--migration_topology"] = "fully_connected"
    fixed_params["--selection_policy"] = "truncation"
    fixed_params["--replacement_policy"] = "truncation"
    fixed_params["--migration_period"] = migration_period
    experiment["fixed_params"] = fixed_params
    variable_params = dict()
    pop = dict()
    pop["type"] = "tuple"
    pop["names"] = ["--population", "--elite_size", "-n", "--epochs", "--log_freq", "--migration_amount"]
    values = list()
    job_count = 0
    estimated_runtime = target_runtimes[data]
    for n in nr_islands:
        population = population_size(n, data)
        job_count += 1
        value = dict()
        multiplier = base_population / population
        epochs = int(round((estimated_runtime * 60 / base_times[data]) * base_epochs * multiplier, 0))
        val = [
            population,
            elite_size(population),
            n,
            epochs,
            log_freq(epochs),
            migration_amount(population)
        ]
        value["value"] = val
        values.append(value)
    pop["values"] = values
    variable_params["pop"] = pop
    variable_params["mode"] = {
        "type": "list",
        "list": ["island", "naive"]
    }
    experiment["variable_params"] = variable_params
    with open(f"scaling_test_{data}.json", mode="w") as file:
        json.dump(experiment, file, indent=2)

    data_runtime = estimated_runtime * job_count * 2 * repetitions / 60
    total_runtime += data_runtime
    print(f"Estimated run time for {data}: {data_runtime} h")
print(f"Estimated total run time: {total_runtime} h")

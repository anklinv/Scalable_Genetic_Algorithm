import json
import math

# in minutes
estimated_runtime = 5
repetitions = 5

# Result of grid search
best_population = {
    "a280": 768,
    "d1291": 64
}
base_n = 4
nr_islands = [1, 2, 4, 8, 12, 16, 24, 32]
population_sizes = [lambda islands, data: int(best_population[data]),
                    lambda islands, data: int(best_population[data] / (islands / base_n) ** 0.5),
                    lambda islands, data: int(best_population[data] / (islands / base_n))]
migration_period = 25
migration_amount = lambda x: x // 8
elite_size = lambda x: x // 2
log_freq = lambda x: x // 1000

base_times = {
    "a280": 4.491,
    "d1291": 16.946
}

base_population = 64
base_epochs = 100000

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
    fixed_params["mode"] = "island"
    fixed_params["--migration_period"] = migration_period
    experiment["fixed_params"] = fixed_params
    variable_params = dict()
    pop = dict()
    pop["type"] = "tuple"
    pop["names"] = ["--population", "--elite_size", "-n", "--epochs", "--log_freq", "--migration_amount"]
    values = list()
    job_count = 0
    for n in nr_islands:
        for population_function in population_sizes:
            population = population_function(n, data)
            job_count += 1
            value = dict()
            multiplier = base_population / population
            epochs = int(round((estimated_runtime * 60 / base_times[data]) * base_epochs * multiplier, -4))
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
    experiment["variable_params"] = variable_params
    with open(f"scaling_test_{data}.json", mode="w") as file:
        json.dump(experiment, file, indent=2)

print(f"Estimated run time: {estimated_runtime * job_count * repetitions / 60} h per problem")

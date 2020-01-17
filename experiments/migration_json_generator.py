import json
import math

# in minutes
estimated_runtime = 2
n = 4
repetitions = 5
population_sizes = [64, 256, 768]
migration_amounts = [lambda x: math.floor(x // (1 * (n - 1))) * (n - 1),
                     lambda x: math.floor(x // (2 * (n - 1))) * (n - 1),
                     lambda x: math.floor(x // (4 * (n - 1))) * (n - 1),
                     lambda x: math.floor(x // (8 * (n - 1))) * (n - 1)]
migration_periods = [10, 25, 50, 100]
elite_size = lambda x: x // 2
log_freq = lambda x: x // 1000

base_times = {
    "a280": 4.491,
    "d1291": 16.946,
    "u2319": 78.710
}

base_population = 64
base_epochs = 100000

for data in base_times.keys():
    experiment = dict()
    experiment["name"] = f"convergence test {data}"
    experiment["repetitions"] = repetitions
    fixed_params = dict()
    fixed_params["--data"] = data + ".csv"
    fixed_params["--mutation"] = 10
    fixed_params["--migration_topology"] = "fully_connected"
    fixed_params["--selection_policy"] = "truncation"
    fixed_params["--replacement_policy"] = "truncation"
    fixed_params["mode"] = "island"
    fixed_params["-n"] = n
    experiment["fixed_params"] = fixed_params
    variable_params = dict()
    pop = dict()
    pop["type"] = "tuple"
    pop["names"] = ["--population", "--elite_size", "--migration_amount", "--epochs", "--log_freq", "--migration_period"]
    values = list()
    job_count = 0
    for population in population_sizes:
        for migration_amount in migration_amounts:
            for migration_period in migration_periods:
                job_count += 1
                value = dict()
                multiplier = base_population / population
                epochs = int(round((estimated_runtime * 60 / base_times[data]) * base_epochs * multiplier, -4))
                val = [
                    population,
                    elite_size(population),
                    migration_amount(population),
                    epochs,
                    log_freq(epochs),
                    migration_period
                ]
                value["value"] = val
                values.append(value)
    pop["values"] = values
    variable_params["pop"] = pop
    experiment["variable_params"] = variable_params
    with open(f"migration_test_{data}.json", mode="w") as file:
        json.dump(experiment, file, indent=2)

print(f"Estimated run time: {estimated_runtime * job_count * repetitions / 60} h per problem")

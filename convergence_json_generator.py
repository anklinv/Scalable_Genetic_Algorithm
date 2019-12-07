import json

# in minutes
estimated_runtime = 15
elite_size = lambda x: x//2
migration_amount = lambda x: x//2
log_freq = lambda x: x // 10000

population_sizes = [4, 8, 16, 32, 64, 128, 256, 512]

base_times = {
    "berlin52": 1.755,
    "bier127": 2.793,
    "a280": 4.491,
    "d1291": 16.946,
    "u2319": 78.710
}

base_population = 64
base_epochs = 100000

for data in base_times.keys():
    experiment = dict()
    experiment["name"] = f"convergence test {data}"
    experiment["repetitions"] = 5
    fixed_params = dict()
    fixed_params["--data"] = data + ".csv"
    fixed_params["--mutation"] = 10
    fixed_params["--migration_topology"] = "fully_connected"
    fixed_params["--selection_policy"] = "truncation"
    fixed_params["--replacement_policy"] = "truncation"
    fixed_params["mode"] = "island"
    fixed_params["-n"] = 4
    fixed_params["--migration_period"] = 50
    experiment["fixed_params"] = fixed_params
    variable_params = dict()
    pop = dict()
    pop["type"] = "tuple"
    pop["names"] = ["--population", "--elite_size", "--migration_amound", "--epochs", "--log_freq"]
    values = list()
    for population in population_sizes:
        value = dict()
        multiplier = base_population / population
        epochs = int(round((estimated_runtime * 60 / base_times[data]) * base_epochs * multiplier, -4))
        val = [
            population,
            elite_size(population),
            migration_amount(population),
            epochs,
            log_freq(epochs)
        ]
        value["value"] = val
        values.append(value)
    pop["values"] = values
    variable_params["pop"] = pop
    experiment["variable_params"] = variable_params
    with open(f"convergence_test_{data}.json", mode="w") as file:
        json.dump(experiment, file, indent=2)

#!/usr/bin/env python

from __future__ import print_function
import argparse
import time
from tqdm import tqdm
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model(data):
    """Stores the data for the problem."""
    with open("../../data/{}.csv".format(data)) as file:
        n_cities = file.readline()
        distances = file.readlines()
        distance_matrix = list(map(lambda x: list(map(int, x.split(";")[0:-1])), distances))
    
    data = {}
    data['distance_matrix'] = distance_matrix  # yapf: disable
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def print_solution(manager, routing, assignment):
    """Prints assignment on console."""
    print('Objective: {} miles'.format(assignment.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = assignment.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)


def solve_tsp(data):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(data)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        return assignment.ObjectiveValue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running some experiments")
    parser.add_argument('--rep', type=int, default=1, dest="rep",
                        help="Number of repetitions to run")
    parser.add_argument('-d', type=str, default=["a280"], nargs="+", dest="data",
                        help="Data files to run")
    args = parser.parse_args()

    measurements = list()
    for i in tqdm(range(args.rep)):
        for data in args.data:
            time1 = time.time()
            length = solve_tsp(data)
            time2 = time.time()
            time_needed = (time2-time1)*1000.0
            measurements.append([data, i, length, time_needed])
    df = pd.DataFrame(measurements, columns=["data", "rep", "length", "time"])
    df.to_csv("google_or_benchmark.csv", index=False)


#!/usr/bin/env python

from __future__ import print_function
import argparse
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
    print('Path length achieved: {}'.format(assignment.ObjectiveValue()))


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
        print_solution(manager, routing, assignment)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running some experiments")
    parser.add_argument('-d', type=str, default="a280", dest="data",
                        help="Data file to run")
    args = parser.parse_args()
    
    solve_tsp(args.data)

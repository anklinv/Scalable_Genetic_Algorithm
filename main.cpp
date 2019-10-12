#include <iostream>
#include <string>
#include <fstream>
#include "sequential/city.hpp"
#include "sequential/travelling_salesman_problem.hpp"

using namespace std;


int main() {
    ifstream input("data/att48.tsp");

    // TODO: Make this nicer
    string name, comment, type, dimension, edge_weight_type, node;
    getline(input, name);
    name = name.substr(7, name.length());
    getline(input, comment);
    comment = comment.substr(10, comment.length());
    getline(input, type);
    type = type.substr(7, type.length());
    getline(input, dimension);
    dimension = dimension.substr(12, dimension.length());
    getline(input, edge_weight_type);
    edge_weight_type = edge_weight_type.substr(18, edge_weight_type.length());
    getline(input, node);

    // Read cities
    int number_cities = stoi(dimension);
    TravellingSalesmanProblem problem(number_cities, 1000);
    cout << "Reading " << dimension << " cities of problem " << name << "... ";
    // Read city coordinates
    for (int i = 0; i < number_cities; ++i) {
        int index;
        double x, y;
        input >> index >> x >> y;
        problem.cities.push_back({x,y});
    }
    input.close();
    cout << "Done!" << endl;


    return 0;
}



import numpy as np
import sys

tsp_file = sys.argv[1]

tsp_name = tsp_file.split('.')[0]

cities_list = []

with open(tsp_file) as f:
    lns = f.readlines()
    dim = int(lns[3].split(' ')[2])
    print(dim)
    node_edge_incidence = np.empty((dim,dim))
    location = np.empty((dim,2))
    for i in range(0,len(lns)-7):
        location[i,:] = lns[i+6].split(' ')[1:]
    for i in range(0,dim-1):
        for j in range(1,dim):
            node_edge_incidence[i,j] = np.sqrt((location[i,0] - location[j,0])**2 + (location[i,1] - location[j,1])**2)
        
    np.savetxt(tsp_name +'.csv', node_edge_incidence)
        

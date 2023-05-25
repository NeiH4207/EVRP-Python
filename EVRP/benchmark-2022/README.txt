This EVRP benchmark set is published in: 

M. Mavrovouniotis, C. Menelaou, S. Timotheou, G. Ellinas, C. Panayiotou and M. Polycarpou, "A Benchmark Test Suite for the Electric Capacitated Vehicle Routing Problem," 2020 IEEE Congress on Evolutionary Computation (CEC), 2020, pp. 1-8, doi: 10.1109/CEC48606.2020.9185753.

If you use the benchmark set in your research, we would appreciate a citation in your publications.


DETAILS of the instance files

Specification part
------------------
All entries in this section are of the form <keyword> : <value>. Below we give a list of all avaiable keywords.

NAME: <string>
Identifies the data file

TYPE: <string>
Specifies the type of the data, i.e., EVRP

COMMENT: <string>
Additional comments

OPTIMAL_VALUE: <integer>
Identifies either the optimal value, upper bound or best known value

VEHICLES: <integer>
It is the minimum number of EVs that can be used

DIMENSION: <integer>
It is the total number of nodes, including customers, depots, and charging stations

ENERGY_CAPACITY: <integer>
Specifies the EV battery capacity

ENERGY_CONSUMPTION: <decimal>
Specifies the energy consumption of the EV when traversing arcs

STATIONS: <integer>
It is the number of charging stations

CAPACITY: <integer>
Specifies the EV cargo capacity

EDGE_WEIGHT_TYPE: <string>
EUC_2D Weights are Euclidean distances in 2-D

EOF: 
Terminates the input data


Data part
----------------
The instance data are given in the corresponding data sections following the specification part. Each data 
begins with the corresponding keyword. The length of the sections depends on the type of the data.

NODE_COORD_SECTION: 
Node coordinates are given in this section. Each line is of the form 
<integer> <real> <real>
The integers give the number of the respective node and the real numbers give the associate coordinates

DEMAND_SECTION:
Customer delivey demands are given in this section. Each line is of the form
<integer> <integer>
The first integer give the number of the respective customer node and the real give its delivery demand. The demand 
of the depot node is always 0.

STATION_COORD_SECTION:
Contains a list of all the recharging station nodes

DEPOT_SECTION:
Contains a list of the depot nodes. This list is terminated by a -1.

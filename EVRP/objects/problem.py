from collections import OrderedDict
from copy import deepcopy
import os
from random import shuffle
import numpy as np
from matplotlib import pyplot as plt

from EVRP.objects.node import Node
from EVRP.objects.solution import Solution

from EVRP.src.utils import get_problem_name, logger

class Problem():
    
    def __init__(self, problem_path=None):
        """
        Initializes an instance of the class with the given parameters.

        :param problem_name: A string representing the name of the problem to be solved (e.g. E-n22-k4)
        :type problem_name: str
        :param dataset_path: A string representing the path of the dataset to be used (default is ./EVRP/benchmark/)
        :type dataset_path: str
        """
        self.problem_path = problem_path
        self.problem_name = get_problem_name(problem_path)
        if not os.path.isfile(problem_path):
            raise ValueError(f"Problem file not found: {problem_path}. Please input a valid problem name.")

        self.max_num_vehicles = None
        self.energy_capacity = None
        self.capacity = None
        self.num_stations = None
        self.num_dimensions = None
        self.optimal_value = None
        self.energy_consumption = None
        self.nodes = []
        self.node_dict = dict()
        self.customers = []
        self.customer_ids = []
        self.stations = []
        self.station_ids = []
        self.demands = []
        self.depot = None
        self.depot_id = None

        self.problem = self.__read_problem(problem_path)
            
    def get_problem_size(self):
        return len(self.nodes)
    
    def get_depot(self):
        return self.depot
    
    def get_num_customers(self):
        return self.num_customers
    
    def get_num_stations(self):
        return self.num_stations
    
    def get_num_dimensions(self):
        return self.num_dimensions
    
    def get_max_num_vehicles(self):
        return self.max_num_vehicles
    
    def get_customer_demand(self, node):
        return node.get_demand()
    
    def get_energy_consumption(self, from_node, to_node):
        return self.energy_consumption * from_node.distance(to_node)
    
    def get_depot_id(self):
        return self.depot_id
    
    def get_customer_ids(self):
        return deepcopy(self.customer_ids)
    
    def get_station_ids(self):
        return deepcopy(self.station_ids)
    
    def get_all_stations(self):
        return deepcopy(self.stations)
    
    def get_battery_capacity(self):
        return self.energy_capacity
    
    def get_capacity(self):
        return self.capacity
    
    def get_all_customers(self):
        return self.customers
    
    def get_node_from_id(self, id):
        return self.node_dict[id]
    
    def get_distance(self, from_node, to_node):
        return from_node.distance(to_node)
        
    def __read_problem(self, problem_file_path):
        """
        Reads a problem file and initializes the problem.
        
        :param problem_file_path: str, the path to the problem file.
        
        :return: None
        
        The function reads the problem file at the specified path and initializes the problem 
        attributes such as number of vehicles, number of dimensions, number of stations, number 
        of customers, capacity, energy capacity, energy consumption, nodes, demands, customer 
        ids, customers, station ids, stations, and depot id. It also sets the type of each node 
        (depot, customer, or station) and its demand. If the edge weight type is not EUC_2D, 
        the function raises a ValueError.
        """
        with open(problem_file_path, 'r') as f:
            lines = f.readlines()
            
            """ Read metadata """
            logger.info(f"Read problem file: {problem_file_path}")
            logger.info("{}".format(lines[0]))
            logger.info("{}".format(lines[1]))
            logger.info("{}".format(lines[2]))
            logger.info("{}".format(lines[3]))
            self.max_num_vehicles = int(lines[4].split()[-1])
            self.num_dimensions = int(lines[5].split()[-1])
            self.num_stations = int(lines[6].split()[-1])
            self.num_customers = self.num_dimensions - self.num_stations
            self.capacity = float(lines[7].split()[-1])
            self.energy_capacity = float(lines[8].split()[-1])
            self.energy_consumption = float(lines[9].split()[-1])
            logger.info("{}".format(lines[10]))
            edge_weight_type = lines[10].split()[-1]
            
            """ Read NODES """
            if edge_weight_type == 'EUC_2D':
                start_line = 12
                end_line = 12 + self.num_dimensions
                for i in range(start_line, end_line):
                    id, x, y = lines[i].split()
                    id = int(id) - 1
                    self.nodes.append(Node(int(id), float(x), float(y)))
                    self.node_dict[id] = self.nodes[-1]
                    
                start_line = end_line + 1
                end_line = start_line + self.num_customers
                for i in range(start_line, end_line):
                    _id, demand = lines[i].split()[-2:]
                    _id = int(_id) - 1
                    demand = float(demand)
                    self.demands.append(demand)
                    self.nodes[_id].set_type('C')
                    self.nodes[_id].set_demand(demand)
                    self.customer_ids.append(_id)
                    self.customers.append(self.nodes[_id])
                    
                start_line = end_line + 1
                end_line = start_line + self.num_stations
                for i in range(start_line, end_line):
                    _id = lines[i].split()[-1]
                    _id = int(_id) - 1
                    self.nodes[_id].set_type('S')
                    self.station_ids.append(_id)
                    self.stations.append(self.nodes[_id])
                    
                self.depot_id = int(lines[end_line + 1].split()[-1]) - 1
                self.nodes[self.depot_id].set_type('D')
                self.depot = self.nodes[self.depot_id]
                # remove depot from customers
                self.customer_ids.remove(self.depot_id)
                for i in range(len(self.customers)):
                    if self.customers[i].is_depot():
                        self.customers.pop(i)
                        break

                self.num_customers -= 1 # skip depot from customers
            else:
                raise ValueError(f"Invalid benchmark, edge weight type: {edge_weight_type} not supported.")
    
    def check_valid_solution(self, solution, verbose=False):
        """
        Check if a given solution is a valid solution for the Vehicle Routing Problem (VRP).
        
        Args:
            solution: A Solution object containing the tours to be checked.
        
        Returns:
            A boolean value indicating whether the solution is valid or not.
        """
        is_valid = solution.set_tour_index()
        if not is_valid:
            if verbose:
                logger.warning("The vehicle has visited a customer more than once.")
            return False
        
        tours = solution.get_tours()

        if len(tours) > self.max_num_vehicles:
            if verbose:
                logger.warning("This solution using more than number of available vehicles.")
            return False
        
        visited = {}

        for tour in tours:
            energy_temp = self.get_battery_capacity()
            capacity_temp = self.get_capacity()
                
            
            for i in range(len(tour) - 1):
                first_node = tour[i]
                second_node = tour[i + 1]

                if first_node.is_customer():
                    if first_node.get_id() in visited:
                        if verbose:
                            logger.warning("The vehicle has visited a customer more than once.")
                        return False
                    visited[first_node.get_id()] = 1
                
                capacity_temp -= self.get_customer_demand(second_node)
                energy_temp -= self.get_energy_consumption(first_node, second_node)
                
                if capacity_temp < 0.0:
                    if verbose:
                        logger.warning("The vehicle exceeds capacity when visiting {}.".format(second_node.get_id()))
                    return False
                
                if energy_temp < 0.0:
                    if verbose:
                        logger.warning("The vehicle exceeds energy when visiting {}.".format(second_node.get_id()))
                    return False
                
                if second_node.is_depot():
                    capacity_temp = self.get_capacity()
                    energy_temp = self.get_battery_capacity()
                    
                if second_node.is_charging_station():
                    energy_temp = self.get_battery_capacity()
                    
        return True 
        
    def random_solution(self):
        """
        Returns:
            solution (Solution): a randomly generated solution for the EVRP problem instance
            
            Example:
            Basic presentation: [0, 1, 2, 3, 0, 4, 5, 0, 6, 0]
            Vehicle tours: 
                Vehicle 1: 0 -> 1 -> 2 -> 3 -> 0
                Vehicle 2: 0 -> 4 -> 5 -> 0
                Vehicle 3: 0 -> 6 -> 0
                
        (*) Note:   The solution generated by the algorithm is not guaranteed to be valid 
                    in terms of capacity and energy constraints.
                    Your task is to modify the solution to a valid one that has the shortest tour length.
        """
        # Create an empty solution object
        solution = Solution()

        # Generate a list of all customer IDs
        temp_solution = self.get_customer_ids()

        # Shuffle the list of customer IDs to randomize the solution
        shuffle(temp_solution)

        splited_tour_indexes = np.random.choice(len(temp_solution), self.max_num_vehicles - 1, replace=False)
        
        splited_tour_indexes = np.append(0, splited_tour_indexes)

        splited_tour_indexes = np.append(splited_tour_indexes, len(temp_solution))

        splited_tour_indexes = np.sort(splited_tour_indexes)

        for i in range(self.max_num_vehicles):
            tour_ids = temp_solution[splited_tour_indexes[i]:splited_tour_indexes[i + 1]]
            tour = [self.get_node_from_id(_id) for _id in tour_ids]
            solution.add_tour(tour)

        solution.set_tour_index()

        return solution
    
    def calculate_tour_length(self, solution: Solution):
        tour_length = 0
        for tour in solution.get_tours():
            tour = [self.get_depot()] + tour + [self.get_depot()]
            for i in range(len(tour) - 1):
                tour_length += tour[i].distance(tour[i + 1])
        
        if self.check_valid_solution(solution):
            return tour_length
        else:
            return tour_length * 2
    
    def plot(self, solution=None, path=None):
        """
        Plot the solution of the vehicle routing problem on a scatter plot.

        Args:
            solution (Solution): A `Solution` object containing the solution of the vehicle routing problem.

        Returns:
            None.
        """

        _, ax = plt.subplots()

        for node in self.nodes:
            if node.is_customer():
                ax.scatter(node.x, node.y, c='green', marker='o',
                        s=30, alpha=0.5, label="Customer")
            elif node.is_depot():
                ax.scatter(node.x, node.y, c='red', marker='s',
                        s=30, alpha=0.5, label="Depot")
            elif node.is_charging_station():
                ax.scatter(node.x, node.y, c='blue', marker='^',
                        s=30, alpha=0.5, label="Station")
            else:
                raise ValueError("Invalid node type")

        # Set title and labels
        ax.set_title(f"Problem {self.problem_name}")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
                loc='upper right',
                prop={'size': 6})

        if solution is not None:
            tours = solution.get_tours()
            for tour in tours:
                if len(tour) == 0:
                    continue
                plt.plot([self.get_depot().x, tour[0].x],
                        [self.get_depot().y, tour[0].y],
                        c='black', linewidth=0.5, linestyle='--')
                
                for i in range(len(tour) - 1):
                    first_node = tour[i]
                    second_node = tour[i + 1]
                    plt.plot([first_node.x, second_node.x],
                            [first_node.y, second_node.y],
                            c='black', linewidth=0.5, linestyle='--')
                    
                plt.plot([tour[-1].x, self.get_depot().x],
                        [tour[-1].y, self.get_depot().y],
                        c='black', linewidth=0.5, linestyle='--')
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()

if __name__ == "__main__":
    # evrp = EVRP('X-n1006-k43-s5', dataset_path='./EVRP/benchmark-2022/')
    evrp = Problem('E-n22-k4', dataset_path='./EVRP/benchmark-2019/')
    solution = evrp.random_solution()
    logger.info("Random solution is {}".format("valid" if evrp.check_valid_solution(solution, verbose=True) else "invalid"))
    print(solution)
    evrp.plot(solution)
        
    
    
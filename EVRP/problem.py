from collections import OrderedDict
import os
from random import shuffle
import numpy as np
from matplotlib import pyplot as plt

from EVRP.node import Node
from EVRP.solution import Solution

from rich.logging import RichHandler
import logging
logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler()]
)
log = logging.getLogger("rich")

class Problem():
    
    def __init__(self, problem_name: str, dataset_path='./EVRP/benchmark-2022/'):
        """
        Initializes an instance of the class with the given parameters.

        :param problem_name: A string representing the name of the problem to be solved (e.g. E-n22-k4)
        :type problem_name: str
        :param dataset_path: A string representing the path of the dataset to be used (default is ./EVRP/benchmark/)
        :type dataset_path: str
        """
        self.problem_name = problem_name
        self.dataset_path = dataset_path
        problem_path = os.path.join(self.dataset_path, problem_name + '.evrp')
        if not os.path.isfile(problem_path):
            raise ValueError(f"Problem file not found: {problem_path}. Please input a valid problem name.")

        self.num_vehicles = None
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
    
    def get_num_vehicles(self):
        return self.num_vehicles
    
    def get_customer_demand(self, node):
        return node.get_demand()
    
    def get_energy_consumption(self, from_node, to_node):
        return self.energy_consumption * from_node.distance(to_node)
    
    def get_depot_id(self):
        return self.depot_id
    
    def get_customer_ids(self):
        return self.customer_ids
    
    def get_station_ids(self):
        return self.station_ids
    
    def get_all_stations(self):
        return self.stations
    
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
            logging.info(f"Read problem file: {problem_file_path}")
            logging.info("{}".format(lines[0]))
            logging.info("{}".format(lines[1]))
            logging.info("{}".format(lines[2]))
            logging.info("{}".format(lines[3]))
            self.num_vehicles = int(lines[4].split()[-1])
            self.num_dimensions = int(lines[5].split()[-1])
            self.num_stations = int(lines[6].split()[-1])
            self.num_customers = self.num_dimensions - self.num_stations
            self.capacity = float(lines[7].split()[-1])
            self.energy_capacity = float(lines[8].split()[-1])
            self.energy_consumption = float(lines[9].split()[-1])
            logging.info("{}".format(lines[10]))
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
            else:
                raise ValueError(f"Invalid benchmark, edge weight type: {edge_weight_type} not supported.")
    
    def check_valid_solution(self, solution):
        """
        Check if a given solution is a valid solution for the Vehicle Routing Problem (VRP).
        
        Args:
            solution: A Solution object containing the tours to be checked.
        
        Returns:
            A boolean value indicating whether the solution is valid or not.
        """
        energy_temp = self.energy_capacity
        capacity_temp = self.capacity
        
        """ Vehicle did not start or end at a depot. """
        if not solution.tours[0].is_depot() or solution.tours[-1].is_depot():
            return False
        flatten_tours = solution.to_array()
        # Check all customer occurrence only 1 time
        counts = np.unique(flatten_tours, return_counts=True)
        for node_id in range(len(counts[0])):
            """ Vehicle visited customer `node_id` more than once. """
            if counts[1][node_id] != 1 and self.nodes[node_id].is_customer():
                return False
            
        """This solution using more than number of available vehicles. """
        num_vehicle_used = counts[1][self.depot_id] - 1
        if  num_vehicle_used> self.num_vehicles:
            return False
        
        for i in range(len(flatten_tours)-1):
            first_id = flatten_tours[i]
            first_node = self.nodes[first_id]
            second_id = flatten_tours[i + 1]
            second_node = self.nodes[second_id]
            
            capacity_temp -= self.get_customer_demand(second_node)
            energy_temp -= self.get_energy_consumption(first_node, second_node)
            
            """ Vehicle exceeds capacity when visiting second_id. """
            if capacity_temp < 0.0:
                return False
            """ Vehicle exceeds energy when visiting second_id. """
            if energy_temp < 0.0:
                return False
            
            if second_node.is_depot():
                capacity_temp = self.capacity
                energy_temp = self.energy_capacity
                
            if second_node.is_charging_station():
                energy_temp = self.energy_capacity
                
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
        temp_solution = np.arange(1, self.get_num_customers())

        # Shuffle the list of customer IDs to randomize the solution
        shuffle(temp_solution)

        # Insert the depot (ID) at the beginning and end of the solution
        temp_solution = np.insert(temp_solution, 0, self.depot_id)
        temp_solution = np.insert(temp_solution, len(temp_solution), self.depot_id)

        # Insert random depots into the solution
        num_insert_depot = self.num_vehicles - 1
        for _ in range(int(num_insert_depot)):
            idx = np.random.randint(0, len(temp_solution))
            temp_solution = np.insert(temp_solution, idx, self.depot_id)

        # Set the final solution and return it
        for node_id in temp_solution:
            solution.append(self.nodes[node_id])
            
        return solution
    
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
            for i in range(len(solution.complete_tours) - 1):
                first_node = solution.complete_tours[i]
                second_node = solution.complete_tours[i + 1]
                plt.plot([first_node.x, second_node.x],
                        [first_node.y, second_node.y],
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
    logging.info("`Random solution is {}".format("valid" if evrp.check_valid_solution(solution) else "invalid"))
    solution.print()
    # evrp.plot(solution)
        
    
    
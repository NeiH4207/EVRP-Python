from collections import OrderedDict
import os
from random import shuffle
import numpy as np
from matplotlib import pyplot as plt
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")

class Node():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = None
        self.demand = 0
        
    def get_type(self):
        return self.type
    
    def set_type(self, type):
        if type not in ['C', 'S', 'D']:
            raise ValueError(f"Invalid type: {type}, must be 'C', 'S', or 'D'.")
        self.type = type
    
    def get_demand(self):
        return self.demand
    
    def set_demand(self, demand):
        self.demand = demand
    
    def distance(self, P):
        return np.sqrt((self.x - P.x)**2 + (self.y - P.y)**2)
    
    def is_customer(self):
        return self.type == 'C'
    
    def is_station(self):
        return self.type == 'S'
    
    def is_depot(self):
        return self.type == 'D'
    
class Solution():
    def __init__(self):
        super().__init__()
        self.tours = []
        
    def add_node(self, id):
        self.tours.append(id)
    
    def set(self, solution):
        self.tours = solution
        
    def get(self, id):
        return self.tours[id]

class EVRP():
    
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
    
    def get_cutomer_id_list(self):
        return self.customer_ids
    
    def get_station_id_list(self):
        return self.station_ids
        
    def __read_problem(self, problem_file_path):
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
                    x, y = lines[i].split()[-2:]
                    self.nodes.append(Node(float(x), float(y)))
                    
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
            else:
                raise ValueError(f"Invalid edge weight type: {edge_weight_type}")
                
    def is_valid_solution(self, solution: Solution):
        energy_temp = self.energy_capacity
        capacity_temp = self.capacity
        
        if solution.tours[0] != 0 or solution.tours[-1] != 0:
            return False
        
        # Check all customer occurrence only 1 time
        counts = np.unique(solution.tours, return_counts=True)
        for node_id in range(len(counts[0])):
            if counts[1][node_id] != 1 and self.nodes[node_id].is_customer():
                return False
        
        for i in range(len(solution.tours)-1):
            first_id = solution.tours[i]
            first_node = self.nodes[first_id]
            second_id = solution.tours[i+1]
            second_node = self.nodes[second_id]
            
            capacity_temp -= self.get_customer_demand(second_node)
            energy_temp -= self.get_energy_consumption(first_node, second_node)
            
            if capacity_temp < 0.0:
                return False
            if energy_temp < 0.0:
                return False
            
            if second_node.is_depot():
                capacity_temp = self.capacity
                energy_temp = self.energy_capacity
                
            if second_node.is_station():
                energy_temp = self.energy_capacity
                
        return True
    
    def get_random_solution(self):
        solution = Solution()
        temp_solution = np.arange(1, evrp.get_num_customers() + 1)
        shuffle(temp_solution)
        
        num_insert_station = int(np.sqrt(evrp.get_num_customers())) * 2
        for _ in range(int(num_insert_station)):
            idx = np.random.randint(0, len(temp_solution))
            rd_station = np.random.choice(evrp.get_station_id_list(), 1)[0]
            temp_solution = np.insert(temp_solution, idx, rd_station)
            
        temp_solution = np.insert(temp_solution, 0, 0)
        temp_solution = np.insert(temp_solution, len(temp_solution), 0)
        num_insert_depot = int(np.sqrt(evrp.get_num_customers())) * 2
        for _ in range(int(num_insert_depot)):
            idx = np.random.randint(0, len(temp_solution))
            temp_solution = np.insert(temp_solution, idx, 0)
        solution.set(temp_solution)
        return solution
        
    
    def plot(self):
        fig, ax = plt.subplots()
        
        for node in self.nodes:
            if node.is_customer():
                ax.scatter(node.x, node.y, c='green', marker='o', s=100, alpha=0.5, label="Customer")
            elif node.is_depot():
                ax.scatter(node.x, node.y, c='red', marker='s', s=100, alpha=0.5, label="Depot")
            elif node.is_station():
                ax.scatter(node.x, node.y, c='blue', marker='^', s=100, alpha=0.5, label="Station")
            else:
                raise ValueError("Invalid node type")
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Set title and labels
        ax.set_title("Problem {}".format(self.problem_name))
        plt.show()
        
        
    def plot_solution(self, solution):
        fig, ax = plt.subplots()
        
        for node in self.nodes:
            if node.is_customer():
                ax.scatter(node.x, node.y, c='green', marker='o', s=100, alpha=0.5, label="Customer")
            elif node.is_depot():
                ax.scatter(node.x, node.y, c='red', marker='s', s=100, alpha=0.5, label="Depot")
            elif node.is_station():
                ax.scatter(node.x, node.y, c='blue', marker='^', s=100, alpha=0.5, label="Station")
            else:
                raise ValueError("Invalid node type")
        ax.legend()
        # Set title and labels
        ax.set_title("Problem {}".format(self.problem_name))
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        
        for i in range(len(solution.tours) - 1):
            first_node = self.nodes[solution.tours[i]]
            second_node = self.nodes[solution.tours[i+1]]
            plt.plot([first_node.x, second_node.x], [first_node.y, second_node.y], c='black', linewidth=0.5, linestyle='--')
            
        plt.show()

if __name__ == "__main__":
    evrp = EVRP('X-n1006-k43-s5', dataset_path='./EVRP/benchmark-2022/')
    # evrp = EVRP('E-n22-k4', dataset_path='./EVRP/benchmark-2019/')
    solution = evrp.get_random_solution()
    print(evrp.is_valid_solution(solution))
    evrp.plot_solution(solution)
        
    
    
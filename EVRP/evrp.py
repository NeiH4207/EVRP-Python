import os
import numpy as np
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")

class Node():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def get_type(self):
        return self.type
    
    def distance(self, P):
        return np.sqrt((self.x - P.x)**2 + (self.y - P.y)**2)

class EVRP():
    
    def __init__(self, problem_name: str, dataset_path='./EVRP/benchmark-2022/'):
        """
        Initializes an instance of the class with the given parameters.

        :param problem_name: A string representing the name of the problem to be solved (e.g. E-n22-k4)
        :type problem_name: str
        :param dataset_path: A string representing the path of the dataset to be used (default is ./EVRP/benchmark/)
        :type dataset_path: str
        """
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
        self.stations = []
        self.demands = []
        self.depot = None

        self.problem = self.__read_problem(problem_path)
        
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
                    self.customers.append(self.nodes[int(_id) -1])
                    self.demands.append(float(demand))
                start_line = end_line + 1
                end_line = start_line + self.num_stations
                for i in range(start_line, end_line):
                    _id = lines[i].split()[-1]
                    self.stations.append(self.nodes[int(_id) - 1])
                self.depot = int(lines[end_line + 1].split()[-1])
        

if __name__ == "__main__":
    evrp = EVRP('E-n37-k4-s4', dataset_path='./EVRP/benchmark-2022/')
    evrp = EVRP('E-n76-k7', dataset_path='./EVRP/benchmark-2019/')
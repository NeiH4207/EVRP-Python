from copy import deepcopy
import numpy as np
from EVRP.utils import logger

class Solution():
    def __init__(self, tours=None):
        """
        The solution contains a list of tours. Each vehicle start and end at the depot
        See full description in the documentation https://mavrovouniotis.github.io/EVRPcompetition2020/TR-EVRP-Competition.pdf
        Example a solution with two vehicles: 0 -> 1 -> 2 -> 3 -> 0 -> 4 -> 0 -> 5 -> 6 -> 0
        tours includes [1, 2, 3, 4] and [5, 6]
        In this problem, the depot can be visited multiple times for each vehicle. (First vehicle tour: 0 -> 1 -> 2 -> 3 -> 4)
        The depot considered as the charging station.
        """
        self.tour_index = {}
        self.tour_length = np.inf
        if tours:
            self.tours = tours
            self.set_tour_index()
        else:
            self.tours = []
        
    def add_tour(self, tour):
        self.tours.append(tour)

    def set_tour_index(self):
        self.tour_index = {}
        for idx, tour in enumerate(self.tours):
            for node in tour:
                if node.is_customer():
                    if node.id not in self.tour_index:
                        self.tour_index[node.id] = idx
                    else:
                        logger.warning('Node {} already in tour {}'.format(node.id, idx))
                        return 0
        return 1
    def __repr__(self) -> str:
        if self.tour_length:
            presentation = "Tour length: {}\n".format(self.tour_length)
        else:
            presentation = ""
        for i, tour in enumerate(self.tours):
            presentation += 'Tour {}: '.format(i) + ' -> '.join(['0'] + [str(node.id) for node in tour] + ['0']) + '\n'
            
        return presentation
    
    def get_tours(self):
        return deepcopy(self.tours)
    
    def get_basic_tours(self):
        tours = []
        for tour in self.tours:
            _tour = [node for node in tour if node.is_customer()]
            tours.append(_tour)
        return tours

    def get_tour_length(self):
        return self.tour_length
    
    def set_tour_length(self, tour_length):
        self.tour_length = tour_length
    
    def to_array(self):
        return np.array([node.id for node in self.tours])
    
    def get_vehicle_tours(self, skip_depot=False, full=True):
        
        if full:
            tours = self.complete_tours
        else:
            tours = self.tours
        """ Vehicle did not start or end depot """
        if len(tours) == 0:
            tours = deepcopy(self.tours)
        if not tours[0].is_depot() or not tours[-1].is_depot():
            return None
        
        vehicle_tours = []
        
        if not skip_depot:
            tour = [tours[0]]
        else:
            tour = []
        
        for idx, node in enumerate(tours):
            if idx == 0 and not skip_depot:
                continue
            if node.is_depot():
                if skip_depot:
                    vehicle_tours.append(tour)
                    continue
                else:
                    tour.append(tours[0])
                    vehicle_tours.append(tour)
                    tour = [tours[0]]
            else:
                tour.append(node)
        return vehicle_tours
    
    def set_vehicle_tours(self, tours):
        self.tours = tours
        self.set_tour_index()
            
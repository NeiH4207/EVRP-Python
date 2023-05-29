from copy import deepcopy
import numpy as np

class Solution():
    def __init__(self):
        self.tours = []
        self.complete_tours = []
        self.tour_index = dict()
        self.current_tour_idx = -1
        
    def append(self, node):
        self.tours.append(node)
        if node.is_depot():
            self.current_tour_idx += 1
        if not node.is_depot():
            self.tour_index[node.id] = self.current_tour_idx
    
    def set(self, solution):
        self.tours = solution
        self.set_tour_index()
        
    def get(self, id):
        return self.tours[id]
    
    def get(self, id):
        return self.tours[id]
    
    def get_tour_length(self):
        self.calculate_tour_length()
        return self.tour_length
    
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
    
    def set_vehicle_tours(self, tours, skip_depot=False, depot=None):
        if skip_depot:
            self.complete_tours = [[depot] + [node for node in tours[0]]]
        else:
            self.complete_tours = [node for node in tours[0]]
            
        for tour in tours[1:]:
            if skip_depot:
                self.complete_tours.extend([depot] + tour + [depot])
            else:
                self.complete_tours.extend(tour[1:])
            
    def set_tour_index(self):
        current_idx = -1
        for idx, node in enumerate(self.tours):
            if node.is_depot():
                current_idx += 1
            self.tour_index[node.id] = current_idx
        
    def calculate_tour_length(self):
        tour_length = 0
        if len(self.complete_tours) == 0:
            self.complete_tours = self.tours
        for i in range(len(self.complete_tours) - 1):
            tour_length += self.complete_tours[i].distance(self.complete_tours[i + 1])
        self.tour_length = tour_length
    
    def print(self, max_visible_tours=15):
        vehicle_tours = self.get_vehicle_tours()
        print("-" * 40)
        print("Tour length: " + str(self.get_tour_length()))
        for i, tour in enumerate(vehicle_tours):
            if len(tour) > max_visible_tours:
                print('Tour {}: '.format(i) +' | '.join(map(str, tour[:max_visible_tours])) + ' | ...')
            else:
                print('Tour {}: '.format(i)+ ' | '.join(map(str, tour)))
        print("-" * 40)
        
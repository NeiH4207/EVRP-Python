import numpy as np

class Node():
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.type = None
        self.demand = 0
    
    def __str__(self):
        return str(self.id)    
    
    def get_type(self):
        return self.type
    
    def get_id(self):
        return self.id
    
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
    
    def is_charging_station(self):
        return self.type == 'S'
    
    def is_depot(self):
        return self.type == 'D'
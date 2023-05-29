
import logging
from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler()]
)
log = logging.getLogger("rich")

from copy import deepcopy

import numpy as np
from EVRP.problem import Problem
from EVRP.solution import Solution


class GreedySearch():
    """
    Algorithm for insert energy stations into all tours for each vehicle.
    
    """
    def __init__(self, problem: Problem) -> None:
        self.problem = problem
    
    def optimize(self, solution: Solution):
        # solution.print()
        # self.problem.plot(solution, './EVRP/figures/step_1_initial_solution.png')
        vehicle_tours = solution.get_vehicle_tours()
        for i, tour in enumerate(vehicle_tours):
            vehicle_tours[i] = self.local_search_2opt(tour)
            
        # solution.set_vehicle_tours(vehicle_tours)
        # solution.print()
        # self.problem.plot(solution, './EVRP/figures/step_2_LS_2opt_solution.png')
        for i, tour in enumerate(vehicle_tours):
            vehicle_tours[i] = self.insert_energy_stations(tour)
            
        # solution.set_vehicle_tours(vehicle_tours)
        # solution.print()
        # self.problem.plot(solution, './EVRP/figures/step_3_insert_energy_stations_solution.png')
        for i, tour in enumerate(vehicle_tours):
            complete_tour = self.greedy_optimize_station(tour)
            vehicle_tours[i] = complete_tour
        solution.set_vehicle_tours(vehicle_tours)
        # solution.print()
        # self.problem.plot(solution, './EVRP/figures/step_4_greedy_optimize_station_solution.png')
        return solution
    
    def nearest_station(self, from_node, to_node, energy):
        min_length = float("inf")
        best_station = -1

        for v in self.problem.get_all_stations():
            if v.is_charging_station():
                length = v.distance(to_node)
                if self.problem.get_energy_consumption(from_node, v) <= energy:
                    if min_length > length:
                        min_length = length
                        best_station = v

        return best_station
    
    def nearest_station_back(self, from_node, to_node, energy, required_energy):
        min_length = float("inf")
        best_station = -1

        for v in self.problem.get_all_stations():
            if v.is_charging_station():
                if self.problem.get_energy_consumption(from_node, v) <= energy and \
                    self.problem.get_energy_consumption(v, to_node) + required_energy < \
                        self.problem.get_battery_capacity():
                    length1 = v.distance(from_node)
                    length2 = v.distance(to_node)
                    if min_length > length1 + length2:
                        min_length = length1 + length2
                        best_station = v

        return best_station
    
    def greedy_optimize_station(self, tour):
        """
        * Note at this function, number of continuous charging stations S and S' is 1. But it can be more than 1.
        valid tour after inserting energy stations
        : depot_L -> c6 -> c5 -> c4 -> c3 -> S(S1 -> S2) -> c2 -> c1 -> depot_R
        Reverse tour
        : depot_R -> c1 -> c2 -> S(S1 -> S2) -> c3 -> c4 -> c5 -> c6 -> depot_L
        Replace S to other:
        step 1. from depot_R, get a subtour that vehicle reach farest from depot_R but not visit any charging station
            : depot_R -> c1 -> c2 -> c3 -> c4 - (not enough energy to reach c5) -> c5
            : delta_L1 = (d(c2, s1) + d(s1, s2) + d(s2, c3) - d(c2, c3))
        step 2: From c2->c3, c3->c4, c4->c5, find S' (>= 1 charging stations):
            : delta_L2 = d(c3, S') + d(S', c3) - d(c2, c3)
            : delta_L2 = d(c3, S') + d(S', c4) - d(c3, c4)
            : delta_L2 = d(c4, S') + d(S', c5) - d(c4, c5)
            if delta_L2 < delta_L1 then replace S with S'
            # see the paper: https://doi.org/10.1007/s10489-022-03555-8 for more details
        """
        if not tour[0].is_depot() or not tour[-1].is_depot() and len(tour) > 2:
            raise Exception("Tour must start and end with depot")
        
        remaining_energy = dict()
        depotID = tour[0].get_id()
        remaining_energy[depotID] = self.problem.get_battery_capacity()
        optimal_tour = []
        
        for i in range(1, len(tour)):
            if tour[i].is_charging_station() or tour[i].is_depot():
                remaining_energy[tour[i].get_id()] = self.problem.get_battery_capacity()
            else:
                previous_energy = remaining_energy[tour[i - 1].get_id()]
                remaining_energy[tour[i].get_id()] = previous_energy - self.problem.get_energy_consumption(tour[i - 1], tour[i])
                if remaining_energy[tour[i].get_id()] < 0:
                    # skip invalid solution
                    return tour
        
        tour = list(reversed(tour))
        energy = self.problem.get_battery_capacity()
        i = 0
        
        """ Optimize from depot_R """
        while i < len(tour) - 1:
            optimal_tour.append(tour[i])
            if not tour[i + 1].is_charging_station():
                energy -= self.problem.get_energy_consumption(tour[i], tour[i + 1])
                i += 1
                continue
            
            # Calculate delta_L1
            from_node = tour[i]
            num_stations_in_row = 0
            original_distance = 0
            
            while tour[i + 1 + num_stations_in_row].is_charging_station():
                original_distance += from_node.distance(tour[i + 1 + num_stations_in_row])
                from_node = tour[i + num_stations_in_row + 1]
                num_stations_in_row += 1
            
            next_customer_idx = i + num_stations_in_row + 1
            original_distance += from_node.distance(tour[next_customer_idx]) 
            delta_L1 = original_distance - tour[i].distance(tour[next_customer_idx])
            from_node = tour[i]
            considered_nodes = []  
            tmp_energy = energy
            for node in tour[next_customer_idx:]:
                considered_nodes.append(node)
                if node.is_charging_station():
                    break
                tmp_energy -= self.problem.get_energy_consumption(from_node, node)
                if tmp_energy <= 0:
                    break
                from_node = node
            
            from_node = tour[i]
            best_station = tour[i + 1]
            best_station_index = 0
            
            for j, node in enumerate(considered_nodes):
                to_node = node
                required_energy = remaining_energy[to_node.get_id()]
                station = self.nearest_station_back(from_node, to_node, energy, required_energy)
                if station != -1:
                    if self.problem.get_distance(best_station, to_node) > self.problem.get_distance(station, to_node):
                        delta_L2 = self.problem.get_distance(from_node, station) + self.problem.get_distance(station, to_node) \
                            - self.problem.get_distance(from_node, to_node)
                        if delta_L2 < delta_L1:
                            delta_L1 = delta_L2
                            best_station = station

                from_node = to_node

            optimal_tour.extend(considered_nodes[:best_station_index])
            optimal_tour.append(best_station)
            i = i + num_stations_in_row + best_station_index + 1
            energy = self.problem.get_battery_capacity()

        optimal_tour.append(self.problem.get_depot())
        return list(reversed(optimal_tour))
    
    def insert_energy_stations(self, tour):
        if not tour[0].is_depot() or not tour[-1].is_depot() and len(tour) > 2:
            raise Exception("Tour must start and end with depot")
        
        remaining_energy = dict()
        required_min_energy = dict()
        complete_tour = []
        skip_node = dict()
        
        depotID = self.problem.get_depot_id()
        remaining_energy[depotID] = self.problem.get_battery_capacity()
        
        # At the current customer node, calculate the minimum energy required for an 
        # electric vehicle to reach the nearest charging station.
        for node in tour:
            nearest_station = self.nearest_station(node, node, self.problem.get_battery_capacity())
            required_min_energy[node.get_id()] = self.problem.get_energy_consumption(node, nearest_station)
        
        i = 0
        from_node = tour[0]
        to_node = tour[1]
        
        while i < len(tour) - 1:
            
            # go ahead util energy is not enough for visiting the next node
            energy_consumption = self.problem.get_energy_consumption(from_node, to_node)
            if energy_consumption <= remaining_energy[from_node.get_id()]:
                remaining_energy[to_node.get_id()] = remaining_energy[from_node.get_id()] - energy_consumption
                complete_tour.append(from_node)
                i += 1
                from_node = tour[i]
                if i < len(tour) - 1:
                    to_node = tour[i + 1]
                continue
            
            find_charging_station = True
            # If there is enough energy, find the nearest station.
            # If there is not enough energy to reach the nearest station, go back to 
            # the previous node and find the next nearest station from there.
            while find_charging_station:
                while i > 0 and required_min_energy[from_node.get_id()] > remaining_energy[from_node.get_id()]:
                    i -= 1
                    from_node = tour[i]
                    complete_tour.pop()
                if i == 0:
                    return tour
                if from_node.id not in skip_node:
                    return tour
                skip_node[from_node.id] = True
                to_node = tour[i + 1]
                best_station = self.nearest_station(from_node, to_node, remaining_energy[from_node.get_id()])
                if best_station == -1:
                    return tour
                
                complete_tour.append(from_node)
                from_node = best_station
                to_node = tour[i + 1]
                remaining_energy[from_node.get_id()] = self.problem.get_battery_capacity()
                required_min_energy[from_node.get_id()] = 0
                find_charging_station = False                    
                        
        complete_tour.append(tour[-1])
        return complete_tour
    
    def local_search_2opt(self, tour):
        n = len(tour)
        while True:
            improvement = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    new_tour = deepcopy(tour)
                    new_tour[i:j] = reversed(new_tour[i:j])
                    new_distance = sum([new_tour[k].distance(new_tour[k + 1]) for k in range(n - 1)])
                    if new_distance < sum([tour[k].distance(tour[k + 1]) for k in range(n - 1)]):
                        tour = new_tour
                        improvement = True
            if not improvement:
                break
        return tour

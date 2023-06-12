
from copy import deepcopy
from random import shuffle

import numpy as np
from EVRP.problem import Problem
from EVRP.solution import Solution


class GreedySearch():
    """
    Algorithm for insert energy stations into all tours for each vehicle.
    
    """
    def __init__(self) -> None:
        pass

    def set_problem(self, problem: Problem):
        self.problem = problem
        self.nearest_matrix_distance = {}
        self.calculate_nearest_matrix_distance()

    def free(self):
        pass

    def run(self):
        solution = self.create_solution()
        solution = self.optimize(solution)
        solution.set_tour_length(self.problem.calculate_tour_length(solution))
        return solution
    
    def calculate_nearest_matrix_distance(self):
        all_customers = self.problem.get_all_customers()
        self.nearest_matrix_distance = {}
        for i in range(len(all_customers)):
            distances_ = []
            for j in range(len(all_customers)):
                distance = all_customers[i].distance(all_customers[j])
                distances_.append(distance)
            argsort_dist = np.argsort(distances_)[1:]
            self.nearest_matrix_distance[all_customers[i].get_id()] = \
                [all_customers[j].get_id() for j in argsort_dist if i != j]
        

    def create_solution(self):
        # Create an empty solution object
        solution = Solution()

        # Generate a list of all customer IDs
        temp_solution = self.problem.get_customer_ids()

        # Shuffle the list of customer IDs to randomize the solution
        shuffle(temp_solution)
        temp_set = set(temp_solution)
        idx = 0
        capacity_max = self.problem.get_capacity()
        n_tours = 0

        while idx < len(temp_solution):
            center_customer = temp_solution[idx]
            if center_customer not in temp_set:
                idx += 1
                continue
            n_tours += 1
            if n_tours == self.problem.get_num_vehicles():
                tour = list(temp_set)
                solution.add_tour([self.problem.get_node_from_id(tour[i]) for i in range(len(tour))])
                break
            temp_set.remove(center_customer)
            tour = [center_customer]
            capacity = self.problem.get_node_from_id(center_customer).get_demand()
            for i in range(len(self.nearest_matrix_distance[center_customer])):
                second_customer_id = self.nearest_matrix_distance[center_customer][i]
                if second_customer_id in temp_set:
                    demand = self.problem.get_node_from_id(second_customer_id).get_demand()
                    if capacity + demand > capacity_max:
                        continue
                    tour.append(second_customer_id)
                    temp_set.remove(second_customer_id)
                    capacity += demand

            solution.add_tour([self.problem.get_node_from_id(tour[i]) for i in range(len(tour))])

        solution.set_tour_index()

        solution.set_tour_length(self.problem.calculate_tour_length(solution))

        return solution
    
    def optimize(self, solution: Solution, verbose=False):
        if verbose:
            print(solution)
            self.problem.plot(solution, './EVRP/figures/step_1_initial_solution.png')
        
        vehicle_tours = solution.get_basic_tours()

        # for i, tour in enumerate(vehicle_tours):
        #     vehicle_tours[i] = self.local_search_2opt(tour)
            
        if verbose:
            solution.set_vehicle_tours(vehicle_tours)
            print(solution)
            self.problem.plot(solution, './EVRP/figures/step_2_LS_2opt_solution.png')

        # This step (back to depot) make sure that the capacity of each vehicle is not exceeded.
        for i, tour in enumerate(vehicle_tours):
            vehicle_tours[i] = self.insert_depots(tour)

        if verbose:
            solution.set_vehicle_tours(vehicle_tours)
            print(solution)
            self.problem.plot(solution, './EVRP/figures/step_2_LS_2opt_solution.png')

        for i, tour in enumerate(vehicle_tours):
            depot_indexes = [idx for idx, node in enumerate(tour) if node.is_depot()] + [len(tour)]
            sub_tours = []
            for j in range(len(depot_indexes)):
                if j == 0:
                    sub_tour = tour[:depot_indexes[j]]
                else:
                    sub_tour = tour[depot_indexes[j-1] + 1:depot_indexes[j]]
                sub_tours.append(self.insert_energy_stations(sub_tour))
            vehicle_tours[i] = []
            for j, sub_tour in enumerate(sub_tours):
                vehicle_tours[i].extend(sub_tour)
                if j < len(sub_tours) - 1:
                    vehicle_tours[i].append(self.problem.get_depot())
            
        if verbose:
            solution.set_vehicle_tours(vehicle_tours)
            print(solution)
            self.problem.plot(solution, './EVRP/figures/step_3_insert_energy_stations_solution.png')

        for i, tour in enumerate(vehicle_tours):
            depot_indexes = [idx for idx, node in enumerate(tour) if node.is_depot()] + [len(tour)]
            sub_tours = []
            for j in range(len(depot_indexes)):
                if j == 0:
                    sub_tour = tour[:depot_indexes[j]]
                else:
                    sub_tour = tour[depot_indexes[j-1] + 1:depot_indexes[j]]
                sub_tours.append(self.greedy_optimize_station(sub_tour))
            vehicle_tours[i] = []
            for j, sub_tour in enumerate(sub_tours):
                vehicle_tours[i].extend(sub_tour)
                if j < len(sub_tours) - 1:
                    vehicle_tours[i].append(self.problem.get_depot())

        solution.set_vehicle_tours(vehicle_tours)

        if verbose:
            print(solution)
            self.problem.plot(solution, './EVRP/figures/step_4_greedy_optimize_station_solution.png')
        
        solution.set_tour_length(self.problem.calculate_tour_length(solution))
        
        return solution
    
    
    def insert_depots(self, tour):
        _tour = []
        cappacity = self.problem.get_capacity()
        for node in tour:
            if node.is_customer():
                if node.get_demand() > cappacity:
                    _tour.append(self.problem.get_depot())
                    _tour.append(node)
                    cappacity = self.problem.get_capacity() - node.get_demand()
                else:
                    _tour.append(node)
                    cappacity -= node.get_demand()
            if node.is_depot():
                _tour.append(node)
                cappacity = self.problem.get_capacity()

        return _tour
    
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

        for v in self.problem.get_all_stations() + [self.problem.get_depot()]:
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
        tour = [self.problem.get_depot()] + tour + [self.problem.get_depot()]
        required_energy = dict()
        depotID = tour[0].get_id()
        required_energy[depotID] = 0
        optimal_tour = [self.problem.get_depot()]
        
        for i in range(1, len(tour)):
            if tour[i].is_charging_station() or tour[i].is_depot():
                required_energy[tour[i].get_id()] = 0
            else:
                previous_required_energy = required_energy[tour[i - 1].get_id()]
                required_energy[tour[i].get_id()] = previous_required_energy + self.problem.get_energy_consumption(tour[i - 1], tour[i])
                if required_energy[tour[i].get_id()] > self.problem.get_battery_capacity():
                    # skip invalid solution
                    return tour[1:-1]
        
        tour = list(reversed(tour))
        energy = self.problem.get_battery_capacity()
        i = 1
        
        """ Optimize from depot_R """
        while i < len(tour) - 1:
            if not tour[i].is_charging_station():
                energy -= self.problem.get_energy_consumption(optimal_tour[-1], tour[i])
                optimal_tour.append(tour[i])
                i += 1
                continue
            if i == len(tour) - 1:
                optimal_tour.append(tour[i])
                break
            # Calculate delta_L1
            _from_node = optimal_tour[-1]
            num_stations_in_row = 0
            original_distance = 0
            
            while i + num_stations_in_row < len(tour) - 1 and \
                    tour[i + num_stations_in_row].is_charging_station():
                original_distance += _from_node.distance(tour[i + num_stations_in_row])
                _from_node = tour[i + num_stations_in_row]
                num_stations_in_row += 1
            
            next_customer_idx = i + num_stations_in_row
            original_distance += _from_node.distance(tour[next_customer_idx]) 
            delta_L1 = original_distance - tour[i].distance(tour[next_customer_idx])
            _from_node = optimal_tour[-1]
            considered_nodes = []  
            tmp_energy = energy
            for node in tour[next_customer_idx:]:
                considered_nodes.append(node)
                if node.is_charging_station():
                    break
                tmp_energy -= self.problem.get_energy_consumption(_from_node, node)
                if tmp_energy <= 0:
                    break
                _from_node = node
            
            from_node = optimal_tour[-1]
            best_station = tour[i]
            best_station_index = 0 # index of the best station inserted right after the considered_nodes[best_station_index] node
            
            for j, node in enumerate(considered_nodes):
                to_node = node
                required_energy_node = required_energy[to_node.get_id()]
                station = self.nearest_station_back(from_node, to_node, energy, required_energy_node)
                if station != -1:
                    delta_L2 = self.problem.get_distance(from_node, station) + self.problem.get_distance(station, to_node) \
                        - self.problem.get_distance(from_node, to_node)
                    if delta_L2 < delta_L1:
                        delta_L1 = delta_L2
                        best_station = station
                        best_station_index = j

                energy -= self.problem.get_energy_consumption(from_node, to_node)
                from_node = to_node

            optimal_tour.extend(considered_nodes[:best_station_index])
            optimal_tour.append(best_station)
            i = i + num_stations_in_row + best_station_index
            energy = self.problem.get_battery_capacity() - self.problem.get_energy_consumption(best_station, tour[i])

        optimal_tour = optimal_tour[1:]
        return list(reversed(optimal_tour))
    
    def insert_energy_stations(self, tour):
        remaining_energy = dict()
        required_min_energy = dict()
        complete_tour = []
        skip_node = dict()
        
        depotID = self.problem.get_depot_id()
        remaining_energy[depotID] = self.problem.get_battery_capacity()
        tour = [self.problem.get_node_from_id(depotID)] + tour + [self.problem.get_node_from_id(depotID)]
        
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
                if to_node.is_charging_station():
                    remaining_energy[to_node.get_id()] = self.problem.get_battery_capacity()
                else:
                    remaining_energy_node = remaining_energy[from_node.get_id()] - energy_consumption
                    if to_node.get_id() in remaining_energy and remaining_energy_node > remaining_energy[to_node.get_id()]:
                        skip_node[to_node.id] = False
                    remaining_energy[to_node.get_id()] = remaining_energy_node
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
                    return tour[1:-1]
                if from_node.id in skip_node:
                    return tour[1:-1]
                skip_node[from_node.id] = True
                to_node = tour[i + 1]
                best_station = self.nearest_station(from_node, to_node, remaining_energy[from_node.get_id()])
                if best_station == -1:
                    return tour[1:-1]
                
                complete_tour.append(from_node)
                from_node = best_station
                to_node = tour[i + 1]
                remaining_energy[from_node.get_id()] = self.problem.get_battery_capacity()
                required_min_energy[from_node.get_id()] = 0
                find_charging_station = False                    
                        
        return complete_tour[1:]
    
    def local_search_2opt(self, tour):
        n = len(tour)
        tour = [self.problem.get_depot()] + tour + [self.problem.get_depot()]
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
        return tour[1:-1]

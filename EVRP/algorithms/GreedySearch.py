
from copy import deepcopy
from random import shuffle

from loguru import logger
import numpy as np
from EVRP.objects.problem import Problem
from EVRP.objects.solution import Solution


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
    
    def optimize(self, solution: Solution) -> Solution:
        solution = self.local_search(solution)
        solution = self.insert_depots(solution)
        solution = self.insert_charging_stations(solution)
        solution = self.greedy_optimize_stations(solution)
        solution.set_tour_length(self.problem.calculate_tour_length(solution))
        return solution

    def solve(self, problem, verbose=False) -> Solution:
        self.set_problem(problem)
        self.verbose = verbose
        solution = self.init_solution()
        solution = self.optimize(solution)
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
    
    def init_solution(self) -> Solution:
        solution = self.create_clustering_solution()
        solution = self.balancing_capacity(solution)
        return solution

    def create_clustering_solution(self) -> Solution:
        solution = Solution()
        node_list = self.problem.get_customer_ids()
        capacity_max = self.problem.get_capacity()
        
        shuffle(node_list)
        temp_set = set(node_list)
        idx = 0
        n_tours = 0

        while idx < len(node_list):
            if node_list[idx] not in temp_set:
                idx += 1
                continue
            
            center_node = node_list[idx]
            
            n_tours += 1
            if n_tours == self.problem.get_max_num_vehicles():
                tour = list(temp_set)
                solution.add_tour([self.problem.get_node_from_id(tour[i]) for i in range(len(tour))])
                break
            
            temp_set.remove(center_node)
            tour = [center_node]
            capacity = self.problem.get_node_from_id(center_node).get_demand()
            
            for candidate_node_id in self.nearest_matrix_distance[center_node]:
                if candidate_node_id in temp_set:
                    demand = self.problem.get_node_from_id(candidate_node_id).get_demand()
                    if capacity + demand > capacity_max:
                        break
                    tour.append(candidate_node_id)
                    temp_set.remove(candidate_node_id)
                    capacity += demand

            solution.add_tour([self.problem.get_node_from_id(tour[i]) for i in range(len(tour))])

        while n_tours < self.problem.get_max_num_vehicles():
            n_tours += 1
            solution.add_tour([])
            
        solution.set_tour_index()
        return solution
    
    def balancing_capacity(self, solution: Solution) -> Solution:
        tours = solution.get_tours()
        last_tour_idx = len(solution.get_tours()) - 1
        last_tour = tours[last_tour_idx]
        is_customer_in_last_tour = {}
        sum_demand = 0
        
        for node in last_tour:
            is_customer_in_last_tour[node.get_id()] = True
            sum_demand += node.get_demand()
            
        if sum_demand >= self.problem.get_capacity():
            return solution
            
        # choose a customer in last tour
        rd_idx = np.random.randint(0, len(last_tour))
        moving_node_id = last_tour[rd_idx].get_id()
        
        for candidate_node_id in self.nearest_matrix_distance[moving_node_id]:
            if candidate_node_id in is_customer_in_last_tour:
                continue
            
            demand = self.problem.get_node_from_id(candidate_node_id).get_demand()
            candidate_tour_index = solution.get_tour_index_by_node(candidate_node_id)
            curr_tour_demand = sum([node.get_demand() for node in tours[candidate_tour_index]])
            
            new_delta = abs((sum_demand + demand) - (curr_tour_demand - demand));
            delta = abs(sum_demand - curr_tour_demand);
            
            if new_delta < delta and sum_demand + demand <= self.problem.get_capacity():
                sum_demand += demand
                is_customer_in_last_tour[candidate_node_id] = True
                last_tour.append(self.problem.get_node_from_id(candidate_node_id))
                tours[candidate_tour_index] = [node for node in tours[candidate_tour_index] if node.get_id() != candidate_node_id]
                break
            
        solution.set_vehicle_tours(tours)
        return solution
    
    def local_search(self, solution: Solution) -> Solution:
        tours = solution.get_basic_tours()
        for i, tour in enumerate(tours):
            tours[i] = self.local_search_2opt(tour)
        solution.set_vehicle_tours(tours)
        return solution
    
    def local_search_2opt(self, tour):
        """
        Perform a local search using the 2-opt algorithm on the given tour.

        Args:
            tour (List[Node]): The initial tour to be optimized.

        Returns:
            List[Node]: The optimized tour after applying the 2-opt algorithm.

        Description:
            The 2-opt algorithm is a heuristic optimization algorithm for finding the shortest path in a graph.
            It repeatedly searches for a pair of edges that, if reversed, would result in a shorter path.
            This process is repeated until no further improvements can be made.

            The tour is initially extended by adding the depot node at the beginning and the end.
            Then, the algorithm iterates over all possible pairs of edges within the tour.
            For each pair, a new tour is created by reversing the order of the edges.
        """
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

    
    def insert_depots(self, solution: Solution) -> Solution:
        vehicle_tours = solution.get_basic_tours()
        for i, tour in enumerate(vehicle_tours):
            vehicle_tours[i] = self.insert_depot_for_single_tour(tour)
        solution.set_vehicle_tours(vehicle_tours)
        return solution
    
    def insert_depot_for_single_tour(self, tour):
        """
        Inserts depots into a tour based on the tour's capacity and demand of nodes.
        The function ensures that the capacity constraint is satisfied for each vehicle.
        If the demand of a node is greater than the vehicle's capacity, the function inserts a depot,
        the vehicle back to the depot to get a new batch of goods and then continues the tour.
        """
        _tour = [self.problem.get_depot()]
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
                
        if not _tour[-1].is_depot():
            _tour.append(self.problem.get_depot())

        """
        Greedy optimization for depot position:
        
        Optimize the depot position if the vehicle needs to return to
        the depot during transportation due to exceeding capacity.
        depot -> c1 -> c2 -> depot -> c3 -> depot
        => 
        depot -> c1 -> depot -> c2 -> c3 -> depot
        if distance(c1, depot) + distance(depot, c2) + distance(c2, c3) < distance(c1, c2) + distance(c2, depot) + distance(depot, c3)
        and demand(c2) + demand(c3) <= max_capacity 
        then swap c2 and depot
        """
        
        curr_capacity = 0
        
        for i in reversed(range(len(_tour))):
            if i < 2 or i == len(_tour) - 1:
                continue
            
            node = _tour[i]
            if node.is_customer():
                curr_capacity += node.get_demand()
                
            if node.is_depot():
                c1 = _tour[i - 2]
                c2 = _tour[i - 1]
                depot = _tour[i]
                c3 = _tour[i + 1]
                
                d1 = self.problem.get_distance(c1, c2)
                d2 = self.problem.get_distance(c2, depot)
                d3 = self.problem.get_distance(depot, c3)
                
                new_d1 = self.problem.get_distance(c1, depot)
                new_d2 = self.problem.get_distance(depot, c2)
                new_d3 = self.problem.get_distance(c2, c3)
                
                demand_condition = c2.get_demand() + c3.get_demand() <= self.problem.get_capacity()
                distance_condition = d1 + d2 + d3 > new_d1 + new_d2 + new_d3
                
                if demand_condition and distance_condition:
                    _tour[i] = c2
                    _tour[i - 1] = depot
                    curr_capacity += c2.get_demand()
                else:
                    curr_capacity = 0
                
        return _tour
    
    def insert_charging_stations(self, solution: Solution) -> Solution:
        vehicle_tours = solution.get_tours()
        for i, tour in enumerate(vehicle_tours):
            vehicle_tours[i] = self.insert_charging_station_for_single_tour(tour)
        solution.set_vehicle_tours(vehicle_tours)
        return solution
    
    def insert_charging_station_for_single_tour(self, tour):
        remaining_energy = dict()
        min_required_energy = dict()
        complete_tour = []
        skip_node = dict()
        
        depotID = self.problem.get_depot_id()
        remaining_energy[depotID] = self.problem.get_battery_capacity()
        """
        At the current customer node, calculate the minimum energy required for an 
        electric vehicle to reach the nearest charging station.
        """
        for node in tour:
            nearest_station = self.nearest_station(node, node, self.problem.get_battery_capacity())
            min_required_energy[node.get_id()] = self.problem.get_energy_consumption(node, nearest_station)
        
        if len(tour) < 2:
            return tour
        
        i = 0
        from_node = tour[0]
        to_node = tour[1]
        
        while i < len(tour) - 1:
            
            """go ahead util energy is not enough for visiting the next node""" 
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
            """
            If there is enough energy, find the nearest station.
            If there is not enough energy to reach the nearest station, go back to 
            the previous node and find the next nearest station from there.
            """
            while find_charging_station:
                while i > 0 and min_required_energy[from_node.get_id()] > remaining_energy[from_node.get_id()]:
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
                min_required_energy[from_node.get_id()] = 0
                find_charging_station = False                    
                      
        if not complete_tour[-1].is_depot():
            complete_tour.append(self.problem.get_depot())     
        
        return complete_tour
    
    def greedy_optimize_stations(self, solution: Solution) -> Solution:
        vehicle_tours = solution.get_tours()
        for i, tour in enumerate(vehicle_tours):
            vehicle_tours[i] = self.greedy_optimize_station_for_single_tour(tour)
        solution.set_vehicle_tours(vehicle_tours)
        return solution
    
    def greedy_optimize_station_for_single_tour(self, tour):
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
        
         # calculate required energy to reach node i if vehicle travel in the reverse tour
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
                    return tour
        
        """ Travel from depot_R """
        tour = list(reversed(tour))
        energy = self.problem.get_battery_capacity()
        i = 1
        
        while i < len(tour) - 1:
            if not tour[i].is_charging_station():
                energy -= self.problem.get_energy_consumption(optimal_tour[-1], tour[i])
                optimal_tour.append(tour[i])
                i += 1
                continue
            
            if tour[i].is_depot():
                energy = self.problem.get_battery_capacity()
                optimal_tour.append(tour[i])
                i += 1
                continue
                
            # tour[i] is a charging station, try to instead it with a better one
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

        if not optimal_tour[-1].is_depot():
            optimal_tour.append(self.problem.get_depot())
            
        return list(reversed(optimal_tour))
    
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
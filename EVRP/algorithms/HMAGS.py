from bisect import bisect_right
from copy import deepcopy
import random
from typing import List
from random import shuffle, randint, uniform, random, choice

import numpy as np

from EVRP.solution import Solution
from EVRP.problem import Problem
from EVRP.algorithms.GreedySearch import GreedySearch

import pandas as pd
import matplotlib.pyplot as plt

class HMAGS():
    """
    Here is a basic implementation of a GA class in Python for solving the VRP problem. 
    It takes a `Problem` object, population size, generations, crossover probability, mutation probability, and elite size as input arguments. 
    It has a `run` method which returns the best solution found by the GA after the specified number of generations. 
    It also has several helper functions for initializing the population, obtaining the elite, tournament selection, crossover, and mutation. 
    Note that this implementation only provides a basic framework for a GA and may need to be modified or extended depending on the specific VRP problem you are attempting to solve.

    """
    def __init__(self, population_size: int, generations: int, crossover_prob: float,
                 mutation_prob: float, elite_size: int):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_size = elite_size
        self.history = {
            'mean_fit': [],
            'min_fit': []
        }
        self.ranks = []
        self.population = []
    
    def set_problem(self, problem: Problem):
        self.problem = problem
        self.gs = GreedySearch()
        self.gs.set_problem(problem)
        self.population = self._initial_population()

    def free(self):
        self.history = {
            'mean_fit': [],
            'min_fit': []
        }
        self.ranks = []
        self.population = self._initial_population()

    def selection(self, population: List[Solution], num: int) -> List[Solution]:
        new_pop = []
        self.compute_rank(population)
        for i in range(num):
            indv = self.choose_by_rank(population)
            new_pop.append(indv)
        return new_pop

    def run(self, verbose=False) -> Solution:
        for i in range(self.generations):
            new_population = []
            self.compute_rank(self.population)
            while len(new_population) < self.population_size:
                child = self.choose_by_rank(self.population)
                if random() < self.mutation_prob:
                    if random() < 0.5:
                        child = self.hmm_mutate(child)
                    else:
                        child = self.hsm_mutate(child)
                child = self.gs.optimize(child)
                new_population.append(child)
            
            elites = self._get_elite(self.population)
            self.population = elites + self.selection(new_population, self.population_size)
            valids = [self.problem.check_valid_solution(indv) for indv in self.population]
            mean_fit = np.mean([indv.get_tour_length() for i, indv in enumerate(self.population) if valids[i]]) 
            min_fit = np.min([indv.get_tour_length() for i, indv in enumerate(self.population) if valids[i]])
            if verbose:
                print(f"Generation: {i}, mean fit: {mean_fit}, min fit: {min_fit}")
            self.history['mean_fit'].append(mean_fit)
            self.history['min_fit'].append(min_fit)
            self.plot_history('./EVRP/figures/history.png')
        return self.population[np.argmin([indv.get_tour_length() for indv in self.population])]
    
    def plot_history(self, path):
        df = pd.DataFrame(self.history)
        df.plot()
        plt.savefig(path)
        plt.close()

    def _initial_population(self) -> List[Solution]:
        return [self.gs.optimize(self.gs.create_solution()) for _ in range(self.population_size)]

    def _get_elite(self, population: List[Solution]) -> List[Solution]:
        return sorted(population, key=lambda s: s.get_tour_length())[:self.elite_size]

    def _tournament_selection(self):
        return self.population[randint(0, len(self.population) - 1)]

    def distribute_crossover(self, parent_1: Solution, parent_2: Solution) -> None:
        rd_node = choice(self.problem.get_all_customers())
        id1 = parent_1.tour_index[rd_node.get_id()]
        id2 = parent_2.tour_index[rd_node.get_id()]
        have = np.zeros(self.problem.get_num_dimensions())
        alens = set()
        child_1 = Solution()
        child_2 = Solution()
        parent_1_tours = parent_1.get_basic_tours()
        parent_2_tours = parent_2.get_basic_tours()
        index = 0
        
        for node in parent_1_tours[id1]:
            alens.add(node.get_id())
            have[node.get_id()] = 1
            
        for node in parent_2_tours[id2]:
            alens.add(node.get_id())
            have[node.get_id()] = 1
        
        alen_list = list(alens)

        # drop random
        for idx in range(len(alen_list)):
            if random() > self.crossover_prob:
                alens.remove(alen_list[idx])
                have[alen_list[idx]] = 0

        index = 0
        alens = list(alens)
        shuffle(alens)

        for idx_tour in range(len(parent_1_tours)):
            for idx in range(len(parent_1_tours[idx_tour])):
                if have[parent_1_tours[idx_tour][idx].get_id()]:
                    parent_1_tours[idx_tour][idx] = self.problem.get_node_from_id(alens[index])
                    index += 1
            child_1.add_tour(parent_1_tours[idx_tour])

        index = 0
        shuffle(alens)
        for idx_tour in range(len(parent_2_tours)):
            for idx in range(len(parent_2_tours[idx_tour])):
                if have[parent_2_tours[idx_tour][idx].get_id()]:
                    parent_2_tours[idx_tour][idx] = self.problem.get_node_from_id(alens[index])
                    index += 1
            child_2.add_tour(parent_2_tours[idx_tour])

        child_1.set_tour_index()
        child_2.set_tour_index()
        
        return child_1, child_2

    def hmm_mutate(self, solution: Solution) -> Solution:
        solution.set_tour_index()
        tours = solution.get_basic_tours()

        if len(tours) == 1:
            return solution
        
        tours = solution.get_basic_tours()
        rd_tour_idx = choice(range(len(tours)))
        if len(tours[rd_tour_idx]) == 0:
            return solution
        rd_customer_idx = choice(range(len(tours[rd_tour_idx])))
        rd_customer = tours[rd_tour_idx][rd_customer_idx]

        tour_idx = solution.tour_index[rd_customer.get_id()]
        mm_customer_list = []
        for customer_id in self.gs.nearest_matrix_distance[rd_customer.get_id()]:
            if solution.tour_index[customer_id] != tour_idx:
                mm_customer_list.append(self.problem.get_node_from_id(customer_id))
                if len(mm_customer_list) > 3:
                    break

        mm_customer = choice(mm_customer_list)
        mm_customer_tour_idx = solution.tour_index[mm_customer.get_id()]
        
        mm_customer_idx = -1
        for idx in range(len(tours[mm_customer_tour_idx])):
            if tours[mm_customer_tour_idx][idx].get_id() == mm_customer.get_id():
                mm_customer_idx = idx
                break
        
        tours[tour_idx].append(mm_customer)
        tours[mm_customer_tour_idx].pop(mm_customer_idx)
        
        return Solution(tours)


    def hsm_mutate(self, solution: Solution) -> Solution:
        solution.set_tour_index()
        tours = solution.get_basic_tours()

        if len(tours) == 1:
            return solution
        
        tours = solution.get_basic_tours()
        rd_tour_idx = choice(range(len(tours)))

        if len(tours[rd_tour_idx]) == 0:
            return solution
        
        rd_customer_idx = choice(range(len(tours[rd_tour_idx])))
        rd_customer = tours[rd_tour_idx][rd_customer_idx]

        tour_idx = solution.tour_index[rd_customer.get_id()]
        mm_customer_list = []
        for customer_id in self.gs.nearest_matrix_distance[rd_customer.get_id()]:
            if solution.tour_index[customer_id] != tour_idx:
                mm_customer_list.append(self.problem.get_node_from_id(customer_id))
                if len(mm_customer_list) > 3:
                    break

        mm_customer = choice(mm_customer_list)
        mm_customer_tour_idx = solution.tour_index[mm_customer.get_id()]
        
        mm_customer_idx = -1
        for idx in range(len(tours[mm_customer_tour_idx])):
            if tours[mm_customer_tour_idx][idx].get_id() == mm_customer.get_id():
                mm_customer_idx = idx
                break
        
        tours[tour_idx][rd_customer_idx], tours[mm_customer_tour_idx][mm_customer_idx] = \
            tours[mm_customer_tour_idx][mm_customer_idx], tours[tour_idx][rd_customer_idx]
        
        return Solution(tours)
    
    def compute_rank(self, pop):
        _sum = 0
        self.ranks = []
        fit_min = min([pop[i].get_tour_length() for i in range(len(pop))])
        fit_max = max([pop[i].get_tour_length() for i in range(len(pop))])
        for i in range(len(pop)):
            temp_fit = ((fit_max - pop[i].get_tour_length()) / (fit_max - fit_min + 1e-6)) ** 2
            _sum += temp_fit
            self.ranks.append(temp_fit)
        for i in range(len(pop)):
            self.ranks[i] /= _sum
            if i > 0:
                self.ranks[i] += self.ranks[i - 1]

    def choose_by_rank(self, population: List[Solution]) -> int:
        prob = random()
        return population[bisect_right(self.ranks, prob, hi=len(population)) - 1]

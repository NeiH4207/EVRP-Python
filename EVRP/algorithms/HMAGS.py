from copy import deepcopy
import random
from typing import List
from random import shuffle, randint, uniform, random, choice

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
    def __init__(self, problem: Problem, population_size: int, generations: int, crossover_prob: float,
                 mutation_prob: float, elite_size: int):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_size = elite_size
        self.gs = GreedySearch(problem)
        self.history = {
            'best_fit': [],
            'valid_best_fit': []
        }

    def run(self) -> Solution:
        population = self._initial_population()
        elite = self._get_elite(population)
        for i in range(self.generations):
            new_population = []
            for j in range(self.population_size):
                if uniform(0, 1) < self.crossover_prob:
                    p1, p2 = self._tournament_selection(population), self._tournament_selection(population)
                    child = self.distribute_crossover(p1, p2)
                else:
                    child = self.problem.random_solution()
                child = self._mutate(child)
                child = self.gs.optimize(child)
                new_population.append(child)
            population = self._get_elite(population) + new_population
            valids = [self.problem.check_valid_solution(indv) for indv in population]
            best_fit = min([indv.get_tour_length() for indv in population])
            valid_best_fit = None
            if sum(valids) > 0:
                valid_best_fit = min([indv.get_tour_length() for i, indv in enumerate(population) if valids[i]])
                
            self.history['best_fit'].append(best_fit)
            self.history['valid_best_fit'].append(valid_best_fit)
            self.plot_history('./EVRP/figures/history.png')
            elite = self._get_elite(population)
        return elite[0]
    
    def plot_history(self, path):
        df = pd.DataFrame(self.history)
        df.plot()
        plt.savefig(path)
        plt.close()

    def _initial_population(self) -> List[Solution]:
        return [self.problem.random_solution() for _ in range(self.population_size)]

    def _get_elite(self, population: List[Solution]) -> List[Solution]:
        return sorted(population, key=lambda s: s.get_tour_length())[:self.elite_size]

    def _tournament_selection(self, population: List[Solution]) -> Solution:
        return population[randint(0, len(population) - 1)]

    def distribute_crossover(self, parent_1: Solution, parent_2: Solution) -> None:
        rd_node = choice(self.problem.get_all_customers())
        id1 = parent_1.tour_index[rd_node.get_id()]
        id2 = parent_2.tour_index[rd_node.get_id()]
        have = dict()
        alens = dict()
        child_1 = deepcopy(parent_1)
        child_2 = deepcopy(parent_2)
        parent_1_vehicle_tours = parent_1.get_vehicle_tours(skip_depot=True, full=False)
        parent_2_vehicle_tours = parent_2.get_vehicle_tours(skip_depot=True, full=False)
        index = 0
        have[self.problem.get_depot().get_id()] = 0
        
        for node in parent_1_vehicle_tours[id1]:
            alens[index] = node.get_id()
            index += 1
            have[node.get_id()] = 1
            
        for node in parent_2_vehicle_tours[id2]:
            if node.get_id() not in have or have[node.get_id()] == 0:
                alens[index] = node.get_id()
                index += 1
                have[node.get_id()] = 1
                
        tours_1 = parent_1.to_array()
        tours_2 = parent_2.to_array()
        index2 = 0
        for idx in self.problem.get_customer_ids():
            if have[tours_1[idx]]:
                tours_1[idx] = alens[index - 1]
                index -= 1
            if have[tours_2[idx]]:
                tours_2[idx] = alens[index2]
                index2 += 1
        node_tours_1 = [self.problem.get_node_from_id(id) for id in tours_1]
        node_tours_2 = [self.problem.get_node_from_id(id) for id in tours_2]
        child_1.set(node_tours_1)
        child_2.set(node_tours_2)
        
        if random() < 0.5:
            return child_1
        else:
            return child_2

    def _mutate(self, solution: Solution) -> Solution:
        tours = solution.to_array()
        rd_id_1 = randint(1, len(tours) - 2)
        rd_id_2 = randint(1, len(tours) - 2)
        tours[rd_id_1], tours[rd_id_2] = tours[rd_id_2], tours[rd_id_1]
        tours = [self.problem.get_node_from_id(id) for id in tours]
        solution.set(tours)
        return solution
                

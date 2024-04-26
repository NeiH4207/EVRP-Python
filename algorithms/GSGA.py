from bisect import bisect_right
from copy import deepcopy
import random
from typing import List
from random import shuffle, randint, uniform, random, choice

import numpy as np

from objects.solution import Solution
from objects.problem import Problem
from algorithms.GreedySearch import GreedySearch

import pandas as pd
import matplotlib.pyplot as plt

class GSGA():
    """
    Here is a basic implementation of a GA class in Python for solving the VRP problem. 
    It takes a `Problem` object, population size, generations, crossover probability, mutation probability, and elite size as input arguments. 
    It has a `run` method which returns the best solution found by the GA after the specified number of generations. 
    It also has several helper functions for initializing the population, obtaining the elite, tournament selection, crossover, and mutation. 
    Note that this implementation only provides a basic framework for a GA and may need to be modified or extended depending on the specific VRP problem you are attempting to solve.

    """
    def __init__(self, population_size: int, generations: int, crossover_prob: float,
                 mutation_prob: float, elite_rate: int):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_size = int(population_size * elite_rate)
        self.history = {
            'Mean Pop Fitness': [],
            'Best Pop Fitness': []
        }
        self.ranks = []
        self.population = []
        self.gs = GreedySearch()
    
    def set_problem(self, problem: Problem):
        self.problem = problem
        self.gs.set_problem(problem)
        self.population = self._initial_population()

    def free(self):
        self.history = {
            'Mean Pop Fitness': [],
            'Best Pop Fitness': []
        }
        self.ranks = []
        self.population = self._initial_population()

    def selection(self, population: List[Solution], num: int) -> List[Solution]:
        population = sorted(population, key=lambda s: s.get_tour_length())
        elites = population[:self.elite_size]
        self.compute_rank(population[self.elite_size:])
        new_pop = elites
        
        while len(new_pop) < num:
            indv = self.choose_by_rank(population[self.elite_size:])
            new_pop.append(indv)
            
        return new_pop

    def solve(self, problem: Problem, verbose=False, plot_path=None) -> Solution:
        
        self.set_problem(problem)
        
        for i in range(self.generations):
            new_population = []
            self.compute_rank(self.population)
            while len(new_population) < self.population_size:
                child_1 = self.choose_by_rank(self.population)
                child_2 = self.choose_by_rank(self.population)
                
                child_1, child_2 = self.distribute_crossover(child_1, child_2)
                
                if random() < self.mutation_prob:
                    if random() < 1/3:
                        child_1 = self.hmm(child_1)
                    elif random() < 2/3:
                        child_1 = self.hsm(child_1)
                    else:
                        child_1 = self.hsm2(child_1)
                
                if random() < self.mutation_prob:
                    if random() < 1/3:
                        child_2 = self.hmm(child_2)
                    elif random() < 2/3:
                        child_2 = self.hsm(child_2)
                    else:
                        child_2 = self.hsm2(child_2)
                        
                child_1 = self.gs.optimize(child_1)
                child_1 = self.gs.optimize(child_2)
                if child_1.get_tour_length() < child_2.get_tour_length():
                    new_population.append(child_1)
                else:
                    new_population.append(child_2)
            
            self.population = self.selection(self.population + new_population, self.population_size)
            
            valids = [self.problem.check_valid_solution(indv) for indv in self.population]
            mean_fit = np.mean([indv.get_tour_length() for i, indv in enumerate(self.population) if valids[i]]) 
            best_fit = np.min([indv.get_tour_length() for i, indv in enumerate(self.population) if valids[i]])
            
            if verbose:
                print(f"Generation: {i}, mean fit: {mean_fit}, min fit: {best_fit}")
                
            self.history['Mean Pop Fitness'].append(mean_fit)
            self.history['Best Pop Fitness'].append(best_fit)
            
            if plot_path is not None:
                self.plot_history(plot_path)
                best_sol = self.population[0]
                self.problem.plot(best_sol, plot_path.replace('.png', '_solution.png'))
                
        return self.population[np.argmin([indv.get_tour_length() for indv in self.population])]
    
    def plot_history(self, path):
        df = pd.DataFrame(self.history)
        df.plot()
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Convergence trending ({})'.format(self.problem.get_name()))
        plt.legend()
        plt.grid()
        plt.savefig(path)
        plt.close()

    def _initial_population(self) -> List[Solution]:
        return [self.gs.optimize(self.gs.init_solution()) for _ in range(self.population_size)]

    def _get_elite(self, population: List[Solution]) -> List[Solution]:
        return sorted(population, key=lambda s: s.get_tour_length())[:self.elite_size]

    def _tournament_selection(self):
        return self.population[randint(0, len(self.population) - 1)]

    def distribute_crossover(self, parent_1: Solution, parent_2: Solution) -> None:
        parent_1_tours = parent_1.get_basic_tours()
        parent_2_tours = parent_2.get_basic_tours()
        
        if random() < self.crossover_prob:
            rd_node_id = choice(self.problem.get_all_customers()).get_id()
            id1 = parent_1.tour_index[rd_node_id]
            id2 = parent_2.tour_index[rd_node_id]
            have = np.zeros(self.problem.get_num_dimensions())
            alens = set()
            index = 0
            
            for node in parent_1_tours[id1]:
                alens.add(node.get_id())
                have[node.get_id()] = 1
                
            for node in parent_2_tours[id2]:
                alens.add(node.get_id())
                have[node.get_id()] = 1

            index = 0
            alens = list(alens)
            shuffle(alens)
            
            child_1_tours = []

            for idx_tour in range(len(parent_1_tours)):
                for idx in range(len(parent_1_tours[idx_tour])):
                    if have[parent_1_tours[idx_tour][idx].get_id()]:
                        parent_1_tours[idx_tour][idx] = self.problem.get_node_from_id(alens[index])
                        index += 1
                        
                child_1_tours.append(parent_1_tours[idx_tour])

            index = 0
            shuffle(alens)
            child_2_tours = []
            
            for idx_tour in range(len(parent_2_tours)):
                for idx in range(len(parent_2_tours[idx_tour])):
                    if have[parent_2_tours[idx_tour][idx].get_id()]:
                        parent_2_tours[idx_tour][idx] = self.problem.get_node_from_id(alens[index])
                        index += 1
                        
                child_2_tours.append(parent_2_tours[idx_tour])

            return Solution(child_1_tours), Solution(child_2_tours)
        else:
            return parent_1, parent_2
        

    def hmm(self, solution: Solution) -> Solution:
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


    def hsm(self, solution: Solution) -> Solution:
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
    
    def hsm2(self, solution: Solution) -> Solution:
        solution.set_tour_index()
        tours = solution.get_basic_tours()

        if len(tours) == 1:
            return solution
        
        tours = solution.get_basic_tours()
        rd_tour_idx = choice(range(len(tours)))

        if len(tours[rd_tour_idx]) == 0:
            return solution
        
        rd_customer_idx_1 = choice(range(len(tours[rd_tour_idx])))
        rd_customer_idx_2 = choice(range(len(tours[rd_tour_idx])))

        tours[rd_tour_idx][rd_customer_idx_1], tours[rd_tour_idx][rd_customer_idx_2] = \
            tours[rd_tour_idx][rd_customer_idx_2], tours[rd_tour_idx][rd_customer_idx_1]
        
        return Solution(tours)
    
    def compute_rank(self, pop):
        _sum = 0
        self.ranks = []
        fit_min = min([pop[i].get_tour_length() for i in range(len(pop))])
        fit_max = max([pop[i].get_tour_length() for i in range(len(pop))])
        for i in range(len(pop)):
            temp_fit = ((fit_max - pop[i].get_tour_length()) / (fit_max - fit_min + 1e-6)) ** np.e
            _sum += temp_fit
            self.ranks.append(temp_fit)
        for i in range(len(pop)):
            self.ranks[i] /= _sum
            if i > 0:
                self.ranks[i] += self.ranks[i - 1]

    def choose_by_rank(self, population: List[Solution]) -> int:
        prob = random()
        return population[bisect_right(self.ranks, prob, hi=len(population)) - 1]

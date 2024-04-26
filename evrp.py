import argparse
import os

import numpy as np
from algorithms.GSGA import GSGA
from objects.problem import Problem
from algorithms.GreedySearch import GreedySearch
from src.utils import get_problem_name, logger
import random

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem-path', type=str, default='./benchmarks/evrp-2019/E-n22-k4.evrp')
    parser.add_argument('-a', '--algorithm', type=str, default='GSGA')
    parser.add_argument('-o', '--result-path', type=str, default='./results/GSGA/')
    parser.add_argument('-n', '--nruns', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    

if __name__ == "__main__":
    args = argparser()
    set_random_seed(args.seed)
    problem_name = get_problem_name(args.problem_path)
    problem = Problem(args.problem_path)
    
    if args.algorithm == 'GreedySearch':
        algorithm = GreedySearch()
        
        kwargs = {
            'problem': problem,
            'verbose': True
        }
        
    elif args.algorithm == 'GSGA':
        algorithm = GSGA(population_size=100, generations=200, 
                          crossover_prob=0.85, mutation_prob=0.75, elite_rate=0.2)
        
        kwargs = {
            'problem': problem,
            'verbose': True,
            'plot_path': os.path.join(args.result_path, problem_name, 'fitness_history.png')
        }
    else:
        raise ValueError(f'Invalid algorithm {args.algorithm}')
        
    results = []
    
    for i in range(args.nruns):
        result_path = os.path.join(args.result_path, problem_name)
        result_file = os.path.join(result_path, f"run_{i}.txt")
        figure_file = os.path.join(result_path, f"run_{i}.png")
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            
        solution = algorithm.solve(**kwargs)
        
        if problem.check_valid_solution(solution, verbose=True):
            tour_length = solution.get_tour_length()
            with open(result_file, 'w') as f:
                f.write(f"{tour_length}\n")
                
            results.append(tour_length)
            algorithm.free()
            problem.plot(solution, figure_file)
            print(solution)
        else:
            logger.error('Invalid solution')
            results.append(np.inf)
            
            with open(result_file, 'w') as f:
                f.write(f"{np.inf}\n")
            
            
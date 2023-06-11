import argparse
import os

import numpy as np
from EVRP.algorithms.HMAGS import HMAGS
from EVRP.problem import Problem
from EVRP.algorithms.GreedySearch import GreedySearch
from EVRP.utils import logger, get_problem_list

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='./EVRP/benchmark-2019/')
    parser.add_argument('--result-path', type=str, default='./results/GreedySearch/')
    parser.add_argument('--algorithm', type=str, default='GreedySearch')
    parser.add_argument('--nruns', type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argparser()
    problem_list = get_problem_list(args.dataset_path)
    for problem_name in problem_list:
        problem = Problem(problem_name, dataset_path=args.dataset_path)
        gs = GreedySearch(problem)
        best = None
        results = []
        
        stats_path = os.path.join(args.result_path, 'Stats', problem_name)
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)

        with open(os.path.join(stats_path, problem_name + '.txt'), 'w') as f:
            for i in range(args.nruns):
                solution = gs.run()
                if problem.check_valid_solution(solution):
                    if best is None or best.get_tour_length() > solution.get_tour_length():
                        best = solution
                    results.append(solution.get_tour_length())
                f.write(f"{solution.get_tour_length()}\n")
            if len(results) > 0:
                f.write(f"Best: {best.get_tour_length()}, Mean: {sum(results)/len(results)}, Std: {np.std(results)}\n")
            
            f.close()
        
        figure_result_path = os.path.join(args.result_path, 'Figures', problem_name)
        if not os.path.exists(figure_result_path):
            os.makedirs(figure_result_path)
        figure_result_path = os.path.join(figure_result_path, 'solution.png')
        
        problem.plot(best, figure_result_path)
    # hmags = HMAGS(problem, population_size=50, generations=100, crossover_prob=0.8, mutation_prob=0.2, elite_size=10)
    # hmags.run()
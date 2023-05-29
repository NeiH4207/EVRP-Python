
import logging
from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler()]
)
log = logging.getLogger("rich")
    
from EVRP.problem import Problem
from EVRP.algorithms.HMAGS import HMAGS


if __name__ == "__main__":
    problem = Problem('E-n22-k4', dataset_path='./EVRP/benchmark-2019/')
    solution = problem.random_solution()
    solution.print()
    hmags = HMAGS(problem, population_size=50, generations=100, crossover_prob=0.8, mutation_prob=0.2, elite_size=10)
    hmags.run()

import logging
from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler()]
)
log = logging.getLogger("rich")
    
from EVRP.problem import Problem
from EVRP.GreedySearch import GreedySearch


if __name__ == "__main__":
    problem = Problem('E-n29-k4-s7', dataset_path='./EVRP/benchmark-2022/')
    solution = problem.create_random_solution()
    gs = GreedySearch(problem)
    solution = gs.optimize(solution)
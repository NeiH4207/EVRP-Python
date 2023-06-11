from loguru import logger
import sys, os
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>", backtrace=False, diagnose=False)
os.remove(".log")
logger.add(".log")

def get_problem_list(dataset_path):
    problem_list = []
    for file in os.listdir(dataset_path):
        if file.endswith(".evrp"):
            problem_list.append(file.split(".")[0])
    return sorted(problem_list, key=lambda x: int(x.split("-")[1][1:]))
from loguru import logger
import os
if os.path.exists(".log"):
    os.remove(".log")
logger.add(".log")

def get_problem_name(problem_path):
    return problem_path.split("/")[-1].split(".")[0]

def get_problem_list(dataset_path):
    problem_list = []
    for file in os.listdir(dataset_path):
        if file.endswith(".evrp"):
            problem_list.append(get_problem_name(file))
    return sorted(problem_list, key=lambda x: int(x.split("-")[1][1:]))
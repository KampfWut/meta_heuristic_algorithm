# !/usr/bin/env python3
"""
    author: Apeiria
    contact: apeiriaStar@outlook.com
    file: TSP_solve.py
    function: solve the traveling salesman problem
"""

# ---------------------------------- import --------------------------------- #
import os
import sys
import random
import ray

# Problem import
import Problem.TSP as TSP
# Method import
import Method.SimulatedAnnealingAlgorithm as SA
import Method.GeneticAlgorithm as GA

# ------------------------------ Main Def ----------------------------------- #
def TSP_solve(method_name,
              use_load,
              file_output=True):
    """TSP problem solving

    Args:
        method_name (string): Method name
        use_load (bool): Load existing results using load
        file_output (bool, optional): file output tag. Defaults to True.

    Raises:
        Exception: illegal parameter
    """
    # Ray set
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_path + "/meta_heuristic_algorithm/Method")
    sys.path.append(root_path + "/meta_heuristic_algorithm/Problem")

    node_num = 30
    paralle_flag = True if method_name in ["PAGA"] else False
    tsp = TSP.TSP(node_num, 100)
    if use_load == False:
        tsp.generate()
        tsp.save()
    else:
        tsp.load()
    solve = list(range(0, node_num))
    random.shuffle(solve)

    # Log
    log_file = open("./Log/TSP_{}.log".format(method_name), 'a')
    if file_output == True:
        sys.stdout = log_file
    
    # Algorithm selection
    print(">> Start Training by {} ...".format(method_name))
    if method_name == "SA":
        # Simulated annealing algorithm
        out = SA.SimulatedAnnealing_SerialProcess(inti_solve=solve,
                                                  fitFunction=tsp.fitFunction,
                                                  fitFunctionInput=[],
                                                  gamma=0.9995,
                                                  MaximumIteration=30000,
                                                  EarlyStopLimited=5000,
                                                  output=True)
    elif method_name == "GA":
        # Genetic algorithm
        out = GA.GeneticAlgorithm_SerialProcess(population_size=500,
                                                individual_member=solve,
                                                fitFunction=tsp.fitFunction,
                                                fitFunctionInput=[],
                                                maximum_iteration=5000,
                                                cross_prob=0.5,
                                                mutation_prob=0.6,
                                                mutimutaion_prob=0.1,
                                                select_mode="championships",
                                                select_candidate_parameters=[0.05],
                                                elite_strategy=True,
                                                preserve_original_population=False,
                                                output=True)
    elif method_name == "PAGA":
        # Paralle Genetic algorithm
        out = GA.GeneticAlgorithm_ParalleProcess(multi_population_num=30,
                                                 serial_population_size=100,
                                                 individual_member=solve,
                                                 fitFunction=tsp.fitFunction,
                                                 fitFunctionInput=[],
                                                 multi_maximum_iteration=20,
                                                 transfer_scale=0.2,
                                                 transfer_method="random",
                                                 replacement_interval=5,
                                                 replacement_scale=10,
                                                 serial_maximum_iteration=50,
                                                 cross_prob_list=[0.3, 0.6],
                                                 mutation_prob_list=[0.8],
                                                 mutimutaion_prob_list=[0, 0.1],
                                                 select_mode_list=["greedy", "championships"],
                                                 select_candidate_parameters_list=[[], [0.05]],
                                                 elite_strategy_list=[True, False],
                                                 preserve_original_population_list=[True, False],
                                                 output=True)
    else:
        raise Exception("[ERROR] Wrong Agent Choose!")
    
    # Result display
    log_file.close()
    solve, fit = out
    tsp.setForDraw(solve, fit)
    tsp.show(method_name)  
    
    return

###############################################################################
if __name__ == "__main__":
    method_name = "PAGA"
    
    # 1. Ray Set
    if method_name in ["PAGA"]:
        ray.init(address='auto', _redis_password='5241590000000000')
    
    # 2. Main Process
    TSP_solve(method_name=method_name,
              use_load=True,
              file_output=True)
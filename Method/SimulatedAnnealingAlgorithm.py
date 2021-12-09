# !/usr/bin/env python3
"""
    author: Apeiria
    contact: apeiriaStar@outlook.com
    file: SimulatedAnnealingAlgorithm.py
    function: Simulated Annealing Algorithm correlation
"""

# ---------------------------------- import --------------------------------- #
import time
import math
import random

from Method.GenerationSolve import GenerationSolve

# ---------------------------------- Tool Class ----------------------------- #
class SimulatedAnnealing_Base(object):
    
    """Simulated Annealing Algorithm, Standard tool class"""
    
    @classmethod
    def __param_check(cls, param,
                      type,
                      param_name):
        """Parameter validity check 

        Args:
            param (Any): Param
            type (Any): Param type
            param_name (string): Param Name
        """
        assert isinstance(param, type), "[ERROR] {} need {}".format(param_name, type)
        return
    
    
    @classmethod
    def Mutation(cls, 
                 individual):
        """Genetic algorithm, mutation process

        Args:
            individual (list): individual individual

        Returns:
            list: new individual
        """
        # Param Check
        cls.__param_check(individual, list, "individual")
        assert len(individual) > 0, "[ERROR] individual is null."
        for i in range(0, len(individual)):
            cls.__param_check(individual[i], int, "individual element")
        
        # Main Process
        if len(individual) > 10:
            p = random.random()
            if p > 0.5:
                # 50% Swap an individual element
                new_individual = GenerationSolve.get_NewSeq_RandomSwitchOne(individual)
            elif p > 0.3:
                # 20% Partial exchange
                new_individual = GenerationSolve.get_NewSeq_RandomSwitchPart(individual)
            elif p > 0.1:
                # 20% Make a jump switch
                new_individual = GenerationSolve.get_NewSeq_PartJump(individual)
            else:
                # 10% Partial inversion
                new_individual = GenerationSolve.get_NewSeq_PartReverse(individual)
        else:
            new_individual = GenerationSolve.get_NewSeq_RandomSwitchOne(individual)
        
        return new_individual
    
# ------------------------------ Main Def ----------------------------------- #
def SimulatedAnnealing_SerialProcess(inti_solve,
                                     fitFunction,
                                     fitFunctionInput,
                                     gamma=0.9997,
                                     MaximumIteration=150000,
                                     EarlyStopLimited=50000,
                                     output=False):
    """Simulated annealing algorithm, serial mainstream, Maximization process

    Args:
        inti_solve (list): Initial solution
        fitFunction (function): Fitness calculation function
        fitFunctionInput (list): Auxiliary input of fitness calculation function
        gamma (float, optional): Temperature decay rate. Defaults to 0.9997.
        MaximumIteration (int, optional): Maximum number of iterations. Defaults to 150000.
        EarlyStopLimited (int, optional): Number of early stops. Defaults to 50000.
        output (bool, optional): Output tag. Defaults to False.

    Returns:
        [list, double]: Best solve，best fit value
    """
    if output == True:
        print("[{}] Start!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        
    # Simulated annealing parameters
    Temperature = 100               # Current temperature
    Solve = inti_solve              # Current solution
    TryLimited = 50                 # Maximum number of cooling attempts
    hist_best_solve = inti_solve    # historical optimal individual
    hist_best_solve_fit = fitFunction(hist_best_solve, fitFunctionInput)
    Fit = fitFunction(Solve, fitFunctionInput)      # Fitness value of current solution

    # Simulated annealing main process
    e = 0                   # Iteration counter
    early_stop_count = 0    # Early stop counter
    for epoch in range(0, MaximumIteration):
        e = epoch
        updateFlag, subCount = False, 0
        while updateFlag == False:
            subCount += 1
            new_solve = SimulatedAnnealing_Base.Mutation(Solve)
            newFit = fitFunction(new_solve, fitFunctionInput)
            # Judge whether to accept the new solution
            if newFit > Fit:
                Solve = new_solve
                Fit = newFit
                if newFit > hist_best_solve_fit:
                    hist_best_solve_fit = Fit
                    hist_best_solve = Solve.copy()
                updateFlag = True
            else:
                # Calculate the difference and calculate the acceptance probability
                Delta = Fit - newFit
                prob = math.exp(- Delta / Temperature)
                p = random.random()
                if p < prob:
                    Solve = new_solve
                    if Fit != newFit:
                        updateFlag = True
                    Fit = newFit
                    break
                else:
                    if subCount >= TryLimited:
                        break
        # Temperature correction
        Temperature *= gamma

        # Early stop judgment
        if updateFlag == False:
            early_stop_count += 1
            if early_stop_count > EarlyStopLimited:
                if output == True:
                    print("   early stop!")
                break
        else:
            early_stop_count = 0

        # Output control
        if epoch % 1000 == 0:
            if output == True:
                print("[{}] Epoch {:5d}: fit value = {:2.4f}, histBest = {:2.4f}".format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    epoch, Fit, hist_best_solve_fit))

    # End part
    if output == True:
        print("[{}] Final Result: Stop epoch = {}, Fit = {:2.4f}, histBest = {:2.4f}".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            e, Fit, hist_best_solve_fit))

    return hist_best_solve, hist_best_solve_fit
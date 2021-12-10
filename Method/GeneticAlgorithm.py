# !/usr/bin/env python3
"""
    author: Apeiria
    contact: apeiriaStar@outlook.com
    file: GeneticAlgorithm.py
    function: Genetic algorithm correlation
"""

# ---------------------------------- import --------------------------------- #
import os
import sys
import time
import random
import copy
import numpy as np
import ray

from Method.GenerationSolve import GenerationSolve

# ---------------------------------- Tool Class ----------------------------- #
class Genetic_Base(object):
    
    """Real Genetic algorithm, Standard tool class"""

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
    def init_population(cls,
                        population_size,
                        individual_member):
        """Initialize population

        Args:
            population_size (int): population size
            individual_member (list): individual member element

        Returns:
            [list, list]: [Initialize population, Individual optimal record]
        """
        # Param Check
        cls.__param_check(population_size, int, "population_size")
        cls.__param_check(individual_member, list, "individual_member")
        for element in individual_member:
            cls.__param_check(element, int, "individual_member element")
        
        # Main Process
        population, hist_individual_best = [], []
        for _ in range(0, population_size):
            individual = copy.deepcopy(individual_member)
            random.shuffle(individual)
            population.append(individual)
            hist_individual_best.append(copy.deepcopy(individual))
            
        return population, hist_individual_best


    @classmethod
    def update_histinfo(cls,
                        population, 
                        hist_info,
                        fitFunction, 
                        fitFunctionInput,
                        output=False):
        """Update history information

        Args:
            population (list): Population list
            hist_info (list): Historical information list
            fitFunction (function): Fitness calculation function
            fitFunctionInput (list): Auxiliary input of fitness function
            output (bool, optional): Output tag. Defaults to False.

        Returns:
            [bool, double, list]: Update marker, optimal fitness of current population, updated historical information
        """
        # Param Check
        cls.__param_check(population, list, "population")
        for element in population:
            cls.__param_check(element, list, "population element")
        cls.__param_check(hist_info, list, "hist_info")
        
        # Main Process
        [hist_best_individual, hist_best_fit] = hist_info
        update_flag  = False
        populationFit = []
        for individual in population:
            Fit = fitFunction(individual, fitFunctionInput)
            populationFit.append(Fit)
        Fit = np.max(populationFit)
        if hist_best_fit == None or hist_best_fit < np.max(populationFit):
            update_flag  = True
            hist_best_fit = np.max(populationFit)
            hist_best_individual = copy.deepcopy(population[np.argmax(populationFit)])
        hist_info = [hist_best_individual, hist_best_fit]
        
        # Output control
        if output == True:
            populationFit = sorted(populationFit, reverse=True)
            output_str = "Best: "
            for i in range(0, 15):
                output_str += "{:2.2f}%, ".format(populationFit[i])
            print(output_str[0:-2])
            output_str = "Low:  "
            for i in range(0, 15):
                output_str += "{:2.2f}%, ".format(populationFit[-i - 1])
            print(output_str[0:-2])
        
        return update_flag, Fit, hist_info


    @classmethod
    def Cross(cls,
              individual1, 
              individual2, 
              cross_mode,
              fitFunction,
              fitFunctionInput):
        """Genetic algorithm, single point crossover process

        Args:
            individual1 (list): Parent individual 1
            individual2 (list): Parent individual 2
            cross_mode (string): Crossover mode, optional "no_bias", "bias"
            fitFunction (function): Fitness calculation function
            fitFunctionInput (list): Auxiliary input of fitness function

        Raises:
            Exception: Overlapping problem

        Returns:
            list: New individuals after crossover
        """
        # Param Check
        cls.__param_check(individual1, list, "individual1")
        cls.__param_check(individual2, list, "individual2")
        assert len(individual1) > 0, "[ERROR] individual1 is null."
        assert len(individual2) > 0, "[ERROR] individual2 is null."
        for i in range(0, len(individual1)):
            cls.__param_check(individual1[i], int, "individual1 element")
        for i in range(0, len(individual2)):
            cls.__param_check(individual2[i], int, "individual2 element")
        
        # Main Process
        if cross_mode == "no_bias":
            # Unbiased crossover mode
            new_individual = []
            if len(individual1) == len(individual2):
                temp = len(individual1) - 1  
            else:
                temp = min(len(individual1) - 1, len(individual2) - 1)
            cross_position = random.randint(1, temp)
            new_individual.extend(individual1[0:cross_position])
            new_individual.extend(individual2[cross_position:])
            # 1. Filter duplicate members
            repeat_member_index = {}
            for i in range(0, len(new_individual)):
                if new_individual[i] not in repeat_member_index.keys():
                    repeat_member_index[new_individual[i]] = [i]
                else:
                    repeat_member_index[new_individual[i]].append(i) 
            # 2. Find missing members
            replace_index = []
            for i in individual1:
                if i not in new_individual:
                    replace_index.append(i)
            # 3. Replace conflicting duplicate elements
            for key in repeat_member_index.keys():
                if len(repeat_member_index[key]) == 2:
                    choice_index = random.choice(repeat_member_index[key])
                    choice_member = random.choice(replace_index)
                    new_individual[choice_index] = choice_member
                    repeat_member_index[key].remove(choice_index)
                    replace_index.remove(choice_member)
                elif len(repeat_member_index[key]) > 2:
                    raise Exception("[ERROR] In 2 individuals, 1 index cannot appear more than 3 times.")
            
        elif cross_mode == "bias":
            # Bias crossover mode
            new_individual = []
            if len(individual1) == len(individual2):
                temp = len(individual1) - 1  
            else:
                temp = min(len(individual1) - 1, len(individual2) - 1)
            cross_position = random.randint(1, temp)
            Fit1, _ = fitFunction(individual1, fitFunctionInput)
            Fit2, _ = fitFunction(individual2, fitFunctionInput)
            better_individual = 1 if Fit1 > Fit2 else 2
            new_individual.extend(individual1[0:cross_position])
            new_individual.extend(individual2[cross_position:])
            # 1. Filter duplicate members
            repeat_member_index = {}
            for i in range(0, len(new_individual)):
                if new_individual[i] not in repeat_member_index.keys():
                    repeat_member_index[new_individual[i]] = [i]
                else:
                    repeat_member_index[new_individual[i]].append(i) 
            # 2. Find missing members
            replace_index = []
            for i in individual1:
                if i not in new_individual:
                    replace_index.append(i)
            # 3. Replace conflicting duplicate elements
            for key in repeat_member_index.keys():
                if len(repeat_member_index[key]) == 2:
                    # Bias
                    tt = 1 if repeat_member_index[key][0] in new_individual[0:cross_position] else 2
                    choice_index = repeat_member_index[key][0] if tt != better_individual else repeat_member_index[key][1]
                    choice_member = random.choice(replace_index)
                    new_individual[choice_index] = choice_member
                    repeat_member_index[key].remove(choice_index)
                    replace_index.remove(choice_member)
                elif len(repeat_member_index[key]) > 2:
                    raise Exception("[ERROR] In 2 individuals, 1 index cannot appear more than 3 times.")
        
        else:
            raise Exception("[ERROR] Unknown Param: cross_mode({})".format(cross_mode))
            
        return new_individual


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


    @classmethod
    def Select(cls,
               population, 
               population_size, 
               select_mode,
               fitFunction, 
               fitFunctionInput,
               candidate_parameters,
               epoch,
               output=True):
        """Genetic algorithm, selection process

        Args:
            population (list): Population list
            population_size (int): Population size
            select_mode (string): Select mode, optional "greedy","championships","roulette"
            fitFunction (function): Fitness calculation function
            fitFunctionInput (list): Auxiliary input of fitness function
            candidate_parameters (list): Candidate parameter list
            epoch (int): Number of iterations, which provides the current number of iterations for select_mode
            output (bool, optional): output tag. Defaults to False.

        Raises:
            Exception: Mismatch data type

        Returns:
            list: new Population
        """
        # Param Check
        cls.__param_check(population, list, "population")
        for element in population:
            cls.__param_check(element, list, "population element")
        cls.__param_check(population_size, int, "population_size")
        cls.__param_check(candidate_parameters, list, "candidate_parameters")
        
        # Main Process
        # 1. Calculation of fitness in population
        populationFit = []
        for individual in population:
            Fit = fitFunction(individual, fitFunctionInput)
            populationFit.append(Fit)
        temp = []
        for i in range(0, len(population)):
            temp.append((population[i], populationFit[i]))
        
        # 2. Population screening (maximum screening)
        new_population = []
        if select_mode == "greedy":
            # A. Greedy strategy, retain the optimal individual
            temp = sorted(temp, key=lambda x: x[1], reverse=True)
            temp = temp[0:population_size]
            for element in temp:
                new_population.append(element[0])
        elif select_mode == "championships":
            # B. Championships strategy (no put back)
            [error_prob] = candidate_parameters
            cls.__param_check(error_prob, float, "error_prob")
            for _ in range(0, population_size):
                sample = random.sample(temp, 2)
                sample = sorted(sample, key=lambda x: x[1], reverse=True)
                if random.random() <= error_prob:
                    individual = sample[1][0]
                    temp.remove(sample[1])
                else:
                    individual = sample[0][0]
                    temp.remove(sample[0])
                new_population.append(individual)
        elif select_mode == "championships_back":
            # C. Championships strategy (put back)
            [error_prob] = candidate_parameters
            cls.__param_check(error_prob, float, "error_prob")
            for _ in range(0, population_size):
                sample = random.sample(temp, 2)
                sample = sorted(sample, key=lambda x: x[1], reverse=True)
                if random.random() <= error_prob:
                    new_population.append(sample[1][0])
                else:
                    new_population.append(sample[0][0])
        elif select_mode == "championships_adaption":
            # D. Championships strategy (self-adaption, no put back)
            [error_prob_init, sigma] = candidate_parameters
            cls.__param_check(error_prob_init, float, "error_prob_init")
            cls.__param_check(sigma, float, "sigma")
            error_prob = error_prob_init * (sigma ** (epoch + 1))
            for _ in range(0, population_size):
                sample = random.sample(temp, 2)
                sample = sorted(sample, key=lambda x: x[1], reverse=True)
                if random.random() <= error_prob:
                    individual = sample[1][0]
                    temp.remove(sample[1])
                else:
                    individual = sample[0][0]
                    temp.remove(sample[0])
                new_population.append(individual)
        elif select_mode == "roulette":
            # E. Roulette strategy (no put back)
            [exponent_coefficient] = candidate_parameters
            cls.__param_check(exponent_coefficient, float, "exponent_coefficient")
            for i in range(0, len(populationFit)):
                populationFit[i] = exponent_coefficient ** populationFit[i]
            for _ in range(0, population_size):
                assert len(population) == len(populationFit)
                individual = random.choices(population, weights=populationFit, k=1)[0]
                t = population.index(individual)
                population.remove(individual)
                del populationFit[t]
                new_population.append(individual)
        elif select_mode == "roulette_back":
            # F. Roulette strategy (put back)
            [exponent_coefficient] = candidate_parameters
            cls.__param_check(exponent_coefficient, float, "exponent_coefficient")
            for i in range(0, len(populationFit)):
                populationFit[i] = exponent_coefficient ** populationFit[i]
            for _ in range(0, population_size):
                individual = random.choices(population, weights=populationFit, k=1)[0]
                new_population.append(individual)
        else:
            raise Exception("[ERROR] Unknown Param: select_mode({})".format(select_mode))
        
        return copy.deepcopy(new_population)


@ray.remote
def population_fit_feature_cal(population,
                               fitFunction,
                               fitFunctionInput,
                               ite):
    """Parallel population fitness calculation function

    Args:
        population (list): Single population
        fitFunction (function): Fitness calculation function
        fitFunctionInput (list): Auxiliary input of fitness calculation function
        ite (int): Current population number

    Returns:
        [int, list]: Current population number, population characteristics (mean, variance, maximum)
    """
    
    # Calculate population fitness
    populationFit = []
    for individual in population:
        Fit = fitFunction(individual, fitFunctionInput)
        populationFit.append(Fit)
    
    # Calculate population fitness characteristics
    mean, std, max = np.mean(populationFit), np.std(populationFit), np.max(populationFit)
    
    return ite, [mean, std, max]


class Genetic_Paralle(object):
    
    """Real genetic algorithm, parallel method tool class"""
    
    @classmethod
    def Transfer(cls,
                 paralle_population,
                 transfer_scale,
                 transfer_method,
                 fitFunction,
                 fitFunctionInput):
        """Parallel coarse-grained genetic algorithm, cyclic migration operation

        Args:
            paralle_population (list): Parallel population list
            transfer_scale (int/float): Migration scale, int is the number of each migration, \
                                        and float is the proportion of migration population
            transfer_method (string): Migration methods, including "random", "tail_random", "head_random"
            fitFunction (function): Fitness function
            fitFunctionInput (list): Auxiliary input list of fitness function

        Raises:
            Exception: Unknown parameter, migration size exceeds population size

        Returns:
            list: List of new parallel populations after migration
        """
        new_paralle_population = copy.deepcopy(paralle_population)
        # Adaptive migration scale
        assert transfer_scale > 0, "[ERROR] transfer_scale can not be negative.({})".format(transfer_scale)
        if isinstance(transfer_scale, int) == True:
            pass
        elif isinstance(transfer_scale, float) == True:
            assert transfer_scale < 1, "[ERROR] float transfer_scale can not over than 1."
            transfer_scale = int(len(new_paralle_population[0]) * transfer_scale)
        print("[Main] transfer_scale = {}".format(transfer_scale))
        # Cyclic migration for each population
        random.shuffle(new_paralle_population)
        for i in range(0, len(new_paralle_population)):
            # A. Migration individual source and migration target
            origin_population = new_paralle_population[i]
            if i + 1 < len(new_paralle_population):
                target_population = new_paralle_population[i + 1]
            else:
                target_population = new_paralle_population[0]
            # B. Fitness preparation process
            if transfer_method != "random":
                populationFit = []
                for individual in origin_population:
                    Fit = fitFunction(individual, fitFunctionInput)
                    populationFit.append(Fit)
                temp = []
                for i in range(0, len(origin_population)):
                    temp.append((origin_population[i], populationFit[i]))
                temp = sorted(temp, key=lambda x:x[1], reverse=True)
            # C. Migration process
            for _ in range(0, transfer_scale):
                assert transfer_scale < len(origin_population), "[ERROR] Oversized transfer_scale."
                assert transfer_scale < len(target_population), "[ERROR] Oversized transfer_scale."
                if transfer_method == "random":
                    # Migration method 1: random migration
                    transfer_individual = random.choice(origin_population)
                    origin_population.remove(transfer_individual)
                    target_population.append(transfer_individual)
                elif transfer_method == "tail_random":
                    # Migration method 2: Tail random
                    step = max(int(len(origin_population) / 4), 1)
                    transfer_individual_tuple = random.choice(temp[-step:])
                    origin_population.remove(transfer_individual_tuple[0])
                    target_population.append(transfer_individual_tuple[0])
                    temp.remove(transfer_individual_tuple)
                elif transfer_method == "head_random":
                    # Migration method 3: random head
                    step = max(int(len(origin_population) / 4), 1)
                    transfer_individual_tuple = random.choice(temp[0:step])
                    origin_population.remove(transfer_individual_tuple[0])
                    target_population.append(transfer_individual_tuple[0])
                    temp.remove(transfer_individual_tuple)
                else:
                    raise Exception("[EEROR] Unknown Param: transfer_method({})".format(transfer_method))
        
        return new_paralle_population
    
    
    @classmethod
    def Replacement(cls,
                    paralle_population,
                    replacement_scale,
                    replacement_method,
                    fitFunction,
                    fitFunctionInput):
        """Parallel coarse-grained genetic algorithm, population replacement

        Args:
            paralle_population (list): Parallel population
            replacement_scale (int): Replacement population size
            replacement_method (string): Replacement method
            fitFunction (function): Fitness calculation function
            fitFunctionInput (list)): Auxiliary input of fitness calculation function

        Raises:
            Exception: Bad data range, unknown data type

        Returns:
            [list]: The new parallel population, the partially killed species group is set as none
        """
        # Param Check
        assert isinstance(replacement_scale, int), "[EEROR] replacement_scale need int."
        assert replacement_scale > 0 and replacement_scale < len(paralle_population), \
            "[ERROR] replacement_scale should between 0 and paralle_population length."
        
        # Parallel calculation of all population fitness
        paralle_feature_id = [
            population_fit_feature_cal.remote(paralle_population[ite], 
                                              fitFunction, 
                                              fitFunctionInput,
                                              ite) 
            for ite in range(0, len(paralle_population))
        ]
        paralle_feature_result = ray.get(paralle_feature_id)
        
        # Population replacement
        new_paralle_population = copy.deepcopy(paralle_population)
        if replacement_method == "mean_first":
            temp = sorted(paralle_feature_result, key=lambda x: x[1][0]) 
            for i in range(0, replacement_scale):
                new_paralle_population[temp[i][0]] = None
        elif replacement_method == "std_first":
            temp = sorted(paralle_feature_result, key=lambda x: x[1][1], reverse=True) 
            for i in range(0, replacement_scale):
                new_paralle_population[temp[i][0]] = None
        elif replacement_method == "max_first":
            temp = sorted(paralle_feature_result, key=lambda x: x[1][2]) 
            for i in range(0, replacement_scale):
                new_paralle_population[temp[i][0]] = None
        elif replacement_method == "standard":
            temp = sorted(paralle_feature_result, key=lambda x: (x[1][2], x[1][0], -x[1][1])) 
            for i in range(0, replacement_scale):
                new_paralle_population[temp[i][0]] = None
        else:
            raise Exception("[ERROR] Unknown Param: replacement_method")       
        
        return new_paralle_population           
    
    
     
# ------------------------------ Main Def ----------------------------------- #

@ray.remote
def GeneticAlgorithm_SerialProcess(population_size,
                                   individual_member,
                                   fitFunction,
                                   fitFunctionInput,
                                   # Genetic algorithm parameters
                                   init_population=None,
                                   maximum_iteration=5000,
                                   cross_prob=0.75,
                                   mutation_prob=0.75,
                                   mutimutaion_prob=0.2,
                                   select_mode="roulette",
                                   select_candidate_parameters=[1.07],
                                   elite_strategy=True,
                                   preserve_original_population=False,
                                   # system parameter
                                   origin_output=False,
                                   epoch_output=None,
                                   output=False):
    """Genetic algorithm, serial mainstream, Maximization process

    Args:
        population_size (int): Population size
        individual_member (list): Individual member information
        fitFunction (function): Fitness calculation function
        fitFunctionInput (list): Auxiliary input of fitness calculation function
        init_population (list, optional): Initialize population. Defaults to None.
        maximum_iteration (int, optional): Maximum number of iterations. Defaults to 5000.
        cross_prob (float, optional): Crossover probability. Defaults to 0.75.
        mutation_prob (float, optional): Mutation probability. Defaults to 0.75.
        mutimutaion_prob (float, optional): Multiple mutation probability. Defaults to 0.2.
        select_mode (string, optional): Select mode. Defaults to "roulette".
        select_candidate_parameters (list, optional): Select mode additional parameters. Defaults to [1.07].
        elite_strategy (bool, optional): Elite strategy marker. Defaults to True.
        preserve_original_population (bool, optional): Preserve the original population tag. Defaults to False.
        origin_output (bool, optional): origin output tag. Defaults to False.
        epoch_output (int, optional): Progress mark on parallel. Defaults to None.
        output (bool, optional): Output tag. Defaults to False.

    Returns:
        [list, double]: Best individual，best fit value
    """
    if output == True and epoch_output is None:
        print("[{}] Start!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    # Ray set
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_path + "/GIT_MHA/Method")
    sys.path.append(root_path + "/GIT_MHA/Problem")
    print(sys)
    
    # Constructing population, individual exchange order, individual historical optimization
    if init_population is None:
        population, _ = Genetic_Base.init_population(population_size, 
                                                     individual_member)
    else:
        population = init_population
    now_best_fit = None                         # Maximum fitness of current population
    hist_best_fit = None                        # Historical optimal fitness
    hist_best_individual = None                 # Historical optimal solution
    earlystop_limited = maximum_iteration       # Early stop upper limit
    hist_info = [hist_best_individual, hist_best_fit]
    _, now_best_fit, hist_info = Genetic_Base.update_histinfo(population, 
                                                              hist_info, 
                                                              fitFunction, 
                                                              fitFunctionInput)
    [hist_best_individual, hist_best_fit] = hist_info

    e = 0                   # Iteration counter
    early_stop_count = 0    # Early stop counter
    for epoch in range(0, maximum_iteration):
        e += 1
        
        # Population retention
        if preserve_original_population == True:
            newpopulation = []
        else:
            newpopulation = copy.deepcopy(population)
        
        # Cross
        for _ in range(0, int(population_size * 1.5)):
            [individual1, individual2] = random.sample(population, 2)
            if random.random() <= cross_prob:
                new_individual = Genetic_Base.Cross(individual1, 
                                                    individual2, 
                                                    "no_bias",
                                                    fitFunction, fitFunctionInput)
            else:
                new_individual = copy.deepcopy(individual1)
            # Mutation
            if random.random() <= mutation_prob:
                new_individual = Genetic_Base.Mutation(new_individual)
                while random.random() <= mutimutaion_prob:
                    new_individual = Genetic_Base.Mutation(new_individual)
            newpopulation.append(new_individual)
        
        # Select
        newpopulation = Genetic_Base.Select(newpopulation, population_size, 
                                            select_mode,
                                            fitFunction, fitFunctionInput, select_candidate_parameters, epoch)
        population = newpopulation
       
        # Optimal individual and fitness update
        hist_info = [hist_best_individual, hist_best_fit]
        update_flag, now_best_fit, _ = Genetic_Base.update_histinfo(population, 
                                                                    hist_info, 
                                                                    fitFunction, 
                                                                    fitFunctionInput)
        
        # Elite strategy, optimal individual retention
        if elite_strategy == True and hist_best_individual not in newpopulation:
            newpopulation.remove(random.choice(newpopulation))
            newpopulation.append(copy.deepcopy(hist_best_individual))
        _, _, hist_info = Genetic_Base.update_histinfo(population, 
                                                       hist_info, 
                                                       fitFunction, 
                                                       fitFunctionInput)
        [hist_best_individual, hist_best_fit] = hist_info
        
        # Early stop judgment
        if update_flag == False:
            early_stop_count += 1
            if early_stop_count > earlystop_limited:
                if output == True:
                    print("   early stop!")
                break
        else:
            early_stop_count = 0
        
        # Output
        if epoch % 25 == 0:
            if output == True:
                if epoch_output is None:
                    print("[{}] Epoch {:5d}: Fit value = {}, HistBest = {}, Individual = {}".format(
                           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                           epoch, now_best_fit, hist_best_fit, hist_best_individual))
                else:
                    print("[{:4d}][{}] Epoch {:5d}: Fit value = {}, HistBest = {}, Individual = {}".format(
                           epoch_output, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                           epoch, now_best_fit, hist_best_fit, hist_best_individual))

    # Final Part
    if origin_output == True:
        return population, hist_best_individual, hist_best_fit
    if output == True and epoch_output is None:
        print("[{}] Final Result: Stop epoch = {}, Fit = {}, HistBest = {}".format(
              time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              e, now_best_fit, hist_best_fit))

    return hist_best_individual, hist_best_fit


class BreakAll(Exception):
    """Global break out flag"""
    pass


def GeneticAlgorithm_ParalleProcess(multi_population_num,
                                    serial_population_size,
                                    individual_member,
                                    fitFunction,
                                    fitFunctionInput,
                                    # Main process related parameters
                                    multi_maximum_iteration=20,
                                    transfer_scale=0.2,
                                    transfer_method="random",
                                    replacement_interval=10,
                                    replacement_scale=10,
                                    # Child process related parameters
                                    serial_maximum_iteration=50,
                                    cross_prob_list=[0.5],
                                    mutation_prob_list=[0.75],
                                    mutimutaion_prob_list=[0.2],
                                    select_mode_list=["roulette"],
                                    select_candidate_parameters_list=[[1.07]],
                                    elite_strategy_list=[True, False],
                                    preserve_original_population_list=[True, False],
                                    # system parameter
                                    output=False):
    """Genetic algorithm, parallel mainstream (coarse-grained parallel method), Maximization process

    Args:
        multi_population_num (int): Number of parallel populations
        serial_population_size (int)): Sub thread (sub population) size
        individual_member (list): Individual member information
        fitFunction (function): Fitness calculation function
        fitFunctionInput (list): Auxiliary input of fitness calculation function
        multi_maximum_iteration (int, optional): Maximum number of iterations of the main thread. Defaults to 20.
        transfer_scale (float/int, optional): Migration scale. Defaults to 0.2.
        transfer_method (string, optional): Migration method. Defaults to "random".
        replacement_interval (int, optional): Replacement interval: how many large iterations will one replacement \
            occur. Defaults to 10.
        replacement_scale (int, optional): Replacement scale. Defaults to 10.
        serial_maximum_iteration (int, optional): Maximum iteration times of sub population. Defaults to 50.
        cross_prob_list (list, optional): Possible list of crossover probabilities. Defaults to [0.5].
        mutation_prob_list (list, optional):Possible list of mutation probabilities. Defaults to [0.75].
        mutimutaion_prob_list (list, optional): Possible list of multiple mutation probabilities. Defaults to [0.2].
        select_mode_list (list, optional): Possible list of Select mode. Defaults to ["roulette"].
        select_candidate_parameters_list (list, optional): List of input parameters corresponding to the selection \
            mode. Defaults to [[1.07]].
        elite_strategy_list (list, optional): Possible list of Elite strategy . Defaults to [True, False].
        preserve_original_population_list (list, optional): Possible list of Population retention. \
            Defaults to [True, False].
        output (bool, optional): Output tag. Defaults to False.

    Raises:
        Exception: The input parameter format is incorrect, the parameter combination scale is too large or empty.

    Returns:
        [list, double]: Best individual，best fit value
    """
    if output == True:
        print("[Main][{}] Start!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    # Prepare parameter combination
    param_list = []
    param_count = 0
    assert isinstance(cross_prob_list, list), "[ERROR] cross_prob_list need a list."
    assert isinstance(mutation_prob_list, list), "[ERROR] mutation_prob_list need a list."
    assert isinstance(mutimutaion_prob_list, list), "[ERROR] mutimutaion_prob_list need a list."
    assert isinstance(select_mode_list, list), "[ERROR] select_mode_list need a list."
    assert isinstance(select_candidate_parameters_list, list), \
        "[ERROR] select_candidate_parameters_list need a list."
    for item in select_candidate_parameters_list:
        assert isinstance(item, list), \
            "[ERROR] select_candidate_parameters_list item need a list."
    assert isinstance(elite_strategy_list, list), "[ERROR] elite_strategy_list need a list."
    assert isinstance(preserve_original_population_list, list), \
        "[ERROR] preserve_original_population_list need a list."
    try:
        # Cartesian product
        for cross_prob in cross_prob_list:
            for mutation_prob in mutation_prob_list:
                for mutimutaion_prob in mutimutaion_prob_list:
                    for elite_strategy in elite_strategy_list:
                        for preserve_original_population in preserve_original_population_list:
                            for ppp in range(0, len(select_mode_list)):
                                select_mode = select_mode_list[ppp]
                                select_candidate_parameters = select_candidate_parameters_list[ppp]
                                param_list.append((cross_prob, 
                                                   mutation_prob, mutimutaion_prob, 
                                                   select_mode, select_candidate_parameters,
                                                   elite_strategy, preserve_original_population
                                                 ))
                                param_count += 1
                                if multi_population_num is not None and \
                                    param_count >= int(multi_population_num * 20):
                                    raise BreakAll
                                if param_count > 100000:
                                    raise Exception("[ERROR] Too many param combination.")
    except BreakAll as ba:
        pass
    random.shuffle(param_list)
    if multi_population_num is not None:
        if len(param_list) >= multi_population_num:
            param_list = param_list[0:multi_population_num]
        else:
            assert len(param_list) > 0, "[ERROR] No Param! May some Param is Null List."
            while len(param_list) < multi_population_num:
                param_list.append(random.choice(param_list))
    else:
        multi_population_num = len(param_list)
    if output == True:
        print("[Main] Param num: {}".format(multi_population_num))
    
    # Parallel mainstream
    paralle_best_individual, paralle_best_fit = None, None
    paralle_population = None
    for epoch in range(0, multi_maximum_iteration):
        # 1. Ray parallel process of each sub thread
        paralle_result_id = [
            GeneticAlgorithm_SerialProcess.remote(serial_population_size, individual_member,
                                                  fitFunction, fitFunctionInput,
                                                  paralle_population[ite] if epoch != 0 else None,
                                                  serial_maximum_iteration, 
                                                  param_list[ite][0], 
                                                  param_list[ite][1], param_list[ite][2], 
                                                  param_list[ite][3], param_list[ite][4],
                                                  param_list[ite][5], param_list[ite][6],
                                                  origin_output=True, epoch_output=ite, output=False) 
            for ite in range(0, multi_population_num)
        ]
        paralle_result_list = ray.get(paralle_result_id)
        # 2. Migration operation
        paralle_population, paralle_individual, paralle_fit = [], [], []
        population_size = ""
        for item in paralle_result_list:
            population_size += str(len(item[0])) + ", "
            paralle_population.append(item[0])
            paralle_individual.append(item[1])
            paralle_fit.append(item[2])
            if paralle_best_fit is None or item[2] > paralle_best_fit:
                paralle_best_fit = item[2]
                paralle_best_individual = copy.deepcopy(item[1])
        if output == True:
            print("[Main] Population Size: " + population_size)
        paralle_population = Genetic_Paralle.Transfer(paralle_population, 
                                                      transfer_scale, 
                                                      transfer_method, 
                                                      fitFunction, 
                                                      fitFunctionInput)
        # 3. Replacement operation
        if (epoch + 1) % replacement_interval == 0:
            paralle_population = Genetic_Paralle.Replacement(paralle_population,
                                                             replacement_scale,
                                                             "standard",
                                                             fitFunction,
                                                             fitFunctionInput)
            if output == True:
                print("[Main] Do repalcement!")
        # 4. Epoch output control
        if output == True:
            print("[Main][{}] Epoch {:3d}: Fit = {}, Hist Best Fit = {}".format(
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                  epoch, max(paralle_fit), paralle_best_fit))
    
    # End part
    if output == True:
        print("[Main][{}] Final Result: HistBest = {}%".format(
              time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              paralle_best_fit))

    return paralle_best_individual, paralle_best_fit
    
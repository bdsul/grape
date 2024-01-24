import re
import math
from operator import attrgetter
import numpy as np
import random
import copy
import statistics

from functions import shuffle_rows_except_first, remove_row, add_index_column, remove_columns, aggregate_rows, represent_matrix_behaviour, remove_equal_rows, remove_equal_columns, find_equal_columns, remove_columns_with_different_value, aggregate_rows_sum, count_zeros_except_first_row, count_zeros

def median_abs_deviation(arr, axis=0):
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Calculate the median along axis 0
    median = np.median(arr, axis=0)

    # Calculate the absolute deviations from the median along axis 0
    abs_deviations = np.abs(arr - median)

    # Calculate the median of the absolute deviations along axis 0
    mad = np.median(abs_deviations, axis=0)

    return mad

def selLexicaseFilter(individuals, k):
    """
   

    """
    selected_individuals = []
    #valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    if len(inds_fitness_zero) > 0:
        for i in range(k):
            selected_individuals.append(random.choice(inds_fitness_zero))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        while len(cases) > 0 and len(pool) > 1:
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selLexicaseFilterCount(individuals, k):
    """
   

    """
    selected_individuals = []
    #valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    if len(inds_fitness_zero) > 0:
        for i in range(k):
            selected_individuals.append(random.choice(inds_fitness_zero))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selLexi2_nodesCountTies(individuals, k):
    """
    same as selLexi2_nodesCount, but also registers the number of ties in the selected individual in the attribute 'ties'

    """
    selected_individuals = []
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        for ind in cands:
            ind.ties = len(cands)
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals
   
def selEpsilonLexi2_nodesCountTies(individuals, k, alpha):
    """
    same as selEpsilonLexi2_nodesCount, but also registers the number of ties in the selected individual in the attribute 'ties'
   
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    #Check if we have found a perfect score already
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix, axis=0)

    mad = median_abs_deviation(fitness_cases_matrix, axis=0)
    epsilon = alpha * mad
    avg_epsilon = np.mean(epsilon)
    
    for i in range(len(candidates)):
        for j in range(l_samples):
            if fitness_cases_matrix[i][j] <= min_[j] + epsilon[j]:
                fitness_cases_matrix[i][j] = 0
                #candidates[i].fitness_each_sample_discrete[j] = 1
            else:
                fitness_cases_matrix[i][j] = 1
                #candidates[i].fitness_each_sample_discrete[j] = 0
        candidates[i].fitness_each_sample_discrete = list(fitness_cases_matrix[i,:])
    
    n_zeros = count_zeros(fitness_cases_matrix) #number of zeros in the matrix with discrete fitness cases
    avg_zeros = n_zeros / len(individuals) #average number of zeros per individual
    avg_zeros = avg_zeros / l_samples #represent as a percentage of the number of samples

    error_vectors = list(fitness_cases_matrix)
    
    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample_discrete == unique_error_vectors[i]]
        for ind in cands:
            ind.ties = len(cands)
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    indexes = []
    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = min
            best_val_for_case = f(map(lambda x: x.fitness_each_sample_discrete[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample_discrete[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        pool[0].avg_zeros = avg_zeros
        pool[0].avg_epsilon = avg_epsilon
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases
        
        index = individuals.index(pool[0])
        indexes.append(index)
        
    selected_individuals[0].unique_selected = len(set(indexes)) / len(individuals) # percentage of unique inds selected

    return selected_individuals
    
def selDynEpsilonLexicase(individuals, k):
    """
    Implements the dynamic version of epsilon Lexicase
   
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix_original = np.array(error_vectors) # inds (rows) x samples (cols)
    fitness_cases_matrix_original = add_index_column(fitness_cases_matrix_original) # add a column with indexes in the beginning
    fitness_cases_matrix_original = fitness_cases_matrix_original.transpose() # samples (rows) x inds (cols); row 0 contains the indexes
    
    for i in range(k):
        l, c = fitness_cases_matrix_original.shape
        #Shuffle fitness cases
        fitness_cases_matrix = shuffle_rows_except_first(fitness_cases_matrix_original)
        
        while l > 1 and c > 1: #we have more than one individual in the pool and more than one fitness case to test
            min_ = np.nanmin(fitness_cases_matrix[1])    
            mad = median_abs_deviation(fitness_cases_matrix[1]) #mad for the second row
            
            fitness_cases_matrix = remove_columns(fitness_cases_matrix, min_ + mad) #filter individuals
            
            fitness_cases_matrix = remove_row(fitness_cases_matrix, 1) #remove the assessed test case (second row, since the first one contains the indexes)
            
            l, c = fitness_cases_matrix.shape

        remaining_candidates = fitness_cases_matrix[0].astype(int) #indexes of the remaining candidates
        selected_ind = candidates[random.choice(remaining_candidates)]
        selected_ind.n_cases = l_samples - l #number of testcases used in the filtering process
        selected_ind.ties = len(remaining_candidates)
        selected_individuals.append(selected_ind) #Select the remaining candidate

    return selected_individuals

def selDynEpsilonLexi2_nodesCountTies(individuals, k):
    """
    Implements the dynamic version of epsilon Lexi^2 reducing nodes
   
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix_original = np.array(error_vectors) # inds (rows) x samples (cols)
    fitness_cases_matrix_original = add_index_column(fitness_cases_matrix_original) # add a column with indexes in the beginning
    fitness_cases_matrix_original = fitness_cases_matrix_original.transpose() # samples (rows) x inds (cols); row 0 contains the indexes
    
    for i in range(k):
        l, c = fitness_cases_matrix_original.shape
        #Shuffle fitness cases
        fitness_cases_matrix = shuffle_rows_except_first(fitness_cases_matrix_original)
        
        while l > 1 and c > 1: #we have more than one individual in the pool and more than one fitness case to test
            if np.all(fitness_cases_matrix[1] == fitness_cases_matrix[1, 0]): #if all individuals have the same fitness value for this case, we won't be able to filter anything
                pass
            else:
                min_ = np.nanmin(fitness_cases_matrix[1])    
                mad = median_abs_deviation(fitness_cases_matrix[1]) #mad for the second row
                fitness_cases_matrix = remove_columns(fitness_cases_matrix, min_ + mad) #filter individuals
            
            fitness_cases_matrix = remove_row(fitness_cases_matrix, 1) #remove the assessed test case (second row, since the first one contains the indexes)
            
            l, c = fitness_cases_matrix.shape

        remaining_candidates = [candidates[j] for j in fitness_cases_matrix[0].astype(int)] #indexes of the remaining candidates
        if len(remaining_candidates) > 1:
            f = min
            best_val_for_nodes = f(map(lambda x: x.nodes, remaining_candidates))
            smallest_size_candidates = [ind for ind in remaining_candidates if ind.nodes == best_val_for_nodes]
            selected_ind = random.choice(smallest_size_candidates) #if there are still more than one candidate with the same size, we choose randomly
        else:
            selected_ind = remaining_candidates[0]
        selected_ind.n_cases = l_samples - l #number of testcases used in the filtering process
        selected_ind.ties = len(remaining_candidates)
        selected_individuals.append(selected_ind) #Select the remaining candidate

    return selected_individuals
     
def selEpsilonLexi2_nodesCount(individuals, k):
    """
        
    """
    selected_individuals = []
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix, axis=0)
    #min_ = np.min(np.where(fitness_cases_matrix != 0, fitness_cases_matrix, np.inf), axis=0)

    #mad = robust.mad(fitness_cases_matrix, axis=0, c=1.0)
    #mad = np.std(fitness_cases_matrix, axis=0)
    #try:
    mad = median_abs_deviation(fitness_cases_matrix, axis=0)
    #except (MemoryError):
    #    pass
    
    for i in range(len(candidates)):
        for j in range(l_samples):
            #if fitness_cases_matrix[i][j] >= min_[j] and fitness_cases_matrix[i][j] <= min_[j] + mad[j]:
            if fitness_cases_matrix[i][j] <= min_[j] + mad[j]:
                fitness_cases_matrix[i][j] = 1
                candidates[i].fitness_each_sample[j] = 1
            else:
                fitness_cases_matrix[i][j] = 0
                candidates[i].fitness_each_sample[j] = 0
                
    error_vectors = list(fitness_cases_matrix)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
     #   print(sum(pool[0].fitness_each_sample)/l_samples, pool[0].mce)
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selBatchLexicase(individuals, k, batch_size=20):
    """
        
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors
	
    n_batches = math.ceil(l_samples / batch_size)
    
    for i in range(len(candidates)):
        candidates[i].fitness_each_batch = [0] * n_batches

    for _ in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            candidate = random.choice(list_)
            candidate.fitness_each_batch = [0] * n_batches
            pool.append(candidate) 
        random.shuffle(cases)
        batch_ = 0
        while batch_ < n_batches - 1 and len(pool) > 1:
            #Build batch of batch_size cases
            for _ in range(batch_size):
                for i in range(len(pool)):
                    pool[i].fitness_each_batch[batch_] += pool[i].fitness_each_sample[cases[0]]
                del cases[0]
            f = max
            best_val_for_batch = f(map(lambda x: x.fitness_each_batch[batch_], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[batch_] == best_val_for_batch]
            batch_ += 1
        if batch_ == n_batches - 1 and len(pool) > 1:
            #Build batch with the remaining cases
            for case in cases:
                for i in range(len(pool)):
                    pool[i].fitness_each_batch[batch_] += pool[i].fitness_each_sample[case]
            f = max
            best_val_for_batch = f(map(lambda x: x.fitness_each_batch[batch_], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[batch_] == best_val_for_batch]
            batch_ += 1
        
        #Despite filtering the individuals initially, we can have more than one remaining in the pool after checking the batches, because inds with different behaviours can have the same batch fitness
        if len(pool) == 1:
            selected_individual = pool[0]
        else:
#            print("error")
            selected_individual = random.choice(pool)
        selected_individual.n_cases = batch_
        selected_individuals.append(selected_individual)
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selBatchEpsilonLexi2_nodesCountTies(individuals, k, batch_size=20):
    """
    It calculates MAD and filters unique vectors before creating batches.
    Then, MAD is calculated once per generation.
    But after creating batches, it might exist inds with the same error vector.
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    error_vectors = [ind.fitness_each_sample for ind in individuals] #real values
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    
    min_ = np.nanmin(fitness_cases_matrix, axis=0)
    epsilon = median_abs_deviation(fitness_cases_matrix, axis=0) #mad

    candidates = individuals
    for i in range(len(candidates)):
        for j in range(l_samples):
            if fitness_cases_matrix[i][j] <= min_[j] + epsilon[j]:
                fitness_cases_matrix[i][j] = 0
                candidates[i].fitness_each_sample[j] = 0
            else:
                fitness_cases_matrix[i][j] = 1
                candidates[i].fitness_each_sample[j] = 1
                
    error_vectors = list(fitness_cases_matrix)
    
    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        #for ind in cands:
        #    ind.ties = len(cands)
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        for ind in cands:
            ind.ties = len(cands)
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    n_batches = math.ceil(l_samples / batch_size)

    fitness_cases_matrix_original = np.array(unique_error_vectors) # inds (rows) x samples (cols)
    fitness_cases_matrix_original = add_index_column(fitness_cases_matrix_original) # add a column with indexes in the beginning; these indexes match to candidates_prefiltered_set
    fitness_cases_matrix_original = fitness_cases_matrix_original.transpose() # samples (rows) x inds (cols); row 0 contains the indexes
    
    for i in range(k):
        #Shuffle fitness cases
        fitness_cases_matrix = shuffle_rows_except_first(fitness_cases_matrix_original)
        #Create batches
        fitness_cases_matrix = aggregate_rows(fitness_cases_matrix, batch_size)
        
        l, c = fitness_cases_matrix.shape
        
        while l > 1 and c > 1: #we have more than one individual in the pool and more than one fitness case to test
            if np.all(fitness_cases_matrix[1] == fitness_cases_matrix[1, 0]): #if all individuals have the same fitness value for this case, we won't be able to filter anything
                pass #we preprocessing the data to have unique vectors, but while creating batches, it's possible to have same vector again
            else: #we do Lexicase as normal
                min_ = np.nanmin(fitness_cases_matrix[1])    
                fitness_cases_matrix = remove_columns(fitness_cases_matrix, min_) #filter individuals
                
            fitness_cases_matrix = remove_row(fitness_cases_matrix, 1) #remove the assessed test case (second row, since the first one contains the indexes)
            
            l, c = fitness_cases_matrix.shape

        winning_indexes = list(fitness_cases_matrix[0].astype(int)) #indexes of the remaining candidates
        if len(winning_indexes) > 1:
            selected_index = random.choice(winning_indexes)
        else:
            selected_index = winning_indexes[0]
        selected_ind = random.choice(candidates_prefiltered_set[selected_index])
        selected_ind.n_cases = n_batches - l #number of batches used in the filtering process
        selected_ind.ties = len(candidates_prefiltered_set[selected_index])
        selected_individuals.append(selected_ind) #Select the remaining candidate

    return selected_individuals
    
def selBatchEpsilonLexi2_nodesCountTies_MADafter(individuals, k, batch_size=20):
    """
    Same as selBatchEpsilonLexi2_nodesCountTies, but calculates MAD after creating batches
    A bit more expensive, since it calculates MAD every time at the beginning of the filtering process, and not once per generation
    But it can save time afterwards because the unique vectors are considered after MAD
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    error_vectors = [ind.fitness_each_sample for ind in individuals] #real values
    
    fitness_cases_matrix_original = np.array(error_vectors) # inds (rows) x samples (cols)
    fitness_cases_matrix_original = add_index_column(fitness_cases_matrix_original) # add a column with indexes in the beginning; these indexes match to candidates_prefiltered_set
    fitness_cases_matrix_original = fitness_cases_matrix_original.transpose() # samples (rows) x inds (cols); row 0 contains the indexes
    
    candidates = individuals
    
    indexes = [] #indexes of the selected inds
    
    for _ in range(k):
        #Shuffle fitness cases
        fitness_cases_matrix = shuffle_rows_except_first(fitness_cases_matrix_original)
        #Create batches
        #fitness_cases_matrix = aggregate_rows_sum(fitness_cases_matrix, batch_size) #we can aggregate rows just summing up, despite we can have a batch with different size (the last one), because we will calculate MAD per batch and represent the matrix with 0's and 1's, so every row is independent
        fitness_cases_matrix = aggregate_rows(fitness_cases_matrix, batch_size)

        fitness_cases_matrix = fitness_cases_matrix.transpose()
        min_ = np.nanmin(fitness_cases_matrix[:,1:], axis=0)
        epsilon = median_abs_deviation(fitness_cases_matrix[:,1:], axis=0) #mad
        fitness_cases_matrix = fitness_cases_matrix.transpose()

        fitness_cases_matrix[1:] = represent_matrix_behaviour(fitness_cases_matrix[1:], min_ + epsilon)
        
        n_zeros = count_zeros_except_first_row(fitness_cases_matrix) #number of zeros in the matrix with dicrete fitness cases
        
        fitness_cases_matrix_reserved = fitness_cases_matrix.copy()
        
        fitness_cases_matrix = remove_equal_columns(fitness_cases_matrix)
        
        l, c = fitness_cases_matrix.shape
        n_batches = l - 1
                
        avg_zeros = n_zeros / len(individuals) #average number of zeros per individual
        avg_zeros = avg_zeros / n_batches #represent as a percentage of the number of batches
        
        avg_epsilon = np.mean(epsilon)
    
        while l > 1 and c > 1: #we have more than one individual in the pool and more than one fitness case to test
            min_ = np.nanmin(fitness_cases_matrix[1])    
            fitness_cases_matrix = remove_columns_with_different_value(fitness_cases_matrix, min_) #filter individuals
                
            fitness_cases_matrix = remove_row(fitness_cases_matrix, 1) #remove the assessed test case (second row, since the first one contains the indexes)
            
            l, c = fitness_cases_matrix.shape

        selected_index = int(fitness_cases_matrix[0])
        candidates_indexes = find_equal_columns(fitness_cases_matrix_reserved, selected_index) #indexes of the candidates with the best vector
    
        tied_candidates = []
        for idx in candidates_indexes:
            tied_candidates.append(candidates[idx])
    #        for ind in tied_candidates:
    #            ind.ties = len(tied_candidates)
    
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, tied_candidates))
        smallest_candidates = [ind for ind in tied_candidates if ind.nodes == best_val_for_nodes]
    
        selected_ind = random.choice(smallest_candidates)
        index = candidates_indexes[tied_candidates.index(selected_ind)]
        indexes.append(index)
        selected_ind.ties = len(tied_candidates)
#        if selected_ind.ties != 1:
 #           print(selected_ind.ties)
        selected_ind.n_cases = n_batches - l #number of batches used in the filtering process
        selected_individuals.append(selected_ind) #Select the remaining candidate
        selected_ind.fitness_each_sample_discrete = list(fitness_cases_matrix_reserved[1:,selected_index])
        selected_ind.avg_zeros = avg_zeros
        selected_ind.avg_epsilon = avg_epsilon
        
    selected_individuals[0].unique_selected = len(set(indexes)) / len(individuals) # percentage of unique inds selected

    return selected_individuals

def selTournamentExtra(individuals, k, tournsize, fit_attr="fitness"):
    chosen = []
    for i in range(k):
        aspirants = [random.choice(individuals) for i in range(tournsize)]
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    
    error_vectors = [ind.fitness_each_sample for ind in chosen] #real values
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix[:,:], axis=0)
    epsilon = median_abs_deviation(fitness_cases_matrix[:,:], axis=0) #mad
    fitness_cases_matrix = fitness_cases_matrix.transpose() # samples (rows) x inds (cols)
#    variance = statistics.variance(fitness_cases_matrix.flatten()) #variance of the original matrix
    fitness_cases_matrix[:] = represent_matrix_behaviour(fitness_cases_matrix[:], min_ + epsilon)
    
    #fitness_cases_matrix_unique = fitness_cases_matrix.copy()
    #fitness_cases_matrix_unique = remove_equal_columns(fitness_cases_matrix_unique)
    
    n_zeros = count_zeros(fitness_cases_matrix) #number of zeros in the matrix with unique fitness cases
    avg_zeros = n_zeros / len(individuals) #average number of zeros per individual
    avg_zeros = avg_zeros / len(fitness_cases_matrix[:,0]) #represent as a percentage of the number of samples
    
    avg_epsilon = np.mean(epsilon)
    
    indexes = []
    for i in range(k):
        chosen[i].fitness_each_sample_discrete = list(fitness_cases_matrix[:,i])
        chosen[i].avg_zeros = avg_zeros    
        chosen[i].avg_epsilon = avg_epsilon
#        chosen[i].variance = variance

        index = individuals.index(chosen[i])
        indexes.append(index)
        
    chosen[0].unique_selected = len(set(indexes)) / len(individuals) # percentage of unique inds selected
    
    return chosen

def selBatchEpsilonLexi2_nodesCountTiesOld(individuals, k, batch_size=20):
    """
        
    """
    selected_individuals = []
    error_vectors = [ind.fitness_each_sample for ind in individuals]
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
   
    pop_size = len(individuals)
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    n_batches = math.ceil(l_samples / batch_size)
    
    cases = list(range(0,l_samples))
    random.shuffle(cases)
    fitness_batches_matrix = np.zeros([pop_size, n_batches], dtype=float) # inds (rows) x samples (cols)
    #partitions
    for i in range(n_batches-1):
        for _ in range(batch_size):
            fitness_batches_matrix[:,i] += fitness_cases_matrix[:,cases[0]]
            del cases[0]
    for case in cases:
        fitness_batches_matrix[:,n_batches-1] += fitness_cases_matrix[:,case]

    min_ = np.nanmin(fitness_batches_matrix, axis=0)
    mad = median_abs_deviation(fitness_batches_matrix, axis=0)

    candidates = individuals
    for i in range(len(candidates)):
        candidates[i].fitness_each_batch = [0] * n_batches
        for j in range(n_batches):
            if fitness_batches_matrix[i][j] <= min_[j] + mad[j]:
                fitness_batches_matrix[i][j] = 1
                candidates[i].fitness_each_batch[j] = 1
            else:
                fitness_batches_matrix[i][j] = 0
                candidates[i].fitness_each_batch[j] = 0
            
    error_vectors = list(fitness_batches_matrix)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_batch == unique_error_vectors[i]]
        for ind in cands:
            ind.ties = len(cands)
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    batches = list(range(0,n_batches))
    
    for _ in range(k):
        random.shuffle(batches)
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        count_ = 0
        while len(batches) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_batch[batches[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[batches[0]] == best_val_for_case]
            del batches[0]
            
        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        
        batches = list(range(0,n_batches))

    return selected_individuals

def selDynBatchEpsilonLexi2_nodesCountTies(individuals, k, batch_size):
    """
        
    """
    selected_individuals = []
    
    #pop_size = len(individuals)
    l_batches = math.ceil(np.shape(individuals[0].fitness_each_sample)[0] / batch_size)
    
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix_original = np.array(error_vectors) # inds (rows) x samples (cols)
    fitness_cases_matrix_original = add_index_column(fitness_cases_matrix_original) # add a column with indexes in the beginning
    fitness_cases_matrix_original = fitness_cases_matrix_original.transpose() # samples (rows) x inds (cols); row 0 contains the indexes
    
    for i in range(k):
        #Shuffle fitness cases
        fitness_cases_matrix = shuffle_rows_except_first(fitness_cases_matrix_original)
        #Create batches
        fitness_cases_matrix = aggregate_rows(fitness_cases_matrix, batch_size)
        
        l, c = fitness_cases_matrix.shape
        
        while l > 1 and c > 1: #we have more than one individual in the pool and more than one fitness case to test
            if np.all(fitness_cases_matrix[1] == fitness_cases_matrix[1, 0]): #if all individuals have the same fitness value for this case, we won't be able to filter anything
                pass
            else:
                min_ = np.nanmin(fitness_cases_matrix[1])    
                mad = median_abs_deviation(fitness_cases_matrix[1]) #mad for the second row
                fitness_cases_matrix = remove_columns(fitness_cases_matrix, min_ + mad) #filter individuals
            
            fitness_cases_matrix = remove_row(fitness_cases_matrix, 1) #remove the assessed test case (second row, since the first one contains the indexes)
            
            l, c = fitness_cases_matrix.shape

        #print(fitness_cases_matrix[0].astype(int))        
        remaining_candidates = [candidates[j] for j in fitness_cases_matrix[0].astype(int)] #indexes of the remaining candidates
        if len(remaining_candidates) > 1:
            f = min
            best_val_for_nodes = f(map(lambda x: x.nodes, remaining_candidates))
            smallest_size_candidates = [ind for ind in remaining_candidates if ind.nodes == best_val_for_nodes]
            selected_ind = random.choice(smallest_size_candidates) #if there are still more than one candidate with the same size, we choose randomly
        else:
            selected_ind = remaining_candidates[0]
        selected_ind.n_cases = l_batches - l #number of batches used in the filtering process
        selected_ind.ties = len(remaining_candidates)
        selected_individuals.append(selected_ind) #Select the remaining candidate

    return selected_individuals
        
def selBatchEpsilonLexi2_nodesCountOld(individuals, k, batch_size=2):
    """
    different batches for select each individual    
    """
    selected_individuals = []
    error_vectors = [ind.fitness_each_sample for ind in individuals]
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    pop_size = len(individuals)
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    n_batches = math.ceil(l_samples / batch_size)
    
    candidates = individuals
    
    for _ in range(k):
        cases = list(range(0,l_samples))
        random.shuffle(cases)
        fitness_batches_matrix = np.zeros([pop_size, n_batches], dtype=float) # inds (rows) x samples (cols)
        #partitions
        for i in range(n_batches-1):
            for _ in range(batch_size):
                fitness_batches_matrix[:,i] += fitness_cases_matrix[:,cases[0]]
                del cases[0]
        for case in cases:
            fitness_batches_matrix[:,n_batches-1] += fitness_cases_matrix[:,case]

        min_ = np.nanmin(fitness_batches_matrix, axis=0)
        mad = median_abs_deviation(fitness_batches_matrix, axis=0)

        for i in range(len(candidates)):
            candidates[i].fitness_each_batch = [0] * n_batches
            for j in range(n_batches):
                if fitness_batches_matrix[i][j] <= min_[j] + mad[j]:
                    fitness_batches_matrix[i][j] = 1
                    candidates[i].fitness_each_batch[j] = 1
                else:
                    fitness_batches_matrix[i][j] = 0
                    candidates[i].fitness_each_batch[j] = 0
                
        error_vectors = list(fitness_batches_matrix)

        unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
        unique_error_vectors = [list(i) for i in unique_error_vectors]
        
        candidates_prefiltered_set = []
        for i in range(len(unique_error_vectors)):
            cands = [ind for ind in candidates if ind.fitness_each_batch == unique_error_vectors[i]]
            f = min
            best_val_for_nodes = f(map(lambda x: x.nodes, cands))
            cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
            candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        count_ = 0
        while count_ < n_batches and len(pool) > 1:
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_batch[count_], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[count_] == best_val_for_case]
            count_ += 1

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate

    return selected_individuals

def selEpsilonLexicaseCount(individuals, k):
    """
        
    """
    selected_individuals = []
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix, axis=0)
    #mad = robust.mad(fitness_cases_matrix, axis=0, c=1.0)
    mad = median_abs_deviation(fitness_cases_matrix, axis=0)
    
    for i in range(len(candidates)):
        for j in range(l_samples):
            if fitness_cases_matrix[i][j] <= min_[j] + mad[j]:
                fitness_cases_matrix[i][j] = 1
                candidates[i].fitness_each_sample[j] = 1
            else:
                fitness_cases_matrix[i][j] = 0
                candidates[i].fitness_each_sample[j] = 0
                
    error_vectors = list(fitness_cases_matrix)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selLexicase(individuals, k):
    """
    """
    selected_individuals = []
    valid_individuals = [i for i in individuals if not i.invalid]
    l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
    
    cases = list(range(0,l_samples))
    candidates = valid_individuals
    
    for i in range(k):
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            candidates_update = [i for i in candidates if i.fitness_each_sample[cases[0]] == True]
            
            if len(candidates_update) == 0:
                #no candidate correctly predicted the case
                pass
            else:
                candidates = candidates_update    
            del cases[0]                    

        #If there is only one candidate remaining, it will be selected
        #If there are more than one, the choice will be made randomly
        selected_individuals.append(random.choice(candidates))
        
        cases = list(range(0,l_samples))
        candidates = valid_individuals

    return selected_individuals

def selLexicaseCount(individuals, k):
    """Same as Lexicase Selection, but counting attempts of filtering and
    updating respective attributes on ind.
    
    If some ind has fitness equal to zero, do not enter in the loop.
    Instead, select randomly within the inds with fitness equal to zero.
    """
    selected_individuals = []
    valid_individuals = [i for i in individuals if not i.invalid]
    l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    
    #For analysing Lexicase selection
    samples_attempted = [0]*l_samples
    samples_used = [0]*l_samples
    samples_unsuccessful1 = [0]*l_samples
    samples_unsuccessful2 = [0]*l_samples
    inds_to_choose = [0]*k
    times_chosen = [0]*4
    
    cases = list(range(0,l_samples))
    #fit_weights = valid_individuals[0].fitness.weights
    candidates = valid_individuals
    
    if len(inds_fitness_zero) > 0:
        for i in range(k):
            selected_individuals.append(random.choice(inds_fitness_zero))
            inds_to_choose[i] = len(inds_fitness_zero)
            if len(inds_fitness_zero) == 1:
                times_chosen[0] += 1 #The choise was made by error
            else:
                times_chosen[3] += 1 #The choise was made by randomly
        samples_attempted = [x+k for x in samples_attempted]
        samples_used = [x+1 for x in samples_used]
        samples_unsuccessful1 = [x+k-1 for x in samples_unsuccessful1]
        
        return selected_individuals, samples_attempted, samples_used, samples_unsuccessful1, samples_unsuccessful2, inds_to_choose, times_chosen

    for i in range(k):
        #cases = list(range(len(valid_individuals[0].fitness.values)))
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            print(cases[0])
            print(candidates[0].fitness_each_sample[cases[0]])
            print(type(True))
            #f = min if fit_weights[cases[0]] < 0 else max
            candidates_update = [i for i in candidates if i.fitness_each_sample[cases[0]] == True]
            
            samples_attempted[cases[0]] += 1
            if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0):
                samples_used[cases[0]] += 1
            if (len(candidates_update) == len(candidates)):
                samples_unsuccessful1[cases[0]] += 1
            if len(candidates_update) == 0:
                samples_unsuccessful2[cases[0]] += 1
            
            if len(candidates_update) == 0:
                #no candidate correctly predicted the case
                pass
            else:
                candidates = candidates_update    
            del cases[0]                    

            #best_val_for_case = f(map(lambda x: x.fitness.values[cases[0]], candidates))

            #candidates = list(filter(lambda x: x.fitness.values[cases[0]] == best_val_for_case, candidates))
            #cases.pop(0)

        #If there is only one candidate remaining, it will be selected
        if len(candidates) == 1:
            selected_individuals.append(candidates[0])
            inds_to_choose[i] = 1
            times_chosen[0] += 1 #The choise was made by fitness
        else: #If there are more than one, the choice will be made randomly
            selected_individuals.append(random.choice(candidates))
            inds_to_choose[i] = len(candidates)
            times_chosen[3] += 1 #The choise was made by randomly
        
        cases = list(range(0,l_samples))
        candidates = valid_individuals

    return selected_individuals, samples_attempted, samples_used, samples_unsuccessful1, samples_unsuccessful2, inds_to_choose, times_chosen

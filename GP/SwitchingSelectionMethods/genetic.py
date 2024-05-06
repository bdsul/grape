# -*- coding: utf-8 -*-
"""
Created on Tue May 10 06:53:28 2022

@author: allan
"""

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

def selEpsilonLexi2_nodesCountTies(individuals, k):
    """
    same as selEpsilonLexi2_nodesCount, but also registers the number of ties in the selected individual in the attribute 'ties'
   
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix, axis=0)

    mad = median_abs_deviation(fitness_cases_matrix, axis=0)
    epsilon = mad
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
    
def selDownSampledEpsilonLexi2_nodesCountTies(individuals, k, s=0.1):
    """
        
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    #Parameters for down-sampling
    num_columns = l_samples
    sample_size = int(num_columns * s)
    
    cases = list(range(0,sample_size))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    #down_sampled_error_vectors = random.sample(error_vectors, int(s * len(individuals)))
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    
    #Down-sampling
    sampled_columns_indices = np.random.choice(num_columns, size=sample_size, replace=False)
    sampled_array = fitness_cases_matrix[:, sampled_columns_indices]
    
    #Pre-filtering    
    min_ = np.nanmin(sampled_array, axis=0)
    mad = median_abs_deviation(sampled_array, axis=0)
    avg_epsilon = np.mean(mad)
    
    for i in range(len(candidates)):
        candidates[i].fitness_each_downsampled = [None] * sample_size
        for j in range(sample_size):
            if sampled_array[i][j] <= min_[j] + mad[j]:
                sampled_array[i][j] = 1
                candidates[i].fitness_each_downsampled[j] = 1
            else:
                sampled_array[i][j] = 0
                candidates[i].fitness_each_downsampled[j] = 0
          
    n_zeros = count_zeros(sampled_array) #number of zeros in the matrix with discrete fitness cases
    avg_zeros = n_zeros / len(individuals) #average number of zeros per individual
    avg_zeros = avg_zeros / sample_size #represent as a percentage of the number of samples
    
    error_vectors = list(sampled_array)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_downsampled == unique_error_vectors[i]]
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
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_downsampled[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_downsampled[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        pool[0].avg_zeros = avg_zeros
        pool[0].avg_epsilon = avg_epsilon
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,sample_size)) #Recreate the list of cases
        
        index = individuals.index(pool[0])
        indexes.append(index)
        
    selected_individuals[0].unique_selected = len(set(indexes)) / len(individuals) # percentage of unique inds selected

    return selected_individuals

def selDownSampledEpsilonLexicase(individuals, k, s=0.1):
    """
        
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    #Parameters for down-sampling
    num_columns = l_samples
    sample_size = int(num_columns * s)
    
    cases = list(range(0,sample_size))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    #down_sampled_error_vectors = random.sample(error_vectors, int(s * len(individuals)))
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    
    #Down-sampling
    sampled_columns_indices = np.random.choice(num_columns, size=sample_size, replace=False)
    sampled_array = fitness_cases_matrix[:, sampled_columns_indices]
    
    #Pre-filtering    
    min_ = np.nanmin(sampled_array, axis=0)
    mad = median_abs_deviation(sampled_array, axis=0)
    
    for i in range(len(candidates)):
        candidates[i].fitness_each_downsampled = [None] * sample_size
        for j in range(sample_size):
            if sampled_array[i][j] <= min_[j] + mad[j]:
                sampled_array[i][j] = 1
                candidates[i].fitness_each_downsampled[j] = 1
            else:
                sampled_array[i][j] = 0
                candidates[i].fitness_each_downsampled[j] = 0
                
    error_vectors = list(sampled_array)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_downsampled == unique_error_vectors[i]]
        for ind in cands:
            ind.ties = len(cands)
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
            best_val_for_case = f(map(lambda x: x.fitness_each_downsampled[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_downsampled[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,sample_size)) #Recreate the list of cases

    return selected_individuals

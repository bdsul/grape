# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:21:08 2021

@author: allan
"""

import grape
import algorithms
from functions import not_, and_, or_, nand_, nor_

from os import path
import pandas as pd
import numpy as np
from deap import creator, base, tools

import random

problem = 'parity3'

if problem == 'parity3':
    X_train = np.zeros([3,8], dtype=bool)
    Y_train = np.zeros([8,], dtype=bool)

    data = pd.read_table(r"datasets/parity3.csv")
    for i in range(3):
        for j in range(8):
            X_train[i,j] = data['d'+ str(i)].iloc[j]
    for i in range(8):
        Y_train[i] = data['output'].iloc[i]
        
    GRAMMAR_FILE = 'parity3.bnf'

elif problem == 'parity4':
    X_train = np.zeros([4,16], dtype=bool)
    Y_train = np.zeros([16,], dtype=bool)

    data = pd.read_table(r"datasets/parity4.csv")
    for i in range(4):
        for j in range(16):
            X_train[i,j] = data['d'+ str(i)].iloc[j]
    for i in range(16):
        Y_train[i] = data['output'].iloc[i]
        
    GRAMMAR_FILE = 'parity4.bnf'

elif problem == 'parity5':
    X_train = np.zeros([5,32], dtype=bool)
    Y_train = np.zeros([32,], dtype=bool)

    data = pd.read_table(r"datasets/parity5.csv")
    for i in range(5):
        for j in range(32):
            X_train[i,j] = data['d'+ str(i)].iloc[j]
    for i in range(32):
        Y_train[i] = data['output'].iloc[i]
        
    GRAMMAR_FILE = 'parity5.bnf'

BNF_GRAMMAR = grape.Grammar(r"grammars/" + GRAMMAR_FILE)

def mae(y, yhat):
    """
    Calculate mean absolute error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean absolute error.
    """
    
    compare = np.equal(y,yhat)

    return 1 - np.mean(compare)

def fitness_eval(individual, points):
    x = points[0]
    Y = points[1]
    
    if individual.invalid == True:
        return np.NaN,

    # Evaluate the expression
    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError):
        # FP err can happen through eg overflow (lots of pow/exp calls)
        # ZeroDiv can happen when using unprotected operators
        return np.NaN,
    assert np.isrealobj(pred)
    
    fitness = mae(Y, pred)
    
    return fitness,

toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

#toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual) 
toolbox.register("populationCreator", grape.random_initialisation, creator.Individual) 
#toolbox.register("populationCreator", grape.PI_Grow, creator.Individual) 

toolbox.register("evaluate", fitness_eval)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=6)

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

POPULATION_SIZE = 1000
MAX_GENERATIONS = 200
P_CROSSOVER = 0.8
P_MUTATION = 0.01
ELITE_SIZE = round(0.01*POPULATION_SIZE) #it should be smaller or equal to HALLOFFAME_SIZE
HALLOFFAME_SIZE = round(0.01*POPULATION_SIZE) #it should be at least 1

CODON_CONSUMPTION = 'lazy'
RANDOM_SEED = 0 #Pay attention that the seed is set up inside the loop of runs, so you are going to have similar runs

INIT_GENOME_LENGTH = 30 #used only for random initialisation
random_initilisation = True #put True if you use random initialisation

MAX_INIT_TREE_DEPTH = 10
MIN_INIT_TREE_DEPTH = 2
MAX_TREE_DEPTH = 90
MAX_WRAPS = 0
CODON_SIZE = 255

N_RUNS = 3

for i in range(N_RUNS):
    print()
    print()
    print("Run:", i+1)
    print()
    
    random.seed(RANDOM_SEED) #Comment this line or set a different RANDOM_SEED each run if you want distinct results
    
    # create initial population (generation 0):
    if random_initilisation:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE, 
                                           bnf_grammar=BNF_GRAMMAR, 
                                           init_genome_length=INIT_GENOME_LENGTH,
                                           max_init_depth=MAX_TREE_DEPTH, 
                                           codon_size=CODON_SIZE,
                                           codon_consumption=CODON_CONSUMPTION
                                           )
    else:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE, 
                                           bnf_grammar=BNF_GRAMMAR, 
                                           min_init_depth=MIN_INIT_TREE_DEPTH,
                                           max_init_depth=MAX_INIT_TREE_DEPTH,
                                           codon_size=CODON_SIZE,
                                           codon_consumption=CODON_CONSUMPTION
                                            )
    
    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALLOFFAME_SIZE)
    
    # prepare the statistics object:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)
    
    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                              bnf_grammar=BNF_GRAMMAR, 
                                              codon_size=CODON_SIZE, 
                                              max_tree_depth=MAX_TREE_DEPTH,
                                              points_train=[X_train, Y_train], 
                                              codon_consumption=CODON_CONSUMPTION,
                                              stats=stats, halloffame=hof, verbose=False)
    
    import textwrap
    best = hof.items[0].phenotype
    print("Best individual: \n","\n".join(textwrap.wrap(best,80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
    print("Depth: ", hof.items[0].depth)
    print("Length of the genome: ", len(hof.items[0].genome))
    print(f'Used portion of the genome: {hof.items[0].used_codons/len(hof.items[0].genome):.2f}')
    
    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")
    best_ind_length = logbook.select("best_ind_length")
    avg_length = logbook.select("avg_length")

    selection_time = logbook.select("selection_time")
    generation_time = logbook.select("generation_time")
    gen, invalid = logbook.select("gen", "invalid")
    avg_used_codons = logbook.select("avg_used_codons")
    best_ind_used_codons = logbook.select("best_ind_used_codons")
    
    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")

    best_ind_depth = logbook.select("best_ind_depth")
    avg_depth = logbook.select("avg_depth")

    structural_diversity = logbook.select("structural_diversity") 
    
    import csv
    import random
    r = random.randint(1,1e10)
    
    header = ['gen', 'invalid', 'avg', 'std', 'min', 'max',
              'best_ind_length', 'avg_length', 
              'best_ind_nodes', 'avg_nodes', 
              'best_ind_depth', 'avg_depth', 
              'avg_used_codons', 'best_ind_used_codons', 
              'structural_diversity',
              'selection_time', 'generation_time']
    with open("results/" + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], invalid[value], mean_fitness_values[value],
                             std_fitness_values[value], min_fitness_values[value],
                             max_fitness_values[value], 
                             best_ind_length[value], 
                             avg_length[value], 
                             best_ind_nodes[value],
                             avg_nodes[value],
                             best_ind_depth[value],
                             avg_depth[value],
                             avg_used_codons[value],
                             best_ind_used_codons[value], 
                             structural_diversity[value],
                             selection_time[value], 
                             generation_time[value]])
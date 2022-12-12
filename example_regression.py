# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:21:08 2021

@author: allan
"""

import grape
import algorithms
from functions import add, sub, mul, pdiv, plog, exp, psqrt

import random

from os import path
import pandas as pd
import numpy as np
from deap import creator, base, tools

import warnings
warnings.filterwarnings("ignore")

problem = 'vladislavleva4'

def setDataSet(problem):
    if problem == 'pagie1':
        X_train = np.zeros([2,676], dtype=float)
        Y_train = np.zeros([676,], dtype=float)
    
        data_train = pd.read_table(r"datasets/Pagie1_train.txt")
        for i in range(2):
            for j in range(676):
                X_train[i,j] = data_train['x'+ str(i)].iloc[j]
        for i in range(676):
            Y_train[i] = data_train['response'].iloc[i]
    
        X_test = np.zeros([2,10000], dtype=float)
        Y_test = np.zeros([10000,], dtype=float)
    
        data_test = pd.read_table(r"datasets/Pagie1_test.txt")
        for i in range(2):
            for j in range(10000):
                X_test[i,j] = data_test['x'+ str(i)].iloc[j]
        for i in range(10000):
            Y_test[i] = data_test['response'].iloc[i]
    
        GRAMMAR_FILE = 'Pagie1.bnf'
        
    elif problem == 'vladislavleva4':
        X_train = np.random.uniform(0.05, 6.05, (5, 1024))
        Y_train = np.zeros([1024,], dtype=float)
        for i in range(1024):
            Y_train[i] = 10/(5 + (X_train[0,i] - 3)**2 + (X_train[1,i] - 3)**2 + (X_train[2,i] - 3)**2 + (X_train[3,i] - 3)**2 + (X_train[4,i] - 3)**2)
    
        X_test = np.random.uniform(-0.25, 6.35, (5, 5000))
        Y_test = np.zeros([5000,], dtype=float)
        for i in range(5000):
            Y_test[i] = 10/(5 + (X_test[0,i] - 3)**2 + (X_test[1,i] - 3)**2 + (X_test[2,i] - 3)**2 + (X_test[3,i] - 3)**2 + (X_test[4,i] - 3)**2)
    
        GRAMMAR_FILE = 'Vladislavleva4.bnf'
    
    elif problem == 'Dow':
        X_train = np.zeros([57,747], dtype=float)
        Y_train = np.zeros([747,], dtype=float)
        
        data_train = pd.read_table(r"datasets/DowNorm_train.txt")
        for i in range(56):
            for j in range(747):
                  X_train[i,j] = data_train['x'+ str(i+1)].iloc[j]
        for i in range(747):
            Y_train[i] = data_train['y'].iloc[i]
        
        X_test = np.zeros([57,319], dtype=float)
        Y_test = np.zeros([319,], dtype=float)
        
        data_test = pd.read_table(r"datasets/DowNorm_test.txt")
        for i in range(56):
            for j in range(319):
                X_test[i,j] = data_test['x'+ str(i+1)].iloc[j]
        for i in range(319):
            Y_test[i] = data_test['y'].iloc[i]
        
        GRAMMAR_FILE = 'Dow.bnf'
        
    BNF_GRAMMAR = grape.Grammar(r"grammars/" + GRAMMAR_FILE)
    
    return X_train, Y_train, X_test, Y_test, BNF_GRAMMAR

def fitness_eval(individual, points):
    #points = [X, Y]
    x = points[0]
    y = points[1]
    
    if individual.invalid == True:
        return np.NaN,

    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        return np.NaN,
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("evaluation error", err)
            raise
    assert np.isrealobj(pred)
    
    try:
        fitness = np.mean(np.square(y - pred))
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = np.NaN
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("fitness error", err)
            raise
        
    if fitness == float("inf"):
        return np.NaN,
    
    return fitness,

toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual) 
#toolbox.register("populationCreator", grape.random_initialisation, creator.Individual) 
#toolbox.register("populationCreator", grape.PI_Grow, creator.Individual) 

toolbox.register("evaluate", fitness_eval)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=7)

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)
    
POPULATION_SIZE = 200
MAX_GENERATIONS = 200
P_CROSSOVER = 0.8
P_MUTATION = 0.01
ELITE_SIZE = 0#round(0.01*POPULATION_SIZE) #it should be smaller or equal to HALLOFFAME_SIZE
HALLOFFAME_SIZE = 1#round(0.01*POPULATION_SIZE) #it should be at least 1

MIN_INIT_GENOME_LENGTH = 30 #used only for random initialisation
MAX_INIT_GENOME_LENGTH = 50
random_initilisation = False #put True if you use random initialisation

MAX_INIT_TREE_DEPTH = 13 #equivalent to 6 in GP with this grammar
MIN_INIT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 35 #equivalent to 17 in GP with this grammar
MAX_WRAPS = 0
CODON_SIZE = 255

CODON_CONSUMPTION = 'lazy'
GENOME_REPRESENTATION = 'list'
MAX_GENOME_LENGTH = None

REPORT_ITEMS = ['gen', 'invalid', 'avg', 'std', 'min', 'max', 
                'fitness_test',
          'best_ind_length', 'avg_length', 
          'best_ind_nodes', 'avg_nodes', 
          'best_ind_depth', 'avg_depth', 
          'avg_used_codons', 'best_ind_used_codons', 
        #  'behavioural_diversity',
          'structural_diversity', #'fitness_diversity',
          'selection_time', 'generation_time']

N_RUNS = 1

for i in range(N_RUNS):
    print()
    print()
    print("Run:", i)
    print()
    
    RANDOM_SEED = i
    
    np.random.seed(RANDOM_SEED)
    X_train, Y_train, X_test, Y_test, BNF_GRAMMAR = setDataSet(problem) #We set up this inside the loop for the case in which the data is defined randomly

    random.seed(RANDOM_SEED) 
    
    # create initial population (generation 0):
    if random_initilisation:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_genome_length=MIN_INIT_GENOME_LENGTH,
                                           max_init_genome_length=MAX_INIT_GENOME_LENGTH,
                                           max_init_depth=MAX_TREE_DEPTH,
                                           codon_size=CODON_SIZE,
                                           codon_consumption=CODON_CONSUMPTION,
                                           genome_representation=GENOME_REPRESENTATION
                                           )
    else:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_depth=MIN_INIT_TREE_DEPTH,
                                           max_init_depth=MAX_INIT_TREE_DEPTH,
                                           codon_size=CODON_SIZE,
                                           codon_consumption=CODON_CONSUMPTION,
                                           genome_representation=GENOME_REPRESENTATION
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
                                              max_genome_length=MAX_GENOME_LENGTH,
                                              points_train=[X_train, Y_train],
                                              points_test=[X_test, Y_test],
                                              codon_consumption=CODON_CONSUMPTION,
                                              report_items=REPORT_ITEMS,
                                              genome_representation=GENOME_REPRESENTATION,
                                              stats=stats, halloffame=hof, verbose=False)
    
    import textwrap
    best = hof.items[0].phenotype
    print("Best individual: \n","\n".join(textwrap.wrap(best,80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
    print("Test Fitness: ", fitness_eval(hof.items[0], [X_test,Y_test])[0])
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
    
    fitness_test = logbook.select("fitness_test")
    
    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")

    best_ind_depth = logbook.select("best_ind_depth")
    avg_depth = logbook.select("avg_depth")

    structural_diversity = logbook.select("structural_diversity") 
    
    import csv
    r = RANDOM_SEED
    
    header = REPORT_ITEMS
    
    with open(r"./results/" + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], invalid[value], mean_fitness_values[value],
                             std_fitness_values[value], min_fitness_values[value],
                             max_fitness_values[value], 
                             fitness_test[value],
                             best_ind_length[value], 
                             avg_length[value], 
                             best_ind_nodes[value],
                             avg_nodes[value],
                             best_ind_depth[value],
                             avg_depth[value],
                             avg_used_codons[value],
                             best_ind_used_codons[value], 
                           #  behavioural_diversity[value],
                             structural_diversity[value],
                          #   fitness_diversity[value],
                             selection_time[value], 
                             generation_time[value]])
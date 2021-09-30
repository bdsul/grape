# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:21:08 2021

@author: allan
"""

#from ponyge2_adapted_files import Grammar, Individual, initialisation_PI_Grow, crossover_onepoint, mutation_int_flip_per_codon
from ponyge2_adapted_files import Grammar, ge
import algorithms
from functions import div, plog, psqrt, exp

from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import creator, base, tools

import warnings
warnings.filterwarnings("ignore")

problem = 'Dow'

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
  X_train = np.zeros([5,1024], dtype=float)
  Y_train = np.zeros([1024,], dtype=float)

  data_train = pd.read_table(r"datasets/Vladislavleva4_train.txt")
  for i in range(5):
      for j in range(1024):
            X_train[i,j] = data_train['x'+ str(i)].iloc[j]
  for i in range(1024):
      Y_train[i] = data_train['response'].iloc[i]

  X_test = np.zeros([5,5000], dtype=float)
  Y_test = np.zeros([5000,], dtype=float)

  data_test = pd.read_table(r"datasets/Vladislavleva4_test.txt")
  for i in range(5):
      for j in range(5000):
          X_test[i,j] = data_test['x'+ str(i)].iloc[j]
  for i in range(5000):
      Y_test[i] = data_test['response'].iloc[i]

  GRAMMAR_FILE = 'Vladislavleva4.bnf'

elif problem == 'randomData':
  X_train = np.zeros([5,135], dtype=float)
  Y_train = np.zeros([135,], dtype=float)

  data_train = pd.read_table(r"datasets/randomData_train2.csv")
  for i in range(5):
      for j in range(135):
            X_train[i,j] = data_train['x'+ str(i)].iloc[j]
  for i in range(135):
      Y_train[i] = data_train['response'].iloc[i]

  X_test = np.zeros([5,15], dtype=float)
  Y_test = np.zeros([15,], dtype=float)

  data_test = pd.read_table(r"datasets/randomData_test2.csv")
  for i in range(5):
      for j in range(15):
          X_test[i,j] = data_test['x'+ str(i)].iloc[j]
  for i in range(15):
      Y_test[i] = data_test['response'].iloc[i]

  GRAMMAR_FILE = 'randomData.bnf'
  
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
  
BNF_GRAMMAR = Grammar(path.join("grammars", GRAMMAR_FILE))

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
    assert np.isrealobj(pred)
    
    try:
        fitness = np.mean(np.square(y - pred))
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = np.NaN
        
    if fitness == float("inf"):
        return np.NaN,
    
    return fitness,

toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create('Individual', ge.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", ge.initialisation_PI_Grow, creator.Individual) 

toolbox.register("evaluate", fitness_eval)#, points=[X_train, Y_train])

# Tournament selection:
toolbox.register("select", ge.selTournament, tournsize=7)

# Single-point crossover:
toolbox.register("mate", ge.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", ge.mutation_int_flip_per_codon)
    

POPULATION_SIZE = 10000
MAX_GENERATIONS = 500
P_CROSSOVER = 0.8
P_MUTATION = 0.01
ELITE_SIZE = round(0.01*POPULATION_SIZE)

HALL_OF_FAME_SIZE = 1
MAX_INIT_TREE_DEPTH = 10
MIN_INIT_TREE_DEPTH = 1
MAX_TREE_DEPTH = 17
MAX_WRAPS = 0
CODON_SIZE = 255

# create initial population (generation 0):
population = toolbox.populationCreator(size=POPULATION_SIZE, 
                                        bnf_grammar=BNF_GRAMMAR, 
                                        min_init_tree_depth=MIN_INIT_TREE_DEPTH,
                                        max_init_tree_depth=MAX_INIT_TREE_DEPTH,
                                        max_tree_depth=MAX_TREE_DEPTH, 
                                        max_wraps=MAX_WRAPS,
                                        codon_size=CODON_SIZE
                                        )

# define the hall-of-fame object:
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

import math 
# prepare the statistics object:
#stats = tools.Statistics(key=lambda ind: ind.fitness.values if math.isnan(ind.fitness.values[0]) else None)#ind.fitness.values != np.inf else None)
#stats = tools.Statistics(key=lambda ind: ind.fitness.values[0] if not math.isnan(ind.fitness.values[0]) else np.NaN)#ind.fitness.values != np.inf else None)
stats = tools.Statistics(key=lambda ind: ind.fitness.values)# if not ind.invalid else (np.NaN,))#ind.fitness.values != np.inf else None)
stats.register("avg", np.nanmean)
stats.register("std", np.nanstd)
stats.register("min", np.nanmin)
stats.register("max", np.nanmax)

# perform the Grammatical Evolution flow:
population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                          ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                          bnf_grammar=BNF_GRAMMAR, codon_size=CODON_SIZE, 
                                          max_tree_depth=MAX_TREE_DEPTH, max_wraps=MAX_WRAPS,
                                          points_train=[X_train, Y_train], 
                                          points_test=[X_test, Y_test], 
                                          stats=stats, halloffame=hof, verbose=True)

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
fitness_test = logbook.select("fitness_test")
best_ind_length = logbook.select("best_ind_length")
avg_length = logbook.select("avg_length")
max_length = logbook.select("max_length")
selection_time = logbook.select("selection_time")
generation_time = logbook.select("generation_time")
gen, invalid = logbook.select("gen", "invalid")

plt.plot(gen, min_fitness_values, color='red', label="Training fitness")
plt.plot(gen, fitness_test, color='blue', label="Test fitness")
plt.legend(fontsize=12)
plt.xlabel('Generations', fontsize=14)
plt.ylabel('Best Fitness', fontsize=14)
plt.title('Best Fitness over Generations', fontsize=16)
plt.show()

plt.plot(gen, max_length, color='red', label="Maximum")
plt.plot(gen, avg_length, color='blue', label="Average")
plt.plot(gen, best_ind_length, color='green', label="Best individual")
plt.legend(fontsize=12)
plt.xlabel('Generations', fontsize=14)
plt.ylabel('Genome Length', fontsize=14)
plt.title('Genome Length over Generations', fontsize=16)
plt.show()
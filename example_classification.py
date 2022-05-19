# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:21:08 2021

@author: allan
"""

import grape
import algorithms
from functions import add, sub, mul, pdiv, neg, and_, or_, not_, less_than_or_equal, greater_than_or_equal

from os import path
import pandas as pd
import numpy as np
from deap import creator, base, tools
import random

from sklearn.model_selection import train_test_split

problem = 'heartDisease'

if problem == 'spambase':
    #There 1813 samples with class 1
    #We'll split into 70% for training and 30% for test, assuring the balanced data
    X = np.zeros([4601, 57], dtype=float)
    Y = np.zeros([4601,], dtype=int)

    data = pd.read_table(r"datasets/spambase.csv")
    for i in range(4601):
        for j in range(57):
            X[i,j] = data['d'+ str(j)].iloc[i]
    for i in range(4601):
        Y[i] = data['class'].iloc[i]
        
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    X_train = np.transpose(X_train)
    X_test = np.transpose(X_test)
        
    GRAMMAR_FILE = 'spambase.bnf'

if problem == 'heartDisease':
    #There are 297 samples
    #We'll split into 70% for training and 30% for test, assuring the balanced data
    #X = np.zeros([297, 13], dtype=float)
    #Y = np.zeros([297,], dtype=int)
    
    data =  pd.read_csv(r"datasets/processed.cleveland.data", sep=",")
    #There are some data missing on columns d11 and d12, so let's remove the rows
    data = data[data.ca != '?']
    data = data[data.thal != '?']
    
    #There are 160 samples with class 0, 54 with class 1, 35 with class 2,
    #35 with class 3 and 13 with class 4
    #Let's consider the class 0 and all the remaining as class 1
    Y = data['class'].to_numpy()
    for i in range(len(Y)):
        Y[i] = 1 if Y[i] > 0 else 0
    data = data.drop(['class'], axis=1)
    
    data.loc[:, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = (data.loc[:, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] - data.loc[:, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].mean())/data.loc[:, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].std()
    
    
    data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])#, prefix = ['cp']) = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])#, prefix = ['cp'])
    
    X = data.to_numpy()
  
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    X_train = np.transpose(X_train)
    X_test = np.transpose(X_test)
        
    GRAMMAR_FILE = 'heartDisease.bnf'

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
    
    try:
        Y_class = [1 if pred[i] > 0 else 0 for i in range(len(Y))]
    except (IndexError, TypeError):
        return np.NaN,
    fitness = mae(Y, Y_class)
    
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
toolbox.register("select", tools.selTournament, tournsize=6)

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

POPULATION_SIZE = 1000
MAX_GENERATIONS = 20
P_CROSSOVER = 0.8
P_MUTATION = 0.01
ELITE_SIZE = round(0.01*POPULATION_SIZE)

INIT_GENOME_LENGTH = 30 #used only for random initialisation
random_initilisation = False #put True if you use random initialisation

MAX_INIT_TREE_DEPTH = 10
MIN_INIT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 90
MAX_WRAPS = 0
CODON_SIZE = 255

N_RUNS = 3

for i in range(N_RUNS):
    print()
    print()
    print("Run:", i+1)
    print()

    # create initial population (generation 0):
    if random_initilisation:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE, 
                                           bnf_grammar=BNF_GRAMMAR, 
                                           init_genome_length=INIT_GENOME_LENGTH,
                                           max_init_depth=MAX_TREE_DEPTH, 
                                           codon_size=CODON_SIZE
                                           )
    else:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE, 
                                           bnf_grammar=BNF_GRAMMAR, 
                                           min_init_depth=MIN_INIT_TREE_DEPTH,
                                           max_init_depth=MAX_INIT_TREE_DEPTH,
                                           codon_size=CODON_SIZE
                                            )
    
    # define the hall-of-fame object:
    hof = tools.HallOfFame(ELITE_SIZE)
    
    # prepare the statistics object:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)
    
    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                              bnf_grammar=BNF_GRAMMAR, codon_size=CODON_SIZE, 
                                              max_tree_depth=MAX_TREE_DEPTH,
                                              points_train=[X_train, Y_train], 
                                              points_test=[X_test, Y_test], 
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
    import random
    r = random.randint(1,1e10)
    
    header = ['gen', 'invalid', 'avg', 'std', 'min', 'max', 'fitness_test', 
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
                             fitness_test[value],
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
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:21:08 2021

@author: allan
"""

#from ponyge2_adapted_files import Grammar, Individual, initialisation_PI_Grow, crossover_onepoint, mutation_int_flip_per_codon
from ponyge2_adapted_files import Grammar, ge
import algorithms
from functions import add, sub, mul, pdiv, neg, and_, or_, not_, less_than_or_equal, greater_than_or_equal

from os import path
import pandas as pd
import numpy as np
from deap import creator, base, tools
import random

from sklearn.model_selection import train_test_split

MAX_INIT_TREE_DEPTH = 10
MIN_INIT_TREE_DEPTH = 2
MAX_TREE_DEPTH = 17
MAX_WRAPS = 0
CODON_SIZE = 255

POPULATION_SIZE = 1000
ELITE_SIZE = round(0.01*POPULATION_SIZE)
P_CROSSOVER = 0.8 # probability for crossover
P_MUTATION = 0.01  # probability for mutating an individual
MAX_GENERATIONS = 500
HALL_OF_FAME_SIZE = 1

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

BNF_GRAMMAR = Grammar(path.join("grammars", GRAMMAR_FILE))

def mae(y, yhat):
    """
    Calculate mean absolute error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean absolute error.
    """
    
    compare = np.equal(y,yhat)

    return 1 - np.mean(compare)

def turnBool(pred):
    return (pred>0)

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

creator.create('Individual', ge.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", ge.initialisation_PI_Grow, creator.Individual) 

toolbox.register("evaluate", fitness_eval)#, points=[x for x in np.linspace(-1, 1, 100)])

# genetic operators:
toolbox.register("select", ge.selTournament, tournsize=7)

# Single-point crossover:
toolbox.register("mate", ge.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", ge.mutation_int_flip_per_codon)

def main():

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

    # prepare the statistics object:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
#    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    # perform the Genetic Algorithm flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                              bnf_grammar=BNF_GRAMMAR, codon_size=CODON_SIZE, 
                                              max_tree_depth=MAX_TREE_DEPTH, max_wraps=MAX_WRAPS,
                                              points_train=[X_train, Y_train], 
                                              points_test=[X_test, Y_test], 
                                              stats=stats, halloffame=hof, verbose=True)

if __name__ == "__main__":
    main() 
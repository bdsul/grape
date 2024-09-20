# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:09:54 2024

@author: Allan.DeLima
"""

import grape
import algorithms

from os import path
import pandas as pd
import numpy as np
from deap import creator, base, tools
import random

GRAMMAR_FILE = 'simpleIncrement.bnf'
BNF_GRAMMAR = grape.Grammar(r"grammars/" + GRAMMAR_FILE)

RANDOM_SEED = 42

toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual) 
#toolbox.register("populationCreator", grape.random_initialisation, creator.Individual) 
#toolbox.register("populationCreator", grape.PI_Grow, creator.Individual) 

#toolbox.register("evaluate", fitness_eval)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=7) #selLexicaseFilter

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

POPULATION_SIZE = 10
MAX_INIT_TREE_DEPTH = 13
MIN_INIT_TREE_DEPTH = 2

MAX_GENERATIONS = 200
P_CROSSOVER = 0.8
P_MUTATION = 0.01
ELITE_SIZE = 0#round(0.01*POPULATION_SIZE) #it should be smaller or equal to HALLOFFAME_SIZE
HALLOFFAME_SIZE = 1#round(0.01*POPULATION_SIZE) #it should be at least 1

MIN_INIT_GENOME_LENGTH = 95#*6
MAX_INIT_GENOME_LENGTH = 115#*6
random_initilisation = False #put True if you use random initialisation

CODON_CONSUMPTION = 'lazy'
GENOME_REPRESENTATION = 'list'
MAX_GENOME_LENGTH = None#'auto'

MAX_TREE_DEPTH = 35 #equivalent to 17 in GP with this grammar
MAX_WRAPS = 0
CODON_SIZE = 255

random.seed(RANDOM_SEED) 

population = toolbox.populationCreator(pop_size=POPULATION_SIZE, 
                                       bnf_grammar=BNF_GRAMMAR, 
                                       min_init_depth=MIN_INIT_TREE_DEPTH,
                                       max_init_depth=MAX_INIT_TREE_DEPTH,
                                       codon_size=CODON_SIZE,
                                       codon_consumption=CODON_CONSUMPTION,
                                       genome_representation=GENOME_REPRESENTATION
                                        )

for i in range(POPULATION_SIZE):
    print(population[i].phenotype)
    
    exec(population[i].phenotype)
    
    print("a = ", a)
    print("b = ", b)
    
    print()
    print()

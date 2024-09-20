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

POPULATION_SIZE = 10
MAX_INIT_TREE_DEPTH = 6
MIN_INIT_TREE_DEPTH = 4

CODON_CONSUMPTION = 'lazy'
GENOME_REPRESENTATION = 'list'

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

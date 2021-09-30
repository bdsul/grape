#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`algorithms` module is intended to contain some specific algorithms
in order to execute very common evolutionary algorithms. The method used here
are more for convenience than reference as the implementation of every
evolutionary algorithm may vary infinitely. Most of the algorithms in this
module use operators registered in the toolbox. Generally, the keyword used are
:meth:`mate` for crossover, :meth:`mutate` for mutation, :meth:`~deap.select`
for selection and :meth:`evaluate` for evaluation.

You are encouraged to write your own algorithms in order to make them do what
you really want them to do.
"""

import random
import math
import numpy as np
import time

from deap import tools

def varAnd(population, toolbox, cxpb, mutpb,
           bnf_grammar, codon_size, max_tree_depth, max_wraps):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i],
                                                          bnf_grammar, max_tree_depth, max_wraps)
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        #if random.random() < mutpb:
        offspring[i], = toolbox.mutate(offspring[i], mutpb,
                                       codon_size, bnf_grammar, max_tree_depth, max_wraps)
        del offspring[i].fitness.values

    return offspring

def replacement(new_pop, old_pop, elite_size, pop_size):
    """
    Replaces the old population with the new population. The ELITE_SIZE best
    individuals from the previous population are appended to new pop regardless
    of whether or not they are better than the worst individuals in new pop.
    
    :param new_pop: The new population (e.g. after selection, variation, &
    evaluation).
    :param old_pop: The previous generation population, from which elites
    are taken.
    :return: The 'POPULATION_SIZE' new population with elites.
    """
    # Sort both populations.
    old_pop.sort(key=lambda x: float('inf') if math.isnan(x.fitness.values[0]) else x.fitness.values[0], reverse=False)
    new_pop.sort(key=lambda x: float('inf') if math.isnan(x.fitness.values[0]) else x.fitness.values[0], reverse=False)
     
    # Append the best ELITE_SIZE individuals from the old population to the
    # new population.
    for ind in old_pop[:elite_size]:
        new_pop.insert(0, ind)

#    for ind in old_pop:
#        if ind.fitness.values[0] == float('inf'):
#            print(ind.fitness.values[0])
#            print(ind.phenotype)
#            ind.fitness.values[0] = np.NaN

    # Return the top POPULATION_SIZE individuals of the new pop, including
    # elites.
    return new_pop[:pop_size]

def ge_eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, elite_size, 
                bnf_grammar, codon_size, max_tree_depth, max_wraps,
                points_train, points_test=None, stats=None, halloffame=None, 
                verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_, and includes Elitism.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param elite_size: The number of best individuals to be copied to the 
                    next generation.
    :params bnf_grammar, codon_size, max_tree_depth, max_wraps: Parameters 
                    used to mapper the individuals after crossover and
                    mutation in order to check if they are valid.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    
    logbook = tools.Logbook()
    if points_test:
        logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['fitness_test', 'best_ind_length', 'avg_length', 'max_length', 'selection_time', 'generation_time']
    else:
        logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['best_ind_length', 'avg_length', 'max_length', 'selection_time', 'generation_time']

    start_gen = time.time()        
    # Evaluate the individuals with an invalid fitness
#    invalid_ind = [ind for ind in population if not ind.fitness.valid]
#    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
#    for ind, fit in zip(invalid_ind, fitnesses):
#        ind.fitness.values = fit
    for ind in population:
        if not ind.fitness.valid:
            invalid_ind = ind
            ind.fitness.values = toolbox.evaluate(invalid_ind, points_train)
        
    invalid = 0
    for ind in population:
        if ind.invalid == True:
            invalid += 1
    
    end_gen = time.time()
    generation_time = end_gen-start_gen
        
    selection_time = 0
    
    population.sort(key=lambda x: float('inf') if math.isnan(x.fitness.values[0]) else x.fitness.values[0], reverse=False)
    
    if points_test:
        fitness_test = toolbox.evaluate(population[0], points_test)[0]
    
    length = [len(ind.genome) for ind in population]

    avg_length = sum(length)/len(length)
    max_length = max(length)
    
    best_ind_length = len(population[0].genome)
    
    record = stats.compile(population) if stats else {}
    if points_test:
        logbook.record(gen=0, invalid=invalid, **record, fitness_test=fitness_test, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       max_length=max_length, selection_time=selection_time, 
                       generation_time=generation_time)
    else:
        logbook.record(gen=0, invalid=invalid, **record,  
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       max_length=max_length, selection_time=selection_time, 
                       generation_time=generation_time)
    if verbose:
        print(logbook.stream)
#        x = logbook.stream.split()
#        print("                     fitness")
#        print(f'{x[0]:3} {x[1]:10} {x[2]:7} {x[3]:7} {x[4]:7} {x[5]:15} {x[6]:16}')
#        print(f'{int(x[7]):3} {int(x[8]):5} {float(x[9]):9.4f} {float(x[10]):7.4f} {float(x[11]):7.4f} {float(x[12]):10.4f} {float(x[13]):16.4f}')

    # Begin the generational process
    for gen in range(1, ngen + 1):
        start_gen = time.time()    
    
        # Select the next generation individuals
        start = time.time()    
        offspring = toolbox.select(population, len(population)-elite_size)
        end = time.time()
        selection_time = end-start

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb,
                           bnf_grammar, codon_size, max_tree_depth, max_wraps)

        # Evaluate the individuals with an invalid fitness
#        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
#        for ind, fit in zip(invalid_ind, fitnesses):
#            ind.fitness.values = fit
        for ind in offspring:
            if not ind.fitness.valid:
                invalid_ind = ind
                ind.fitness.values = toolbox.evaluate(invalid_ind, points_train)
            
        invalid = 0
        for ind in offspring:
            if ind.invalid == True:
                invalid += 1
            
        # Replace the current population by the offspring
        population[:] = replacement(offspring, population, elite_size=elite_size, pop_size=len(population))
        
#        fitness_test = toolbox.evaluate(population[0], points_test)[0]
        #i = 0
        #for ind in population:
        #    i += 1
        #    print(i, ind.fitness.values[0])
        
        valid = [ind for ind in population if not math.isnan(ind.fitness.values[0])]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(valid)
        
        length = [len(ind.genome) for ind in population]
        
        avg_length = sum(length)/len(length)
        max_length = max(length)
        best_ind_length = len(halloffame.items[0].genome)
        
        if points_test:
            fitness_test = toolbox.evaluate(halloffame.items[0], points_test)[0]        

        end_gen = time.time()
        generation_time = end_gen-start_gen
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        if points_test:
            logbook.record(gen=gen, invalid=invalid, **record, fitness_test=fitness_test, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       max_length=max_length, selection_time=selection_time, 
                       generation_time=generation_time)
        else:
            logbook.record(gen=gen, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       max_length=max_length, selection_time=selection_time, 
                       generation_time=generation_time)
                
        if verbose:
            print(logbook.stream)
#            x = logbook.stream.split("\t")
#            print(f'{int(x[0]):3} {int(x[1]):5} {float(x[2]):9.4f} {float(x[3]):7.4f} {float(x[4]):7.4f} {float(x[5]):10.4f} {float(x[6]):16.4f}')

    return population, logbook

import random

from deap import tools, gp
import math
import numpy as np

from functions import shuffle_rows_except_first, remove_row, add_index_column, remove_columns, aggregate_rows, represent_matrix_behaviour, remove_equal_rows, remove_equal_columns, find_equal_columns, remove_columns_with_different_value, aggregate_rows_sum, count_zeros_except_first_row, count_zeros, median_abs_deviation
import statistics

def varAnd(population, toolbox, cxpb, mutpb):
    r"""Part of an evolutionary algorithm applying only the variation part
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
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, ngen, points_train, 
             points_test=None, 
             report_items=None,
             stats=None,
             halloffame=None):

    logbook = tools.Logbook()
    if report_items:
        logbook.header = report_items
    else:
        logbook.header = ['gen', 'nevals', 'best_train_fitness', 'best_ind_mce', 
                      'best_ind_depth', 'best_ind_nodes', 'fitness_test']
    if 'lexicase_avg_ties_chosen_ind' in report_items: 
        if 'lexicase_avg_steps' not in report_items:
            raise ValueError("When reporting 'lexicase_avg_ties_chosen_ind', 'lexicase_avg_steps' must also be in report_items")

    # Evaluate the individuals with an invalid fitness
    new_inds = [ind for ind in population if not ind.fitness.valid]
    n_evals = len(new_inds)
    for ind in new_inds:
        ind.fitness.values = toolbox.evaluate(ind, points_train)
        
    if 'variance' in report_items or 'avg_epsilon' in report_items or 'avg_zeros' in report_items:
        error_vectors = [ind.fitness_each_sample for ind in new_inds] #real values
        fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
        min_ = np.nanmin(fitness_cases_matrix[:,:], axis=0)
        
    if 'variance' in report_items:
        variance = statistics.variance(fitness_cases_matrix.flatten()) #variance of the original matrix
    else:
        variance = 0

    if 'avg_epsilon' in report_items:
        epsilon = median_abs_deviation(fitness_cases_matrix[:,:], axis=0) #mad
        avg_epsilon = np.mean(epsilon)
    else:
        avg_epsilon = 0
        
    if 'avg_zeros' in report_items:
        fitness_cases_matrix = fitness_cases_matrix.transpose() # samples (rows) x inds (cols)
        fitness_cases_matrix[:] = represent_matrix_behaviour(fitness_cases_matrix[:], min_ + epsilon)
        
        n_zeros = count_zeros(fitness_cases_matrix) #number of zeros in the matrix with unique fitness cases
        avg_zeros = n_zeros / len(population) #average number of zeros per individual
        avg_zeros = avg_zeros / len(fitness_cases_matrix[:,0]) #represent as a percentage of the number of samples
    else:
        avg_zeros = 0
        
    if 'behavioural_diversity_fitness_cases' in report_items:
        unique_behaviours_fitness_cases = np.unique(fitness_cases_matrix, axis=0)
        behavioural_diversity_fitness_cases = len(unique_behaviours_fitness_cases)/len(population)            
    else:
        behavioural_diversity_fitness_cases = 0

    nodes = [getattr(ind, 'nodes') for ind in new_inds]
    avg_nodes = sum(nodes) / len(nodes)
    
    mce = [getattr(ind, 'mce') for ind in new_inds]
    avg_mce = sum(mce) / len(mce)    
    
    if 'behavioural_diversity' in report_items:
        behaviours = np.zeros([len(population), len(population[0].behaviour)], dtype=float)
    
    #for ind in offspring:
    for idx, ind in enumerate(population):
        if 'behavioural_diversity' in report_items:
            behaviours[idx, :] = ind.behaviour
        
    if 'behavioural_diversity' in report_items:
        unique_behaviours = np.unique(behaviours, axis=0)
    
    behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0

    if halloffame is not None:
        halloffame.update(population)
        best_train_fitness = halloffame.items[0].fitness.values[0]
        best_ind_mce = halloffame.items[0].mce
        best_ind_depth = halloffame.items[0].height
        best_ind_nodes = len(halloffame.items[0])
        print("gen =", 0, ", fitness =", best_train_fitness, ", MCE =", best_ind_mce, ", evals =", n_evals)
        
    fitness_test = np.NaN
    
    logbook.record(gen=0, nevals=n_evals, best_train_fitness=best_train_fitness,
                best_ind_mce=best_ind_mce, avg_mce=avg_mce,
                best_ind_depth=best_ind_depth,
                best_ind_nodes=best_ind_nodes,
                avg_nodes=avg_nodes,
                fitness_test=fitness_test,
                behavioural_diversity=behavioural_diversity,
                lexicase_avg_steps=0,
                lexicase_avg_ties_chosen_ind=0,
                avg_zeros=avg_zeros,
                avg_epsilon=avg_epsilon,
                variance=variance,
                unique_selected=0,
                behavioural_diversity_fitness_cases=behavioural_diversity_fitness_cases)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        if 'unique_selected' in report_items:
            unique_selected = offspring[0].unique_selected
        else:
            unique_selected = 0

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        new_inds = [ind for ind in offspring if not ind.fitness.valid]
        n_evals = len(new_inds)
        for ind in new_inds:
            ind.fitness.values = toolbox.evaluate(ind, points_train)
            
        nodes = [getattr(ind, 'nodes') for ind in offspring]
        avg_nodes = sum(nodes) / len(nodes)
        
        mce = [getattr(ind, 'mce') for ind in new_inds]
        avg_mce = sum(mce) / len(mce)    
        
        # Replace the current population by the offspring
        population[:] = offspring
        
        if 'behavioural_diversity' in report_items:
            behaviours = np.zeros([len(population), len(population[0].behaviour)], dtype=float)
        
        #for ind in offspring:
        for idx, ind in enumerate(population):
            if 'behavioural_diversity' in report_items:
                behaviours[idx, :] = ind.behaviour
            
        if 'behavioural_diversity' in report_items:
            unique_behaviours = np.unique(behaviours, axis=0)
            
        if 'behavioural_diversity_fitness_cases' in report_items:
            behaviours_fitness_cases = np.zeros([len(population), len(population[0].fitness_each_sample_discrete)], dtype=float)
            for idx, ind in enumerate(population):
                behaviours_fitness_cases[idx, :] = ind.fitness_each_sample_discrete
            unique_behaviours_fitness_cases = np.unique(behaviours_fitness_cases, axis=0)
            behavioural_diversity_fitness_cases = len(unique_behaviours_fitness_cases)/len(population)            
        else:
            behavioural_diversity_fitness_cases = 0
            
        if 'avg_zeros' in report_items:
            avg_zeros = [getattr(ind, 'avg_zeros') for ind in offspring] #for Lexicase it's the same value for every individual, because we have a single matrix of discrete values, but for batch Lexicase it's different, because we create the matrix only after the shuffle
            avg_zeros = sum(avg_zeros) / len(avg_zeros) #avg of the population
        else:
            avg_zeros = 0
        
        if 'avg_epsilon' in report_items:
            avg_epsilon = [getattr(ind, 'avg_epsilon') for ind in offspring] #for Lexicase it's the same value for every individual, because we have a single matrix of discrete values, but for batch Lexicase it's different, because we create the matrix only after the shuffle
            avg_epsilon = sum(avg_epsilon) / len(avg_epsilon) #avg of the population
        else:
            avg_epsilon = 0
            
        if 'variance' in report_items:
            error_vectors = [ind.fitness_each_sample for ind in offspring] #real values
            fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
            variance = statistics.variance(fitness_cases_matrix.flatten()) #variance of the original matrix
            
        behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0
        
        if halloffame is not None:
            halloffame.update(population)
            best_train_fitness = halloffame.items[0].fitness.values[0]
            best_ind_mce = halloffame.items[0].mce
            best_ind_depth = halloffame.items[0].height
            best_ind_nodes = len(halloffame.items[0])
            print("gen =", gen, ", fitness =", best_train_fitness, ", MCE =", best_ind_mce, ", evals =", n_evals)
            
        if points_test:
            if gen < ngen:
                fitness_test = np.NaN
            else:
                _ = toolbox.evaluate(halloffame.items[0], points_test)
                fitness_test = halloffame.items[0].mce
        else:            
            fitness_test = np.NaN

        if 'lexicase_avg_ties_chosen_ind' in report_items:
            attribute_values = [getattr(ind, 'ties') for ind in offspring]
            lexicase_avg_ties_chosen_ind = sum(attribute_values) / len(attribute_values)
        else:
            lexicase_avg_ties_chosen_ind = np.NaN
        if 'lexicase_avg_steps' in report_items:
            attribute_values = [getattr(ind, 'n_cases') for ind in offspring]
            lexicase_avg_steps = sum(attribute_values) / len(attribute_values)
        else:
            lexicase_avg_steps = np.NaN
        logbook.record(gen=gen, nevals=n_evals, best_train_fitness=best_train_fitness,
                   best_ind_mce=best_ind_mce, avg_mce=avg_mce,
                   best_ind_depth=best_ind_depth,
                   best_ind_nodes=best_ind_nodes,
                   avg_nodes=avg_nodes,
                   fitness_test=fitness_test,
                   behavioural_diversity=behavioural_diversity,
                   lexicase_avg_steps=lexicase_avg_steps,
                   lexicase_avg_ties_chosen_ind=lexicase_avg_ties_chosen_ind,
                   avg_zeros=avg_zeros,
                   avg_epsilon=avg_epsilon,
                   variance=variance,
                   unique_selected=unique_selected,
                   behavioural_diversity_fitness_cases=behavioural_diversity_fitness_cases)

    return population, logbook

def eaSimpleDistanceBatch(population, toolbox, cxpb, mutpb, ngen, points_train, 
                          normalised_distances, order_distances, batch_size,
             points_test=None, 
             report_items=None,
             stats=None,
             halloffame=None):

    logbook = tools.Logbook()
    if report_items:
        logbook.header = report_items
    else:
        logbook.header = ['gen', 'nevals', 'best_train_fitness', 'best_ind_mce', 
                      'best_ind_depth', 'best_ind_nodes', 'fitness_test']

    # Evaluate the individuals with an invalid fitness
    new_inds = [ind for ind in population if not ind.fitness.valid]
    n_evals = len(new_inds)
    for ind in new_inds:
        ind.fitness.values = toolbox.evaluate(ind, points_train)
        
    nodes = [getattr(ind, 'nodes') for ind in new_inds]
    avg_nodes = sum(nodes) / len(nodes)

    if halloffame is not None:
        halloffame.update(population)
        best_train_fitness = halloffame.items[0].fitness.values[0]
        best_ind_mce = halloffame.items[0].mce
        best_ind_depth = halloffame.items[0].height
        best_ind_nodes = len(halloffame.items[0])
        print("gen =", 0, ", MCE =", best_ind_mce, ", evals =", n_evals, 
              ", depth =", best_ind_depth, ", nodes =", best_ind_nodes, ", avg_nodes =", avg_nodes)
        
    fitness_test = np.NaN

    logbook.record(gen=0, nevals=n_evals, best_train_fitness=best_train_fitness,
                   best_ind_mce=best_ind_mce, best_ind_depth=best_ind_depth,
                   best_ind_nodes=best_ind_nodes,
                   fitness_test=fitness_test,
                   lexicase_avg_steps=0,
                   lexicase_avg_ties_chosen_ind=0)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population), normalised_distances, order_distances, batch_size)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        new_inds = [ind for ind in offspring if not ind.fitness.valid]
        n_evals = len(new_inds)
        for ind in new_inds:
            ind.fitness.values = toolbox.evaluate(ind, points_train)
            
        nodes = [getattr(ind, 'nodes') for ind in offspring]
        avg_nodes = sum(nodes) / len(nodes)

        # Replace the current population by the offspring
        population[:] = offspring
        
        if halloffame is not None:
            halloffame.update(population)
            best_train_fitness = halloffame.items[0].fitness.values[0]
            best_ind_mce = halloffame.items[0].mce
            best_ind_depth = halloffame.items[0].height
            best_ind_nodes = len(halloffame.items[0])
            print("gen =", gen, ", fitness =", best_train_fitness, ", MCE =", best_ind_mce, ", evals =", n_evals, 
                  ", depth =", best_ind_depth, ", nodes =", best_ind_nodes, ", avg_nodes =", avg_nodes)
            
        if points_test:
            if gen < ngen:
                fitness_test = np.NaN
            else:
                _ = toolbox.evaluate(halloffame.items[0], points_test)
                fitness_test = halloffame.items[0].mce
        else:            
            fitness_test = np.NaN

        attribute_values = [getattr(ind, 'n_cases') for ind in offspring]
        lexicase_avg_steps = sum(attribute_values) / len(attribute_values)
#        attribute_values = [getattr(ind, 'ties') for ind in offspring]
#        lexicase_avg_ties_chosen_ind = sum(attribute_values) / len(attribute_values)
        logbook.record(gen=0, nevals=n_evals, best_train_fitness=best_train_fitness,
                   best_ind_mce=best_ind_mce, best_ind_depth=best_ind_depth,
                   best_ind_nodes=best_ind_nodes,
                   fitness_test=fitness_test,
                   lexicase_avg_steps=lexicase_avg_steps)#,
 #                  lexicase_avg_ties_chosen_ind=lexicase_avg_ties_chosen_ind)
 
    return population, logbook


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    r"""Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population.

    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.

    This variation is named *Or* because an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    r"""This is the :math:`(\mu + \lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.

    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)

    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                    stats=None, halloffame=None, verbose=__debug__):
    r"""This is the :math:`(\mu~,~\lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(offspring, mu)

    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from **only** the offspring. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Care must be taken when the lambda:mu ratio is 1 to 1 as a
        non-stochastic selection will result in no selection at all as the
        operator selects *lambda* individuals from a pool of *mu*.


    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    return population, logbook


def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__):
    """This is algorithm implements the ask-tell model proposed in
    [Colette2010]_, where ask is called `generate` and tell is called `update`.

    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm generates the individuals using the :func:`toolbox.generate`
    function and updates the generation method with the :func:`toolbox.update`
    function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The pseudocode goes as follow ::

        for g in range(ngen):
            population = toolbox.generate()
            evaluate(population)
            toolbox.update(population)


    This function expects :meth:`toolbox.generate` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Colette2010] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and
       R. Le Riche (2010). On Object-Oriented Programming of Optimizers -
       Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.:
       Multidisciplinary Design Optimization in Computational Mechanics,
       Wiley, pp. 527-565;

    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(ngen):
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

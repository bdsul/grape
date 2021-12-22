import numpy as np

from collections import deque

from math import floor
from re import match, finditer, DOTALL, MULTILINE
from sys import maxsize
from random import random, choice, randrange, randint, shuffle, sample
from operator import attrgetter

class ge(object):
    
    class Individual(object):
        """
        A GE individual.
        """
    
        def __init__(self, genome, ind_tree, bnf_grammar, max_tree_depth, max_wraps, map_ind=True):
            """
            Initialise an instance of the individual class (i.e. create a new
            individual).
    
            :param genome: An individual's genome.
            :param ind_tree: An individual's derivation tree, i.e. an instance
            of the representation.tree.Tree class.
            :param map_ind: A boolean flag that indicates whether or not an
            individual needs to be mapped.
            """
            
            if map_ind:
                # The individual needs to be mapped from the given input
                # parameters.
                self.phenotype, self.genome, self.tree, self.nodes, self.invalid, \
                    self.depth, self.used_codons, self.n_wraps = mapper(genome, ind_tree, bnf_grammar, max_tree_depth, max_wraps)
    
            else:
                # The individual does not need to be mapped.
                self.genome, self.tree, self.n_wraps = genome, ind_tree, -1
            
    #        self.evaluation_time = 0
    
            #self.fitness = creator.FitnessMin#float('nan')#params['FITNESS_FUNCTION'].default_fitness
            self.runtime_error = False
            self.name = None
            self.fitness_val = None
            self.fitness_each_sample = [] #len = population_size; Each position receives 1 if prediction is correct, 0 if false.
                                     #if LEXICASE_EACH_BIT is True, each position receives the number of bits predicted correctly
            self.n_gates = None #number of gates in the phenotype
            self.n_gates_longest_path = None #number of gates in the longest path
            
            self.samples_attempted = None 
            self.samples_used = None 
            self.samples_unsuccessful1 = None 
            self.samples_unsuccessful2 = None 
            
                                 
    
        def deep_copy(self):
            """
            Copy an individual and return a unique version of that individual.
    
            :return: A unique copy of the individual.
            """
    
            # Create a new unique copy of the tree.
            new_tree = self.tree.__copy__()
    
            # Create a copy of self by initialising a new individual.
            new_ind = Individual(self.genome.copy(), new_tree, bnf_grammar, max_tree_depth, max_wraps, map_ind=False)
    
            # Set new individual parameters (no need to map genome to new
            # individual).
            new_ind.phenotype, new_ind.invalid = self.phenotype, self.invalid
            new_ind.depth, new_ind.nodes = self.depth, self.nodes
            new_ind.used_codons = self.used_codons
            new_ind.runtime_error = self.runtime_error
            
            new_ind.fitness_val = self.fitness_val
            #New attributes (maybe It is not necessary to update all here)
            new_ind.predict_result = self.predict_result
            new_ind.n_gates = self.n_gates
            new_ind.n_gates_longest_path = self.n_gates_longest_path
            
            new_ind.samples_attempted = self.samples_attempted
            new_ind.samples_used = self.samples_used
            new_ind.samples_unsuccessful1 = self.samples_unsuccessful1
            new_ind.samples_unsuccessful2 = self.samples_unsuccessful2
            
            new_ind.n_wraps = self.n_wraps
            
            
            return new_ind
    
    def initialisation_PI_Grow(ind_class, size, bnf_grammar, min_init_tree_depth, max_init_tree_depth, max_tree_depth, max_wraps, codon_size):
        """
        Create a population of size using Position Independent Grow and return.
    
        :param size: The size of the required population.
        :return: A full population of individuals.
        """
    
        # Calculate the range of depths to ramp individuals from.
        depths = range(min_init_tree_depth+1, max_init_tree_depth+1)
        population = []
    
        if size < 2:
            # If the population size is too small, can't use PI Grow
            # initialisation.
            print("Error: population size too small for PI Grow initialisation.")
            exit()
    
        elif not depths:
            # If we have no depths to ramp from, then params['MAX_INIT_DEPTH'] is
            # set too low for the specified grammar.
            s = "operators.initialisation.PI_grow\n" \
                "Error: Maximum initialisation depth too low for specified " \
                "grammar."
            raise Exception(s)
    
        else:
            if size < len(depths):
                # The population size is too small to fully cover all ramping
                # depths. Only ramp to the number of depths we can reach.
                depths = depths[:int(size)]
    
            # Calculate how many individuals are to be generated by each
            # initialisation method.
            times = int(floor(size/len(depths)))
            remainder = int(size - (times * len(depths)))
    
            # Iterate over depths.
            for depth in depths:
                # Iterate over number of required individuals per depth.
                for i in range(times):
    
                    # Generate individual using "Grow"
                    ind = generate_PI_ind_tree(ind_class, depth, bnf_grammar, max_tree_depth, max_wraps, codon_size)
    
                    # Append individual to population
                    population.append(ind)
    
            if remainder:
                # The full "size" individuals were not generated. The population
                #  will be completed with individuals of random depths.
                depths = list(depths)
                shuffle(depths)
    
            for i in range(remainder):
                depth = depths.pop()
    
                # Generate individual using "Grow"
                ind = generate_PI_ind_tree(ind_class, depth, bnf_grammar, max_tree_depth, max_wraps, codon_size)
    
                # Append individual to population
                population.append(ind)
    
            return population

    def crossover_onepoint(p_0, p_1, bnf_grammar, max_tree_depth, max_wraps):#, max_tree_nodes, max_genome_length): #variable_onepoint
        """
        Given two individuals, create two children using one-point crossover and
        return them. A different point is selected on each genome for crossover
        to occur. Note that this allows for genomes to grow or shrink in
        size. Crossover points are selected within the used portion of the
        genome by default (i.e. crossover does not occur in the tail of the
        individual).
        
        :param p_0: Parent 0
        :param p_1: Parent 1
        :return: A list of crossed-over individuals.
        """
    
        # Get the chromosomes.
        genome_0, genome_1 = p_0.genome, p_1.genome
    
        # Uniformly generate crossover points.
        max_p_0, max_p_1 = get_max_genome_index(p_0, p_1)
            
        # Select unique points on each genome for crossover to occur.
        pt_0, pt_1 = randint(1, max_p_0), randint(1, max_p_1)
        
        # Make new chromosomes by crossover: these slices perform copies.
        c_0 = genome_0[:pt_0] + genome_1[pt_1:]
        c_1 = genome_1[:pt_1] + genome_0[pt_0:]
        
        p_0 = reMap(p_0,c_0,bnf_grammar, max_tree_depth, max_wraps)
        p_1 = reMap(p_1,c_1,bnf_grammar, max_tree_depth, max_wraps)
        
        inds = [p_0, p_1]
        
        # Check each individual is ok (i.e. does not violate specified limits).
        checks = [check_ind(ind, "crossover", max_tree_depth) for ind in inds]
    
        while any(checks):
            # An individual violates a limit.
            pt_0, pt_1 = randint(1, max_p_0), randint(1, max_p_1)
            c_0 = genome_0[:pt_0] + genome_1[pt_1:]
            c_1 = genome_1[:pt_1] + genome_0[pt_0:]
            p_0 = reMap(p_0,c_0,bnf_grammar, max_tree_depth, max_wraps)
            p_1 = reMap(p_1,c_1,bnf_grammar, max_tree_depth, max_wraps)
            inds = [p_0, p_1]
            
            checks = [check_ind(ind, "crossover", max_tree_depth) for ind in inds]
                    
        return p_0, p_1                
    
    def mutation_int_flip_per_codon(ind, indpb, codon_size, bnf_grammar, max_tree_depth, max_wraps):
        """
        Mutate the genome of an individual by randomly choosing a new int with
        probability p_mut. Works per-codon. Mutation is performed over the
        effective length (i.e. within used codons, not tails).
    
        :param ind: An individual to be mutated.
        :return: A mutated individual.
        """
    
        # Set effective genome length over which mutation will be performed.
        eff_length = get_effective_length(ind)
    
        # Mutation probability works per-codon over the used portion of the genome
        for i in range(eff_length):
            if random() < indpb:
                ind.genome[i] = randint(0, codon_size)
    
        new_ind = reMap(ind, ind.genome, bnf_grammar, max_tree_depth, max_wraps)
    
        # Check ind does not violate specified limits.
        check = check_ind(new_ind, "mutation", max_tree_depth)
    
        while check:
            # Perform mutation until the individual passes all tests.
            eff_length = get_effective_length(ind)
            for i in range(eff_length):
                if random() < indpb:
                    ind.genome[i] = randint(0, codon_size)
            new_ind = reMap(ind, ind.genome, bnf_grammar, max_tree_depth, max_wraps)
            check = check_ind(new_ind, "mutation")
    
        return new_ind,

    def selTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the best individual among *tournsize* randomly chosen
        valid individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        The original functions has been changed to prevent selection of 
        invalid individuals.
    
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
    
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        valid_individuals = [i for i in individuals if not i.invalid]
        while len(chosen) < k:
            aspirants = sample(valid_individuals, tournsize)
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen
    
    def selLexicase(individuals, k):
        """Returns an individual that does the best on the fitness cases when
        considered one at a time in random order.
        http://faculty.hampshire.edu/lspector/pubs/lexicase-IEEE-TEC.pdf
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.
        """
        selected_individuals = []
        valid_individuals = [i for i in individuals if not i.invalid]
        l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
        
        cases = list(range(0,l_samples))
        #fit_weights = valid_individuals[0].fitness.weights
        candidates = valid_individuals
    
        for i in range(k):
            #cases = list(range(len(valid_individuals[0].fitness.values)))
            shuffle(cases)
    
            while len(cases) > 0 and len(candidates) > 1:
                #f = min if fit_weights[cases[0]] < 0 else max
                candidates_update = [i for i in candidates if i.fitness_each_sample[cases[0]] == True]
                
                if len(candidates_update) == 0:
                    #no candidate correctly predicted the case
                    pass
                else:
                    candidates = candidates_update    
                del cases[0]                    
    
                #best_val_for_case = f(map(lambda x: x.fitness.values[cases[0]], candidates))
    
                #candidates = list(filter(lambda x: x.fitness.values[cases[0]] == best_val_for_case, candidates))
                #cases.pop(0)
    
            #If there is only one candidate remaining, it will be selected
            #If there are more than one, the choice will be made randomly
            selected_individuals.append(choice(candidates))
            
            cases = list(range(0,l_samples))
            candidates = valid_individuals
    
        return selected_individuals
    
    def selLexicaseCount(individuals, k):
        """Same as Lexicase Selection, but counting attempts of filtering and
        updating respective attributes on ind.
        """
        selected_individuals = []
        valid_individuals = [i for i in individuals if not i.invalid]
        l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
        
        #For analysing Lexicase selection
        samples_attempted = [0]*l_samples
        samples_used = [0]*l_samples
        samples_unsuccessful1 = [0]*l_samples
        samples_unsuccessful2 = [0]*l_samples
        inds_to_choose = [0]*k
        times_chosen = [0]*4
        
        cases = list(range(0,l_samples))
        #fit_weights = valid_individuals[0].fitness.weights
        candidates = valid_individuals
    
        for i in range(k):
            #cases = list(range(len(valid_individuals[0].fitness.values)))
            shuffle(cases)
    
            while len(cases) > 0 and len(candidates) > 1:
                #f = min if fit_weights[cases[0]] < 0 else max
                candidates_update = [i for i in candidates if i.fitness_each_sample[cases[0]] == True]
                
                samples_attempted[cases[0]] += 1
                if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0):
                    samples_used[cases[0]] += 1
                if (len(candidates_update) == len(candidates)):
                    samples_unsuccessful1[cases[0]] += 1
                if len(candidates_update) == 0:
                    samples_unsuccessful2[cases[0]] += 1
                
                if len(candidates_update) == 0:
                    #no candidate correctly predicted the case
                    pass
                else:
                    candidates = candidates_update    
                del cases[0]                    
    
                #best_val_for_case = f(map(lambda x: x.fitness.values[cases[0]], candidates))
    
                #candidates = list(filter(lambda x: x.fitness.values[cases[0]] == best_val_for_case, candidates))
                #cases.pop(0)
    
            #If there is only one candidate remaining, it will be selected
            if len(candidates) == 1:
                selected_individuals.append(candidates[0])
                inds_to_choose[i] = 1
                times_chosen[0] += 1 #The choise was made by fitness
            else: #If there are more than one, the choice will be made randomly
                selected_individuals.append(choice(candidates))
                inds_to_choose[i] = len(candidates)
                times_chosen[3] += 1 #The choise was made by randomly
            
            cases = list(range(0,l_samples))
            candidates = valid_individuals
    
        return selected_individuals, samples_attempted, samples_used, samples_unsuccessful1, samples_unsuccessful2, inds_to_choose, times_chosen
    
    def selLexicaseCountSecondObjective(individuals, k):
        """Same as Lexicase Selection, but counting attempts of filtering and
        updating respective attributes on ind.
        """
        selected_individuals = []
        valid_individuals = [i for i in individuals if not i.invalid]
        l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
        
        #For analysing Lexicase selection
        samples_attempted = [0]*l_samples
        samples_used = [0]*l_samples
        samples_unsuccessful1 = [0]*l_samples
        samples_unsuccessful2 = [0]*l_samples
        inds_to_choose = [0]*k
        
        cases = list(range(0,l_samples))
        #fit_weights = valid_individuals[0].fitness.weights
        candidates = valid_individuals
    
        for i in range(k):
            #cases = list(range(len(valid_individuals[0].fitness.values)))
            shuffle(cases)
    
            while len(cases) > 0 and len(candidates) > 1:
                #f = min if fit_weights[cases[0]] < 0 else max
                candidates_update = [i for i in candidates if i.fitness_each_sample[cases[0]] == True]
                
                samples_attempted[cases[0]] += 1
                if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0):
                    samples_used[cases[0]] += 1
                if (len(candidates_update) == len(candidates)):
                    samples_unsuccessful1[cases[0]] += 1
                if len(candidates_update) == 0:
                    samples_unsuccessful2[cases[0]] += 1
                
                if len(candidates_update) == 0:
                    #no candidate correctly predicted the case
                    pass
                else:
                    candidates = candidates_update    
                del cases[0]                    
    
                #best_val_for_case = f(map(lambda x: x.fitness.values[cases[0]], candidates))
    
                #candidates = list(filter(lambda x: x.fitness.values[cases[0]] == best_val_for_case, candidates))
                #cases.pop(0)
    
            #If there is only one candidate remaining, it will be selected
            if len(candidates) == 1:
                selected_individuals.append(candidates[0])
            else:
            #If there are more than one, the choice will be made by a second objective
                candidates.sort(key=lambda x: x.n_gates, reverse=False)
                selected_individuals.append(candidates[0])
            inds_to_choose[i] = len(candidates)
            
            cases = list(range(0,l_samples))
            candidates = valid_individuals
    
        return selected_individuals, samples_attempted, samples_used, samples_unsuccessful1, samples_unsuccessful2, inds_to_choose

    def selLexicaseEachBitCount(individuals, k):
        """
        """
        selected_individuals = []
        valid_individuals = [i for i in individuals if not i.invalid]
        l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
        
        #For analysing Lexicase selection
        samples_attempted = [0]*l_samples
        samples_used = [0]*l_samples
        samples_unsuccessful1 = [0]*l_samples
        samples_unsuccessful2 = [0]*l_samples
        inds_to_choose = [0]*k
        
        cases = list(range(0,l_samples))
        candidates = valid_individuals
    
        for i in range(k):
            shuffle(cases)
    
            while len(cases) > 0 and len(candidates) > 1:
                f = max
                best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], candidates))
                
                samples_attempted[cases[0]] += 1 #attempt of filtering
                
                if best_val_for_case == 0: #no candidate correctly predicted any bit
                    samples_unsuccessful2[cases[0]] += 1
                
                else: #at least one candidate correctly predicted at least one bit
                    candidates_update = list(filter(lambda x: x.fitness_each_sample[cases[0]] == best_val_for_case, candidates))
                
                    if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0): #a successful attempt of filtering happened
                        samples_used[cases[0]] += 1
                    elif (len(candidates_update) == len(candidates)): #all candidates correctly predicted the same quantity of bits for this case and it is not useful
                        samples_unsuccessful1[cases[0]] += 1
                    
                    candidates = candidates_update    

                del cases[0]                    
    
            #If there is only one candidate remaining, it will be selected
            #If there are more than one, the choice will be made randomly
            selected_individuals.append(choice(candidates))
            inds_to_choose[i] = len(candidates)
            
            cases = list(range(0,l_samples))
            candidates = valid_individuals
            
        return selected_individuals, samples_attempted, samples_used, samples_unsuccessful1, samples_unsuccessful2, inds_to_choose

    def selLexicaseEachBitCountSecondObjective(individuals, k):
        """
        """
        selected_individuals = []
        valid_individuals = [i for i in individuals if not i.invalid]
        l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
        
        #For analysing Lexicase selection
        samples_attempted = [0]*l_samples
        samples_used = [0]*l_samples
        samples_unsuccessful1 = [0]*l_samples
        samples_unsuccessful2 = [0]*l_samples
        inds_to_choose = [0]*k
        
        cases = list(range(0,l_samples))
        candidates = valid_individuals
    
        for i in range(k):
            shuffle(cases)
    
            while len(cases) > 0 and len(candidates) > 1:
                f = max
                best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], candidates))
                
                samples_attempted[cases[0]] += 1 #attempt of filtering
                
                if best_val_for_case == 0: #no candidate correctly predicted any bit
                    samples_unsuccessful2[cases[0]] += 1
                
                else: #at least one candidate correctly predicted at least one bit
                    candidates_update = list(filter(lambda x: x.fitness_each_sample[cases[0]] == best_val_for_case, candidates))
                
                    if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0): #a successful attempt of filtering happened
                        samples_used[cases[0]] += 1
                    elif (len(candidates_update) == len(candidates)): #all candidates correctly predicted the same quantity of bits for this case and it is not useful
                        samples_unsuccessful1[cases[0]] += 1
                    
                    candidates = candidates_update    

                del cases[0]                    
    
            #If there is only one candidate remaining, it will be selected
            if len(candidates) == 1:
                selected_individuals.append(candidates[0])
                inds_to_choose[i] = 1
            else:
            #If there are more than one, the choice will be made by a second objective
            #In case of tie in the second objective, choose randomly
                f = min
                best_val = f(map(lambda x: x.n_gates, candidates))
                candidates_update = list(filter(lambda x: x.n_gates == best_val, candidates))
                
                selected_individuals.append(choice(candidates_update))
                inds_to_choose[i] = len(candidates_update)
            
            cases = list(range(0,l_samples))
            candidates = valid_individuals
            
        return selected_individuals, samples_attempted, samples_used, samples_unsuccessful1, samples_unsuccessful2, inds_to_choose
    
    def selLexicaseOtherObjectives(individuals, k, secondObjective, thirdObjective=None, choose_random=False):
        """
        When choose_random is True, choose randomly which objective will be used first
        Put choose_random as True only if there is a thirdObjective
        """
        selected_individuals = []
        valid_individuals = [i for i in individuals if not i.invalid]
        l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
        
        #For analysing Lexicase selection
        samples_attempted = [0]*l_samples
        samples_used = [0]*l_samples
        samples_unsuccessful1 = [0]*l_samples
        samples_unsuccessful2 = [0]*l_samples
        inds_to_choose = [0]*k
        times_chosen = [0]*4
        
        cases = list(range(0,l_samples))
        candidates = valid_individuals
    
        for i in range(k):
            shuffle(cases)
    
            while len(cases) > 0 and len(candidates) > 1:
                f = max
                best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], candidates))
                
                samples_attempted[cases[0]] += 1 #attempt of filtering
                
                if best_val_for_case == 0: #no candidate correctly predicted any bit
                    samples_unsuccessful2[cases[0]] += 1
                
                else: #at least one candidate correctly predicted at least one bit
                    candidates_update = list(filter(lambda x: x.fitness_each_sample[cases[0]] == best_val_for_case, candidates))
                
                    if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0): #a successful attempt of filtering happened
                        samples_used[cases[0]] += 1
                    elif (len(candidates_update) == len(candidates)): #all candidates correctly predicted the same quantity of bits for this case and it is not useful
                        samples_unsuccessful1[cases[0]] += 1
                    
                    candidates = candidates_update    

                del cases[0]                    
    
            #If there is only one candidate remaining, it will be selected
            if len(candidates) == 1:
                selected_individuals.append(candidates[0])
                inds_to_choose[i] = 1
                times_chosen[0] += 1 #The choise was made by fitness
                #print("Chosen by fitness")
            else:
            #If there are more than one, the choice will be made by a second objective
                f = min
#                a = candidates[0].n_gates

                if choose_random:
                    r = random()
                    if r > 0.5:
                        moo = [secondObjective, thirdObjective]
                    else:
                        moo = [thirdObjective, secondObjective]
                else:
                    moo = [secondObjective, thirdObjective]
                
                best_val = f(map(lambda x: getattr(x, moo[0]), candidates))
                candidates_update = list(filter(lambda x: getattr(x, moo[0]) == best_val, candidates))
                if len(candidates_update) == 1:
                    selected_individuals.append(candidates_update[0])
                    inds_to_choose[i] = 1
                    times_chosen[1] += 1 #The choise was made by second objective
#                    print("Chosen by", secondObjective)
#                    print("Original:", a)
#                    print("Now:", candidates_update[0].n_gates)
                else:
                    #In case of tie in the second objective, choose using the third objective
#                    b = candidates_update[0].n_gates_longest_path
                    if moo[1]:
                        best_val = f(map(lambda x: getattr(x, moo[1]), candidates_update))
                        candidates_update = list(filter(lambda x: getattr(x, moo[1]) == best_val, candidates))
                        if len(candidates_update) == 1:
                            selected_individuals.append(candidates_update[0])
                            inds_to_choose[i] = 1
                            times_chosen[2] += 1 #The choise was made by third objective
    #                        print("Chosen by", thirdObjective)
    #                        print("Original:", b)
    #                        print("Now:", candidates_update[0].n_gates_longest_path)
                        else:
                            #If the tie still remains, choose randomly                
                            selected_individuals.append(choice(candidates_update))
                            inds_to_choose[i] = len(candidates_update)
                            times_chosen[3] += 1 #The choise was made randomly
                            #print("Chosen randomly")
                    else:
                        #If the tie still remains, choose randomly                
                        selected_individuals.append(choice(candidates_update))
                        inds_to_choose[i] = len(candidates_update)
                        times_chosen[3] += 1 #The choise was made randomly
            cases = list(range(0,l_samples))
            candidates = valid_individuals
            
        return selected_individuals, samples_attempted, samples_used, samples_unsuccessful1, samples_unsuccessful2, inds_to_choose, times_chosen
    
    def selLexicaseMixAllObjectives(individuals, k, secondObjective, thirdObjective=None):
        """
        """
        selected_individuals = []
        valid_individuals = [i for i in individuals if not i.invalid]
        l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
        
        inds_to_choose = [0]*k
        times_chosen = [0]*4
        
        cases = list(range(0,l_samples))
        candidates = valid_individuals
    
        for i in range(k):
            shuffle(cases)
            
            r = random()
            
            if r <= 1/6:
                moo = ['error', secondObjective, thirdObjective]
            elif r > 1/6 and r <= 1/3:
                moo = ['error', thirdObjective, secondObjective]
            elif r > 2/6 and r <= 3/6:
                moo = [secondObjective, 'error', thirdObjective]
            elif r > 3/6 and r <= 4/6:
                moo = [secondObjective, thirdObjective, 'error']
            elif r > 4/6 and r <= 5/6:
                moo = [thirdObjective, 'error', secondObjective]
            elif r > 5/6:
                moo = [thirdObjective, secondObjective, 'error']
          
            while len(cases) > 0 and len(candidates) > 1:
                for j in range(3):
                    if moo[j] == 'error':
                        f = max
                        best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], candidates))
                        if best_val_for_case == 0: #no candidate correctly predicted any bit
                            pass
                        else: #at least one candidate correctly predicted at least one bit
                            candidates_update = list(filter(lambda x: x.fitness_each_sample[cases[0]] == best_val_for_case, candidates))
                            candidates = candidates_update
                        del cases[0]
        
                    elif moo[j] == secondObjective or moo[j] == thirdObjective:
                        f = min
                        best_val = f(map(lambda x: getattr(x, moo[j]), candidates))
                        candidates_update = list(filter(lambda x: getattr(x, moo[j]) == best_val, candidates))
                        candidates = candidates_update
                        
                    if len(candidates) == 1:
                        #We had a perfect filtering, then we need to stop
                        #Otherwise, it would continue in the loop of the objectives
                        times_chosen[j] += 1 #The choice was made by the i-th objective
                        break 
                        
                                    
    
            #If there is only one candidate remaining, it will be selected
            if len(candidates) == 1:
                selected_individuals.append(candidates[0])
                inds_to_choose[i] = 1
            else:
                #If the tie still remains, choose randomly                
                selected_individuals.append(choice(candidates))
                inds_to_choose[i] = len(candidates)
                times_chosen[3] += 1 #The choice was made randomly
            
            #Re-initilise to select the next individual
            cases = list(range(0,l_samples))
            candidates = valid_individuals
            
        return selected_individuals, inds_to_choose, times_chosen
    
    
def mapper(genome, tree, bnf_grammar, max_tree_depth, max_wraps):
    """
    Wheel for mapping. Calls the correct mapper for a given _input. Checks
    the params dict to ensure the correct type of individual is being created.

    If a genome is passed in with no tree, all tree-related information is
    generated. If a tree is passed in with no genome, the genome is
    sequenced from the tree.

    :param genome: Genome of an individual.
    :param tree: Tree of an individual.
    :return: All components necessary for a fully mapped individual.
    """

    # one or other must be passed in, but not both
    assert (genome or tree)
    assert not (genome and tree)

    if genome:
        # We have a genome and need to map an individual from that genome.

        genome = list(genome)
        # This is a fast way of creating a new unique copy of the genome
        # (prevents cross-contamination of information between individuals).

        # Can generate tree information faster using
        # algorithm.mapper.map_ind_from_genome() if we don't need to
        # store the whole tree.
        phenotype, genome, tree, nodes, invalid, depth, \
            used_codons, wraps = map_ind_from_genome(genome, bnf_grammar, max_tree_depth, max_wraps)

    else:
        # We have a tree.
        
        # genome, output, invalid, depth, and nodes can all be
        # generated by recursing through the tree once.

        genome, output, invalid, depth, \
        nodes = tree.get_tree_info(bnf_grammar.non_terminals.keys(),
                                   [], [])
        used_codons, phenotype = len(genome), "".join(output)

    if invalid:
        # Set values for invalid individuals.
        phenotype, nodes, depth, used_codons = None, np.NaN, np.NaN, np.NaN

    return phenotype, genome, tree, nodes, invalid, depth, used_codons, wraps


def map_ind_from_genome(genome, bnf_grammar, max_tree_depth, max_wraps):
    """
    A fast genotype to phenotype mapping process. Map input via rules to
    output. Does not require the recursive tree class, but still calculates
    tree information, e.g. number of nodes and maximum depth.

    :param genome: A genome to be mapped.
    :return: Output in the form of a phenotype string ('None' if invalid),
             Genome,
             None (this is reserved for the derivation tree),
             The number of nodes in the derivation,
             A boolean flag for whether or not the individual is invalid,
             The maximum depth of any node in the tree, and
             The number of used codons.
    """

    n_input = len(genome)

    # Depth, max_depth, and nodes start from 1 to account for starting root
    # Initialise number of wraps at -1 (since
    used_input, current_depth, max_depth, nodes, wraps = 0, 1, 1, 1, -1

    # Initialise output as empty deque list (deque is a list-like container
    # with fast appends and pops on either end).
    output = deque()

    # Initialise the list of unexpanded non-terminals with the start rule.
    unexpanded_symbols = deque([(bnf_grammar.start_rule, 1)])

    while (wraps < max_wraps) and unexpanded_symbols:
        # While there are unexpanded non-terminals, and we are below our
        # wrapping limit, we can continue to map the genome.

        if max_tree_depth and (max_depth > max_tree_depth):
            # We have breached our maximum tree depth limit.
            break

        if used_input % n_input == 0 and \
                        used_input > 0 and \
                any([i[0]["type"] == "NT" for i in unexpanded_symbols]):
            # If we have reached the end of the genome and unexpanded
            # non-terminals remain, then we need to wrap back to the start
            # of the genome again. Can break the while loop.
            wraps += 1

        # Expand a production from the list of unexpanded non-terminals.
        current_item = unexpanded_symbols.popleft()
        current_symbol, current_depth = current_item[0], current_item[1]

        if max_depth < current_depth:
            # Set the new maximum depth.
            max_depth = current_depth

        # Set output if it is a terminal.
        if current_symbol["type"] != "NT":
            output.append(current_symbol["symbol"])

        else:
            # Current item is a new non-terminal. Find associated production
            # choices.
            production_choices = bnf_grammar.rules[current_symbol[
                "symbol"]]["choices"]
            no_choices = bnf_grammar.rules[current_symbol["symbol"]][
                "no_choices"]

            # Select a production based on the next available codon in the
            # genome.
            current_production = genome[used_input % n_input] % no_choices

            # Use an input
            used_input += 1

            # Initialise children as empty deque list.
            children = deque()
            nt_count = 0

            for prod in production_choices[current_production]['choice']:
                # iterate over all elements of chosen production rule.

                child = [prod, current_depth + 1]

                # Extendleft reverses the order, thus reverse adding.
                children.appendleft(child)
                if child[0]["type"] == "NT":
                    nt_count += 1

            # Add the new children to the list of unexpanded symbols.
            unexpanded_symbols.extendleft(children)

            if nt_count > 0:
                nodes += nt_count
            else:
                nodes += 1

    # Generate phenotype string.
    output = "".join(output)
    
    if wraps == -1:
        wraps += 1 #Because we started with -1
    
    if len(unexpanded_symbols) > 0:
        # All non-terminals have not been completely expanded, invalid
        # solution.
        return None, genome, None, nodes, True, max_depth, used_input, wraps

    return output, genome, None, nodes, False, max_depth, used_input, wraps


class Grammar(object):
    """
    Parser for Backus-Naur Form (BNF) Context-Free Grammars.
    """

    def __init__(self, file_name): #codon_size min_init_tree_depth
        """
        Initialises an instance of the grammar class. This instance is used
        to parse a given file_name grammar.

        :param file_name: A specified BNF grammar file.
        """

        if file_name.endswith("pybnf"):
            # Use python filter for parsing grammar output as grammar output
            # contains indented python code.
            self.python_mode = True

        else:
            # No need to filter/interpret grammar output, individual
            # phenotypes can be evaluated as normal.
            self.python_mode = False

        # Initialise empty dict for all production rules in the grammar.
        # Initialise empty dict of permutations of solutions possible at
        # each derivation tree depth.
        self.rules, self.permutations = {}, {}

        # Initialise dicts for terminals and non terminals, set params.
        self.non_terminals, self.terminals = {}, {}
        self.start_rule = None
        #, self.codon_size = None, codon_size
        self.min_path, self.max_arity = None, None
        #, self.min_ramp = None, None, None

        # Set regular expressions for parsing BNF grammar.
        self.ruleregex = '(?P<rulename><\S+>)\s*::=\s*(?P<production>(?:(?=\#)\#[^\r\n]*|(?!<\S+>\s*::=).+?)+)'
        self.productionregex = '(?=\#)(?:\#.*$)|(?!\#)\s*(?P<production>(?:[^\'\"\|\#]+|\'.*?\'|".*?")+)'
        self.productionpartsregex = '\ *([\r\n]+)\ *|([^\'"<\r\n]+)|\'(.*?)\'|"(.*?)"|(?P<subrule><[^>|\s]+>)|([<]+)'

        # Read in BNF grammar, set production rules, terminals and
        # non-terminals.
        self.read_bnf_file(file_name)

        # Check the minimum depths of all non-terminals in the grammar.
        self.check_depths()

        # Check which non-terminals are recursive.
        self.check_recursion(self.start_rule["symbol"], [])

        # Set the minimum path and maximum arity of the grammar.
        self.set_arity()

        # Generate lists of recursive production choices and shortest
        # terminating path production choices for each NT in the grammar.
        # Enables faster tree operations.
        self.set_grammar_properties()

        # Calculate the total number of derivation tree permutations and
        # combinations that can be created by a grammar at a range of depths.
#        self.check_permutations()

#        if params['MIN_INIT_TREE_DEPTH']:
        # Set the minimum ramping tree depth from the command line.
#        self.min_ramp = min_init_tree_depth

#        elif hasattr(params['INITIALISATION'], "ramping"):
            # Set the minimum depth at which ramping can start where we can
            # have unique solutions (no duplicates).
#            self.get_min_ramp_depth()

#        if params['REVERSE_MAPPING_TARGET'] or params['TARGET_SEED_FOLDER']:
            # Initialise dicts for reverse-mapping GE individuals.
#            self.concat_NTs, self.climb_NTs = {}, {}

        # Find production choices which can be used to concatenate
        # subtrees.
  #      self.find_concatenation_NTs()

    def read_bnf_file(self, file_name):
        """
        Read a grammar file in BNF format. Parses the grammar and saves a
        dict of all production rules and their possible choices.

        :param file_name: A specified BNF grammar file.
        :return: Nothing.
        """

        with open(file_name, 'r') as bnf:
            # Read the whole grammar file.
            content = bnf.read()

            for rule in finditer(self.ruleregex, content, DOTALL):
                # Find all rules in the grammar

                if self.start_rule is None:
                    # Set the first rule found as the start rule.
                    self.start_rule = {"symbol": rule.group('rulename'),
                                       "type": "NT"}

                # Create and add a new rule.
                self.non_terminals[rule.group('rulename')] = {
                    'id': rule.group('rulename'),
                    'min_steps': maxsize,
                    'expanded': False,
                    'recursive': True,
                    'b_factor': 0}

                # Initialise empty list of all production choices for this
                # rule.
                tmp_productions = []

                for p in finditer(self.productionregex,
                                  rule.group('production'), MULTILINE):
                    # Iterate over all production choices for this rule.
                    # Split production choices of a rule.

                    if p.group('production') is None or p.group(
                            'production').isspace():
                        # Skip to the next iteration of the loop if the
                        # current "p" production is None or blank space.
                        continue

                    # Initialise empty data structures for production choice
                    tmp_production, terminalparts = [], None

                    # special cases: GE_RANGE:dataset_n_vars will be
                    # transformed to productions 0 | 1 | ... |
                    # n_vars-1, and similar for dataset_n_is,
                    # dataset_n_os
#                    GE_RANGE_regex = r'GE_RANGE:(?P<range>\w*)'
#                    m = match(GE_RANGE_regex, p.group('production'))
#                    if m:
#                        try:
#                            if m.group('range') == "dataset_n_vars":
                                # number of columns from dataset
#                                n = params['FITNESS_FUNCTION'].n_vars
#                            elif m.group('range') == "dataset_n_is":
                                # number of input symbols (see
                                # if_else_classifier.py)
#                                n = params['FITNESS_FUNCTION'].n_is
#                            elif m.group('range') == "dataset_n_os":
                                # number of output symbols
#                                n = params['FITNESS_FUNCTION'].n_os
#                            else:
                                # assume it's just an int
#                                n = int(m.group('range'))
#                        except (ValueError, AttributeError):
#                            raise ValueError("Bad use of GE_RANGE: "
#                                             + m.group())

#                        for i in range(n):
                            # add a terminal symbol
#                            tmp_production, terminalparts = [], None
#                            symbol = {
#                                "symbol": str(i),
#                                "type": "T",
#                                "min_steps": 0,
#                                "recursive": False}
#                            tmp_production.append(symbol)
#                            if str(i) not in self.terminals:
#                                self.terminals[str(i)] = \
#                                    [rule.group('rulename')]
#                            elif rule.group('rulename') not in \
#                                self.terminals[str(i)]:
#                                self.terminals[str(i)].append(
#                                    rule.group('rulename'))
#                            tmp_productions.append({"choice": tmp_production,
#                                                    "recursive": False,
#                                                    "NT_kids": False})
                        # don't try to process this production further
                        # (but later productions in same rule will work)
#                        continue

                    for sub_p in finditer(self.productionpartsregex,
                                          p.group('production').strip()):
                        # Split production into terminal and non terminal
                        # symbols.

                        if sub_p.group('subrule'):
                            if terminalparts is not None:
                                # Terminal symbol is to be appended to the
                                # terminals dictionary.
                                symbol = {"symbol": terminalparts,
                                          "type": "T",
                                          "min_steps": 0,
                                          "recursive": False}
                                tmp_production.append(symbol)
                                if terminalparts not in self.terminals:
                                    self.terminals[terminalparts] = \
                                        [rule.group('rulename')]
                                elif rule.group('rulename') not in \
                                    self.terminals[terminalparts]:
                                    self.terminals[terminalparts].append(
                                        rule.group('rulename'))
                                terminalparts = None

                            tmp_production.append(
                                {"symbol": sub_p.group('subrule'),
                                 "type": "NT"})

                        else:
                            # Unescape special characters (\n, \t etc.)
                            if terminalparts is None:
                                terminalparts = ''
                            terminalparts += ''.join(
                                [part.encode().decode('unicode-escape') for
                                 part in sub_p.groups() if part])

                    if terminalparts is not None:
                        # Terminal symbol is to be appended to the terminals
                        # dictionary.
                        symbol = {"symbol": terminalparts,
                                  "type": "T",
                                  "min_steps": 0,
                                  "recursive": False}
                        tmp_production.append(symbol)
                        if terminalparts not in self.terminals:
                            self.terminals[terminalparts] = \
                                [rule.group('rulename')]
                        elif rule.group('rulename') not in \
                            self.terminals[terminalparts]:
                            self.terminals[terminalparts].append(
                                rule.group('rulename'))
                    tmp_productions.append({"choice": tmp_production,
                                            "recursive": False,
                                            "NT_kids": False})

                if not rule.group('rulename') in self.rules:
                    # Add new production rule to the rules dictionary if not
                    # already there.
                    self.rules[rule.group('rulename')] = {
                        "choices": tmp_productions,
                        "no_choices": len(tmp_productions)}

                    if len(tmp_productions) == 1:
                        # Unit productions.
                        print("Warning: Grammar contains unit production "
                              "for production rule", rule.group('rulename'))
                        print("         Unit productions consume GE codons.")
                else:
                    # Conflicting rules with the same name.
                    raise ValueError("lhs should be unique",
                                     rule.group('rulename'))

    def check_depths(self):
        """
        Run through a grammar and find out the minimum distance from each
        NT to the nearest T. Useful for initialisation methods where we
        need to know how far away we are from fully expanding a tree
        relative to where we are in the tree and what the depth limit is.

        :return: Nothing.
        """

        # Initialise graph and counter for checking minimum steps to Ts for
        # each NT.
        counter, graph = 1, []

        for rule in sorted(self.rules.keys()):
            # Iterate over all NTs.
            choices = self.rules[rule]['choices']

            # Set branching factor for each NT.
            self.non_terminals[rule]['b_factor'] = self.rules[rule][
                'no_choices']

            for choice in choices:
                # Add a new edge to our graph list.
                graph.append([rule, choice['choice']])

        while graph:
            removeset = set()
            for edge in graph:
                # Find edges which either connect to terminals or nodes
                # which are fully expanded.
                if all([sy["type"] == "T" or
                        self.non_terminals[sy["symbol"]]['expanded']
                        for sy in edge[1]]):
                    removeset.add(edge[0])

            for s in removeset:
                # These NTs are now expanded and have their correct minimum
                # path set.
                self.non_terminals[s]['expanded'] = True
                self.non_terminals[s]['min_steps'] = counter

            # Create new graph list and increment counter.
            graph = [e for e in graph if e[0] not in removeset]
            counter += 1

    def check_recursion(self, cur_symbol, seen):
        """
        Traverses the grammar recursively and sets the properties of each rule.

        :param cur_symbol: symbol to check.
        :param seen: Contains already checked symbols in the current traversal.
        :return: Boolean stating whether or not cur_symbol is recursive.
        """

        if cur_symbol not in self.non_terminals.keys():
            # Current symbol is a T.
            return False

        if cur_symbol in seen:
            # Current symbol has already been seen, is recursive.
            return True

        # Append current symbol to seen list.
        seen.append(cur_symbol)

        # Get choices of current symbol.
        choices = self.rules[cur_symbol]['choices']
        nt = self.non_terminals[cur_symbol]

        recursive = False
        for choice in choices:
            for sym in choice['choice']:
                # Recurse over choices.
                recursive_symbol = self.check_recursion(sym["symbol"], seen)
                recursive = recursive or recursive_symbol

        # Set recursive properties.
        nt['recursive'] = recursive
        seen.remove(cur_symbol)

        return nt['recursive']

    def set_arity(self):
        """
        Set the minimum path of the grammar, i.e. the smallest legal
        solution that can be generated.

        Set the maximum arity of the grammar, i.e. the longest path to a
        terminal from any non-terminal.

        :return: Nothing
        """

        # Set the minimum path of the grammar as the minimum steps to a
        # terminal from the start rule.
        self.min_path = self.non_terminals[self.start_rule["symbol"]][
            'min_steps']

        # Set the maximum arity of the grammar as the longest path to
        # a T from any NT.
        self.max_arity = max(self.non_terminals[NT]['min_steps']
                             for NT in self.non_terminals)

        # Add the minimum terminal path to each production rule.
        for rule in self.rules:
            for choice in self.rules[rule]['choices']:
                NT_kids = [i for i in choice['choice'] if i["type"] == "NT"]
                if NT_kids:
                    choice['NT_kids'] = True
                    for sym in NT_kids:
                        sym['min_steps'] = self.non_terminals[sym["symbol"]][
                            'min_steps']

        # Add boolean flag indicating recursion to each production rule.
        for rule in self.rules:
            for prod in self.rules[rule]['choices']:
                for sym in [i for i in prod['choice'] if i["type"] == "NT"]:
                    sym['recursive'] = self.non_terminals[sym["symbol"]][
                        'recursive']
                    if sym['recursive']:
                        prod['recursive'] = True

    def set_grammar_properties(self):
        """
        Goes through all non-terminals and finds the production choices with
        the minimum steps to terminals and with recursive steps.

        :return: Nothing
        """

        for nt in self.non_terminals:
            # Loop over all non terminals.
            # Find the production choices for the current NT.
            choices = self.rules[nt]['choices']

            for choice in choices:
                # Set the maximum path to a terminal for each produciton choice
                choice['max_path'] = max([item["min_steps"] for item in
                                      choice['choice']])

            # Find shortest path to a terminal for all production choices for
            # the current NT. The shortest path will be the minimum of the
            # maximum paths to a T for each choice over all chocies.
            min_path = min([choice['max_path'] for choice in choices])

            # Set the minimum path in the self.non_terminals dict.
            self.non_terminals[nt]['min_path'] = [choice for choice in
                                                  choices if choice[
                                                      'max_path'] == min_path]

            # Find recursive production choices for current NT. If any
            # constituent part of a production choice is recursive,
            # it is added to the recursive list.
            self.non_terminals[nt]['recursive'] = [choice for choice in
                                                   choices if choice[
                                                       'recursive']]

#    def check_permutations(self):
#        """
#        Calculates how many possible derivation tree combinations can be
#        created from the given grammar at a specified depth. Only returns
#        possible combinations at the specific given depth (if there are no
#        possible permutations for a given depth, will return 0).

#        :param ramps:
#        :return: Nothing.
#        """

        # Set the number of depths permutations are calculated for
        # (starting from the minimum path of the grammar)
#        ramps = params['PERMUTATION_RAMPS']

#        perms_list = []
#        if self.max_arity > self.min_path:
#            for i in range(max((self.max_arity + 1 - self.min_path), ramps)):
#                x = self.check_all_permutations(i + self.min_path)
#                perms_list.append(x)
#                if i > 0:
#                    perms_list[i] -= sum(perms_list[:i])
#                    self.permutations[i + self.min_path] -= sum(perms_list[:i])
#        else:
#            for i in range(ramps):
#                x = self.check_all_permutations(i + self.min_path)
#                perms_list.append(x)
#                if i > 0:
##                    perms_list[i] -= sum(perms_list[:i])
#                    self.permutations[i + self.min_path] -= sum(perms_list[:i])

    def check_all_permutations(self, depth):
        """
        Calculates how many possible derivation tree combinations can be
        created from the given grammar at a specified depth. Returns all
        possible combinations at the specific given depth including those
        depths below the given depth.

        :param depth: A depth for which to calculate the number of
        permutations of solution that can be generated by the grammar.
        :return: The permutations possible at the given depth.
        """

        if depth < self.min_path:
            # There is a bug somewhere that is looking for a tree smaller than
            # any we can create
            s = "representation.grammar.Grammar.check_all_permutations\n" \
                "Error: cannot check permutations for tree smaller than the " \
                "minimum size."
            raise Exception(s)

        if depth in self.permutations.keys():
            # We have already calculated the permutations at the requested
            # depth.
            return self.permutations[depth]

        else:
            # Calculate permutations at the requested depth.
            # Initialise empty data arrays.
            pos, depth_per_symbol_trees, productions = 0, {}, []

            for NT in self.non_terminals:
                # Iterate over all non-terminals to fill out list of
                # productions which contain non-terminal choices.
                a = self.non_terminals[NT]

                for rule in self.rules[a['id']]['choices']:
                    if rule['NT_kids']:
                        productions.append(rule)

            # Get list of all production choices from the start symbol.
            start_symbols = self.rules[self.start_rule["symbol"]]['choices']

            for choice in productions:
                # Generate a list of the symbols of each production choice
                key = str([sym['symbol'] for sym in choice['choice']])

                # Initialise permutations dictionary with the list
                depth_per_symbol_trees[key] = {}

            for i in range(2, depth + 1):
                # Find all the possible permutations from depth of min_path up
                # to a specified depth

                for choice in productions:
                    # Iterate over all production choices
                    sym_pos = 1

                    for j in choice['choice']:
                        # Iterate over all symbols in a production choice.
                        symbol_arity_pos = 0

                        if j["type"] is "NT":
                            # We are only interested in non-terminal symbols
                            for child in self.rules[j["symbol"]]['choices']:
                                # Iterate over all production choices for
                                # each NT symbol in the original choice.

                                if len(child['choice']) == 1 and \
                                   child['choice'][0]["type"] == "T":
                                    # If the child choice leads directly to
                                    # a single terminal, increment the
                                    # permutation count.
                                    symbol_arity_pos += 1

                                else:
                                    # The child choice does not lead
                                    # directly to a single terminal.
                                    # Generate a key for the permutations
                                    # dictionary and increment the
                                    # permutations count there.
                                    key = [sym['symbol'] for sym in child['choice']]
                                    if (i - 1) in depth_per_symbol_trees[str(key)].keys():
                                        symbol_arity_pos += depth_per_symbol_trees[str(key)][i - 1]

                            # Multiply original count by new count.
                            sym_pos *= symbol_arity_pos

                    # Generate new key for the current production choice and
                    # set the new value in the permutations dictionary.
                    key = [sym['symbol'] for sym in choice['choice']]
                    depth_per_symbol_trees[str(key)][i] = sym_pos

            # Calculate permutations for the start symbol.
            for sy in start_symbols:
                key = [sym['symbol'] for sym in sy['choice']]
                if str(key) in depth_per_symbol_trees:
                    pos += depth_per_symbol_trees[str(key)][depth] if depth in depth_per_symbol_trees[str(key)] else 0
                else:
                    pos += 1

            # Set the overall permutations dictionary for the current depth.
            self.permutations[depth] = pos

            return pos

#    def get_min_ramp_depth(self):
#        """
#        Find the minimum depth at which ramping can start where we can have
#        unique solutions (no duplicates).#

#        :param self: An instance of the representation.grammar.grammar class.
#        :return: The minimum depth at which unique solutions can be generated
#        """

#        max_tree_depth = params['MAX_INIT_TREE_DEPTH']
#        size = params['POPULATION_SIZE']

        # Specify the range of ramping depths
#        depths = range(self.min_path, max_tree_depth + 1)

#        if size % 2:
            # Population size is odd
#            size += 1

#        if size / 2 < len(depths):
            # The population size is too small to fully cover all ramping
            # depths. Only ramp to the number of depths we can reach.
#            depths = depths[:int(size / 2)]

        # Find the minimum number of unique solutions required to generate
        # sufficient individuals at each depth.
#        unique_start = int(floor(size / len(depths)))
#        ramp = None

#        for i in sorted(self.permutations.keys()):
            # Examine the number of permutations and combinations of unique
            # solutions capable of being generated by a grammar across each
            # depth i.
#            if self.permutations[i] > unique_start:
                # If the number of permutations possible at a given depth i is
                # greater than the required number of unique solutions,
                # set the minimum ramp depth and break out of the loop.
#                ramp = i
#                break
#        self.min_ramp = ramp

    def find_concatenation_NTs(self):
        """
        Scour the grammar class to find non-terminals which can be used to
        combine/reduce_trees derivation trees. Build up a list of such
        non-terminals. A concatenation non-terminal is one in which at least
        one production choice contains multiple non-terminals. For example:

            <e> ::= (<e><o><e>)|<v>

        is a concatenation NT, since the production choice (<e><o><e>) can
        reduce_trees multiple NTs together. Note that this choice also includes
        a combination of terminals and non-terminals.

        :return: Nothing.
        """

        # Iterate over all non-terminals/production rules.
        for rule in sorted(self.rules.keys()):

            # Find rules which have production choices leading to NTs.
            concat = [choice for choice in self.rules[rule]['choices'] if
                      choice['NT_kids']]

            if concat:
                # We can reduce_trees NTs.
                for choice in concat:

                    symbols = [[sym['symbol'], sym['type']] for sym in
                               choice['choice']]

                    NTs = [sym['symbol'] for sym in choice['choice'] if
                           sym['type'] == "NT"]

                    for NT in NTs:
                        # We add to our self.concat_NTs dictionary. The key is
                        # the root node we want to reduce_trees with another
                        # node. This way when we have a node and wish to see
                        # if we can reduce_trees it with anything else, we
                        # simply look up this dictionary.
                        conc = [choice['choice'], rule, symbols]

                        if NT not in self.concat_NTs:
                            self.concat_NTs[NT] = [conc]
                        else:
                            if conc not in self.concat_NTs[NT]:
                                self.concat_NTs[NT].append(conc)

    def __str__(self):
        return "%s %s %s %s" % (self.terminals, self.non_terminals,
                                self.rules, self.start_rule)

def ret_true(obj):
    """
    Returns "True" if an object is there. E.g. if given a list, will return
    True if the list contains some data, but False if the list is empty.
    
    :param obj: Some object (e.g. list)
    :return: True if something is there, else False.
    """

    if obj:
        return True
    else:
        return False
    
def get_nodes_and_depth(tree, bnf_grammar, nodes=0, max_depth=0):
    """
    Get the number of nodes and the max depth of the tree.
    
    :param tree: An individual's derivation tree.
    :param nodes: The number of nodes in a tree.
    :param max_depth: The maximum depth of any node in the tree.
    :return: number, max_depth.
    """

    # Increment number of nodes in the tree.
    nodes += 1

    # Set the depth of the current node.
    if tree.parent:
        tree.depth = tree.parent.depth + 1
    else:
        tree.depth = 1
        
    # Check the recorded max_depth.
    if tree.depth > max_depth:
        max_depth = tree.depth
        
    # Create list of all non-terminal children of current node.
    NT_kids = [kid for kid in tree.children if kid.root in
               bnf_grammar.non_terminals]
    
    if not NT_kids and get_output(tree):
        # Current node has only terminal children.
        nodes += 1
        
        # Terminal children increase the current node depth by one.
        # Check the recorded max_depth.
        if tree.depth + 1 > max_depth:
            max_depth = tree.depth + 1
    
    else:
        for child in NT_kids:
            # Recurse over all children.
            nodes, max_depth = get_nodes_and_depth(child, bnf_grammar, nodes, max_depth)
    
    return nodes, max_depth

def get_output(ind_tree):
    """
    Calls the recursive build_output(self) which returns a list of all
    node roots. Joins this list to create the full phenotype of an
    individual. This two-step process speeds things up as it only joins
    the phenotype together once rather than at every node.

    :param ind_tree: a full tree for which the phenotype string is to be built.
    :return: The complete built phenotype string of an individual.
    """
    
    def build_output(tree):
        """
        Recursively adds all node roots to a list which can be joined to
        create the phenotype.

        :return: The list of all node roots.
        """
        
        output = []
        for child in tree.children:
            if not child.children:
                # If the current child has no children it is a terminal.
                # Append it to the output.
                output.append(child.root)
            
            else:
                # Otherwise it is a non-terminal. Recurse on all
                # non-terminals.
                output += build_output(child)
        
        return output
    
    return "".join(build_output(ind_tree))


    
def generate_PI_ind_tree(ind_class, max_depth, bnf_grammar, max_tree_depth, max_wraps, codon_size):
    """
    Generate an individual using a given Position Independent subtree
    initialisation method.

    :param max_depth: The maximum depth for the initialised subtree.
    :return: A fully built individual.
    """

    # Initialise an instance of the tree class
    ind_tree = Tree(str(bnf_grammar.start_rule["symbol"]), None)

    # Generate a tree
    genome, output, nodes, depth = pi_grow(ind_tree, max_depth, bnf_grammar, codon_size)

    # Get remaining individual information
    phenotype, invalid, used_cod = "".join(output), False, len(genome)

    # Initialise individual
    ind = ind_class(genome, ind_tree, bnf_grammar, max_tree_depth, max_wraps, map_ind=False)
    #, map_ind=False) VOLTA

    # Set individual parameters
    ind.phenotype, ind.nodes = phenotype, nodes
    ind.depth, ind.used_codons, ind.invalid = depth, used_cod, invalid

    # Generate random tail for genome.
    ind.genome = genome + [randint(0, codon_size) for
                           _ in range(int(ind.used_codons / 2))]

    return ind

def pi_grow(tree, max_depth, bnf_grammar, codon_size):
    """
    Grows a tree until a single branch reaches a specified depth. Does this
    by only using recursive production choices until a single branch of the
    tree has reached the specified maximum depth. After that any choices are
    allowed.
    
    :param tree: An instance of the representation.tree.Tree class.
    :param max_depth: The maximum depth to which to derive a tree.
    :return: The fully derived tree.
    """

    # Initialise derivation queue.
    queue = [[tree, ret_true(bnf_grammar.non_terminals[
                                 tree.root]['recursive'])]]

    # Initialise empty genome. With PI operators we can't use a depth-first
    # traversal of the tree to build the genome, we need to build it as we
    # encounter each node.
    genome = []

    while queue:
        # Loop until no items remain in the queue.

        # Pick a random item from the queue.
        chosen = randint(0, len(queue) - 1)

        # Pop the next item from the queue.
        all_node = queue.pop(chosen)
        node, recursive = all_node[0], all_node[0]

        # Get depth of current node.
        if node.parent is not None:
            node.depth = node.parent.depth + 1

        # Get maximum depth of overall tree.
        _, overall_depth = get_nodes_and_depth(tree, bnf_grammar)
        
        # Find the productions possible from the current root.
        productions = bnf_grammar.rules[node.root]
        
        # Set remaining depth.
        remaining_depth = max_depth - node.depth
        
        if (overall_depth < max_depth) or \
                (recursive and (not any([item[1] for item in queue]))):
            # We want to prevent the tree from creating terminals until a
            # single branch has reached the full depth. Only select recursive
            # choices.
            # Find which productions can be used based on the derivation method.
            available = legal_productions("full", remaining_depth, node.root,
                                          productions['choices'], bnf_grammar)
        else:
            # Any production choices can be made.
            
            # Find which productions can be used based on the derivation method.
            available = legal_productions("random", remaining_depth, node.root,
                                          productions['choices'], bnf_grammar)
        
        # Randomly pick a production choice.
        chosen_prod = choice(available)

        # Find the index of the chosen production and set a matching codon
        # based on that index.
        prod_index = productions['choices'].index(chosen_prod)
        codon = randrange(productions['no_choices'],
                          codon_size,#bnf_grammar.codon_size,
                          productions['no_choices']) + prod_index

        # Set the codon for the current node and append codon to the genome.
        node.codon = codon

        # Insert codon into the genome.
        genome.append(codon)
            
        # Initialise empty list of children for current node.
        node.children = []

        for i, symbol in enumerate(chosen_prod['choice']):
            # Iterate over all symbols in the chosen production.

            # Create new child.
            child = Tree(symbol["symbol"], node)

            # Append new node to children.
            node.children.append(child)

            if symbol["type"] == "NT":
                # The symbol is a non-terminal.
    
                # Check whether child is recursive
                recur_child = ret_true(bnf_grammar.non_terminals
                                       [child.root]['recursive'])
    
                # Insert new child into the correct position in the queue.
                queue.insert(chosen + i, [child, recur_child])

    # genome, output, invalid, depth, and nodes can all be generated by
    # recursing through the tree once.
    _, output, invalid, depth, \
    nodes = tree.get_tree_info(bnf_grammar.non_terminals.keys(),
                               [], [])
    
    return genome, output, nodes, depth

def legal_productions(method, depth_limit, root, productions, bnf_grammar):
    """
    Returns the available production choices for a node given a specific
    depth limit.
    
    :param method: A string specifying the desired tree derivation method.
    Current methods are "random" or "full".
    :param depth_limit: The overall depth limit of the desired tree from the
    current node.
    :param root: The root of the current node.
    :param productions: The full list of production choices from the current
    root node.
    :return: The list of available production choices based on the specified
    derivation method.
    """

    # Get all information about root node
    root_info = bnf_grammar.non_terminals[root]
    
    if method == "random":
        # Randomly build a tree.
        
        if not depth_limit:
            # There is no depth limit, any production choice can be used.
            available = productions
        
        elif depth_limit > bnf_grammar.max_arity + 1:
            # If the depth limit is greater than the maximum arity of the
            # grammar, then any production choice can be used.
            available = productions

        elif depth_limit < 0:
            # If we have already surpassed the depth limit, then list the
            # choices with the shortest terminating path.
            available = root_info['min_path']
        
        else:
            # The depth limit is less than or equal to the maximum arity of
            # the grammar + 1. We have to be careful in selecting available
            # production choices lest we generate a tree which violates the
            # depth limit.
            available = [prod for prod in productions if prod['max_path'] <=
                         depth_limit - 1]

            if not available:
                # There are no available choices which do not violate the depth
                # limit. List the choices with the shortest terminating path.
                available = root_info['min_path']
    
    elif method == "full":
        # Build a "full" tree where every branch extends to the depth limit.
        
        if not depth_limit:
            # There is no depth limit specified for building a Full tree.
            # Raise an error as a depth limit HAS to be specified here.
            s = "representation.derivation.legal_productions\n" \
                "Error: Depth limit not specified for `Full` tree derivation."
            raise Exception(s)
        
        elif depth_limit > bnf_grammar.max_arity + 1:
            # If the depth limit is greater than the maximum arity of the
            # grammar, then only recursive production choices can be used.
            available = root_info['recursive']

            if not available:
                # There are no recursive production choices for the current
                # rule. Pick any production choices.
                available = productions

        else:
            # The depth limit is less than or equal to the maximum arity of
            # the grammar + 1. We have to be careful in selecting available
            # production choices lest we generate a tree which violates the
            # depth limit.
            available = [prod for prod in productions if prod['max_path'] ==
                         depth_limit - 1]
                        
            if not available:
                # There are no available choices which extend exactly to the
                # depth limit. List the NT choices with the longest terminating
                # paths that don't violate the limit.
                available = [prod for prod in productions if prod['max_path']
                             < depth_limit - 1]

    return available

class Tree:

    def __init__(self, expr, parent):
        """
        Initialise an instance of the tree class.
        
        :param expr: A non-terminal from the params['BNF_GRAMMAR'].
        :param parent: The parent of the current node. None if node is tree
        root.
        """
        
        self.parent = parent
        self.codon = None
        self.depth = 1
        self.root = expr
        self.children = []
        self.snippet = None

    def __str__(self):
        """
        Builds a string of the current tree.
        
        :return: A string of the current tree.
        """
        
        # Initialise the output string.
        result = "("
        
        # Append the root of the current node to the output string.
        result += str(self.root)
        
        for child in self.children:
            # Iterate across all children.
            
            if len(child.children) > 0:
                # Recurse through all children.
                result += " " + str(child)
            
            else:
                # Child is a terminal, append root to string.
                result += " " + str(child.root)
        
        result += ")"
        
        return result

    def __copy__(self):
        """
        Creates a new unique copy of self.
        
        :return: A new unique copy of self.
        """

        # Copy current tree by initialising a new instance of the tree class.
        tree_copy = Tree(self.root, self.parent)
        
        # Set node parameters.
        tree_copy.codon, tree_copy.depth = self.codon, self.depth

        tree_copy.snippet = self.snippet

        for child in self.children:
            # Recurse through all children.
            new_child = child.__copy__()
            
            # Set the parent of the copied child as the copied parent.
            new_child.parent = tree_copy
            
            # Append the copied child to the copied parent.
            tree_copy.children.append(new_child)

        return tree_copy

    def __eq__(self, other, same=True):
        """
        Set the definition for comparison of two instances of the tree
        class by their attributes. Returns True if self == other.

        :param other: Another instance of the tree class with which to compare.
        :return: True if self == other.
        """

        # Get attributes of self and other.
        a_self, a_other = vars(self), vars(other)
                
        # Don't look at the children as they are class instances themselves.
        taboo = ["parent", "children", "snippet", "id"]
        self_no_kids = {k: v for k, v in a_self.items() if k not in taboo}
        other_no_kids = {k: v for k, v in a_other.items() if k not in taboo}
                
        # Compare attributes
        if self_no_kids != other_no_kids:
            # Attributes are not the same.
            return False
            
        else:
            # Attributes are the same
            child_list = [self.children, other.children]
            
            if len(list(filter(lambda x: x is not None, child_list))) % 2 != 0:
                # One contains children, the other doesn't.
                return False

            elif self.children and len(self.children) != len(other.children):
                # Number of children differs between self and other.
                return False

            elif self.children:
                # Compare children recursively.
                for i, child in enumerate(self.children):
                    same = child.__eq__(other.children[i], same)

        return same

#    def get_target_nodes(self, array, target=None):
#        """
#        Returns the all NT nodes which match the target NT list in a
#        given tree.
        
#        :param array: The array of all nodes that match the target.
#        :param target: The target nodes to match.
#        :return: The array of all nodes that match the target.
#        """
            
#        if self.root in target:
            # Check if the current node matches the target.
            
            # Add the current node to the array.
#            array.append(self)
        
        # Find all non-terminal children of the current node.
#        NT_kids = [kid for kid in self.children if kid.root in
#                   params['BNF_GRAMMAR'].non_terminals]
        
#        for child in NT_kids:
#            if NT_kids:
#                # Recursively call function on any non-terminal children.
#                array = child.get_target_nodes(array, target=target)
        
#        return array

    def get_node_labels(self, labels):
        """
        Recurses through a tree and appends all node roots to a set.
        
        :param labels: The set of roots of all nodes in the tree.
        :return: The set of roots of all nodes in the tree.
        """
        
        # Add the current root to the set of all labels.
        labels.add(self.root)

        for child in self.children:
            # Recurse on all children.
            labels = child.get_node_labels(labels)
        
        return labels

    def get_tree_info(self, nt_keys, genome, output, invalid=False,
                      max_depth=0, nodes=0):
        """
        Recurses through a tree and returns all necessary information on a
        tree required to generate an individual.
        
        :param genome: The list of all codons in a subtree.
        :param output: The list of all terminal nodes in a subtree. This is
        joined to become the phenotype.
        :param invalid: A boolean flag for whether a tree is fully expanded.
        True if invalid (unexpanded).
        :param nt_keys: The list of all non-terminals in the grammar.
        :param nodes: the number of nodes in a tree.
        :param max_depth: The maximum depth of any node in the tree.
        :return: genome, output, invalid, max_depth, nodes.
        """

        # Increment number of nodes in tree and set current node id.
        nodes += 1
        
        if self.parent:
            # If current node has a parent, increment current depth from
            # parent depth.
            self.depth = self.parent.depth + 1
        
        else:
            # Current node is tree root, set depth to 1.
            self.depth = 1
        
        if self.depth > max_depth:
            # Set new max tree depth.
            max_depth = self.depth

        if self.codon:
            # If the current node has a codon, append it to the genome.
            genome.append(self.codon)

        # Find all non-terminal children of current node.
        NT_children = [child for child in self.children if child.root in
                       nt_keys]
        
        if not NT_children:
            # The current node has only terminal children, increment number
            # of tree nodes.
            nodes += 1

            # Terminal children increase the current node depth by one.
            # Check the recorded max_depth.
            if self.depth + 1 > max_depth:
                # Set new max tree depth.
                max_depth = self.depth + 1

        if self.root in nt_keys and not self.children:
            # Current NT has no children. Invalid tree.
            invalid = True

        for child in self.children:
            # Recurse on all children.

            if not child.children:
                # If the current child has no children it is a terminal.
                # Append it to the phenotype output.
                output.append(child.root)
                
                if child.root in nt_keys:
                    # Current non-terminal node has no children; invalid tree.
                    invalid = True
            
            else:
                # The current child has children, recurse.
                genome, output, invalid, max_depth, nodes = \
                    child.get_tree_info(nt_keys, genome, output, invalid,
                                        max_depth, nodes)

        return genome, output, invalid, max_depth, nodes

    def print_tree(self):
        """
        Prints out all nodes in the tree, indented according to node depth.
        
        :return: Nothing.
        """

        print(self.depth, "".join([" " for _ in range(self.depth)]), self.root)

        for child in self.children:
            if not child.children:
                print(self.depth + 1,
                      "".join([" " for _ in range(self.depth + 1)]),
                      child.root)
            else:
                child.print_tree()

def check_ind(ind, check, max_tree_depth):#, max_tree_nodes, max_genome_length):
    """
    Check all shallow aspects of an individual to ensure everything is correct.
    
    :param ind: An individual to be checked.
    :return: False if everything is ok, True if there is an issue.
    """

    if ind.genome == []:
        # Ensure all individuals at least have a genome.
        return True

#    if ind.invalid:
        # We have an invalid.
   #     print(ind.invalid)
#        return True

    elif ind.depth > max_tree_depth:
        # Tree is too deep.
        return True

#    elif ind.nodes > max_tree_nodes:
        # Tree has too many nodes.
#        return True

#    elif len(ind.genome) > max_genome_length:
        # Genome is too long.
#        return True
    
def reMap(ind, genome, bnf_grammar, max_tree_depth, max_wraps):
    ind.phenotype, ind.genome, ind.tree, ind.nodes, ind.invalid, \
                ind.depth, ind.used_codons, ind.n_wraps = mapper(genome, None, bnf_grammar, max_tree_depth, max_wraps)
    return ind



def get_max_genome_index(ind_0, ind_1):
    """
    Given two individuals, return the maximum index on each genome across
    which operations are to be performed. We consider only the used portion 
    of the genome.
    
    :param ind_0: Individual 0.
    :param ind_1: Individual 1.
    :return: The maximum index on each genome across which operations are to be
             performed.
    """

    # Get used codons range.
    
    if ind_0.invalid:
        # ind_0 is invalid. Default to entire genome.
        max_p_0 = len(ind_0.genome)
    
    else:
        max_p_0 = ind_0.used_codons

    if ind_1.invalid:
        # ind_1 is invalid. Default to entire genome.
        max_p_1 = len(ind_1.genome)
    
    else:
        max_p_1 = ind_1.used_codons
    
    return max_p_0, max_p_1

def get_effective_length(ind):
    """
    Return the effective length of the genome for linear mutation.

    :param ind: An individual.
    :return: The effective length of the genome.
    """

    if not ind.genome:
        # The individual does not have a genome; linear mutation cannot be
        # performed.
        return None

    elif ind.invalid:
        # Individual is invalid.
        eff_length = len(ind.genome)

    else:
        eff_length = min(len(ind.genome), ind.used_codons)

    return eff_length
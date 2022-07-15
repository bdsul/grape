# -*- coding: utf-8 -*-
"""
Created on Tue May 10 06:53:28 2022

@author: allan
"""

import re
import math
from operator import attrgetter
import numpy as np
import random

class Individual(object):
    """
    A GE individual.
    """

    def __init__(self, genome, grammar, max_depth, codon_consumption):
        """
        """
        
        self.genome = genome
        if codon_consumption == 'lazy':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_lazy(genome, grammar, max_depth)
        elif codon_consumption == 'eager':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_eager(genome, grammar, max_depth)
        else:
            raise ValueError("Unknown mapper")

class Grammar(object):
    """
    Attributes:
    - non_terminals: list with each non-terminal (NT);
    - start_rule: first non-terminal;
    - production_rules: list with each production rule (PR), which contains in each position:
        - the PR itself as a string
        - 'non-terminal' or 'terminal'
        - the arity (number of NTs in the PR)
        - production choice label
        - True, if it is recursive, and False, otherwise
        - the minimum depth to terminate the mapping of all NTs of this PR
    - n_rules: df
    
    """
    def __init__(self, file_address):
        #Reading the file
        with open(file_address, "r") as text_file:
            bnf_grammar = text_file.read()
        #Getting rid of all the duplicate spaces
        bnf_grammar = re.sub(r"\s+", " ", bnf_grammar)

        #self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>\s*::=",bnf_grammar)]
        self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>\s*::=",bnf_grammar)]
        self.start_rule = self.non_terminals[0]
        for i in range(len(self.non_terminals)):
            bnf_grammar = bnf_grammar.replace(self.non_terminals[i] + " ::=", "  ::=")
        rules = bnf_grammar.split("::=")
        del rules[0]
        rules = [item.replace('\n',"") for item in rules]
        rules = [item.replace('\t',"") for item in rules]
        
        #list of lists (set of production rules for each non-terminal)
        self.production_rules = [i.split('|') for i in rules]
        for i in range(len(self.production_rules)):
            #Getting rid of all leading and trailing whitespaces
            self.production_rules[i] = [item.strip() for item in self.production_rules[i]]
            for j in range(len(self.production_rules[i])):
                #Include in the list the PR itself, NT or T, arity and the production choice label
                if re.findall(r"\<(\w+)\>",self.production_rules[i][j]):
                    arity = len(re.findall(r"\<(\w+)\>",self.production_rules[i][j]))
                    self.production_rules[i][j] = [self.production_rules[i][j] , "non-terminal", arity, j]
                else:
                    self.production_rules[i][j] = [self.production_rules[i][j] , "terminal", 0, j] #arity 0
        #number of production rules for each non-terminal
        self.n_rules = [len(list_) for list_ in self.production_rules]
        
        #Building list of non-terminals with recursive production-rules
        check_recursiveness = [self.start_rule]
        check_size = len(check_recursiveness)
        continue_ = True
        while continue_:
            for i in range(len(self.production_rules)):
                for j in range(len(self.production_rules[i])):
                    for k in range(len(check_recursiveness)):
                        if check_recursiveness[k] in self.production_rules[i][j][0]:
                            if self.non_terminals[i] not in check_recursiveness:
                                check_recursiveness.append(self.non_terminals[i])
            if len(check_recursiveness) != check_size:
                check_size = len(check_recursiveness)
            else:
                continue_ = False
        
        #Including information of recursiveness in each production-rule
        #True if the respective non-terminal has recursive production-rules. False, otherwise
        for i in range(len(self.production_rules)):
            for j in range(len(self.production_rules[i])):
                if self.production_rules[i][j][1] == 'terminal': #a terminal is never recursive
                    recursive = False
                    self.production_rules[i][j].append(recursive)
                else: #a non-terminal can be recursive
                    for k in range(len(check_recursiveness)):
                        #Check if a recursive NT is in the current list of PR
                        #TODO I just changed from self.non_terminals[k] to check_recursiveness[k]
                        if check_recursiveness[k] in self.production_rules[i][j][0]:
                            recursive = True
                            break #since we already found a recursive NT in this PR, we can stop
                        else:
                            recursive = False #if there is no recursive NT in this PR
                    self.production_rules[i][j].append(recursive)
        
        #minimum depth from each non-terminal to terminate the mapping of all symbols
        NT_depth_to_terminate = [None]*len(self.non_terminals)
        #minimum depth from each production rule to terminate the mapping of all symbols
        part_PR_depth_to_terminate = list() #min depth for each non-terminal or terminal to terminate
        isolated_non_terminal = list() #None, if the respective position has a terminal
        #Separating the non-terminals within the same production rule
        for i in range(len(self.production_rules)):
            part_PR_depth_to_terminate.append( list() )
            isolated_non_terminal.append( list() )
            for j in range(len(self.production_rules[i])):
                part_PR_depth_to_terminate[i].append( list() )
                isolated_non_terminal[i].append( list() )
                if self.production_rules[i][j][1] == 'terminal':
                    isolated_non_terminal[i][j].append(None)
                    part_PR_depth_to_terminate[i][j] = 1
                    if not NT_depth_to_terminate[i]:
                        NT_depth_to_terminate[i] = 1
                else:
                    for k in range(self.production_rules[i][j][2]): #arity
                        part_PR_depth_to_terminate[i][j].append( list() )
                        term = re.findall(r"\<(\w+)\>",self.production_rules[i][j][0])[k]
                        isolated_non_terminal[i][j].append('<' + term + '>')
        continue_ = True
        while continue_:
            #after filling up NT_depth_to_terminate, we need to run the loop one more time to
            #fill up part_PR_depth_to_terminate, so we check in the beginning
            if None not in NT_depth_to_terminate:
                continue_ = False 
            for i in range(len(self.non_terminals)):
                for j in range(len(self.production_rules)):
                    for k in range(len(self.production_rules[j])):
                        for l in range(len(isolated_non_terminal[j][k])):
                            if self.non_terminals[i] == isolated_non_terminal[j][k][l]:
                                if NT_depth_to_terminate[i]:
                                    if not part_PR_depth_to_terminate[j][k][l]:
                                        part_PR_depth_to_terminate[j][k][l] = NT_depth_to_terminate[i] + 1
                                        if [] not in part_PR_depth_to_terminate[j][k]:
                                            if not NT_depth_to_terminate[j]:
                                                NT_depth_to_terminate[j] = part_PR_depth_to_terminate[j][k][l]
        PR_depth_to_terminate = []
        for i in range(len(part_PR_depth_to_terminate)):
            for j in range(len(part_PR_depth_to_terminate[i])):
                #the min depth to terminate a PR is the max depth within the items of that PR
                if type(part_PR_depth_to_terminate[i][j]) == int:
                    depth_ = part_PR_depth_to_terminate[i][j]
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)
                else:
                    depth_ = max(part_PR_depth_to_terminate[i][j])
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)
        
def mapper(genome, grammar, max_depth):
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
        structure.append(index_production_chosen)
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_eager(genome, grammar, max_depth):
    """
    Identical to the previous one.
    Solve the names later.
    """    

    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
        structure.append(index_production_chosen)
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_lazy(genome, grammar, max_depth):
    """
    This mapper is similar to the previous one, but it does not consume codons
    when mapping a production rule with a single option."""
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0        
        else: #we consume one codon, and add the index to the structure
            index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
            structure.append(index_production_chosen)
            idx_genome += 1
        
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
            
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def random_initialisation(ind_class, pop_size, bnf_grammar, init_genome_length, max_init_depth, codon_size, codon_consumption):
        """
        
        """
        population = []
        
        for i in range(pop_size):
            genome = []
            for j in range(init_genome_length):
                genome.append(random.randint(0, codon_size))
            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
            population.append(ind)
            
        return population
    
def sensible_initialisation(ind_class, pop_size, bnf_grammar, min_init_depth, max_init_depth, codon_size, codon_consumption):
        """
        
        """
        
        #Calculate the number of individuals to be generated with each method
        is_odd = pop_size % 2
        n_grow = int(pop_size/2)
        
        n_sets_grow = max_init_depth - min_init_depth + 1
        set_size = int(n_grow/n_sets_grow)
        remaining = n_grow % n_sets_grow
        
        n_full = n_grow + is_odd + remaining #if pop_size is odd, generate an extra ind with "full"
        
        #TODO check if it is possible to generate inds with max_init_depth
        
        population = []
        #Generate inds using "Grow"
        for i in range(n_sets_grow):
            max_init_depth_ = min_init_depth + i
            for j in range(set_size):
                remainders = [] #it will register the choices
                possible_choices = [] #it will register the respective possible choices
    
                phenotype = bnf_grammar.start_rule
                remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)] #
                depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
                idx_branch = 0 #index of the current branch being grown
                while len(remaining_NTs) != 0:
                    idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                    total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                    actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth_]
                    Ch = random.choice(actual_options)
                    phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                    depths[idx_branch] += 1
                    remainders.append(Ch[3])
                    possible_choices.append(len(total_options))
                    
                    if Ch[2] > 1:
                        if idx_branch == 0:
                            depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                        else:
                            depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    if Ch[1] == 'terminal':
                        idx_branch += 1
                    
                    remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)]
                
                #Generate the genome
                genome = []
                for j in range(len(remainders)):
                    codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[j])) * possible_choices[j]) + remainders[j]
                    genome.append(codon)
                
                #Include a tail with 50% of the genome's size
                size_tail = int(0.5*len(genome))
                for j in range(size_tail):
                    genome.append(random.randint(0,codon_size))
                    
                #Initialise the individual and include in the population
                ind = ind_class(genome, bnf_grammar, max_init_depth_, codon_consumption)
                
                #Check if the individual was mapped correctly
                if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                    raise Exception('error in the mapping')
                    
                population.append(ind)    
            
        for i in range(n_full):
            remainders = [] #it will register the choices
            possible_choices = [] #it will register the respective possible choices

            phenotype = bnf_grammar.start_rule
            remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)] #
            depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
            idx_branch = 0 #index of the current branch being grown

            while len(remaining_NTs) != 0:
                idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth]
                recursive_options = [PR for PR in actual_options if PR[4]]
                if len(recursive_options) > 0:
                    Ch = random.choice(recursive_options)
                else:
                    Ch = random.choice(actual_options)
                phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                depths[idx_branch] += 1
                remainders.append(Ch[3])
                possible_choices.append(len(total_options))
                
                if Ch[2] > 1:
                    if idx_branch == 0:
                        depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    else:
                        depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                if Ch[1] == 'terminal':
                    idx_branch += 1
                
                remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)]
            
            #Generate the genome
            genome = []
            for j in range(len(remainders)):
                codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[j])) * possible_choices[j]) + remainders[j]
                genome.append(codon)
            
            #Include a tail with 50% of the genome's size
            size_tail = int(0.5*len(genome))
            for j in range(size_tail):
                genome.append(random.randint(0,codon_size))
                
            #Initialise the individual and include in the population
            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
            
            #Check if the individual was mapped correctly
            if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                raise Exception('error in the mapping')
                
            population.append(ind)    
    
        return population
    
def PI_Grow(ind_class, pop_size, bnf_grammar, min_init_depth, max_init_depth, codon_size, codon_consumption):
    
    #Calculate the number of individuals to be generated with each depth
    n_sets = max_init_depth - min_init_depth + 1 
    set_size = int(pop_size/n_sets)
    remaining = pop_size % n_sets #the size of the last set, to be initialised with a random max depth between min_init_depth and max_init_depth
    n_sets += 1 #including the last set, which will have random init depth
    
    #TODO check if it is possible to generate inds with max_init_depth and min_init_depth
    
    population = []
    for i in range(n_sets):
        if i == n_sets - 1:
            max_init_depth_ = random.randint(min_init_depth, max_init_depth + 1)
            set_size = remaining
        else:
            max_init_depth_ = min_init_depth + i

        for j in range(set_size):
            
            remainders = [] #it will register the choices
            possible_choices = [] #it will register the respective possible choices

            phenotype = bnf_grammar.start_rule
            remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)] #
            list_phenotype = remaining_NTs #it keeps in each position a terminal if the respective branch was terminated or a <NT> otherwise
            depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
            branches = [False]*len(remaining_NTs) #False if the respective branch was terminated. True, otherwise
            n_expansions = [0]*len(remaining_NTs) #Number of expansions used in each branch
            while len(remaining_NTs) != 0:
                choose_ = False
                if max(depths) < max_init_depth_:
                    #Count the number of NT with recursive options remaining
                    NT_with_recursive_options = 0
                    l = 0
                    for k in range(len(branches)):
                        if not branches[k]:
                            idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[l])
                            actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[k] <= max_init_depth_]
                            recursive_options = [PR for PR in actual_options if PR[4]]
                            if len(recursive_options) > 0:
                                NT_with_recursive_options += 1
                                idx_recursive = idx_NT
                                idx_branch = k
                                PI_index = l
                            if NT_with_recursive_options == 2:
                                break
                            l += 1
                    if NT_with_recursive_options == 1: #if there is just one NT with recursive options remaining, choose between them
                        total_options = [PR for PR in bnf_grammar.production_rules[idx_recursive]]
                        recursive_options = [PR for PR in bnf_grammar.production_rules[idx_recursive] if PR[5] + depths[idx_branch] <= max_init_depth_ and PR[4]]
                        Ch = random.choice(recursive_options)
                        n_similar_NTs = remaining_NTs[:PI_index+1].count(remaining_NTs[PI_index])
                    
                        phenotype = replace_nth(phenotype, remaining_NTs[PI_index], Ch[0], n_similar_NTs)
                        
                        new_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>", Ch[0])]
                        list_phenotype = list_phenotype[0:idx_branch] + new_NTs + list_phenotype[idx_branch+1:]
                        
                        if remainders == []:
                            remainders.append(Ch[3])
                            possible_choices.append(len(total_options))
                        else:
                            remainder_position = sum(n_expansions[:idx_branch+1])
                            remainders = remainders[:remainder_position] + [Ch[3]] + remainders[remainder_position:]
                            
                            possible_choices = possible_choices[:remainder_position] + [len(total_options)] + possible_choices[remainder_position:]
                        
                        depths[idx_branch] += 1
                        n_expansions[idx_branch] += 1
                        
                        if Ch[2] > 1:
                            if idx_branch == 0:
                                depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                            else:
                                depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                        if Ch[1] == 'terminal':
                            branches[idx_branch] = True
                        else:
                            branches = branches[0:idx_branch] + [False,]*Ch[2] + branches[idx_branch+1:]
                            n_expansions = n_expansions[0:idx_branch+1] + [0,]*(Ch[2]-1) + n_expansions[idx_branch+1:]
                            
                        remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)]
                    
                    else: #choose within any other branch
                        choose_ = True
                else:
                    choose_ = True
                            
                if choose_: #at least one branch has reached the max depth
                    PI_index = random.choice(range(len(remaining_NTs))) #index of the current branch being grown
                    count_ = 0
                    for k in range(len(branches)):
                        if not branches[k]:
                            if count_ == PI_index:
                                idx_branch = k
                                break
                            count_ += 1
                    
                    idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[PI_index])
                    total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                    actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth_]
    
                    Ch = random.choice(actual_options)
                    n_similar_NTs = remaining_NTs[:PI_index+1].count(remaining_NTs[PI_index])
                    phenotype = replace_nth(phenotype, remaining_NTs[PI_index], Ch[0], n_similar_NTs)

                    new_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>", Ch[0])]
                    list_phenotype = list_phenotype[0:idx_branch] + new_NTs + list_phenotype[idx_branch+1:]
                    
                    if remainders == []:
                        remainders.append(Ch[3])
                        possible_choices.append(len(total_options))
                    else:
                        remainder_position = sum(n_expansions[:idx_branch+1])
                        remainders = remainders[:remainder_position] + [Ch[3]] + remainders[remainder_position:]
                        
                        possible_choices = possible_choices[:remainder_position] + [len(total_options)] + possible_choices[remainder_position:]
                        
                    depths[idx_branch] += 1
                    n_expansions[idx_branch] += 1
                    
                    if Ch[2] > 1:
                        if idx_branch == 0:
                            depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                        else:
                            depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    if Ch[1] == 'terminal':
                        branches[idx_branch] = True
                    else:
                        branches = branches[0:idx_branch] + [False,]*Ch[2] + branches[idx_branch+1:]
                        n_expansions = n_expansions[0:idx_branch+1] + [0,]*(Ch[2]-1) + n_expansions[idx_branch+1:]
                        
                    remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)]
            
            #Generate the genome
            genome = []
            for k in range(len(remainders)):
                codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                genome.append(codon)
            
            #Include a tail with 50% of the genome's size
            size_tail = int(0.5*len(genome))
            for k in range(size_tail):
                genome.append(random.randint(0,codon_size))
                
            #Initialise the individual and include in the population
            ind = ind_class(genome, bnf_grammar, max_init_depth_, codon_consumption)
            
            #Check if the individual was mapped correctly
            if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                raise Exception('error in the mapping')
                
            population.append(ind)   
    
    return population
    
def crossover_onepoint(parent0, parent1, bnf_grammar, max_depth, codon_consumption):
    """
    
    """
    if parent0.invalid: #used_codons = 0
        possible_crossover_codons0 = len(parent0.genome)
    else:
        possible_crossover_codons0 = min(len(parent0.genome), parent0.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    if parent1.invalid:
        possible_crossover_codons1 = len(parent1.genome)
    else:
        possible_crossover_codons1 = min(len(parent1.genome), parent1.used_codons)
    
    continue_ = True
    while continue_:
        #Set points for crossover within the effective part of the genomes
        point0 = random.randint(1, possible_crossover_codons0)
        point1 = random.randint(1, possible_crossover_codons1)
        
        #Operate crossover
        new_genome0 = parent0.genome[0:point0] + parent1.genome[point1:]
        new_genome1 = parent1.genome[0:point1] + parent0.genome[point0:]
        
        new_ind0 = reMap(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption)
        new_ind1 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
        
        continue_ = new_ind0.depth > max_depth or new_ind1.depth > max_depth

    return new_ind0, new_ind1   

def mutation_int_flip_per_codon(ind, mut_probability, codon_size, bnf_grammar, max_depth, codon_consumption):
    """

    """
    # Operation mutation within the effective part of the genome
    if ind.invalid: #used_codons = 0
        possible_mutation_codons = len(ind.genome)
    else:
        possible_mutation_codons = min(len(ind.genome), ind.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    continue_ = True

    while continue_:
        for i in range(possible_mutation_codons):
            if random.random() < mut_probability:
                ind.genome[i] = random.randint(0, codon_size)
    
        new_ind = reMap(ind, ind.genome, bnf_grammar, max_depth, codon_consumption)
        
        continue_ = new_ind.depth > max_depth

    return new_ind,

def reMap(ind, genome, bnf_grammar, max_tree_depth, codon_consumption):
    
    ind.genome = genome
    if codon_consumption == 'lazy':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_lazy(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'eager':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_eager(genome, bnf_grammar, max_tree_depth)
    else:
        raise ValueError("Unknown mapper")
        
    return ind

def replace_nth(string, substring, new_substring, nth):
    find = string.find(substring)
    i = find != -1
    while find != -1 and i != nth:
        find = string.find(substring, find + 1)
        i += 1
    if i == nth:
        return string[:find] + new_substring + string[find+len(substring):]
    return string
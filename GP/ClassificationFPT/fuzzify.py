#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:37:00 2019

@author: allan
"""

import numpy as np
import pandas as pd
import re
import skfuzzy as fuzz
import math 

def matrixDomain(DataFrame, numeric_columns=[], categorical_columns=[]):
    """
    Return a list of lists in the format:
        [
            ['c', 'S', 'M', 'A'], => it means it's categorical, and it's followed by the categories
            ['n', 0, 5] => it means it's numerical and it's followed by the domain
            
            ]
    """
    
    df = DataFrame.copy()
    _, numColumns = np.shape(df)
    
    if numColumns != len(numeric_columns) + len(categorical_columns):
        raise ValueError("There are columns not defined as numeric or categorical")
    
    df.dropna(inplace = True)  
    
    if numeric_columns:
        if categorical_columns:
            dfNumeric = df.drop(columns=categorical_columns)
        else:
            dfNumeric = df
            numColumnsCategorical = 0
        _, numColumnsReal = np.shape(dfNumeric)
        nameColReal = dfNumeric.columns
        minimum = np.zeros([numColumnsReal], dtype=float)
        maximum = np.zeros([numColumnsReal], dtype=float)
        array = dfNumeric.values
        for i in range(numColumnsReal):
            minimum[i] = min(array[:,i])
            maximum[i] = max(array[:,i])
    if categorical_columns:
        if numeric_columns:
            dfCategorical = df.drop(columns=numeric_columns)
        else:
            dfCategorical = df
            numColumnsReal = 0
        _, numColumnsCategorical = np.shape(dfCategorical)
        nameColCategorical = dfCategorical.columns
                    
    #Matrix domain, where the number of rows is the number of features
    matrixDomain = np.empty([numColumns,2], dtype=float)
    
    #Warning
    if (numColumnsReal + numColumnsCategorical) != numColumns:
        print("Attention! Presence of discreet non-categorical columns.")
        print("The domain matrix will not be filled.")
        print()
        return
    
    nameCol = df.columns #all feature names

    j, k = 0, 0 #indexes of numerical and categorical columns, respectively
    for i in range(numColumns):
        if j < numColumnsReal: 
            if nameCol[i] == nameColReal[j]:
                #fill row i with min and max of feature j
                matrixDomain[i][:] = minimum[j], maximum[j]
                j += 1
        if k < numColumnsCategorical: 
            if nameCol[i] == nameColCategorical[k]:
                #fill row with the number of categories
                matrixDomain[i] = len(dfCategorical[nameColCategorical[k]].unique())
                k += 1 
               
    listDomain = []             
    j, k = 0, 0 #indexes of numerical and categorical columns, respectively
    for i in range(numColumns):
        if j < numColumnsReal: 
            if nameCol[i] == nameColReal[j]:
                #fill list i with min and max of feature j
                listDomain.append(['n', minimum[j], maximum[j]])
                j += 1
        if k < numColumnsCategorical: 
            if nameCol[i] == nameColCategorical[k]:
                list_ = ['c']
                #fill list i with categories
                for l in range(len(dfCategorical[nameColCategorical[k]].unique())):
                    list_.append(str(dfCategorical[nameColCategorical[k]].unique()[l]))
                listDomain.append(list_)
                k += 1 
    
    return listDomain

def fuzzifyDataFrame(DataFrame, nSets, matrixDomain):
    
    df = DataFrame.copy() 
    
    numRows, numColumns = np.shape(df)
    
    #check if all features are referred in the matrixDomain
    if len(matrixDomain) == ():
        numVariables = 0
    else:    
        numVariables = len(matrixDomain)
    if numVariables != numColumns:
        print("Domain matrix does not represent all the variables")
        return
    
    totalIndexes = list(df.index)
    
    df.dropna(inplace = True) 
    
    numRowsAposRemocao,_ = np.shape(df)
    numLinhasEliminadas = numRows - numRowsAposRemocao
    if numLinhasEliminadas != 0:
        print("Warning: %i lines were deleted because they didn't contain all the attributes" %numLinhasEliminadas)
    
    #indexes of non-removed data
    validIndexes = list(df.index)
    
    validNumberRows, _ = np.shape(df) #new number of rows (the number of columns didn't change)
    
    #keep in totalIndexes only the removed values
    for i in range(validNumberRows):
        totalIndexes.remove(validIndexes[i])

    dataReal = df.copy()
    dataCategorical = df.copy()
    
    nonReal = [] #eliminate categorical data in dataReal
    nonCategorical = [] #eliminate non-categorical data in dataCategorical
    
    for i in range(numVariables):
        if matrixDomain[i][0] == 'c': #categorical data
            nonReal.append(i)
        else: #non-categorical data
            nonCategorical.append(i)
    
    dataReal.drop(dataReal.columns[nonReal], axis=1, inplace=True)
    dataCategorical.drop(dataCategorical.columns[nonCategorical], axis=1, inplace=True)
    
    _, numColumnsReal = np.shape(dataReal) #the number of lines doesn't matter, because it's the same in validNumberRows
        
    _, numColumnsCategorical = np.shape(dataCategorical)
        
    nameCol = df.columns #names of all features
    nameColReal = dataReal.columns #names of fuzzy features
    nameColCategorical = dataCategorical.columns #names of categorical features
    
    arraySets = np.empty(numColumns, dtype=int)
    #arraySets keeps the number of sets to split each feature
    #If nSets is integer, all non-categorical features will be splitted with the same number of sets
    #If it's an array, each position refers to each column
    #The position reffering to a categorical feature should have the number of categories of that feature
    j, k = 0, 0
    if type(nSets) == int:
        if nSets < 2:
            print("Number of sets must be greater than or equal to 2")
            return
        else:
            for i in range(numColumns):
                if numColumnsReal > j: #if there are any more columns to verify
                    if nameCol[i] == nameColReal[j]:
                        arraySets[i] = nSets
                        j += 1
                if numColumnsCategorical > k: #same for non-categorical
                    if nameCol[i] == nameColCategorical[k]:
                        arraySets[i] = len(matrixDomain[i]) - 1 #-1 because the first position is 'n' or 'c'
                        k += 1
                        
    else: #if it's an array
        nSetsSize = len(nSets)
        if numVariables != nSetsSize:
            print("Size of the array nSets must be equal to the number of variables.")
            return
        for i in range(numColumns):
            if nSets[i] < 2:
                print("Number of sets must be greater than or equal to 2")
                return
            arraySets[i] = nSets[i]
    
    #if some rows in the dataframe were removed (for missing values), the pointing could be difficult,
    #so, we send the values of the dataframe to a matrix
    #The respective positions with removed data in the dataframe will get zero in the matrix
    matrixDataReal = np.zeros([numRows,numColumnsReal], dtype=float)
    
    sumSets = int(sum(arraySets)) #total of sets
    for i in range(numColumns):
        if matrixDomain[i][0] == 'c' and arraySets[i] == 2:
            sumSets -= 1
    pertinenceMatrix = np.zeros([numRows,sumSets], dtype=float)
    pertinenceMatrixDF = {} #final dataframe
    
    i = 0 #we cannot use the index as pointer because of possible missing rows
    for index, row in dataReal.iterrows(): 
        for j in range(numColumnsReal):
            matrixDataReal[validIndexes[i]][j] = row[nameColReal[j]]
        i += 1
    
    actualColumn = 0 #column to the filled now
    actualIndexSets = 0
    for i in range(numColumns):
        if matrixDomain[i][0] == 'c': #categorical data
            if arraySets[i] != 2:
                arrayCategories = []
                for j in range(arraySets[i]):
                    arrayCategories.append(matrixDomain[i][j+1])
                j = 0 #position in the array of valid data
                for index in range(validNumberRows): #for each valid row
                    for k in range(arraySets[i]): #for each set of the current feature
                        if arrayCategories[k] == str(df.loc[validIndexes[j]][i]): #quando as categorias são números, às vezes ocorrem problemas. Verificar se str() corrigiu
                            #se o objeto pertence à categoria atual, a posição na matriz de pertinência será 1
                            pertinenceMatrix[validIndexes[j],actualColumn+k] = 1
                        else:
                            #senão será 0
                            pertinenceMatrix[validIndexes[j],actualColumn+k] = 0
                    j += 1
                actualColumn += arraySets[i] #todas as colunas referentes aos conjuntos da variável atual foram preenchidas
                actualIndexSets += 1
            else: #only two categories, we codified one columns with 0 and 1
                arrayCategories = []
                for j in range(arraySets[i]):
                    arrayCategories.append(matrixDomain[i][j+1])
                j = 0 #posição no vetor de índices válidos
                for index in range(validNumberRows): #para cada linha válida
                    if arrayCategories[0] == str(df.loc[validIndexes[j]][i]):
                        pertinenceMatrix[validIndexes[j],actualColumn] = 0
                    elif arrayCategories[1] == str(df.loc[validIndexes[j]][i]):
                        pertinenceMatrix[validIndexes[j],actualColumn] = 1
                    else:
                        raise TypeError("check fuzzification")
                    j += 1        
                actualColumn += 1 #todas as colunas referentes aos conjuntos da variável atual foram preenchidas
                actualIndexSets += 1                
        else:# matrixDomain[i,0] != matrixDomain[i,1]: #dados reais
            lowerBound = matrixDomain[i][1] #início do domínio
            upperBound = matrixDomain[i][2] #fim do domínio
            width = (upperBound - lowerBound) / (arraySets[i] - 1) #largura do conjunto fuzzy, isto é, a largura da subida ou da descida
            step = (upperBound - lowerBound) / 1000
            
            #fuzzificação
            
            x = np.arange(lowerBound, upperBound + step, step)

            qual = [[[] for _ in range(validNumberRows)] for _ in range(arraySets[i])] #conjuntos fuzzy
            qual_level = [[] for _ in range(arraySets[i])] #valores de pertinência
    
            #primeiro termo fuzzy
            a = lowerBound - step
            b = lowerBound
            c = lowerBound + width
            qual[0] = fuzz.trimf(x, [a, b, c])
            
            #termos fuzzy do meio
            if arraySets[i] > 2:
                for j in range(arraySets[i]-2):#-1): #com o -1 vale para os do meio e o último
                    a = b
                    b = c
                    c = c + width
                    qual[j+1] = fuzz.trimf(x, [a, b, c])

            #último termo fuzzy
            a = upperBound - width
            b = upperBound
            c = upperBound + step
            qual[arraySets[i]-1] = fuzz.trimf(x, [a, b, c])
            
            m = 0
            for index in range(validNumberRows):
                data = DataFrame.loc[validIndexes[m]][i]
                #para evitar problemas com as extremidades
                if data <= lowerBound:
                    qual_level[0] = 1
                    pertinenceMatrix[validIndexes[m],actualColumn] = 1
                    for k in range(arraySets[i]-1):
                        qual_level[k+1] = 0
                        pertinenceMatrix[validIndexes[m],actualColumn+k+1] = 0
                elif data >= upperBound:
                    qual_level[arraySets[i]-1] = 1
                    pertinenceMatrix[validIndexes[m],actualColumn+arraySets[i]-1] = 1
                    for k in range(arraySets[i]-1):
                        qual_level[k] = 0
                        pertinenceMatrix[validIndexes[m],actualColumn+k] = 0
                else:
                    for k in range(arraySets[i]):
                        qual_level[k] = fuzz.interp_membership(x, qual[k], data)
                        pertinenceMatrix[validIndexes[m],actualColumn+k] = qual_level[k]
                m += 1
            actualColumn += arraySets[i]
            actualIndexSets += 1
            
    #cria dataframe a partir da matriz
    actualColumn = 0
    for i in range(numColumns):
        if arraySets[i] == 2:
            pertinenceMatrixDF['{0}'.format(nameCol[i])] = pertinenceMatrix[:,actualColumn]
            actualColumn += 1
        else:
            for j in range(arraySets[i]):
                pertinenceMatrixDF['{0}{1}{2}'.format(nameCol[i],'-',j)] = pertinenceMatrix[:,actualColumn+j]
            actualColumn += arraySets[i]
    pertinenceDataFrame = pd.DataFrame(pertinenceMatrixDF)      
    
    #elimina do dataframe final as mesmas linhas que foram removidas do dataframe inicial
    finalPertinenceDataFrame = pertinenceDataFrame.drop(totalIndexes)
    
    return finalPertinenceDataFrame
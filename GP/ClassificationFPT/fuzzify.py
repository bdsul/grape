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
        if matrixDomain[i][0] == 'c': #dados categóricos
            nonReal.append(i)
        else: #dados não categóricos
            nonCategorical.append(i)
    
    dataReal.drop(dataReal.columns[nonReal], axis=1, inplace=True)
    dataCategorical.drop(dataCategorical.columns[nonCategorical], axis=1, inplace=True)
    
    #dimensões da parte não categórica do dataframe
    #dataReal = df.select_dtypes(include=['float64'])
    _, numColumnsReal = np.shape(dataReal) #o número de linhas não interessa, pois é o mesmo em validNumberRows
        
    #dimensões da parte categórica do dataframe
    #dataCategorical = df.select_dtypes(include=['object'])
    _, numColumnsCategorical = np.shape(dataCategorical)
        
    #nomes das variáveis
    nameCol = df.columns #todos os nomes
    nameColReal = dataReal.columns #nomes das variáveis reais
    nameColCategorical = dataCategorical.columns #nomes das variáveis categóricas
    
    arraySets = np.empty(numColumns, dtype=int)
    #arraySets armazena o número de conjuntos em que cada variável se dividirá
    #Caso a entrada nSets seja um valor inteiro, quer dizer que todas as variáveis não categóricas
    #se  dividirão no mesmo número de conjuntos
    #Caso seja um array, cada posição faz referência a uma coluna com dados no dataframe
    #A posição referente a uma coluna categórica deve conter exatamente o mesmo número de 
    #categorias em que os dados estão divididos
    j, k = 0, 0 #índices dos nomes de colunas com dados reais e categóricos, respectivamente
    if type(nSets) == int:
        if nSets < 2:
            print("Number of sets must be greater than or equal to 2")
            return
        else:
            for i in range(numColumns):
                if numColumnsReal > j: #se ainda há colunas reais a verificar
                    if nameCol[i] == nameColReal[j]:
                        arraySets[i] = nSets
                        j += 1
                if numColumnsCategorical > k: #idem para colunas categóricas
                    if nameCol[i] == nameColCategorical[k]:
                        #se o número de categorias indicado na matriz é realmente o número de categorias em que os dados se dividem
                        arraySets[i] = len(matrixDomain[i]) - 1 #-1 because the first position is 'n' or 'c'
                        k += 1
                        
    else: #se o valor passado foi um vetor
        nSetsSize = len(nSets)
        if numVariables != nSetsSize:
            print("Size of the array nSets must be equal to the number of variables.")
            return
        for i in range(numColumns):
            if nSets[i] < 2:
                print("Number of sets must be greater than or equal to 2")
                return
            arraySets[i] = nSets[i]
    
    #é necessário passar os dados do dataframe para uma matriz, pois o dataframe pode ter
    #tido linhas deletadas (por falta de dados), o que dificulta o seu endereçamento
    #Nesta matriz, as posições do dataframe com dados deletados ficarão zeradas
    matrixDataReal = np.zeros([numRows,numColumnsReal], dtype=float)
    
    sumSets = int(sum(arraySets)) #total de conjuntos a dividir as variáveis
    for i in range(numColumns):
        if matrixDomain[i][0] == 'c' and arraySets[i] == 2:
            sumSets -= 1
    pertinenceMatrix = np.zeros([numRows,sumSets], dtype=float) #matriz de pertinência
    pertinenceMatrixDF = {} #dataframe final
    
    i = 0 #não uso o index para referenciar, porque pode ter linhas deletadas
    for index, row in dataReal.iterrows(): 
        for j in range(numColumnsReal):
            matrixDataReal[validIndexes[i]][j] = row[nameColReal[j]]
       # print(index,i)
        i += 1
    
    #hora de preencher a matriz de pertinência
    actualColumn = 0 #coluna que será preenchida
    actualIndexSets = 0
    for i in range(numColumns):
        if matrixDomain[i][0] == 'c': #dados categóricos
            if arraySets[i] != 2:
                arrayCategories = []
                for j in range(arraySets[i]):
                    arrayCategories.append(matrixDomain[i][j+1])
#                arrayCategories = np.empty(arraySets[i], dtype=str) #arraySets[i] é o número de categorias
#                arrayCategories = df[nameCol[i]].unique() #nome de cada categoria da coluna i
                #Se as categorias são números, especialmente 0 e 1, e pela leitura das classes, 
                #tenham sido ordenadas como 1 e 0, vai dar erro, se o que se espera é uma fuzzificação 
                #0=10 e 1=01, então para isso ordena-se antes
#                arrayCategories = sorted(arrayCategories) 
                #for j in range(arraySets[i]): #necessário, caso as categorias sejam números
                #    arrayCategories[j] = str(df[nameCol[i]].unique()[j])
                j = 0 #posição no vetor de índices válidos
                for index in range(validNumberRows): #para cada linha válida
                    for k in range(arraySets[i]): #para cada conjunto da variável atual
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
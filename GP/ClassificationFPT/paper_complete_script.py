import algorithms_gp as algorithms
from functions import WA, OWA, minimum, maximum, dilator, concentrator, complement

#from os import path
import pandas as pd
import numpy as np
import math
from deap import creator, base, tools, gp
import operator

import random
import sys
import itertools
import os

from sklearn.model_selection import train_test_split

from fuzzify import matrixDomain, fuzzifyDataFrame
from category_encoders.target_encoder import TargetEncoder

import scipy.io

from selection import selEpsilonLexicaseCount, selEpsilonLexi2_nodesCountTies, selBatchEpsilonLexi2_nodesCountTies, selDynEpsilonLexi2_nodesCountTies, selDynBatchEpsilonLexi2_nodesCountTies, selBatchEpsilonLexi2_nodesCountTies_MADafter, selTournamentExtra

import warnings
warnings.filterwarnings("ignore")

scenario = 0#int(sys.argv[1])
problem = 'heartDisease'#sys.argv[2]
run = 1#int(sys.argv[3])

N_RUNS = 1#int(sys.argv[4])

def setDataSet(problem, RANDOM_SEED, FUZZY_SETS):
    fuzzy_sets = FUZZY_SETS

    if problem == 'heartDisease':
        data =  pd.read_csv(r"../../datasets/processed.cleveland.data", sep=",")
        #There are some data missing on columns d11 and d12 represented by ?, so let's remove the ?
        data = data[data.ca != '?']
        data = data[data.thal != '?']
        
        data['class'] = data['class'].map(lambda x: 1 if x in [2, 3, 4] else x)
        
        Ypd = data['class'] 
        #Y = data['class'].to_numpy()
        
        outputsOneHot = pd.get_dummies(Ypd, columns=['class'])
        Y = outputsOneHot.to_numpy()
        
        data = data.drop(['class'], axis=1)
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)
        
        domain = matrixDomain(df_train, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'australian': #66
        data =  pd.read_csv(r"datasets/australian.dat", sep=" ")    
        data = data[data.d3 != 3] #there are only 2 samples with d3=3, so let's remove them
        
        Ypd = data['output'] 
        data = data.drop(['output'], axis=1)
        
        outputsOneHot = pd.get_dummies(Ypd, columns=['output'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        domain = matrixDomain(df_train, ['d1', 'd2', 'd4', 'd5', 'd6', 'd9', 'd12', 'd13'], ['d0', 'd3', 'd7', 'd8', 'd10', 'd11'])
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'segment': #66
        data_training =  pd.read_csv(r"datasets/segmentation.data", sep=",")
        data_test =  pd.read_csv(r"datasets/segmentation.test", sep=",")
        labels_training = data_training[['class']]
        labels_test = data_test[['class']]
        data_training.pop('class')
        data_training.pop('region-pixel-count') #same value in every sample
        data_test.pop('class')
        data_test.pop('region-pixel-count') #same value in every sample
        
        outputsOneHot = pd.get_dummies(labels_training, columns=['class'])
        Y_train = outputsOneHot.to_numpy()
        
        outputsOneHot = pd.get_dummies(labels_test, columns=['class'])
        Y_test = outputsOneHot.to_numpy()
        
        domain = matrixDomain(data_training, ['region-centroid-col', 'region-centroid-row', 'vedge-mean', 'vegde-sd', 'hedge-mean', 'hedge-sd', 'intensity-mean', 'rawred-mean', 'rawblue-mean', 'rawgreen-mean', 'exred-mean', 'exblue-mean', 'exgreen-mean', 'value-mean', 'saturatoin-mean', 'hue-mean'], ['short-line-density-5', 'short-line-density-2'])
        domain[2] = domain[3] #because col 2 on training misses a category
        
        fuzzy_train = fuzzifyDataFrame(data_training, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(data_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'satellite': #66
        data_train =  pd.read_csv(r"datasets/satellite_train.csv", sep=" ")
        data_test =  pd.read_csv(r"datasets/satellite_test.csv", sep=" ")
        labels_train =  data_train[['class']]
        labels_test =  data_test[['class']]
        data_train.pop('class')
        data_test.pop('class')
        
        outputsOneHotTrain = pd.get_dummies(labels_train, columns=['class'])
        outputsOneHotTest = pd.get_dummies(labels_test, columns=['class'])
        Y_train = outputsOneHotTrain.to_numpy()
        Y_test = outputsOneHotTest.to_numpy()
        
        columns = data_train.columns
        
        domain = matrixDomain(data_train, list(columns))
        
        fuzzy_train = fuzzifyDataFrame(data_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(data_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'cover': 
    
        data =  pd.read_csv(r"datasets/cover.data", sep=",")
        labels =  data[['class']]
        data.pop('class')
        
        outputsOneHot = pd.get_dummies(labels, columns=['class'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.98, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        domain = matrixDomain(df_train, list(columns)[0:10], list(columns)[10:54])
        domain[28,0] = 2
        domain[28,1] = 2
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'vowel': 
    
        data =  pd.read_csv(r"datasets/vowel.csv", sep=",")
        labels =  data[['class']]
        data.pop('class')
        
        outputsOneHot = pd.get_dummies(labels, columns=['class'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        domain = matrixDomain(df_train, list(columns))
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'pima':
        data =  pd.read_csv(r"datasets/pima.csv", sep=",")
        labels =  pd.read_csv(r"datasets/pima_labels.csv")
                    
        outputsOneHot = pd.get_dummies(labels, columns=['y'])
        
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        domain = matrixDomain(df_train, ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'])
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
            
    if problem == 'haberman': #66
        data =  pd.read_csv(r"datasets/haberman.csv", sep=",")
        labels =  pd.read_csv(r"datasets/haberman_labels.csv")
        
        outputsOneHot = pd.get_dummies(labels, columns=['class'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        domain = matrixDomain(df_train, ['d1', 'd2', 'd3'])
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
    
    if problem == 'transfusion': #66
        data =  pd.read_csv(r"datasets/transfusion.csv", sep=",")
        labels =  pd.read_csv(r"datasets/transfusion_labels.csv")
        
        outputsOneHot = pd.get_dummies(labels, columns=['class'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        domain = matrixDomain(df_train, ['d1', 'd2', 'd3', 'd4'])
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'lupus': #66
        data =  pd.read_csv(r"datasets/lupus.csv", sep=",")
        labels =  pd.read_csv(r"datasets/lupus_labels.csv")
        
        outputsOneHot = pd.get_dummies(labels, columns=['class'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        domain = matrixDomain(df_train, ['d1', 'd2', 'd3'])
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'germanCredit': #66
        data =  pd.read_csv(r"datasets/german.data", sep=",")    
        
        Ypd = data['class'] 
        data = data.drop(['class'], axis=1)
        
        outputsOneHot = pd.get_dummies(Ypd, columns=['class'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        domain = matrixDomain(df_train, ['d2', 'd5', 'd8', 'd11', 'd13', 'd16', 'd18'], ['d1', 'd3', 'd4', 'd6', 'd7', 'd9', 'd10', 'd12', 'd14', 'd15', 'd17', 'd19', 'd20'])
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
    
    if problem == 'wine': #66
        data =  pd.read_csv(r"datasets/wine.data", sep=",")    
        
        Ypd = data['class'] 
        data = data.drop(['class'], axis=1)
        
        outputsOneHot = pd.get_dummies(Ypd, columns=['class'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        domain = matrixDomain(df_train, ['Alcohol','Malic-acid','Ash','Alcalinity-of-ash','Magnesium','Total-phenols','Flavanoids','Nonflavanoid-phenols','Proanthocyanins','Color-intensity','Hue','OD280/OD315-of-diluted-wines','Proline'])
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
    
    if problem == 'iris': #66
        data =  pd.read_csv(r"datasets/iris.data", sep=",")    
        
        Ypd = data['class'] 
        data = data.drop(['class'], axis=1)
        
        outputsOneHot = pd.get_dummies(Ypd, columns=['class'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        domain = matrixDomain(df_train, ['sepal-length','sepal-width','petal-length','petal-width'])
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'shuttle': #66
        data_train =  pd.read_csv(r"datasets/shuttle.trn", sep=" ")
        data_test =  pd.read_csv(r"datasets/shuttle.tst", sep=" ")
        labels_train =  data_train[['class']]
        labels_test =  data_test[['class']]
        data_train.pop('class')
        data_test.pop('class')
        
        outputsOneHotTrain = pd.get_dummies(labels_train, columns=['class'])
        outputsOneHotTest = pd.get_dummies(labels_test, columns=['class'])
        Y_train = outputsOneHotTrain.to_numpy()
        Y_test = outputsOneHotTest.to_numpy()
        
        columns = data_train.columns
        
        domain = matrixDomain(data_train, list(columns))
        
        fuzzy_train = fuzzifyDataFrame(data_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(data_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'adult': #66
        data_training =  pd.read_csv(r"datasets/adult.data", sep=",")
        data_test =  pd.read_csv(r"datasets/adult.test", sep=",")
        data_training['class'] = data_training['class'].replace({'>50K': 1, '<=50K': 0})
        data_test['class'] = data_test['class'].replace({'>50K': 1, '<=50K': 0})
        labels_training = data_training[['class']]
        labels_test = data_test[['class']]
        data_training.pop('class')
        data_test.pop('class')
        
        # Replace '?' with NaN for numerical operations
        data_training.replace('?', pd.NA, inplace=True)
        data_test.replace('?', pd.NA, inplace=True)
        
        # Convert columns to numeric type
        data_training = data_training.apply(pd.to_numeric, errors='ignore')
        data_test = data_test.apply(pd.to_numeric, errors='ignore')
        
        # Replace NaN values with corresponding column medians (numeric features)
        median_values = data_training.median()
        data_training.fillna(median_values, inplace=True)
        median_values = data_test.median()
        data_test.fillna(median_values, inplace=True)
        
        # Replace NaN values with corresponding column modes (categorical features)
        for column in data_training.columns:
            if data_training[column].dtype == 'object':  # Check if the column is categorical
                mode_value = data_training[column].mode()[0]
                data_training[column].fillna(mode_value, inplace=True)
                mode_value = data_test[column].mode()[0]
                data_test[column].fillna(mode_value, inplace=True)
        
        outputsOneHot = pd.get_dummies(labels_training, columns=['class'])
        Y_train = outputsOneHot.to_numpy()
        
        outputsOneHot = pd.get_dummies(labels_test, columns=['class'])
        Y_test = outputsOneHot.to_numpy()
        
        enc_auto = TargetEncoder(verbose=0, 
                                 cols=['workclass', 'education', 'marital-status', 
                                       'occupation', 'relationship', 'race', 
                                       'sex', 'native-country'], 
                                 drop_invariant=False, 
                                 return_df=True, handle_missing='value', 
                                 handle_unknown='value', min_samples_leaf=20, 
                                 smoothing=10, hierarchy=None)
        enc = enc_auto.fit(data_training, Y_train[:,1]) #col 1 of Y is 0 if class=0 and 1 if class=1
        X_train_coded = enc.transform(data_training)
        X_test_coded = enc.transform(data_test)
        
        #columns = data.columns
        #columns_num = data_numerical.columns.to_list()
        #columns_cat = data_categorical.columns.to_list()
        #df_train = pd.DataFrame(np.concatenate((X_train_num, X_train_cat_coded), axis=1), columns=columns_num+columns_cat)
        #df_test = pd.DataFrame(np.concatenate((X_test_num, X_test_cat_coded), axis=1), columns=columns_num+columns_cat)
        
        #df_train = df_train.reset_index(drop=True)
        #df_test = df_test.reset_index(drop=True)
        
        
        domain = matrixDomain(X_train_coded, ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country'], ['sex'])
        
        fuzzy_train = fuzzifyDataFrame(X_train_coded, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(X_test_coded, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'lawsuit':
        data = scipy.io.loadmat(r"datasets/data_lawsuit.mat")
        labels = scipy.io.loadmat(r"datasets/labels_lawsuit.mat")
        
        data = data['data_lawsuit']
        labels = labels['labs_lawsuit']
                    
        outputsOneHot = pd.get_dummies(labels[:,0])
        
        Y = outputsOneHot.to_numpy()
        
        X = data#.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = ['d1', 'd2', 'd3', 'd4']
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        domain = matrixDomain(df_train, ['d1', 'd2', 'd3'], ['d4'])
        
        fuzzy_train = fuzzifyDataFrame(df_train, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(df_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()

    if problem == 'bankMarketing': #66
        data =  pd.read_csv(r"datasets/bank-additional-full.csv", sep=",")    
        
        Ypd = data['class'] 
        data = data.drop(['class'], axis=1)
        
        outputsOneHot = pd.get_dummies(Ypd, columns=['class'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.75, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)
        
        enc_auto = TargetEncoder(verbose=0, 
                                 cols=['job','marital','education','default',
                                       'housing','loan','contact','month',
                                       'day_of_week','poutcome'], 
                                 drop_invariant=False, 
                                 return_df=True, handle_missing='value', 
                                 handle_unknown='value', min_samples_leaf=20, 
                                 smoothing=10, hierarchy=None)
        enc = enc_auto.fit(df_train, Y_train[:,1]) #col 1 of Y is 0 if class=0 and 1 if class=1
        X_train_coded = enc.transform(df_train)
        X_test_coded = enc.transform(df_test)
        
        domain = matrixDomain(X_train_coded, ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','job','marital','education','default','housing','loan','month','day_of_week','poutcome'], ['contact'])
        
        fuzzy_train = fuzzifyDataFrame(X_train_coded, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(X_test_coded, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()

    if problem == 'recidivism': #66
        data =  pd.read_csv(r"datasets/compas-scores-two-years.csv", sep=",")
        #raw_data, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, 
        #            days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out
        data.pop('id')
        data.pop('name')
        data.pop('first')
        data.pop('last')
        data.pop('compas_screening_date')
        data.pop('dob')
        data.pop('c_jail_in')
        data.pop('c_jail_out')
        data.pop('c_case_number')
        data.pop('c_offense_date')
        data.pop('c_arrest_date')
        data.pop('r_case_number')
        data.pop('r_offense_date')
        data.pop('r_jail_in')
        data.pop('r_jail_out')
        data.pop('r_charge_degree')
        data.pop('r_days_from_arrest')
        data.pop('r_charge_desc')
        data.pop('violent_recid')
        data.pop('vr_case_number')
        data.pop('vr_charge_degree')
        data.pop('vr_offense_date')
        data.pop('vr_charge_desc')
        data.pop('type_of_assessment')
        data.pop('screening_date')
        data.pop('v_type_of_assessment')
        data.pop('v_screening_date')
        data.pop('in_custody')
        data.pop('out_custody')
        data.pop('is_violent_recid')
        data.pop('event')
        data.pop('two_year_recid')
        data.pop('start')
        data.pop('end')
        data.pop('juv_fel_count')
        data.pop('juv_other_count')
        data.pop('juv_misd_count')
        data.pop('c_days_from_compas')
        data.pop('c_charge_desc')
        data.pop('decile_score.1')
        data.pop('v_decile_score')
        data.pop('v_score_text')
        data.pop('priors_count.1')
        
        # Filter rows where 'days_b_screening_arrest' is between -30 and 30
        data = data[(data['days_b_screening_arrest'] >= -30) & (data['days_b_screening_arrest'] <= 30)]
        data = data[(data['is_recid'] != -1)]
        data = data[(data['c_charge_degree'] != "O")]
        data = data[(data['score_text'] != 'N/A')]
        
        # Convert columns to numeric type
        data = data.apply(pd.to_numeric, errors='ignore')
        
        # Replace NaN values with corresponding column medians (numeric features)
        median_values = data.median()
        data.fillna(median_values, inplace=True)
        
        Ypd = data['is_recid'] 
        data = data.drop(['is_recid'], axis=1)
        
        outputsOneHot = pd.get_dummies(Ypd, columns=['is_recid'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)
        
        enc_auto = TargetEncoder(verbose=0, 
                                 cols=['sex', 'age_cat', 'race', 'c_charge_degree',
                                        'score_text'], 
                                 drop_invariant=False, 
                                 return_df=True, handle_missing='value', 
                                 handle_unknown='value', min_samples_leaf=20, 
                                 smoothing=10, hierarchy=None)
        enc = enc_auto.fit(df_train, Y_train[:,1]) #col 1 of Y is 0 if class=0 and 1 if class=1
        X_train_coded = enc.transform(df_train)
        X_test_coded = enc.transform(df_test)
        
        domain = matrixDomain(X_train_coded, ['race', 'c_charge_degree', 'age', 
                                              'decile_score', 'priors_count', 
                                              'days_b_screening_arrest'], 
                              ['sex', 'age_cat', 'score_text'])
        
        fuzzy_train = fuzzifyDataFrame(X_train_coded, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(X_test_coded, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'violentRecidivism': #66
        data =  pd.read_csv(r"datasets/compas-scores-two-years-violent.csv", sep=",")
        #raw_data, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, 
        #            days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out
        data.pop('id')
        data.pop('name')
        data.pop('first')
        data.pop('last')
        data.pop('compas_screening_date')
        data.pop('dob')
        data.pop('c_jail_in')
        data.pop('c_jail_out')
        data.pop('c_case_number')
        data.pop('c_offense_date')
        data.pop('c_arrest_date')
        data.pop('r_case_number')
        data.pop('r_offense_date')
        data.pop('r_jail_in')
        data.pop('r_jail_out')
        data.pop('r_charge_degree')
        data.pop('r_days_from_arrest')
        data.pop('r_charge_desc')
        data.pop('violent_recid')
        data.pop('vr_case_number')
        data.pop('vr_charge_degree')
        data.pop('vr_offense_date')
        data.pop('vr_charge_desc')
        data.pop('type_of_assessment')
        data.pop('screening_date')
        data.pop('v_type_of_assessment')
        data.pop('v_screening_date')
        data.pop('in_custody')
        data.pop('out_custody')
        data.pop('is_violent_recid')
        data.pop('event')
        data.pop('two_year_recid.1')
        data.pop('start')
        data.pop('priors_count.1')
        data.pop('juv_fel_count')
        data.pop('end')
        data.pop('decile_score')
        data.pop('decile_score.1')
        data.pop('c_days_from_compas')
        data.pop('juv_other_count')
        data.pop('juv_misd_count')
        data.pop('c_charge_desc')
        data.pop('score_text')
        data.pop('two_year_recid')
        
        # Filter rows where 'days_b_screening_arrest' is between -30 and 30
        data = data[(data['days_b_screening_arrest'] >= -30) & (data['days_b_screening_arrest'] <= 30)]
        data = data[(data['is_recid'] != -1)]
        data = data[(data['c_charge_degree'] != "O")]
        data = data[(data['v_score_text'] != 'N/A')]
        
        # Convert columns to numeric type
        data = data.apply(pd.to_numeric, errors='ignore')
        
        # Replace NaN values with corresponding column medians (numeric features)
        median_values = data.median()
        data.fillna(median_values, inplace=True)
        
        Ypd = data['is_recid'] 
        data = data.drop(['is_recid'], axis=1)
        
        outputsOneHot = pd.get_dummies(Ypd, columns=['is_recid'])
        Y = outputsOneHot.to_numpy()
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        df_train = pd.DataFrame(X_train, columns=columns)
        df_test = pd.DataFrame(X_test, columns=columns)

        enc_auto = TargetEncoder(verbose=0, 
                                 cols=['sex', 'age_cat', 'race',  'c_charge_degree',
                                        'v_score_text'], 
                                 drop_invariant=False, 
                                 return_df=True, handle_missing='value', 
                                 handle_unknown='value', min_samples_leaf=20, 
                                 smoothing=10, hierarchy=None)
        enc = enc_auto.fit(df_train, Y_train[:,1]) #col 1 of Y is 0 if class=0 and 1 if class=1
        X_train_coded = enc.transform(df_train)
        X_test_coded = enc.transform(df_test)
        
        domain = matrixDomain(X_train_coded, ['race', 'age', 
                                              'priors_count', 'days_b_screening_arrest', 
                                              'v_decile_score'], 
                              ['sex', 'age_cat', 'c_charge_degree', 'v_score_text'])
        
        fuzzy_train = fuzzifyDataFrame(X_train_coded, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(X_test_coded, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'spect': #66
        data_training =  pd.read_csv(r"datasets/SPECT.train", sep=",")
        data_test =  pd.read_csv(r"datasets/SPECT.test", sep=",")
        labels_training = data_training[['class']]
        labels_test = data_test[['class']]
        data_training.pop('class')
        data_test.pop('class')
        
        outputsOneHot = pd.get_dummies(labels_training, columns=['class'])
        Y_train = outputsOneHot.to_numpy()
        
        outputsOneHot = pd.get_dummies(labels_test, columns=['class'])
        Y_test = outputsOneHot.to_numpy()
        
        columns = data_training.columns
        
        domain = matrixDomain(data_training, [], list(columns)) #all data are categorical
        
        fuzzy_train = fuzzifyDataFrame(data_training, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(data_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'vehicle': #66
        data =  pd.read_csv(r"datasets/vehicle.csv", sep=",")
        labels = data[['class']]
        data.pop('class')
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        data_training = pd.DataFrame(X_train, columns=columns)
        data_test = pd.DataFrame(X_test, columns=columns)
        
        # Replace '?' with NaN for numerical operations
        data_training.replace('?', pd.NA, inplace=True)
        data_test.replace('?', pd.NA, inplace=True)
        
        # Convert columns to numeric type
        data_training = data_training.apply(pd.to_numeric, errors='ignore')
        data_test = data_test.apply(pd.to_numeric, errors='ignore')
        
        # Replace NaN values with corresponding column medians (numeric features)
        median_values = data_training.median()
        data_training.fillna(median_values, inplace=True)
        median_values = data_test.median()
        data_test.fillna(median_values, inplace=True)
        
        outputsOneHot = pd.get_dummies(Y_train)
        Y_train = outputsOneHot.to_numpy()
        
        outputsOneHot = pd.get_dummies(Y_test)
        Y_test = outputsOneHot.to_numpy()
        
        domain = matrixDomain(data_training, list(columns), []) #all data are numerical
        
        fuzzy_train = fuzzifyDataFrame(data_training, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(data_test, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'credit': #66
        data =  pd.read_csv(r"datasets/crx.data", sep=",")
        labels = data[['class']]
        data.pop('class')
        
        X = data.to_numpy()
        X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.25, random_state=RANDOM_SEED)
        
        columns = data.columns
        data_training = pd.DataFrame(X_train, columns=columns)
        data_test = pd.DataFrame(X_test, columns=columns)
        
        # Replace '?' with NaN for numerical operations
        data_training.replace('?', pd.NA, inplace=True)
        data_test.replace('?', pd.NA, inplace=True)
        
        # Convert columns to numeric type
        data_training = data_training.apply(pd.to_numeric, errors='ignore')
        data_test = data_test.apply(pd.to_numeric, errors='ignore')

        # Replace NaN values with corresponding column medians (numeric features)
        median_values = data_training.median()
        data_training.fillna(median_values, inplace=True)
        median_values = data_test.median()
        data_test.fillna(median_values, inplace=True)
        
        # Replace NaN values with corresponding column modes (categorical features)
        for column in data_training.columns:
            if data_training[column].dtype == 'object':  # Check if the column is categorical
                mode_value = data_training[column].mode()[0]
                data_training[column].fillna(mode_value, inplace=True)
                mode_value = data_test[column].mode()[0]
                data_test[column].fillna(mode_value, inplace=True)
        
        outputsOneHot = pd.get_dummies(Y_train)
        Y_train = outputsOneHot.to_numpy()
        
        outputsOneHot = pd.get_dummies(Y_test)
        Y_test = outputsOneHot.to_numpy()
        
        enc_auto = TargetEncoder(verbose=0, 
                                 cols=['A4', 'A5', 'A6', 'A7', 'A13'], 
                                 drop_invariant=False, 
                                 return_df=True, handle_missing='value', 
                                 handle_unknown='value', min_samples_leaf=20, 
                                 smoothing=10, hierarchy=None)
        enc = enc_auto.fit(data_training, Y_train[:,1]) #col 1 of Y is 0 if class=0 and 1 if class=1
        X_train_coded = enc.transform(data_training)
        X_test_coded = enc.transform(data_test)
        
        domain = matrixDomain(X_train_coded, ['A2', 'A3', 'A8', 'A11', 'A14', 
                                              'A15', 'A4', 'A5', 'A6', 'A7', 
                                              'A13'], 
                              ['A1', 'A9', 'A10', 'A12']) 
        
        fuzzy_train = fuzzifyDataFrame(X_train_coded, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(X_test_coded, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    if problem == 'horse': #66
        data_training =  pd.read_csv(r"datasets/horse-colic.data", sep=",")
        data_test =  pd.read_csv(r"datasets/horse-colic.test", sep=",")
        data_training = data_training[data_training['outcome'] != '?'] #we eliminate rows where the class is not identified
        data_test = data_test[data_test['outcome'] != '?']
        data_training = data_training.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)
        labels_training = data_training[['outcome']]
        labels_test = data_test[['outcome']]
        
        columns = list(data_training.columns)
        for col in [2, 22, 23, 24, 25, 26, 27]: #2 is the hospital number, 27 is of no significance since pathology data is not included or collected for these cases, and the others are possible class columns
            data_training.pop(columns[col])
            data_test.pop(columns[col])
        
        data_training.pop('nasogastric_reflux_PH') #too many values missing
        data_test.pop('nasogastric_reflux_PH')
        
        # Replace '?' with NaN for numerical operations
        data_training.replace('?', pd.NA, inplace=True)
        data_test.replace('?', pd.NA, inplace=True)
        
        # Convert columns to numeric type
        data_training = data_training.apply(pd.to_numeric, errors='ignore')
        data_test = data_test.apply(pd.to_numeric, errors='ignore')
        
        # Replace NaN values with corresponding column medians (numeric features)
        median_values = data_training.median()
        data_training.fillna(median_values, inplace=True)
        median_values = data_test.median()
        data_test.fillna(median_values, inplace=True)
        
        outputsOneHot = pd.get_dummies(labels_training)
        Y_train = outputsOneHot.to_numpy()
        
        outputsOneHot = pd.get_dummies(labels_test)
        Y_test = outputsOneHot.to_numpy()
        
        columns = list(data_training.columns)
        
        enc_auto = TargetEncoder(verbose=0, 
                                 cols=[columns[0], columns[8], columns[12], 
                                       columns[13], columns[18], 
                                       columns[5], columns[6], columns[7], 
                                       columns[9], columns[10], columns[11], 
                                       columns[14], columns[15]], 
                                 drop_invariant=False, 
                                 return_df=True, handle_missing='value', 
                                 handle_unknown='value', min_samples_leaf=20, 
                                 smoothing=10, hierarchy=None)
        enc = enc_auto.fit(data_training, Y_train[:,1]) #col 1 of Y is 0 if class=0 and 1 if class=1
        X_train_coded = enc.transform(data_training)
        X_test_coded = enc.transform(data_test)
        
        domain = matrixDomain(X_train_coded, 
                              ['Age', 'rectal_temperature', 'pulse', 
                               'temperature_of_extremities', 'peripheral_pulse',
                               'mucous_membranes', 'pain', 'peristalsis', 
                               'abdomen', 
                               'abdominal_distension', 'rectal_examination', 
                               'respiratory_rate', 'packed_cell_volume', 
                               'total_protein', 'abdomcentesis_total_protein'], 
                              ['surgery', 
                               'capillary_refill_time', 
                               'nasogastric_tube', 
                               'nasogastric_reflux', 
                               'abdominocentesis_appearance'])
        
        fuzzy_train = fuzzifyDataFrame(X_train_coded, fuzzy_sets, domain)
        fuzzy_test = fuzzifyDataFrame(X_test_coded, fuzzy_sets, domain)
        X_train = fuzzy_train.to_numpy()#.transpose()
        X_test = fuzzy_test.to_numpy()#.transpose()
        
    return X_train, Y_train, X_test, Y_test

def fitness_eval0(string_features, individual, points):
    """Fitness function used for plain tournament"""
    #points = [X, Y]
    X = points[0]
    y = points[1]
    
    exec(string_features)
    
    try:
        pred = np.array(eval(str(individual))).transpose()
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError):
        return np.NaN,
    assert np.isrealobj(pred)
    
    fitness = math.sqrt(np.mean(np.square(np.subtract(y, pred))))
    
    labels = np.argmax(y, axis=1)
    labels_pred = np.argmax(pred, axis=1)
    individual.mce = 1 - np.mean(np.equal(labels, labels_pred)) 
    
    individual.behaviour = list(labels_pred) #list of predicted outputs
    
    #Calculate the number of nodes
    expr = toolbox.individual()
    nodes, _, _ = gp.graph(expr) #nodes, edges, labels
    individual.nodes = len(nodes)

    return fitness,

def fitness_eval6(string_features, penalty, individual, points):
    """Fitness function used for plain tournament with penalised score (number 
    of nodes divided by the penalty factor).
    It considers we are minimising the fitness.
    """
    #points = [X, Y]
    X = points[0]
    y = points[1]
    
    exec(string_features)
    
    try:
        pred = np.array(eval(str(individual))).transpose()
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError):
        return np.NaN,
    assert np.isrealobj(pred)
    
    raw_fitness = math.sqrt(np.mean(np.square(np.subtract(y, pred))))
    
    labels = np.argmax(y, axis=1)
    labels_pred = np.argmax(pred, axis=1)
    individual.mce = 1 - np.mean(np.equal(labels, labels_pred)) 
    
    individual.behaviour = list(labels_pred) #list of predicted outputs
    
    #Calculate the number of nodes
    expr = toolbox.individual()
    nodes, _, _ = gp.graph(expr) #nodes, edges, labels
    individual.nodes = len(nodes)
    
    fitness = raw_fitness + individual.nodes / penalty
    
    return fitness,

def fitness_eval4(string_features, individual, points):
    """Fitness function used for Lexicase selection."""
    #points = [X, Y]
    X = points[0]
    y = points[1]
    
    exec(string_features)
    
    try:
#        func = toolbox.compile(expr=individual)
  #      if problem == 'shuttle':
        pred = np.array(eval(str(individual))).transpose()
        #else:
 #       pred = list(map(func, *(X[:, i] for i in range(X.shape[1]))))
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError):
        return np.NaN,
    assert np.isrealobj(pred)
    
    fitness = math.sqrt(np.mean(np.square(np.subtract(y, pred))))
    
    labels = np.argmax(y, axis=1)
    labels_pred = np.argmax(pred, axis=1)
    individual.mce = 1 - np.mean(np.equal(labels, labels_pred)) 
    
    individual.fitness_each_sample = np.mean(np.square(y - pred), axis=1)
    individual.fitness_each_sample = list(individual.fitness_each_sample)
    
    individual.behaviour = list(labels_pred) #list of predicted outputs
    
    #Calculate the number of nodes
    expr = toolbox.individual()
    nodes, _, _ = gp.graph(expr) #nodes, edges, labels
    individual.nodes = len(nodes)
    
    return fitness,

def fitness_eval8(string_features, individual, points):
    """Fitness function used for plain tournament"""
    #points = [X, Y]
    X = points[0]
    y = points[1]
    
    exec(string_features)
    
    try:
        pred = np.array(eval(str(individual))).transpose()
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError):
        return np.NaN,
    assert np.isrealobj(pred)
    
    fitness = math.sqrt(np.mean(np.square(np.subtract(y, pred))))
    
    labels = np.argmax(y, axis=1)
    labels_pred = np.argmax(pred, axis=1)
    individual.mce = 1 - np.mean(np.equal(labels, labels_pred)) 
    
    individual.fitness_each_sample = np.mean(np.square(y - pred), axis=1)
    individual.fitness_each_sample = list(individual.fitness_each_sample)
    
    individual.behaviour = list(labels_pred) #list of predicted outputs
    
    #Calculate the number of nodes
    expr = toolbox.individual()
    nodes, _, _ = gp.graph(expr) #nodes, edges, labels
    individual.nodes = len(nodes)

    return fitness,

def fitness_eval10(string_features, penalty, individual, points):
    """Fitness function used for plain tournament with penalised score (number 
    of nodes divided by the penalty factor).
    It considers we are minimising the fitness.
    """
    #points = [X, Y]
    X = points[0]
    y = points[1]
    
    exec(string_features)
    
    try:
        pred = np.array(eval(str(individual))).transpose()
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError):
        return np.NaN,
    assert np.isrealobj(pred)
    
    raw_fitness = math.sqrt(np.mean(np.square(np.subtract(y, pred))))
    
    labels = np.argmax(y, axis=1)
    labels_pred = np.argmax(pred, axis=1)
    individual.mce = 1 - np.mean(np.equal(labels, labels_pred)) 
    
    individual.fitness_each_sample = np.mean(np.square(y - pred), axis=1)
    individual.fitness_each_sample = list(individual.fitness_each_sample)
    
    individual.behaviour = list(labels_pred) #list of predicted outputs
    
    #Calculate the number of nodes
    expr = toolbox.individual()
    nodes, _, _ = gp.graph(expr) #nodes, edges, labels
    individual.nodes = len(nodes)
    
    fitness = raw_fitness + individual.nodes / penalty
    
    return fitness,

def WTA(a, b):
    return [a, b]

def WTA3(a, b, c):
    return [a, b, c]

def WTA7(a, b, c, d, e, f, g):
    return [a, b, c, d, e, f, g]

def WTA6(a, b, c, d, e, f):
    return [a, b, c, d, e, f]

def WTA11(a, b, c, d, e, f, g, h, i, j, l):
    return [a, b, c, d, e, f, g, h, i, j, l]


#creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)
#toolbox.register("populationCreator", grape.random_initialisation, creator.Individual) 

def mutUniform(individual, expr, pset):
    """
    Changing the original function from DEAP to avoid mutation the root node.
    """
    index = random.randrange(1, len(individual)) #setting the start to 1, we avoid choosing the node 0 with WTA 
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual,

def mutLeaves(individual, expr, pset):
    """
    Changing the original function from DEAP to avoid mutation the root node.
    """
    terminals = []
    for node in individual:
        if isinstance(node, gp.Terminal):
            terminals.append(node)
    leaf = random.choice(terminals)
    index = terminals.index(leaf)
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual,

def normalise_probs(input_array):
    _, c = input_array.shape
    row_sums = np.sum(input_array, axis=1, keepdims=True)
    probs_array = np.where(row_sums > 0, input_array / row_sums, 1 / c)
    return probs_array

MAX_DEPTH = 17    
POPULATION_SIZE = 500
GENERATIONS = 50
TOURNSIZE = 7
MIN_INIT_DEPTH = 2#3
MIN_MUT_DEPTH = 0
MAX_MUT_DEPTH = 3

MAX_INIT_DEPTH = 6
P_CROSSOVER = 0.8
P_MUTATION = 0.05

def extend(f):
    return f

if scenario % 2 == 0: #even scenarios (3 triangles)
    FUZZY_SETS = 3
elif scenario % 2 == 1: #odd scenarios (5 triangles)
    FUZZY_SETS = 5

def create_toolbox(problem, scenario, n_features, RANDOM_SEED):
    if problem == 'heartDisease': 
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'australian':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'segment':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA7, [float, float, float, float, float, float, float], list)
    elif problem == 'satellite':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA6, [float, float, float, float, float, float], list)
    elif problem == 'cover':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA7, [float, float, float, float, float, float, float], list)    
    elif problem == 'vowel':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA11, [float, float, float, float, float, float, float, float, float, float, float], list)
    elif problem == 'haberman':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'transfusion':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'lupus':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'pima':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'germanCredit':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'iris':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA3, [float, float, float], list)    
    elif problem == 'wine':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA3, [float, float, float], list)    
    elif problem == 'shuttle':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA7, [float, float, float, float, float, float, float], list)
    elif problem == 'adult':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'lawsuit':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'bankMarketing':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'recidivism':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'violentRecidivism':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'spect':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'vehicle':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA3, [float, float, float], list)
    elif problem == 'credit':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA, [float, float], list)
    elif problem == 'horse':
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, n_features), list, "IN")
        pset.addPrimitive(WTA3, [float, float, float], list)
        
    pset.addPrimitive(WA, [float, float, str], float)
    pset.addPrimitive(OWA, [float, float, str], float)
    pset.addPrimitive(minimum, [float, float], float)
    pset.addPrimitive(maximum, [float, float], float)
    pset.addPrimitive(dilator, [float], float)
    pset.addPrimitive(concentrator, [float], float)
    pset.addPrimitive(complement, [float], float)
    
    pset.addPrimitive(extend, [str], str)
    pset.addTerminal('0.1', str)
    pset.addTerminal('0.2', str)
    pset.addTerminal('0.3', str)
    pset.addTerminal('0.4', str)
    pset.addTerminal('0.5', str)
    pset.addTerminal('0.6', str)
    pset.addTerminal('0.7', str)
    pset.addTerminal('0.8', str)
    pset.addTerminal('0.9', str)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_INIT_DEPTH, max_=MAX_INIT_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)



    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genGrow, min_=MIN_MUT_DEPTH, max_=MAX_MUT_DEPTH)
    #toolbox.register("expr_mut", gp.genFull, min_=1, max_=1)
    toolbox.register("mutate", mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
    
    string_features = ''
    for i in range(n_features):
        string_features = string_features + "IN" + str(i) +" = X[:," + str(i) +"]; "

    if scenario == 0:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval0, string_features)
    if scenario == 1:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval0, string_features)
    if scenario == 14: #using mad
        toolbox.register("select", selEpsilonLexi2_nodesCountTies, alpha=1)
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 15: #using mad
        toolbox.register("select", selEpsilonLexi2_nodesCountTies, alpha=1)
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 16: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies, batch_size=2) #batch=2
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 17: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies, batch_size=2) #batch=2
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 18: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies, batch_size=3) #batch=3
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 19: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies, batch_size=3) #batch=3
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 20: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies, batch_size=5) #batch=5
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 21: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies, batch_size=5) #batch=5
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 22: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies, batch_size=10) #batch=10
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 23: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies, batch_size=10) #batch=10
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 24: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies, batch_size=20) #batch=20
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 25: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies, batch_size=20) #batch=20
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 26: #using mad
        toolbox.register("select", selEpsilonLexicaseCount) 
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 27: #using mad
        toolbox.register("select", selEpsilonLexicaseCount) 
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 28:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval6, string_features, 1000000) #parsimony 1,000,000
    if scenario == 29:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval6, string_features, 1000000) #parsimony 1,000,000 
    if scenario == 30:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval6, string_features, 100000) #parsimony 100,000
    if scenario == 31:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval6, string_features, 100000) #parsimony 100,000 
    if scenario == 32:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval6, string_features, 10000) #parsimony 10,000
    if scenario == 33:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval6, string_features, 10000) #parsimony 10,000 
    if scenario == 34:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval6, string_features, 1000) #parsimony 1,000
    if scenario == 35:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval6, string_features, 1000) #parsimony 1,000 
    if scenario == 36:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval6, string_features, 500) #parsimony 500
    if scenario == 37:
        toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval6, string_features, 500) #parsimony 500 
    if scenario == 90: #using 2*mad
        toolbox.register("select", selEpsilonLexi2_nodesCountTies, alpha=2)
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 92: #using 2.5*mad
        toolbox.register("select", selEpsilonLexi2_nodesCountTies, alpha=2.5)
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 94: #using 1.5*mad
        toolbox.register("select", selEpsilonLexi2_nodesCountTies, alpha=1.5)
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 96: #using 0.5*mad
        toolbox.register("select", selEpsilonLexi2_nodesCountTies, alpha=0.5)
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 98: 
        toolbox.register("select", selDynEpsilonLexi2_nodesCountTies)
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 100: #using mad
        toolbox.register("select", selDynBatchEpsilonLexi2_nodesCountTies, batch_size=2) #batch=2
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 102: #using mad
        toolbox.register("select", selDynBatchEpsilonLexi2_nodesCountTies, batch_size=3) #batch=3
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 104: #using mad
        toolbox.register("select", selDynBatchEpsilonLexi2_nodesCountTies, batch_size=5) #batch=5
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 106: #using mad
        toolbox.register("select", selDynBatchEpsilonLexi2_nodesCountTies, batch_size=10) #batch=10
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 108: #using mad
        toolbox.register("select", selDynBatchEpsilonLexi2_nodesCountTies, batch_size=20) #batch=2
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 110: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies_MADafter, batch_size=2) #batch=2
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 112: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies_MADafter, batch_size=3) #batch=3
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 114: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies_MADafter, batch_size=5) #batch=5
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 116: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies_MADafter, batch_size=10) #batch=10
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 118: #using mad
        toolbox.register("select", selBatchEpsilonLexi2_nodesCountTies_MADafter, batch_size=20) #batch=20
        toolbox.register("evaluate", fitness_eval4, string_features) #not using sqrt in fitness_cases
    if scenario == 120: 
        toolbox.register("select", selTournamentExtra, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval8, string_features) #not using sqrt in fitness_cases
    if scenario == 122: 
        toolbox.register("select", selTournamentExtra, tournsize=TOURNSIZE)
        toolbox.register("evaluate", fitness_eval10, string_features, 500) #parsimony 500 #not using sqrt in fitness_cases
    return toolbox
    
if scenario <= 1 or scenario == 28 or scenario == 29 or scenario == 30 or scenario == 31 or scenario == 32 or scenario == 33 or scenario == 34 or scenario == 35 or scenario == 36 or scenario == 37:
    REPORT_ITEMS = ['gen', 'nevals', 'best_train_fitness', 'best_ind_mce', 
                    'avg_mce',
                  'best_ind_depth', 'best_ind_nodes', 'avg_nodes', 'fitness_test', 
                  'accuracy_test', 'best_phenotype', 'behavioural_diversity']
elif scenario == 26 or scenario == 27: #epsilon-Lexicase
    REPORT_ITEMS = ['gen', 'nevals', 'best_train_fitness', 'best_ind_mce', 
                    'avg_mce',
                  'best_ind_depth', 'best_ind_nodes', 'avg_nodes', 'fitness_test', 
                  'accuracy_test', 'best_phenotype', 'behavioural_diversity', 
                  'lexicase_avg_steps']
elif scenario == 14 or scenario == 15 or scenario == 16 or scenario == 17 or scenario == 18 or scenario == 19 or scenario == 20 or scenario == 21 or scenario == 22 or scenario == 23 or scenario == 24 or scenario == 25 or scenario == 90 or scenario == 92 or scenario == 94 or scenario == 96 or scenario == 98 or scenario == 100 or scenario == 102 or scenario == 104 or scenario == 106 or scenario == 108 or scenario == 110 or scenario == 112 or scenario == 114 or scenario == 116 or scenario == 118: #epsilon-Lexi2
    REPORT_ITEMS = ['gen', 'nevals', 'best_train_fitness', 'best_ind_mce', 
                    'avg_mce',
                  'best_ind_depth', 'best_ind_nodes', 'avg_nodes', 'fitness_test', 
                  'accuracy_test', 'best_phenotype', 'behavioural_diversity', 
                  'lexicase_avg_steps', 'lexicase_avg_ties_chosen_ind',
                  'avg_zeros',
                  'avg_epsilon',
                  'variance',
                  'unique_selected',
                  'behavioural_diversity_fitness_cases']
elif scenario == 120 or scenario == 122:
    REPORT_ITEMS = ['gen', 'nevals', 'best_train_fitness', 'best_ind_mce', 
                    'avg_mce',
                  'best_ind_depth', 'best_ind_nodes', 'avg_nodes', 'fitness_test', 
                  'accuracy_test', 'best_phenotype', 'behavioural_diversity',
                  'avg_zeros',
                  'avg_epsilon',
                  'variance',
                  'unique_selected',
                  'behavioural_diversity_fitness_cases']

for i in range(N_RUNS):
    print()
    print()
    print("Run:", i + run)
    print()
    
    RANDOM_SEED = i + run
    
    np.random.seed(RANDOM_SEED)
    X_train, Y_train, X_test, Y_test = setDataSet(problem, RANDOM_SEED, FUZZY_SETS) #We set up this inside the loop for the case in which the data is defined randomly
    
    n_features = X_train.shape[1]
    print(n_features)
    toolbox = create_toolbox(problem, scenario, n_features, RANDOM_SEED)
        
    random.seed(RANDOM_SEED) 
    
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    population, logbook = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, 
                                                ngen=GENERATIONS, points_train=[X_train, Y_train],
                                                points_test=[X_test, Y_test], report_items=REPORT_ITEMS,
                                                halloffame=hof)
    
    gen = logbook.select("gen")
    nevals = logbook.select("nevals")
    best_train_fitness = logbook.select("best_train_fitness")
    best_ind_mce = logbook.select("best_ind_mce")
    avg_mce = logbook.select("avg_mce")
    best_ind_depth = logbook.select("best_ind_depth")
    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")
    fitness_test = logbook.select("fitness_test")
    
    behavioural_diversity = logbook.select("behavioural_diversity")
    
    print("fitness test = ", fitness_test[-1])
    
    best_phenotype = [float('nan')] * GENERATIONS
    best_phenotype.append(hof.items[0])
    
    accuracy_test = [float('nan')] * GENERATIONS
    accuracy_test.append(1 - hof.items[0].mce)   
                      
    import csv
    r = RANDOM_SEED
    
    header = REPORT_ITEMS
    address = r"./results/GP/" + problem + "/" + str(scenario) + "/"
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(address)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(address)
    
    if scenario <= 1 or scenario == 28 or scenario == 29 or scenario == 30 or scenario == 31 or scenario == 32 or scenario == 33 or scenario == 34 or scenario == 35 or scenario == 36 or scenario == 37:
        with open(address + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(header)
            for value in range(len(gen)):
                writer.writerow([gen[value], 
                                 nevals[value], 
                                 best_train_fitness[value], 
                                 best_ind_mce[value], 
                                 avg_mce[value],
                                 best_ind_depth[value], 
                                 best_ind_nodes[value], 
                                 avg_nodes[value], 
                                 fitness_test[value],
                                 accuracy_test[value],
                                 best_phenotype[value],
                                 behavioural_diversity[value]
                                 ])
    elif scenario == 14 or scenario == 15 or scenario == 16 or scenario == 17 or scenario == 18 or scenario == 19 or scenario == 20 or scenario == 21 or scenario == 22 or scenario == 23 or scenario == 24 or scenario == 25 or scenario == 90 or scenario == 92 or scenario == 94 or scenario == 96 or scenario == 98 or scenario == 100 or scenario == 102 or scenario == 104 or scenario == 106 or scenario == 108 or scenario == 110 or scenario == 112 or scenario == 114 or scenario == 116 or scenario == 118: #epsilon-Lexi2
        lexicase_avg_steps = logbook.select("lexicase_avg_steps")
        lexicase_avg_ties_chosen_ind = logbook.select("lexicase_avg_ties_chosen_ind")
        avg_zeros = logbook.select("avg_zeros")
        avg_epsilon = logbook.select("avg_epsilon")
        variance = logbook.select("variance")
        unique_selected = logbook.select("unique_selected")
        behavioural_diversity_fitness_cases = logbook.select("behavioural_diversity_fitness_cases")
        with open(address + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(header)
            for value in range(len(gen)):
                writer.writerow([gen[value], 
                                 nevals[value], 
                                 best_train_fitness[value], 
                                 best_ind_mce[value], 
                                 avg_mce[value],
                                 best_ind_depth[value], 
                                 best_ind_nodes[value], 
                                 avg_nodes[value], 
                                 fitness_test[value],
                                 accuracy_test[value],
                                 best_phenotype[value],
                                 behavioural_diversity[value],
                                 lexicase_avg_steps[value],
                                 lexicase_avg_ties_chosen_ind[value],
                                 avg_zeros[value],
                                 avg_epsilon[value],
                                 variance[value],
                                 unique_selected[value],
                                 behavioural_diversity_fitness_cases[value]
                                 ])
    elif scenario == 26 or scenario == 27: #epsilon-Lexicase
        lexicase_avg_steps = logbook.select("lexicase_avg_steps")
        with open(address + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(header)
            for value in range(len(gen)):
                writer.writerow([gen[value], 
                                 nevals[value], 
                                 best_train_fitness[value], 
                                 best_ind_mce[value], 
                                 avg_mce[value],
                                 best_ind_depth[value], 
                                 best_ind_nodes[value], 
                                 avg_nodes[value], 
                                 fitness_test[value],
                                 accuracy_test[value],
                                 best_phenotype[value],
                                 behavioural_diversity[value],
                                 lexicase_avg_steps[value]
                                 ])
    elif scenario == 120 or scenario == 122:
        behavioural_diversity_fitness_cases = logbook.select("behavioural_diversity_fitness_cases")
        avg_zeros = logbook.select("avg_zeros")
        avg_epsilon = logbook.select("avg_epsilon")
        variance = logbook.select("variance")
        unique_selected = logbook.select("unique_selected")
        with open(address + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(header)
            for value in range(len(gen)):
                writer.writerow([gen[value], 
                                 nevals[value], 
                                 best_train_fitness[value], 
                                 best_ind_mce[value], 
                                 avg_mce[value],
                                 best_ind_depth[value], 
                                 best_ind_nodes[value], 
                                 avg_nodes[value], 
                                 fitness_test[value],
                                 accuracy_test[value],
                                 best_phenotype[value],
                                 behavioural_diversity[value],
                                 avg_zeros[value],
                                 avg_epsilon[value],
                                 variance[value],
                                 unique_selected[value],
                                 behavioural_diversity_fitness_cases[value]
                                 ])        
    else:
        raise ValueError("The scenario was not registered.")
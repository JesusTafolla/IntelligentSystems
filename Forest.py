from audioop import cross
from cProfile import label
from ctypes.wintypes import SIZE
from enum import unique
from pickle import FALSE
from tkinter.tix import COLUMN
from turtle import color, shape
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from pprint import pprint
import graphviz
from sklearn.metrics import precision_recall_fscore_support

def checkPurity(data):
    #data=data.values
    label_colum = data[:,-1]
    unique_classes = np.unique(label_colum)
    if len(unique_classes) == 1:
        return True
    else:
        return False

def createLeaf(data, ml_task):
    #data=data.values
    label_column = data[:,-1]
    if ml_task == "regression":
        leaf = np.mean(label_column)
    #Classification Task
    else:
        unique_classes, count_unique_classes=np.unique(label_column,return_counts=True)
        index = count_unique_classes.argmax()
        leaf = unique_classes[index]
    return leaf

def potentialSplits(data, random_subspace):

    #data=data.values
    potential_splits = {}
    _, n_colums =data.shape
    column_indices = list(range(n_colums - 1))

    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)

    for column_index in column_indices:
        values = data[:,column_index]
        unique_val = np.unique(values)
        potential_splits[column_index] = unique_val
    return potential_splits

"""        typeOfFeature = FEATURE_TYPES[column_index]
        if typeOfFeature == "continuous":
            if len(unique_val) > 1:
                potential_splits[column_index] = []
                for index in range(len(unique_val)):
                    if index != 0:
                        curr_val = unique_val[index]
                        prev_val = unique_val[index-1]
                        potential_split = (curr_val + prev_val) / 2

                        potential_splits[column_index].append(potential_split)
        elif len(unique_val) > 1:
        potential_splits[column_index] = unique_val"""



def splitData(data,split_column, split_val):

    #data = data.values
    split_column_values = data[:, split_column]

    typeOfFeature = FEATURE_TYPES[split_column]
    if typeOfFeature == "continuous":
        data_below = data[split_column_values <= split_val]
        data_above = data[split_column_values > split_val]
    else:
        data_below = data[split_column_values == split_val]
        data_above = data[split_column_values != split_val]

    return data_below, data_above

def calculateMSE(data):
    actual_val = data[:,-1]
    if len(actual_val) == 0: #Empty Data
        mse = 0
    else:
        prediction = np.mean(actual_val)
        mse = np.mean((actual_val - prediction)**2)
    return mse

def calculateEntropy(data):

    #data=data.values
    label_colum = data[:, -1]
    _, counts= np.unique(label_colum,return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy

def calculateOverallMetric(data_below, data_above, metric_func):

    n_data_points = len(data_below) + len(data_above)
    
    p_data_bellow = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points

    overalMetric = (p_data_bellow * metric_func(data_below) + p_data_above * metric_func(data_above))

    return overalMetric

def determineBestSplit(data,potential_splits, ml_task):

    firstIteration = True
    for column_index in potential_splits:
        #print(COLUMN_HEADERS[column_index],"-", len(np.unique(data[:,column_index])))
        for value in potential_splits[column_index]:
            data_below,data_above = splitData(data, split_column=column_index, split_val=value)

            if ml_task == "regression":
                current_overal_metric = calculateOverallMetric(data_below,data_above, metric_func=calculateMSE)
            #Classification
            else:
                current_overal_metric = calculateOverallMetric(data_below,data_above, metric_func=calculateEntropy) 

            if firstIteration or current_overal_metric <= bestOveralMetric:
                firstIteration = False

                bestOveralMetric = current_overal_metric
                bestSplitCol = column_index
                bestSplitVal = value

    return bestSplitCol, bestSplitVal

def data_analysis():
    #columns = ['Gender','Satisfaction','Business Travel','Department','EducationField','Salary','Home-Office','label']
    columns = ['SnoringRate', 'RespirationRate', 'BodyTemperature', 'LimbMovement', 'BloodOxygen', 'EyeMovement', 'SleepingHours', 'HeartRate', 'label']
    df = pd.read_csv('StressLevel.csv',names = columns)
    #df.replace({'label':{1:'yes', 0:'no'}}, inplace=True)
    #columns = ["age","sex","bmi","children","smoker","region","charges"]
    #df = pd.read_csv('insurance.csv',names = columns)
    #df.replace({'Gender':{'Male':1, 'Female':0}}, inplace=True)
    #df.replace({'Satisfaction':{'High':3, 'Medium':2, 'Low':1}}, inplace=True) 
    #df.replace({'BusinessTravel':{'Frequent':3, 'Rare':2, 'No':1}}, inplace=True)
    #df.replace({'Department':{'R&D':1, 'Sales':0}}, inplace=True) 
    #df.replace({'EducationField':{'Engineering':3, 'Medical':2, 'Technical Degree':1, 'Other':0}}, inplace=True)
    #df.replace({'Salary':{'High':3, 'Medium':2, 'Low':1}}, inplace=True) 
    #df.replace({'Home-Office':{'Far':1, 'Near':0}}, inplace=True)
    #df.replace({'label':{'Yes':3, 'No':0}}, inplace=True)
    #print(df.shape)
    print(df)
    return df

def bootstrapping(train_df, n_bootsrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size= n_bootsrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]

    return df_bootstrapped

def train_test_split(df, test_size):
    #Check indexes to make a random test and train set
    ##indexes = df.index.tolist()
    ##test_indexes = random.sample(population=indexes,k=df.shape[0]/2)

    if isinstance(test_size,float):
        test_size = round(test_size * len(df))

    indexes = df.index.tolist()
    test_indexes = random.sample(population=indexes,k=test_size)

    test_df = df.loc[test_indexes]
    train_df = df.drop(test_indexes)
    return train_df, test_df

def predictExample(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    #Ask Question
    if comparison_operator =="<=":  
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    #Base Case
    if not isinstance(answer,dict):
        return answer

    #Recurrent Part
    else:
        residualTree=answer
        return predictExample(example, residualTree)

def calculareRSquared(df, tree):
    labels = df.label
    mean = labels.mean()
    predictions = df.apply(predictExample, args=(tree,), axis=1)

    ss_res = sum((labels - predictions) ** 2)
    ss_tot = sum((labels - mean) ** 2)
    rSquared = 1 - ss_res / ss_tot
    return rSquared

def createPlot(df, tree, title):
    predictions = df.apply(predictExample, args=(tree,), axis=1)
    actual = df.label
    plot_df = pd.DataFrame({"actual": actual, "predictions": predictions})  
    plot_df.plot(title=title)
    
    return

def calculateAccuracy(df, tree):

    df["Prediction"] = df.apply(predictExample, axis=1, args=(tree,))
    df["Correct"] = df.Prediction == df.label

    accuracy = df.Correct.mean()

    return accuracy

def calcAcc(predictions, labels):
    predictions_correct = predictions == labels
    accuracy = predictions_correct.mean()
    
    return accuracy

def determineFeatureType(df):

    feature_types = []
    n_unique_values_threshold = 15

    for column in df.columns:
        unique_values = df[column].unique()
        example_value = unique_values[0]

        if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_threshold):
            feature_types.append("categorical")
        else:
            feature_types.append("continuous")
    return feature_types


def decisionTreeAlg(df, ml_task, counter=0, min_samples=2, max_depth=10, random_subspace=None):

    #Data preparations
    if counter ==0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determineFeatureType(df)
        data=df.values
    else:
        data = df
    
    #Base case
    if(checkPurity(data)) or (len(data) < min_samples) or(counter == max_depth):
        leaf = createLeaf(data, ml_task)
        return leaf
    
    #Recursive part
    else:
        counter += 1

        #Helper functions
        potential_splits = potentialSplits(data, random_subspace)
        split_column, split_value = determineBestSplit(data,potential_splits,ml_task)
        data_below, data_above = splitData(data, split_column, split_value)

        #Check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            leaf = createLeaf(data, ml_task)
            return leaf

        #Instintiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        typeOfFeature = FEATURE_TYPES[split_column]
        if typeOfFeature == "continuous":
            question = "{} <= {}".format(feature_name,split_value)
        else:
            question = "{} = {}".format(feature_name,split_value)
        subtree = {question: []}

        #Find answers (recursion)
        yes_answer = decisionTreeAlg(data_below,ml_task,counter, min_samples, max_depth, random_subspace)
        no_answer = decisionTreeAlg(data_above,ml_task,counter, min_samples, max_depth, random_subspace)

        if yes_answer ==no_answer:
            subtree = yes_answer
        else:
            subtree[question].append(yes_answer)
            subtree[question].append(no_answer)

        return subtree

def decision_tree_predictions(test_df, tree):
    predictions = test_df.apply(predictExample, args=(tree,), axis=1)
    return predictions

def randomForestAlg(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []

    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decisionTreeAlg(df_bootstrapped, ml_task="classification", max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)

    return forest

def randomForestPred(test_df, forest):

    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df,tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    randomForestPredictions = df_predictions.mode(axis=1)[0]

    return randomForestPredictions

def main():
    employeeData = data_analysis()

    train_df, test_df=train_test_split(employeeData,test_size=0.7)

    forest = randomForestAlg(train_df, n_trees=4, n_bootstrap=len(train_df), n_features=999, dt_max_depth=4)


    predictions = randomForestPred(test_df, forest)

    accuracy = calcAcc(predictions, test_df.label)

    print("Accuracy of the model is: " + str(accuracy))

    print("Predictions")
    acc = calculateAccuracy(test_df,forest[0])
    print(test_df)

    for i in range (len(forest)):
        pprint(forest[i])


if __name__ == "__main__":
    main()
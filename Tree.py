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

def potentialSplits(data):

    #data=data.values
    potential_splits = {}
    _, n_colums =data.shape
    for column_index in range(n_colums -1):
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

def calcAcc(predictions, labels):
    predictions_correct = predictions == labels
    accuracy = predictions_correct.mean()
    
    return accuracy

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
    columns = ['SnoringRate', 'RespirationRate', 'BodyTemperature', 'LimbMovement', 'BloodOxygen', 'EyeMovement', 'SleepingHours', 'HeartRate', 'label']
    df = pd.read_csv('StressLevel.csv',names = columns)
    #df.replace({'label':{1:'yes', 0:'no'}}, inplace=True)
    #columns = ["age","sex","bmi","children","smoker","region","charges"]
    #df = pd.read_csv('insurance.csv',names = columns)
    #df.replace({'diabetes':{1:'yes', 0:'no'}}, inplace=True)
    #df.replace({'smoker':{'yes':0, 'no':1}}, inplace=True) 
    #df.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}}, inplace=True)
    #print(df.shape)
    return df

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

def decisionTreeAlg(df, ml_task, counter=0, min_samples=2, max_depth=10):

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
        potential_splits = potentialSplits(data)
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
        yes_answer = decisionTreeAlg(data_below,ml_task,counter, min_samples, max_depth)
        no_answer = decisionTreeAlg(data_above,ml_task,counter, min_samples, max_depth)

        if yes_answer ==no_answer:
            subtree = yes_answer
        else:
            subtree[question].append(yes_answer)
            subtree[question].append(no_answer)

        return subtree

def decision_tree_predictions(test_df, tree):
    predictions = test_df.apply(predictExample, args=(tree,), axis=1)
    return predictions

def main():
    
    #Function to preproces the data
    stressData=data_analysis()
    
    train_df, test_df=train_test_split(stressData,test_size=0.5)

    print(train_df)

    tree = decisionTreeAlg(train_df,ml_task="classification" , max_depth=4)
    pprint(tree)


    print(test_df)

    example = test_df.iloc[1]
    print(example)
    testEx = predictExample(example,tree)
    print("Stress Prediction: " + str(testEx))

    #allPredictions = decision_tree_predictions(test_df, tree)
    #print("Stress Predictions: " + str(allPredictions))

    acc = calculateAccuracy(test_df,tree)
    print("Accuracy: " + str(acc))
    print(test_df)

    #feature_types = determineFeatureType(insurance_data)
    #i=0
    #for column in insurance_data.columns:
    #    print(column, "-", feature_types[i])
    #    i += 1

    #print(train_df.dtypes)

    createPlot(train_df,tree,title="Trained Data")
    #plt.xlim(pd.to_numeric(0),pd.to_numeric(200))
    createPlot(test_df,tree,title="Test Data")
    #plt.xlim(pd.to_numeric(0),pd.to_numeric(250))
    #checkPurity(train_df[train_df.children<3])
    #classy = clasifyData(train_df[train_df.children < 3])
    #print(classy)
 
    #a = np.unique(label_column,return_counts=True)
    #print(a)

    #print(test_df.head())
    
    #potential_Splits = potentialSplits(train_df)
    #print(potential_Splits)
    #sns.lmplot(data=train_df,x="age",y="bmi",hue="diabetes", fit_reg=False)
    #sns.lmplot(data=train_df,x="glucose",y="bmi",hue="diabetes", fit_reg=False)
    #sns.lmplot(data=train_df,x="bp",y="bmi",hue="diabetes", fit_reg=False)
    #sns.lmplot(data=train_df,x="insulin",y="bmi",hue="diabetes", fit_reg=False)
    #sns.lmplot(data=train_df,x="age",y="insulin",hue="diabetes", fit_reg=False)
    #sns.lmplot(data=train_df,x="age",y="pedigree",hue="diabetes", fit_reg=False)
    #sns.lmplot(data=train_df,x="insulin",y="glucose",hue="diabetes", fit_reg=False)
    #plt.vlines(x=potential_Splits[5],ymin=1, ymax=7)

    #split_column = 3
    #split_val = 0.8

    #data_below,data_above = splitData(train_df,split_column,split_val)

    #plotting_df = pd.DataFrame(data_below, columns=insurance_data.columns)
    #sns.lmplot(data=plotting_df,x="age",y="bmi",hue="diabetes", fit_reg=True, size=6, aspect=1.5)

    #entropy=calculateEntropy(data_below)
    #print(entropy)
    #print("this is other\n")
    #print(calculateEntropy(data_above))
    #print(calculateOverallEntropy(data_below, data_above))
    #print(determineBestSplit(train_df,potential_Splits))

    #sns.lmplot(data=train_df,x="age",y="charges",hue="sex", fit_reg=True)
    #sns.lmplot(data=train_df,x="bmi",y="charges",hue="sex", fit_reg=True)
    
    #sns.lmplot(data=train_df,x="age",y="charges",hue="smoker", fit_reg=True)
    #sns.lmplot(data=train_df,x="bmi", y="charges",hue="smoker", fit_reg=True)
    
    #sns.lmplot(data=train_df,x="age",y="charges",hue="children", fit_reg=True)
    #sns.lmplot(data=train_df,x="bmi",y="charges",hue="children", fit_reg=True)

    plt.show()


if __name__ == "__main__":
    main()
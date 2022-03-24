from statistics import mode
import pandas as pd
from six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.metrics import classification_report

def main():
    columns = ['SnoringRate', 'RespirationRate', 'BodyTemperature', 'LimbMovement', 'BloodOxygen', 'EyeMovement', 'SleepingHours', 'HeartRate', 'label']
    # load dataset
    data = pd.read_csv('StressLevel.csv',names = columns)
    print(data.head())
    x = data.drop(['label'],axis=1)
    y=data.label

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    #Create a Decision Tree Classifier Object
    model = DecisionTreeClassifier()

    #Train the Tree
    model = model.fit(x_train,y_train)

    #Predict the response for the dataset
    y_pred = model.predict(x_test)

    #Accuracy using metrics
    acc1 = metrics.accuracy_score(y_test,y_pred)*100
    print("Accuracy of the model is " + str(acc1))

    #Classification Report
    report = classification_report(y_test, y_pred)
    print(report)

    #Plotting of the tree
    features = x.columns
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['0','1','2','3','4'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('stress_set_1.png')
    Image(graph.create_png())

    # Create Decision Tree classifer object
    model = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    # Train Decision Tree Classifer
    model = model.fit(x_train,y_train)

    #Predict the response for test dataset
    y_pred = model.predict(x_test)

    # Model Accuracy
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

    #Better Decision Tree Visualisation
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,filled=True, rounded=True,special_characters=True, feature_names = features,class_names=['0','1','2','3','4'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('stress_set_2.png')
    Image(graph.create_png())

if __name__ == "__main__":
    main()
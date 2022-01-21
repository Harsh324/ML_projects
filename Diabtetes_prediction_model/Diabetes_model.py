import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold

from sklearn.metrics import accuracy_score, precision_score, f1_score, log_loss


from joblib import dump, load

## Function to print the data overview
def Print_data_overview(Dataframe):
    print('Shape :  ',Dataframe.shape)
    print('Columns :  ',Dataframe.columns)
    print('Data-missing :  ',Dataframe.isnull().sum().values.sum())

## Function to Impute data if data is nan or 0
def Impute(Dataframe, Features):
    for feature in Features:
        Dataframe[feature].replace(0, np.nan, inplace = True)
        Dataframe[feature].replace(np.nan, Dataframe[feature].median(), inplace = True)


Kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

## Function to check the different models test ther accuracy
## and other variable and deciding the best model 

def Model_Selection(Model,Data_list, name):
    Model.fit(Data_list[0],Data_list[2])
    y_Pred = Model.predict(Data_list[1])
    Accuracy = np.mean(cross_val_score(Model, Data_list[0],Data_list[2], cv = Kf, scoring='accuracy'))
    f1Score = np.mean(cross_val_score(Model, Data_list[0],Data_list[2], cv = Kf, scoring='f1'))
    Precision = np.mean(cross_val_score(Model, Data_list[0],Data_list[2], cv = Kf, scoring='precision'))
    
    Logloss = log_loss(Data_list[3], y_Pred)
    
    frame = pd.DataFrame({'Model': [name], 'Accuracy': [Accuracy], 'f1Score': [f1Score], 'Precision': [Precision], 'Logloss': [Logloss]})
    return frame


## Taking Data in dataframe
Data = pd.read_csv('diabetes.csv')

## This line print the number of 0 and 1 values in the Outcome columns
#print(Data.Outcome.value_counts())


## This line plots the 0 and 1 values 
Data['Outcome'].value_counts().plot(kind = 'bar').set_title('Diabetes Outcome')


## Below lines gives correlation vlaues
Correlation = Data.corr()
Correlation['Outcome'].sort_values(ascending = False)

## Plotting the correlation of features with the label(outcome)
sns.pairplot(Data, hue='Outcome', plot_kws=dict(alpha=.3, edgecolor='none'), height=2, aspect=1.1)
sns.pairplot(Data.dropna(), vars = ['Glucose', 'Insulin','BMI','SkinThickness'], height= 2.0, diag_kind='kde', hue='Outcome')

## Dropping the outcome from the dataframe
Data_Features = Data.drop('Outcome', axis = 1)
Data_Labels = Data['Outcome']

Features_list = Data_Features.columns.tolist()
Impute(Data_Features, Features_list)


## Splitting the data in training and testing set
X_train, X_test, y_train, y_test = train_test_split(Data_Features, Data_Labels, test_size=.2, random_state=42, stratify=Data_Labels)


## Calling differne classifer models
model1 = LogisticRegression()
model2 = KNeighborsClassifier()
model3 = GaussianNB()
model4 = DecisionTreeClassifier()
model5 = RandomForestClassifier()


Data_list = [X_train, X_test, y_train, y_test]

## Concatinating all the model outputs in one dataframe
Model_data = pd.concat([
                       Model_Selection(model2, Data_list, 'KNeighborsClassifier'),
                       Model_Selection(model3, Data_list, 'GaussianNB'),
                       Model_Selection(model4, Data_list, 'DecisionTreeClassifier'),
                       Model_Selection(model5, Data_list, 'RandomForestClassifier')],axis = 0).reset_index()

## Print the data of models
#print(Model_data)

## Selecting the bste model since it comes out to be 
## RandomForestClassifier, hence we are using that.

Model = RandomForestClassifier()

## Fittin the model with trainin data and trainin labels
Model.fit(X_train, y_train)

## Now our model is ready 
## Lets take some data from the file and check

some_data = Data_Features.iloc[:5]
some_labels = Data_Labels.iloc[:5]

## Predciting the outcome
Arr = Model.predict(some_data)
Lst = list(some_labels)

## Check the predicted and correct outcome
#print(Arr)
#print(Lst)

## Now time to test the testing dataset

Predicted = Model.predict(X_test)
#print(Predicted)

#print(y_test)


## Saving the model

dump(Model, 'Diabetes_predictor.joblib')



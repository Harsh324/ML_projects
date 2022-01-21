import pandas as pd
import numpy as np

from joblib import dump, load

def Main():
    Preg = float(input("Enter the value of Pregnancies: "))
    Glu = float(input("Enter the value of Glucose: "))
    BP = float(input("Enter the value of Blood Pressure: "))
    Skin_Thick = float(input("Enter the value of SkinThickness: "))
    Ins = float(input("Enter the value of Insulin: "))
    BMI = float(input("Enter the value of BMI: "))
    DiabPed = float(input("Enter the value of DiabetesPedigreeFunction: "))
    Age = float(input("Enter the value of Age: "))
    
    
    from joblib import dump, load
    model = load('Diabetes_predictor.joblib')

    Val = model.predict(np.array([[Preg, Glu, BP, Skin_Thick, Ins, BMI, DiabPed, Age]]))
    if(Val[0] == 1):
        print('Yes', 'Model Predicted the diabetes')
    else:
        print('No', 'Model predicted no diabetes')


Main()

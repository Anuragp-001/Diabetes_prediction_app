import streamlit as st
import pandas as pd 
import numpy as np 
import pickle 
from sklearn.preprocessing import StandardScaler


#Connecting the code with mongodb to store the data 
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = "mongodb+srv://Anurag:Anurag1234@cluster0.bagh3cm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" 
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["diabetes"]
collection = db["diabetes_prediction"]


# Writing the function to load the model 
def load_model():
    with open("diabetes_prediction.pkl","rb") as file:
        model, scaler = pickle.load(file)
        return model, scaler
    
# Writing a function where user fill their choice of data and then get converted into our transformed data 
def preprocessing_input_data(data, scaler):
    # Create DataFrame with proper column names and values
    df = pd.DataFrame([data])
    df_transform = scaler.transform(df)
    return df_transform

# Writing a predict function 
def predict_data(data):
    lasso_model, scaler = load_model()
    process_data = preprocessing_input_data(data, scaler)
    prediction = lasso_model.predict(process_data)
    return prediction

# Creating the UI for app 
def main():
    st.title("Diabetes Prediction App")
    st.write("Enter your data to get a prediction for your diabetes risk")

    # Now we are going to create fields where user can fill the data
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=100, value=0)
    Glucose = st.number_input("Glucose", min_value=10.0, max_value=50.0, value=25.0)  # Fixed: was "sex" instead of "BMI"
    BloodPressure = st.number_input("Blood Pressure", min_value=40, max_value=300, value=120)
    SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=500, value=200)
    Insulin = st.number_input("Insulin", min_value=0, max_value=300, value=100)
    BMI = st.number_input("BMI", min_value=0, max_value=200, value=50)
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0, max_value=20, value=4)
    Age = st.number_input("Age", min_value=0, max_value=10, value=4)

    # Now we are going to create a button at the bottom which when clicked sends all the user input into the model
    if st.button("Predict Your Score"):
        # Fixed: Values and keys were swapped
        user_data = {
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "SkinThickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
            "Age": Age
        }
        prediction = predict_data(user_data)
        st.success(f"Your prediction result is {prediction}")
        user_data["prediction"] = float(prediction)                                                                                                                            #user data me prediction data ko add kro 

        # Imagine you have a big box of toys (that's called user_data), and each toy has a name tag on it. But sometimes the name tags are written in a special computer language that's hard to read.
        # This code is like having a helpful friend who goes through your toy box and rewrites all the name tags in regular English so everyone can understand them better.
        user_data = {key: int(value) if isinstance(value , np.integer) else float(value) if isinstance(value , np.floating) else value for key , value in user_data.items()}   

        collection.insert_one(user_data)                                                                                                                      
#To execute the above url function line 
if __name__ == "__main__" :
    main()
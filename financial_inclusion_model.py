import pandas as pd
import streamlit as st
import pypickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
data = pypickle.load("financial_inclusion_model2.pkl")

st.title("Financial Inclusion Prediction Model")
country = st.selectbox("Country ", ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
st.write("You selected ", country)
year = st.selectbox("Date ", [2018, 2016, 2017])
st.write("You selected ", year)
location_type = st.selectbox("location_type ", ['Rural', 'Urban'])
st.write("You selected ", location_type)
cellphone_access = st.selectbox("cellphone_access", ['Yes', 'No'])
st.write("You selected ", cellphone_access)
household_size = st.number_input("household_size")
st.write(household_size)
age_of_respondent = st.number_input("age_of_respondent")
st.write(age_of_respondent)
gender_of_respondent = st.selectbox("gender_of_respondent", ['Female', 'Male'])
st.write("You selected ", gender_of_respondent)
relationship_with_head = st.selectbox("relationship_with_head", (
    ['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent', 'Other non-relatives']))
st.write("You selected ", relationship_with_head)
marital_status = st.selectbox("marital_status",
    ['Married/Living together', 'Widowed', 'Single/Never Married', 'Divorced/Seperated', 'Dont know'])
st.write("You selected ", marital_status)
educational_level = st.selectbox("educational_level ",
    ['Secondary education', 'No formal education', 'Vocational/Specialised training', 'Primary education',
     'Tertiary education', 'Other/Dont know/RTA'])
st.write("You selected ", educational_level)
job_type = st.selectbox("job_level ",
    ['Self employed', 'Government Dependent', 'Formally employed Private', 'Informally employed',
     'Formally employed Government', 'Farming and Fishing', 'Remittance Dependent', 'Other Income',
     'Dont Know/Refuse to answer', 'No Income'])

#cols = ["country", "date", "location_type", "cellphone_access", "household_size", "age_of_respondent", "gender_of_respondent",
     #"relationship_with_head", "marital_status", "educational_level", "job_type"]


# Create a DataFrame with a single row and one-hot encode the categorical variables
x = pd.DataFrame({
    "country": [country],
    "year": [year],
    "location_type": [location_type],
    "cellphone_access": [cellphone_access],
    "household_size": [household_size],
    "age_of_respondent": [age_of_respondent],
    "gender_of_respondent": [gender_of_respondent],
    "relationship_with_head": [relationship_with_head],
    "marital_status": [marital_status],
    "education_level": [educational_level],
    "job_type": [job_type]
})

a = pd.get_dummies(x)

x = x.drop([country, year, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent,relationship_with_head, marital_status,educational_level,job_type], axis=1)

x = x.join(a)
if st.button("Predict"):
    result = data.predict(x)
    st.write(result)

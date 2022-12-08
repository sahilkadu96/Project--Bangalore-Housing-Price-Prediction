# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:12:54 2022

@author: Sahil
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title('Bangalore Housing Price Prediction')

X = pd.read_csv(r'C:\Users\Sahil\.spyder-py3\Bangalore_data_preprocessed.csv')
st.write(X[0:5])
pickled_model = pickle.load(open(r'C:\Users\Sahil\.spyder-py3\real_state_price_pred_model.pickle', 'rb'))

def predict_price(location,sqft,bath, balcony, bhk):    
    loc_index = np.where(X.columns==location)[0][0]    #finding the loc_index of location

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = balcony
    x[3] = bhk
    if loc_index >= 0:          
        x[loc_index] = 1                      #finding the location & making value 1
 
    return pickled_model.predict([x])[0] 

sqft = st.slider('Square_ft', X['total_sqft'].min(), X['total_sqft'].max(), 1000.0)
bath = st.slider('bath', X['bath'].min(), X['bath'].max(), 5.0, 1.0)
balcony = st.selectbox('balcony', [0.0, 1.0, 2.0, 3.0])
bhk = st.slider('bhk', X['bhk'].min(), X['bhk'].max(), 5, 1)
locations =  X.columns[4:]
#print(locations)
location = st.selectbox('locations', locations)

st.write('Predicted price', predict_price(location,sqft,bath, balcony, bhk))
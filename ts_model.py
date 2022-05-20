from ctypes import alignment
from email.policy import default
from tkinter import CENTER
from turtle import color
from matplotlib.pyplot import margins
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pylab as plt
import seaborn as sns
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

def app():
  st.subheader("Sales Forecasting App")
  ###### Import CSS style file to apply formatting ######
  with open ('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
######### Time series model to predict future sales ##########
    # Load data from data folder
    data = pd.read_csv('data//sales_data.csv')
    # Parse order date field
    data['ORDERDATE'] = pd.to_datetime(data['ORDERDATE'], format="%m/%d/%Y")
    # Group data by date and sales
    data_ts = data.groupby(by=['ORDERDATE']).sum()[['SALES']].reset_index()

    dt = go.Figure(data=[go.Table(
            columnwidth = [80,80],
            header=dict(values=list(data_ts.columns),
                        #fill_color='#272953',
                        align='center'),
            cells=dict(values=[data['ORDERDATE'], data['SALES']],
                    fill_color='#ededed',
                    align='left'))]
            )
    dt.update_layout(width=200, height=400, 
                         margin=dict(t=0, b=30, l=0, r=0),
                         )
    st.plotly_chart(dt, use_container_width=False)
    ###### Plot Original data ########
    fig1 = px.line(data_ts, x= data_ts['ORDERDATE'], y=data_ts['SALES'], title="Time series original | Monthly Sales")
    fig1.update_layout(height=300),
    st.plotly_chart(fig1, use_container_width=True)
    ###### split into train and test (70/30) ######
    train_set, test_set= np.split(data_ts, [int(.70 *len(data_ts))])
    train_set['type'] = 'train'
    test_set['type'] = 'test'
    train_test_data = pd.concat([train_set, test_set])
    fig2 = px.line(train_test_data, x= 'ORDERDATE', y='SALES', color='type')
    fig2.update_layout(height=300),
    st.plotly_chart(fig2, use_container_width=True)
    ####### Fit Arima Model on train data set ######
    model=sm.tsa.statespace.SARIMAX(train_set['SALES'],order=(1, 1, 1),seasonal_order=(0,1,0,12))
    results=model.fit()
    ####### Forecast the fitted model on the test set ######
    data_ts['forecast']=results.predict(start=90,end=103,dynamic=True)
    data_forecast = data_ts[['ORDERDATE','forecast']]
    data_ts = data_ts[['ORDERDATE','SALES']]
    data_forecast['type'] = 'forecast'
    data_forecast = data_forecast.rename(columns={'forecast':'value'})
    data_ts['type'] = 'original'
    data_ts = data_ts.rename(columns={'SALES':'value'})
    data_ts_forecast = pd.concat([data_forecast, data_ts])
    fig3 = px.line(data_ts_forecast, x= 'ORDERDATE', y='value', color='type')
    fig3.update_layout(height=300),
    st.plotly_chart(fig3)    

    
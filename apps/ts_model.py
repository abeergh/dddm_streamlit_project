import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pylab as plt
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

def app():
  st.subheader("Sales Forecasting App")
  ###### Import CSS style file to apply formatting ######
  with open ('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
######### Time series model to predict future sales ##########
    # Load data from data folder
  data = pd.read_csv('data//sales_data.csv', parse_dates=True)
  #  Group data by date and sales
  data_ts = data.groupby(by=['ORDERDATE']).sum()[['SALES']].reset_index()
  data_ts['ORDERDATE'] = pd.to_datetime(data_ts['ORDERDATE'])
  data_ts_monthly = data_ts.resample('M', on='ORDERDATE').sum().reset_index()
  frm, fig2 = st.columns([1,2])
  with frm:
    with st.form('form1', clear_on_submit=False):
       start_date = data_ts_monthly['ORDERDATE'].iloc[-1]
       start_date = st.text_input('Start Date:',start_date)
       horizon = st.number_input('Enter the number of months to forecast',min_value=1,step=1)
       forecast = st.form_submit_button("Forecast Sales") 
    ###### Checking if data is stationary #######
    #Perform Dickey-Fuller test:
    #print ('Results of Dickey-Fuller Test:')
    #dftest = adfuller(data_ts_monthly)
    #print("ADF stats", dftest[0])
    #print("ADF stats", dftest[1])
    ###### Differencing ########
    #def auto_correlation(df, prefix, lags):
    #plt.rcParams.update({'figure.figsize':(7,7), 'figure.dpi':120})
    # Define the plot grid
    #fig, axes = plt.subplots(3,2, sharex=False)
    # First Difference
    #axes[0, 0].plot(df)
    #axes[0, 0].set_title('Original' + prefix)
    #plot_acf(df, lags=lags, ax=axes[0, 1])
    # First Difference
    #df_first_diff = df.diff().dropna()
    #axes[1, 0].plot(df_first_diff)
    #axes[1, 0].set_title('First Order Difference' + prefix)
    #plot_acf(df_first_diff, lags=lags - 1, ax=axes[1, 1])
    # Second Difference
    #df_second_diff = df.diff().diff().dropna()
    #axes[2, 0].plot(df_second_diff)
    #axes[2, 0].set_title('Second Order Difference' + prefix)
    #plot_acf(df_second_diff, lags=lags - 2, ax=axes[2, 1])
    #plt.tight_layout()
    #plt.show()
    #auto_correlation(data_ts_monthly['SALES'], '', 10)
  ###### Split the data into train and test (70/30) ######
  train_set, test_set= np.split(data_ts_monthly, [int(.70 *len(data_ts_monthly))])
  ###### Fit ARIMA model ######
  model = ARIMA(train_set["SALES"], order=(2,1,3))
  model = model.fit()
  model.summary()
  ##### Model Prediction ######
  start = len(train_set)
  end = len(train_set) + len(test_set) -1 
  pred = model.predict(start=start, end=end, typ ='levels')
  ##### Fit the model in all the data ######
  model2 = ARIMA(data_ts_monthly['SALES'], order = (2,1,3))
  model2 = model2.fit()
  data_ts_monthly.tail()
  ##### Model Prediction ######
  index_future_dates = pd.date_range(start=start_date, periods=horizon+1, freq='M')
  pred= model2.predict(start= len(data_ts_monthly), end=len(data_ts_monthly)+int(horizon), typ='levels').rename('ARIMA Predictions')
  pred.index=index_future_dates
  #print(pred)
  predictions_df = pd.DataFrame(pred).reset_index()
  predictions_df['type'] = 'Predictions'
  predictions_df = predictions_df.rename(columns={'ARIMA Predictions':'SALES', 'index':'ORDERDATE'})
  data_ts_monthly['type'] = 'Actual data'
  data_all = pd.concat([data_ts_monthly, predictions_df])
  data_all = data_all.reset_index().rename(columns={'ORDERDATE':'DATE'})
  data_all['DATE'] = pd.to_datetime(data_all['DATE'], format ='%Y-%m')
  with fig2:
    fig2 = px.line(data_all, x= data_all['DATE'], y=data_all['SALES'], color='type',title="Sales Predictions")
    fig2.update_xaxes(showgrid=False, zeroline=False)
    fig2.update_yaxes(ticksuffix='  ', showgrid=False)
    fig2.update_layout(height=250, xaxis_title='', yaxis_title='', title_y=0.2,
                        title="<span style='font-size:16px; font-family:Times New Roman'>Sales Prediction in Million USD($)</span>",         
                        plot_bgcolor='#272953', paper_bgcolor='#272953',
                        title_font=dict(color='#FFFFFF'),
                        margin=dict(t=40, b=0, l=0, r=0), 
                        font=dict(size=13, color='#FFFFFF'),
                        legend=dict(title="", orientation="v", yanchor="bottom", xanchor="center", x=0.83, y=0.8,
                        bordercolor="#fff", borderwidth=0.5, font_size=13))
    st.plotly_chart(fig2)
  fig, tbl = st.columns([1,2])
  with tbl:
    predictions_table = predictions_df[['ORDERDATE','SALES']]
    predictions_table = predictions_table.rename(columns={'ORDERDATE':'DATE'})
    st.write(predictions_table)

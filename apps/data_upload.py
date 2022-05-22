import streamlit as st
import pandas as pd
import numpy as np
def app():
    st.markdown("### About Dataset:")
    st.markdown("###### Sample Sales Data, Order Info, Sales, Customer, Shipping, etc., Used for Segmentation, Customer Analytics, Clustering and More. Inspired for retail analytics.")
    st.markdown("## Data Upload")
    # Upload the dataset and save as csv
    st.markdown("#### Upload the sales data set in a csv or excel file for analysis.") 
    st.write("\n")
###### Upload data in csv or excel format ######
    # Code to read a single file 
    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
    global data
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file,sep=",", encoding='Latin-1')
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)
    ''' Load the data and save the columns with categories as a dataframe. 
    This section also allows changes in the numerical and categorical columns. '''
    if st.button("Load Data"):
        # Save raw data in csv file
        st.dataframe(data)
        data.to_csv('../data/sales_data.csv', index=False)

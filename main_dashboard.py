from email.policy import default
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from millify import millify # used to format numbers in a proper way to read
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2#function to convert to alpah2 country codes and continents
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static 

def app():
  ###### Import CSS style file to apply formatting ######
  with open ('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
  ###### Load the csv file from data folder where we saved the upload and save it in dataframe ######
  data = pd.read_csv('data//sales_data.csv')
  ###### Create side bar filters #####
  # Create country filter, that includes list of unique countries
  country = st.sidebar.multiselect("Country:",
                                          data['COUNTRY'].unique(),
                                          default = data['COUNTRY'].unique())
  # Create country filter, that includes list of unique years
  year = st.sidebar.multiselect("Year",
                                  data['YEAR_ID'].unique(),
                                  default = data['YEAR_ID'].unique())
  # Create country filter, that includes list of unique status
  status = st.sidebar.multiselect("Status:", 
                                  data['STATUS'].unique(),
                                  data['STATUS'].unique())
  # Filter the dashboard based on the user input/selected filters
  data_selection = data.query(
      "COUNTRY == @country & YEAR_ID == @year & STATUS == @status")
  ##### Define KPI's variables #####
    #caluclate the total sales amount
  total_sales =int(data_selection['SALES'].sum())
    #calculate the total number of orders
  total_orders =int(data_selection['ORDERNUMBER'].sum())
    #calculate the total number of products
  total_customers = int(data_selection['CUSTOMERNAME'].nunique())
    #create 4 columns to place the KPIs
  sales_by_product = data_selection.groupby(by=['PRODUCTLINE']).sum()[['SALES']].sort_values(by="SALES", ascending=False).reset_index()
  ##### KPIs cards to plot in the first row, split into 3 columns of the same size
  KPI1, KPI2, KPI3 = st.columns(3)
  KPI1.metric("Total Sales", f"$ {millify(total_sales)}","20")
  KPI2.metric("Total Orders", millify(total_orders),"25")
  KPI3.metric("Total Customers", millify(total_customers),"25")
  ##### Two side by side charts to plot on the second row after the KPIS, the first chart width is twice the second one
  fig1, fig2 = st.columns([2,1])
  ##### First chart plot(line chart)
  with fig1:
    import calendar 
    # Convert order date field into date time field
    data_selection['ORDERDATE'] = pd.to_datetime(data_selection['ORDERDATE'])
    # Extract month number from the order date field
    data_selection['month_num'] = data_selection['ORDERDATE'].dt.month
    # Extract month name from the order month number created earlier
    data_selection['OrderMonth'] = data_selection['month_num'].apply(lambda x: calendar.month_abbr[x])
    # Group the datafram by order month and month number, calculating the total sales and sorting in acending order by month number
    sales_by_week = data_selection.groupby(by=['OrderMonth','month_num']).sum()[['SALES']].sort_values(by="month_num", ascending=True).reset_index()
    # Line chart to plot sales by order month
    fig1 = px.line(x= sales_by_week['OrderMonth'], y= sales_by_week['SALES'])
    # Remove gridlines from both x axis and y axis
    fig1.update_xaxes(showgrid=False, zeroline=False)
    fig1.update_yaxes(ticksuffix='  ', showgrid=False)
    fig1.update_traces(line_color='#87CEEB')
    # Format the chart line color, width, heigth, margin, background etc..
    fig1.update_layout(width=450,height=250, xaxis_title='', yaxis_title='', title_y=0.2,
                      title="<span style='font-size:14px; font-family:Times New Roman'>Total Monthly Sales in Million USD($)</span>",         
                      plot_bgcolor='#272953', paper_bgcolor='#272953',
                      title_font=dict(size=14, color='#FFFFFF', family="Lato, sans-serif"),
                      margin=dict(t=40, b=0, l=0, r=10), 
                      font=dict(size=13, color='#FFFFFF'),
                      legend=dict(title="", orientation="v", yanchor="bottom", xanchor="center", x=0.83, y=0.8,
                      bordercolor="#fff", borderwidth=0.5, font_size=13))
    # Plot the first chart
    st.plotly_chart(fig1)
  ##### Second figure plot(gauge chart)
  with fig2:
    # Calculate the percentage of shipped products out of all products
    shipped_products = int((data_selection[data_selection['STATUS']=='Shipped'].count()["STATUS"]/data_selection['STATUS'].count())*100)
    # Create gauge plot, passing the shipped_product calculated field as value and specifying the target variable
    fig2 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = shipped_products,
        gauge= {'axis': {'range': [None, 100],'tickwidth': 1, 'tickcolor': "#01d1ff"},
                'bar': {'color': "#87CEEB"},
                'borderwidth':2
                },
        title = {'text': "% Shipped Products",'font': {'size': 13}}))
    # Format the gauge background color, width, heigth, margin, etc..
    fig2.update_layout(height=250, width=250, paper_bgcolor='#16113a', font={'color':"white"}, margin=dict(t=50, b=50, l=30, r=30))
    st.plotly_chart(fig2, use_container_width=True)
  ##### Two side by side charts to plot on the third row after the charts, the two charts have the same column size
  fig3, fig4 = st.columns(2)
  with fig3:
    # Group data by product on total sales, sorting by sales on descending order
    sales_by_product = data_selection.groupby(by=['PRODUCTLINE']).sum()[['SALES']].sort_values(by="SALES", ascending=False).reset_index()
    # Plot the scatter plot, product y axis, total sales on x axis
    fig3 = px.scatter(sales_by_product, y='PRODUCTLINE', x='SALES')
    # Add lines between products and sales value (circle marker) through a loop that pass to all products
    for i in range(0, len(sales_by_product)):
      fig3.add_shape(type='line',
                                y0 = sales_by_product['PRODUCTLINE'][i],
                                x0 = i,
                                x1 = sales_by_product['SALES'][i],
                                y1 = i,
                                line=dict(color='#656565', width = 2))
    # Remove gridlines from both x axis and y axis
    fig3.update_xaxes(showgrid=False, zeroline=False)
    fig3.update_yaxes(ticksuffix='  ', showgrid=False)
    fig3.update_traces(hovertemplate=None, marker=dict(line=dict(width=0)),marker_color='#87CEEB', marker_size=32)
    # Format plot, adjust width, height background color, margins etc..
    fig3.update_layout(width=400,height=250, xaxis_title='', yaxis_title='', title_y=0.2,
                      title="<span style='font-size:14px; font-family:Times New Roman'>Total Sales by Product</span>",
                      margin=dict(t=40, b=0, l=0, r=0),               
                      plot_bgcolor='#272953', paper_bgcolor='#272953',
                      title_font=dict(size=14, color='#FFFFFF', family="Lato, sans-serif"),
                      font=dict(size=13, color='#FFFFFF'))
    # Plot the chart
    st.plotly_chart(fig3, use_container_width=True)
  with fig4:
    # Group data by customer name and products, on total sales sorting sales values in descending order
    sales_by_customer = data_selection.groupby(by=['CUSTOMERNAME','PRODUCTLINE']).sum()[['SALES']].sort_values(by="SALES", ascending=False).reset_index()
    # Filter the dataframe on the top 10 customers
    sales_by_customer = sales_by_customer.nlargest(10,'SALES')
    # Create histogram with customer names by sales and color by product(legend)
    fig4 = px.histogram(sales_by_customer, x='CUSTOMERNAME', y='SALES', color='PRODUCTLINE',text_auto='.2s',
                      color_discrete_sequence=['#496595','#6082B6','#87CEEB'])
    fig4.update_yaxes(visible=False)
    fig4.update_traces(hovertemplate=None, marker=dict(line=dict(width=0)))
    # Format plot, adjust width, height background color, margins etc..
    fig4.update_layout(height=250, width=300,xaxis_title='', hovermode='x unified', 
                      title="<span style='font-size:14px; font-family:Times New Roman'>Top Companies by Sales</span>",
                      margin=dict(t=40, b=0, l=0, r=0),               
                      plot_bgcolor='#272953', paper_bgcolor='#272953',
                      title_font=dict(size=14, color='#FFFFFF', family="Lato, sans-serif"),
                      font=dict(size=10, color='#FFFFFF'),
                      legend=dict(title="", orientation="v", yanchor="bottom", xanchor="center", x=0.90, y=0.6,
                                  bordercolor="#fff", borderwidth=0, font_size=10)
    )
    # Plot the chart
    st.plotly_chart(fig4,use_container_width=True)     
  
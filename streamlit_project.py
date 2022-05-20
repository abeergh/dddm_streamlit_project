import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import matplotlib.pylab as plt
from millify import millify
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2#function to convert to alpah2 country codes and continents
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static 
###### Configure Page Layout ######
st.set_page_config(page_title="Sales Dashboard", 
                   page_icon=":bar_chart:",
                   layout="wide",
                   )
st.markdown( "<div style='background-color:#EBF5FB;  font-size:30px; text-align: center; color: #2E86C1 ; width: 100%'>Sales Dashboard</div>",
    unsafe_allow_html=True)
###### Upload data in csv format ######
st.title(":bar_chart: Sales Dashboard")
uploaded_file = st.sidebar.file_uploader("Choose a file")
st.sidebar.header("Please choose your filters:")
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file, sep=",", encoding='Latin-1')
###### Create side bar filters #####
  year = st.sidebar.multiselect(
        "Year:",
        options=data['YEAR_ID'].unique(),
        default=data['YEAR_ID'].unique()
)
  country = st.sidebar.multiselect(
        "Country:",
        options=data['COUNTRY'].unique(),
        default=data['COUNTRY'].unique()
)
  status = st.sidebar.multiselect(
        "Status:",
        options=data['STATUS'].unique(),
        default=data['STATUS'].unique()
)
  product = st.sidebar.multiselect(
        "Product:",
        options=data['PRODUCTLINE'].unique(),
        default=data['PRODUCTLINE'].unique()
)
  data_selection = data.query(
    "COUNTRY == @country & STATUS == @status & PRODUCTLINE == @product"
)
##### Define KPI's variables #####
  #caluclate the total sales amount
  total_sales =int(data_selection['SALES'].sum())
  #calculate the total number of orders
  total_orders =int(data_selection['ORDERNUMBER'].sum())
  #calculate the total number of products
  total_products = int(data_selection['PRODUCTCODE'].count())
  #calculate the percentage of shipped products out of all products
  shipped_products = int((data_selection[data_selection['STATUS']=='Shipped'].count()["STATUS"]/data_selection['STATUS'].count())*100)
  #create 4 columns to place the KPIs
  KPI1, KPI2, KPI3, KPI4 = st.columns(4)
  KPI1.metric("Total Sales", f"$ {millify(total_sales)}","100")
  KPI2.metric("Total Orders", millify(total_orders),"100")
  KPI3.metric("Total Products", millify(total_products),"100")
  import plotly.graph_objects as go
  fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = shipped_products,
    #domain = {'x': [0, 1], 'y': [0, 1]},
    gauge= {'axis': {'range': [None, 100]}},
    title = {'text': "Shipped Products"}))
  fig.update_layout(height=300, width=300)

  #col4.metric("Products Shipped", f"{shipped_products}%", "100")
  #st.markdown("-----")
  #st.dataframe(data_selection)
  
  
##### Display the following message if the user didn't upload any data ####
else:
  st.write("Please upload a data set")


##### Total Sales by products bar chart #####

#plot the bar chart
fig1, fig2= st.columns(2)
with fig1:
  blue_colors = ['rgb(0, 71, 171)', 'rgb(65, 105, 225)', 'rgb(96, 130, 182)',
                  'rgb(167, 199, 231)', 'rgb(182, 208, 226)','rgb(240, 255, 255)']
  #create new groubed data frame that includes each product line with the total sales amount
  order_by_product = data_selection.groupby(by=['PRODUCTLINE']).count()[['ORDERNUMBER']].reset_index().rename(columns={'orders':'ORDERNUMBER','product':'PRODUCTLINE'})
  #create bar chart using the sales_by_product data frame
  fig1_sales_by_prodcut = go.Figure(data=[go.Pie(labels=order_by_product['PRODUCTLINE'], 
                                                values=order_by_product['ORDERNUMBER'], 
                                                textinfo='label+percent',
                                                marker_colors= blue_colors,
                                                insidetextorientation='radial'
                              )])
  #fig1_sales_by_prodcut.update_layout(margin=dict(b=0),autosize=False,title = "Maternal mortality leading causes",width=500,height=400)
  fig1_sales_by_prodcut.update_layout(showlegend=False) 
  st.plotly_chart(fig1_sales_by_prodcut,use_container_width=True)
with fig2:
  ##### Total Sales by products bar chart #####
  #create new groubed data frame that includes each product line with the total sales amount
  sales_by_product = data_selection.groupby(by=['PRODUCTLINE']).sum()[['SALES']].sort_values(by="SALES")
  #create bar chart using the sales_by_product data frame
  fig2_sales_by_prodcut = px.bar(
      sales_by_product,
      x="SALES",
      y=sales_by_product.index,
      orientation="h",
      title="<b>Sales by Product Line",
      color_discrete_sequence = ["#008388"]
  )
  st.plotly_chart(fig2_sales_by_prodcut,use_container_width=True)

  #KPI4.metric("Shipped items",{
   # st.plotly_chart(fig)}
    
    #)
##### Total Sales by Country #####
#create new groubed data frame that includes each country with the total sales amount
sales_by_country = data_selection.groupby('COUNTRY',as_index=False)['SALES'].sum().reset_index().rename(columns={'sum':'SALES','country':'COUNTRY'})
longitude = []
latitude = []
# function to find the coordinate from country name
def findGeocode(country):
       
    # try and catch is used to overcome
    # the exception thrown by geolocator
    # using geocodertimedout  
    try:
          
        # Specify the user_agent as your
        # app name it should not be none
        geolocator = Nominatim(user_agent="your_app_name")
          
        return geolocator.geocode(country)
      
    except GeocoderTimedOut:
          
        return findGeocode(country)    
# each value from city column
# will be fetched and sent to
# function find_geocode   
for i in (sales_by_country["COUNTRY"]):
      
    if findGeocode(i) != None:
           
        loc = findGeocode(i)
          
        # coordinates returned from 
        # function is stored into
        # two separate list
        latitude.append(loc.latitude)
        longitude.append(loc.longitude)
       
    # if coordinate for a city not
    # found, insert "NaN" indicating 
    # missing value 
    else:
        latitude.append(np.nan)
        longitude.append(np.nan)
# now add this column to dataframe
sales_by_country["Longitude"] = longitude
sales_by_country["Latitude"] = latitude
import folium 
#map = folium.Map(location=[sales_by_country.Latitude.mean(), sales_by_country.Longitude.mean()], zoom_start=14, control_scale=True)
map = folium.Map(location=[20,0], tiles="OpenStreetMap", zoom_start=2)
#for index, location_info in sales_by_country.iterrows():
for i in range(0,len(sales_by_country)):
    folium.Marker(
      location=[sales_by_country.iloc[i]['Latitude'], sales_by_country.iloc[i]['Longitude']],
      popup=sales_by_country.iloc[i]['COUNTRY'],
      radius=sales_by_country.iloc[i]['SALES']/1000000,
      color='#69b3a2',
      fill_color='#69b3a2').add_to(map)
    
folium_static(map)
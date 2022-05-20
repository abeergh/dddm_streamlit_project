from email.policy import default
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from millify import millify # used to format numbers in a proper way to read
#from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2#function to convert to alpah2 country codes and continents
# Libraries to extract latitude and longitude from country name
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
# Import libraries for map: folium
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static 
def app():
  ###### Import CSS style file to apply formatting ######
  with open ('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
  ###### Load the csv file from data folder where we saved the upload and save it in dataframe ######
  data = pd.read_csv('data//sales_data.csv')
  # Group data by country and total sales
  ############################# The following code was used to extract latitude and longitude from country field in the data, 
  # but after running the script manytimes I got an error that there is limitation in iteration, 
  # so I exported the lat/long values extracted and save in csv that was used later. 
  # Thats why the code is commented
  #sales_by_country = data_selection.groupby('COUNTRY',as_index=False)['SALES'].sum().reset_index().rename(columns={'sum':'SALES','country':'COUNTRY'})
    # Create new columns, longitude and latitude 
  #longitude = []
  #latitude = []
  # function to find the coordinate from country name
  #def findGeocode(country): 
    # try and catch is used to overcome
    # the exception thrown by geolocator
    # using geocodertimedout  
  #  try:
        # Specify the user_agent as your
        # app name it should not be none
  #      geolocator = Nominatim(user_agent="your_app_name")
  #     return geolocator.geocode(country)
  #  except GeocoderTimedOut:     
  #      return findGeocode(country)    
# each value from city column
# will be fetched and sent to
# function find_geocode   
  #for i in (sales_by_country["COUNTRY"]):
  #  if findGeocode(i) != None:
  #     loc = findGeocode(i)
        # coordinates returned from 
        # function is stored into
        # two separate list
  #     latitude.append(loc.latitude)
  #      longitude.append(loc.longitude)
    # if coordinate for a city not
    # found, insert "NaN" indicating 
    # missing value 
  #  else:
  #      latitude.append(np.nan)
  #      longitude.append(np.nan)
    # now add this column to dataframe
  #sales_by_country["Longitude"] = longitude
  #sales_by_country["Latitude"] = latitude
  ##################################################################
  # Load the dataframe with lat and lon values saved in data folder
  sales_by_country = pd.read_csv('data//geoinfo.csv')
  #map = folium.Map(location=[sales_by_country.Latitude.mean(), sales_by_country.Longitude.mean()], zoom_start=14, control_scale=True)
  map = folium.Map(location=[20,0], tiles="Cartodbdark_matter", zoom_start=1)
        #for index, location_info in sales_by_country.iterrows():
  for i in range(0,len(sales_by_country)):
        folium.CircleMarker(
                location=[sales_by_country.iloc[i]['Latitude'], sales_by_country.iloc[i]['Longitude']],
                popup=sales_by_country.iloc[i]['COUNTRY'],
                radius=float(sales_by_country.iloc[i]['SALES'])/100000,
                icon=folium.DivIcon(html=f"""<div style="font-family: courier new; color: blue">{data.iloc[i]['COUNTRY']}</div>"""),
                color='#87CEEB',
                fill=True,
                fill_color='#87CEEB',
                tooltip= sales_by_country.iloc[i]['COUNTRY']).add_to(map)
  folium_static(map, width=600)

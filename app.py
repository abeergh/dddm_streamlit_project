import streamlit as st
from streamlit_option_menu import option_menu
# Custom imports 
from multiapp import MultiApp
from apps import data_upload, main_dashboard, country_sales, ts_model # import your pages here
import streamlit as st
st.set_page_config(layout="wide")
with open ('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Create an instance of the app 
app = MultiApp()
selected = option_menu(
      menu_title="Sales Dashboard",
      options=["Upload Data","Overview", "Country Sales", "Forecast Sales"],
      orientation="horizontal",
      styles= {
          "container": {"background-color":"#272953"},
          "nav-link": {"font-size":"15px","--hover-color":"#c2d5e8"},
          "icon": {"color":"white"}}
    )
# Add all your applications (pages) here
if selected == "Upload Data":
    app.add_app("Upload Data", data_upload.app)
if selected == "Overview":
    app.add_app("Main Sales Dashboard", main_dashboard.app)
if selected == "Country Sales":
    app.add_app("Country Sales", country_sales.app)
if selected == "Forecast Sales":
    app.add_app("Sales Prediction Model", ts_model.app)
#app.add_page("Data Prediction Model", model_prediction.app)
# The main app
app.run()
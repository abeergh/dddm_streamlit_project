##### Used to run multiple apps in one app as a form of pages
import streamlit as st
class MultiApp:
    
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.selectbox(
            '',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()
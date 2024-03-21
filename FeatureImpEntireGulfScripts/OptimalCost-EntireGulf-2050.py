import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import multiprocessing  # Import the multiprocessing library
import plotly.express as px


from collections import OrderedDict
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

random.seed(100000)

FeatureImp = pd.read_csv("FeatureImpEntireGulfFiles/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_126.csv") 

# SSPRCP 126

# Create a density map using the 'movefactor_s' feature importance
fig = px.density_mapbox(FeatureImp, lat='lat', lon='lon', z='movefactor_s', radius=10,
                        center=dict(lat=0, lon=180), zoom=0,
                        mapbox_style="open-street-map",
                        color_continuous_scale="Inferno",  # Adjust the color scale
                        opacity=1,
                        hover_data={'lat': False, 'lon': False})

fig.update_layout(title_text="FeatureImp OptimalCost EntireGulf 2050 126 Density")


# Save the figure as an image
fig.write_html("FeatureImpEntireGulfPlots/FeatureImpEntireGulfDensityPlots/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_126_Density.html") 


# Create a density map using the 'movefactor_s' feature importance
fig = px.scatter_mapbox(FeatureImp, lat='lat', lon='lon', color='movefactor_s', 
                        center=dict(lat=0, lon=180), zoom=0,
                        mapbox_style="open-street-map",
                        color_continuous_scale="Inferno",  # Adjust the color scale
                        opacity=0.5,
                        hover_data={'lat': False, 'lon': False})
fig.update_traces(marker_size=10)
fig.update_layout(title_text="FeatureImp OptimalCost EntireGulf 2050 126 Scatter")


# Save the figure as an image
fig.write_html("FeatureImpEntireGulfPlots/FeatureImpEntireGulfScatterPlots/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_126_Scatter.html") 

# SSPRCP 245

FeatureImp = pd.read_csv("FeatureImpEntireGulfFiles/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_245.csv") 

# Create a density map using the 'movefactor_s' feature importance
fig = px.density_mapbox(FeatureImp, lat='lat', lon='lon', z='movefactor_s', radius=10,
                        center=dict(lat=0, lon=180), zoom=0,
                        mapbox_style="open-street-map",
                        color_continuous_scale="Inferno",  # Adjust the color scale
                        opacity=1,
                        hover_data={'lat': False, 'lon': False})

fig.update_layout(title_text="FeatureImp OptimalCost EntireGulf 2050 245 Density")


# Save the figure as an image
fig.write_html("FeatureImpEntireGulfPlots/FeatureImpEntireGulfDensityPlots/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_245_Density.html") 


# Create a density map using the 'movefactor_s' feature importance
fig = px.scatter_mapbox(FeatureImp, lat='lat', lon='lon', color='movefactor_s', 
                        center=dict(lat=0, lon=180), zoom=0,
                        mapbox_style="open-street-map",
                        color_continuous_scale="Inferno",  # Adjust the color scale
                        opacity=0.5,
                        hover_data={'lat': False, 'lon': False})
fig.update_traces(marker_size=10)
fig.update_layout(title_text="FeatureImp OptimalCost EntireGulf 2050 245 Scatter")


# Save the figure as an image
fig.write_html("FeatureImpEntireGulfPlots/FeatureImpEntireGulfScatterPlots/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_245_Scatter.html") 

# SSPRCP 460

FeatureImp = pd.read_csv("FeatureImpEntireGulfFiles/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_460.csv") 

# Create a density map using the 'movefactor_s' feature importance
fig = px.density_mapbox(FeatureImp, lat='lat', lon='lon', z='movefactor_s', radius=10,
                        center=dict(lat=0, lon=180), zoom=0,
                        mapbox_style="open-street-map",
                        color_continuous_scale="Inferno",  # Adjust the color scale
                        opacity=1,
                        hover_data={'lat': False, 'lon': False})

fig.update_layout(title_text="FeatureImp OptimalCost EntireGulf 2050 460 Density")


# Save the figure as an image
fig.write_html("FeatureImpEntireGulfPlots/FeatureImpEntireGulfDensityPlots/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_460_Density.html") 


# Create a density map using the 'movefactor_s' feature importance
fig = px.scatter_mapbox(FeatureImp, lat='lat', lon='lon', color='movefactor_s', 
                        center=dict(lat=0, lon=180), zoom=0,
                        mapbox_style="open-street-map",
                        color_continuous_scale="Inferno",  # Adjust the color scale
                        opacity=0.5,
                        hover_data={'lat': False, 'lon': False})
fig.update_traces(marker_size=10)
fig.update_layout(title_text="FeatureImp OptimalCost EntireGulf 2050 460 Scatter")


# Save the figure as an image
fig.write_html("FeatureImpEntireGulfPlots/FeatureImpEntireGulfScatterPlots/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_460_Scatter.html") 

# SSPRCP 585

FeatureImp = pd.read_csv("FeatureImpEntireGulfFiles/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_585.csv") 

# Create a density map using the 'movefactor_s' feature importance
fig = px.density_mapbox(FeatureImp, lat='lat', lon='lon', z='movefactor_s', radius=10,
                        center=dict(lat=0, lon=180), zoom=0,
                        mapbox_style="open-street-map",
                        color_continuous_scale="Inferno",  # Adjust the color scale
                        opacity=1,
                        hover_data={'lat': False, 'lon': False})

fig.update_layout(title_text="FeatureImp OptimalCost EntireGulf 2050 585 Density")


# Save the figure as an image
fig.write_html("FeatureImpEntireGulfPlots/FeatureImpEntireGulfDensityPlots/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_585_Density.html") 


# Create a density map using the 'movefactor_s' feature importance
fig = px.scatter_mapbox(FeatureImp, lat='lat', lon='lon', color='movefactor_s', 
                        center=dict(lat=0, lon=180), zoom=0,
                        mapbox_style="open-street-map",
                        color_continuous_scale="Inferno",  # Adjust the color scale
                        opacity=0.5,
                        hover_data={'lat': False, 'lon': False})
fig.update_traces(marker_size=10)
fig.update_layout(title_text="FeatureImp OptimalCost EntireGulf 2050 585 Scatter")


# Save the figure as an image
fig.write_html("FeatureImpEntireGulfPlots/FeatureImpEntireGulfScatterPlots/OptimalCost2050/FeatureImp_OptimalCost_EntireGulf_2050_585_Scatter.html") 
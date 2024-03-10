import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import seaborn as sns
import numpy as np
from flask import Flask, render_template, request
import plotly.graph_objects as go
import json
from bokeh.embed import components
import plotly.graph_objects as go

df_test = pd.read_csv("data/wordbank_administration_data.csv") # testing result

# demographic population
df_demo = pd.read_csv("data/18district.csv") 
df_demo_filter = df_demo.groupby(['district'])['number'].aggregate('sum').reset_index()
# Read the JSON file for district boundary
with open('data/18district_boundary.json') as file:
    data = json.load(file)

# Flask constructor  
app = Flask(__name__) 

@app.route("/")
def regression_plot():
    p = figure(title="ABC", x_axis_label='age', y_axis_label='Word count')
    coefficients = np.polyfit(df_test['age'], df_test['comprehension'], 2)
    poly = np.poly1d(coefficients)

    # Generate the regression line points
    x_values = np.linspace(df_test['age'].min(), df_test['age'].max(), 48)
    y_values = poly(x_values)
    # Plot the regression line
    p.line(x_values, y_values, line_color='red', legend_label="Comprehension", line_width=2)

    coefficients = np.polyfit(df_test['age'], df_test['production'], 2)
    poly = np.poly1d(coefficients)

    # Generate the regression line points
    x_values = np.linspace(df_test['age'].min(), df_test['age'].max(), 48)
    y_values = poly(x_values)
    # Plot the regression line
    p.line(x_values, y_values, line_color='blue', legend_label="Production", line_width=2)
    # Generate the HTML components of the plot
    script1, div1 = components(p)


    df_demo_filter = df_demo.groupby(['district'])['number'].aggregate('sum').reset_index()
    fig = go.Figure(go.Choroplethmapbox(
        geojson=data,
        featureidkey='properties.地區',
        locations = df_demo_filter['district'],
        # locations=[feature['properties']['地區'] for feature in data['features']],
        z=df_demo['number'],  # Replace with your own data for the choropleth color scale
        colorscale="Viridis",  # Replace with the desired colorscale
        marker_line_width=0.5,
        marker_opacity=0.5
    ))

    fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=8.5,
    mapbox_center={"lat": 22.364, "lon": 114.15},
    )   
    # Generate the HTML components of the plot
    plotly_html = fig.to_html(full_html=False)


    # Return the components to the HTML template 
    return render_template( 
        template_name_or_list='charts.html', 
        script1=script1, 
        div1=div1,
        plotly_html=plotly_html,
    ) 

# Main Driver Function  
if __name__ == '__main__':
    app.run(debug=True)
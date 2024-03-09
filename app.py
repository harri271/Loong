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

df_test = pd.read_csv("data/wordbank_instrument_data.csv") # testing result

# demographic population
df_demo = pd.read_csv("data/18district.csv") 
df_demo_filter = df_demo.groupby(['district'])['number'].aggregate('sum').reset_index()
# Read the JSON file for district boundary
with open('data/18district_boundary.json') as file:
    data = json.load(file)

# Flask constructor  
app = Flask(__name__) 

@app.route('/input_form', methods=['GET', 'POST'])
def input_form():
    if request.method == 'POST':
        age = int(request.form['age'])
        comprehension = int(request.form['comprehension'])
        production = int(request.form['production'])

        X = df_test['age'].values.reshape(-1, 1)
        yc = df_test['comprehension'].values
        comprehension_model = LinearRegression(fit_intercept=False)
        yp = df_test['production'].values
        production_model = LinearRegression(fit_intercept=False)

        # Fit the model to the data
        comprehension_model.fit(X, yc)
        production_model.fit(X, yp)

        age_input = np.array(age).reshape(-1, 1)
        meanc = comprehension_model.predict(age_input)
        residuals = yc - meanc
        sdc = np.std(residuals)
        percentilec = (comprehension - meanc) / sdc

        meanp = production_model.predict(age_input)
        residuals = yp - meanp
        sdp = np.std(residuals)
        percentilep = (production - meanp) / sdp

        result = {
            'average_comprehension': round(meanc[0]),
            'comprehension_score': round(norm.cdf(percentilec[0]) * 100),
            'average_production': round(meanp[0]),
            'production_score': round(norm.cdf(percentilep[0]) * 100)
        }

        return render_template('result.html', result=result)
    else:
        return render_template('input_form.html')


## Regression prediction
@app.route("/regression_plot")
def regression_plot():
    p = figure(title="ABC", x_axis_label='age', y_axis_label='Word count')
    coefficients = np.polyfit(df_test['age'], df_test['comprehension'], 2)
    poly = np.poly1d(coefficients)

    # slope, intercept = np.polyfit(df_test['age'], df_test['comprehension'], 1)

    # Generate the regression line points
    x_values = np.linspace(df_test['age'].min(), df_test['age'].max(), 48)
    y_values = poly(x_values)
    # Plot the regression line
    p.line(x_values, y_values, line_color='red', legend_label="Comprehension", line_width=2)

    coefficients = np.polyfit(df_test['age'], df_test['production'], 2)
    poly = np.poly1d(coefficients)

    # slope, intercept = np.polyfit(df_test['age'], df_test['comprehension'], 1)

    # Generate the regression line points
    x_values = np.linspace(df_test['age'].min(), df_test['age'].max(), 48)
    y_values = poly(x_values)
    # Plot the regression line
    p.line(x_values, y_values, line_color='blue', legend_label="Production", line_width=2)
    return p

@app.route("/regression_plot")
def regression_plot():
    fig = go.Figure(go.Choroplethmapbox(
        geojson=data,
        featureidkey='properties.地區',
        locations = df_demo['district'],
        z=df_demo['number'],  # Replace with your own data for the choropleth color scale
        colorscale="Viridis",  # Replace with the desired colorscale
        marker_line_width=0.2,
        marker_opacity=0.5
    ))

    # Set the mapbox style and center the map on Hong Kong
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10,
        mapbox_center={"lat": 22.3964, "lon": 114.1095}
    )

    return fig


# Main Driver Function  
if __name__ == '__main__':
    app.run('localhost', port=9115)
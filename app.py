import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import seaborn as sns
import numpy as np




df_test = pd.read_csv("wordbank_instrument_data.csv")

X = df_test['age'].values.reshape(-1,1)
yc = df_test['comprehension'].values
comprehension_model = LinearRegression(fit_intercept=False)
yp = df_test['production'].values
production_model = LinearRegression(fit_intercept=False)

# Fit the model to the data
comprehension_model.fit(X, yc)
production_model.fit(X, yp)

age, comprehension, production = np.array(int(input("Age (month): "))).reshape(-1, 1), int(input("Comprehension: ")), int(input("Production: "))
meanc = comprehension_model.predict(age)
residuals = yc - meanc
sdc = np.std(residuals)
percentilec = (comprehension - meanc)/sdc

meanp = production_model.predict(age)
residuals = yc - meanp
sdc = np.std(residuals)
percentilep = (production - meanp)/sdc

print(f"Average comprehension is {round(meanc[0])}. Your score in your age group {round(norm.cdf(percentilec[0]) * 100)}% \n\
Average production is {round(meanp[0])}. Your score in your age group {round(norm.cdf(percentilep[0]) * 100)}%")

## Regression prediction
def regression():
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

# c = regression('comprehension')
p = regression()


# Show the plot
output_notebook()
show(p)
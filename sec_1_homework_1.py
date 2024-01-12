# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
# ---

# %%
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from rich import inspect
from icecream import ic
from src.print import print
# %% [markdown]
# 1.) Import Data from FRED

# %%
data = pd.read_csv(
    os.path.join('data', 'TaylorRuleData.csv'),
    index_col = 0
)

# %%
# Checking the last rows of the data.
data.tail()

# %%
# Converting the index to datetime.
data.index = pd.to_datetime(data.index)

# %%
# Checking amount of nan values and dropping them.
_ = ic(data.isna().sum())

# %%
data = data.dropna()

# %% [markdown]
# # 2.) Do Not Randomize, split your data into Train, Test Holdout

# %%
# Names of the features and target.
x_names = ['Unemployment', 'HousingStarts', 'Inflation']
y_names = ['FedFunds']

# %%
# Defining the percentage of the data to be used for training, 
# testing and holdout.
 
train_size = 0.6
test_size = 0.2
hold_size = 0.2

_ = ic(train_size + test_size + hold_size)
# %%
split_1 = np.ceil(data.shape[0] * train_size).astype(int)
split_2 = split_1 + np.ceil(data.shape[0] * test_size).astype(int)
data_in = data.iloc[:split_1]
data_out = data.iloc[split_1:split_2]
data_hold = data.iloc[split_2:]

# %%
X_in = data_in[x_names]
y_in = data_in[y_names]
X_out = data_out[x_names]
y_out = data_out[y_names]
X_hold = data_hold[x_names]
y_hold = data_hold[y_names]

# %%
# Add Constants
X_in = sm.add_constant(X_in)
X_out = sm.add_constant(X_out)
X_hold = sm.add_constant(X_hold)

# %% [markdown]
# # 3.) Build a model that regresses FF~Unemp, HousingStarts, Inflation

# %%
model1 = sm.OLS(y_in, X_in).fit()
print(model1.summary())

# %% [markdown]
# # 4.) Recreate the graph for your model
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (12,5))

# Observed Data
ax.plot(y_in, label = 'Training Data')
ax.plot(y_out, label = 'Testing Data')

# Predicted Data
ax.plot(model1.predict(X_in), label = 'Fitted Values')
ax.plot(model1.predict(X_out), label = 'Prediction - Testing Data')

# Testing Data Separation Line
ax.axvline(x = y_out.index[0], color = 'black', linestyle = '--')

# Legend, axes, title
plt.grid()
ax.legend()
ax.set_ylabel('Fed Funds (Percent)')
ax.set_xlabel('Time')
ax.set_title('Visualizing Model Accuracy')
fig.suptitle('Fed Funds Rate', fontsize = 18, weight = 'bold')

# %% [markdown]
# ## "All Models are wrong but some are useful" - 1976 George Box

# %% [markdown]
# # 5.) What are the in/out of sample MSEs

# %%
from sklearn.metrics import mean_squared_error

# %%
in_mse_1 = mean_squared_error(y_in, model1.fittedvalues)
out_mse_1 = mean_squared_error(y_out, model1.predict(X_out))

# %%
print('Insample MSE : ', round(in_mse_1, 2))
print('Outsample MSE : ', round(out_mse_1, 2))

# %% [markdown]
# # 6.) Using a for loop. Repeat 3,4,5 for polynomial degrees 1,2,3

# %%
from sklearn.preprocessing import PolynomialFeatures

max_degrees = 3

results = {}
for degree in range(1,max_degrees+1): 
    
    poly = PolynomialFeatures(degree = degree)
    
    X_in_poly = pd.DataFrame(
        poly.fit_transform(X_in)
    ).set_index(X_in.index)
    
    X_out_poly = pd.DataFrame(
        poly.fit_transform(X_out)
    ).set_index(X_out.index)

    temp_model = sm.OLS(y_in, X_in_poly).fit()
    
    results[degree] = {
        'model' : temp_model,
        'y_in_pred' : temp_model.fittedvalues,
        'y_out_pred' : temp_model.predict(X_out_poly),
        'in_mse' : mean_squared_error(y_in, temp_model.fittedvalues),
        'out_mse' : mean_squared_error(y_out, temp_model.predict(X_out_poly))
    }

fig, ax = plt.subplots(figsize = (12,5))

# Observed Data
ax.plot(y_in, label = 'Training Data')
ax.plot(y_out, label = 'Testing Data')

# Predicted Data
for degree in results.keys():
    ax.plot(
        results[degree]['y_in_pred'],
        label = f'Fitted Values - Degree {degree}'
    )
    # Extra color for the testing data
    color = ax.lines[-1].get_color()
    
    ax.plot(
        results[degree]['y_out_pred'],
        label = f'Prediction - Testing Data - Degree {degree}',
        color = color,
        linestyle = '--'
    )

# Testing Data Separation Line
ax.axvline(x = y_out.index[0], color = 'black', linestyle = '--')

# Legend, axes, title
plt.grid()
ax.set_ylabel('Fed Funds (Percent)')
ax.set_xlabel('Time')
ax.set_title('Visualizing Model Accuracy')
fig.suptitle('Fed Funds Rate', fontsize = 18, weight = 'bold')
# %%
# Plot in and out of sample MSEs
fig, ax = plt.subplots(figsize = (12,5))

ax.plot(
    [results[degree]['in_mse'] for degree in results.keys()],
    label = 'Insample MSE'
)
ax.plot(
    [results[degree]['out_mse'] for degree in results.keys()],
    label = 'Outsample MSE'
)

ax.set_xticks(range(0,max_degrees))
ax.set_xticklabels(range(1,max_degrees+1))
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('MSE')
ax.set_title('MSEs for Different Polynomial Degrees')
ax.legend()

# %% [markdown]
# # 7.) State your observations :

# %% [markdown]
"""
As the degree of the polynomial increases, the in-sample MSE decreases, while the out-of-sample MSE increases, indicating a clear sign of overfitting. In the plot of observed versus predicted values, it is clear that even though a higher-degree polynomial fits the training data better, higher-degree polynomials typically have highly volatile predictions.
"""

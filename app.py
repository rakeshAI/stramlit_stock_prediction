import streamlit as st
# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


st.title("AMAZON stock price")
stock = yf.Ticker("AMZN")
df_stock = stock.history(period="max")
#df_stock = yf.download("AMZN", start="2019-01-01", end="2020-10-30")


col1, col2 = st.beta_columns(2)
col1.write(df_stock)
col2.line_chart(df_stock['Close'])

# lets only concentrate on close price
df = df_stock[['Close']]
# create a variable to predict 'x' days out into the future
future_days = 15
# create a new column (target) shifted 'x' units/days up
df['Prediction'] = df[['Close']].shift(-future_days)
#st.write(df)

# create the future data set (X) and convert it to a numpy array and remove the last 'x' rows/days
X = np.array(df.drop(['Prediction'], 1))[:-future_days]
#st.write(X)

# create the target data set(y) and convert it to a numpy array and get all of the target values except the last 'x' rows/days
y = np.array(df['Prediction'])[:-future_days]
#st.write(y)

# split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# create the models
# create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)
# create the LinearRegression model
lr = LinearRegression().fit(x_train, y_train)

# get the last 'x' rows of the feature data set
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

# show the model tree prediction
tree_prediction = tree.predict(x_future)
#st.write(tree_prediction)

# show the model linear regression prediction
lr_prediction = lr.predict(x_future)
#st.write(lr_prediction)

st.info("Decision tree prediction")
predictions = tree_prediction
valid = df[X.shape[0]:]
valid['Prediction'] = predictions
#st.line_chart(valid)
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close price')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Prediction']])
plt.legend(['Orig', 'Val', 'Pred'])
st.pyplot(plt)

st.info("Linear regression prediction")
predictions = lr_prediction
valid = df[X.shape[0]:]
valid['Prediction'] = predictions
#st.line_chart(valid)
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close price')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Prediction']])
plt.legend(['Orig', 'Val', 'Pred'])
st.pyplot(plt)





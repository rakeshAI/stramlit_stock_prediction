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
import base64


Activity = ["NIFTY_50", "Single stock"]
activity_choice = st.sidebar.selectbox("Select Activity", Activity)
if activity_choice == "NIFTY_50":
    st.markdown(
        """
    This app retrieves the list of the **NSE** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
    * **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_National_Stock_Exchange_of_India).
    """
    )

    st.sidebar.header("User Input Features")

    # Web scraping of S&P 500 data
    #
    @st.cache
    def load_data():
        url = "https://en.wikipedia.org/wiki/NIFTY_50"
        html = pd.read_html(url, header=0)
        df = html[1]
        return df

    df = load_data()
    # st.write(df)

    sector = df.groupby("Sector")

    # Sidebar - Sector selection
    sorted_sector_unique = sorted(df["Sector"].unique())
    selected_sector = st.sidebar.multiselect(
        "Sector", sorted_sector_unique, sorted_sector_unique
    )

    # Filtering data
    df_selected_sector = df[(df["Sector"].isin(selected_sector))]

    st.header("Display Companies in Selected Sector")
    st.write(
        "Data Dimension: "
        + str(df_selected_sector.shape[0])
        + " rows and "
        + str(df_selected_sector.shape[1])
        + " columns."
    )
    st.dataframe(df_selected_sector)

    # download NIFTY_50 data
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="nifty50.csv">Download CSV File</a>'
        return href

    st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

    data = yf.download(
        tickers = list(df_selected_sector[:].Symbol),
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )
    #st.write(data["NTPC.NS"].Close)

    # Plot Closing Price of Query Symbol
    def price_plot(symbol):
        df = pd.DataFrame(data[symbol].Close)
        df['Date'] = df.index
        fig = plt.figure()
        #plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
        plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
        plt.xticks(rotation=45)
        plt.title(symbol, fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Closing Price', fontweight='bold')
        return st.pyplot(fig)

    #num_company = st.sidebar.slider('Number of Companies', 1, 5)
    #num_company = 50

    #if st.button('Show close price'):
    #    st.header('Stock Closing Price')
    #    for i in list(df_selected_sector.Symbol)[:-1]:
    #        price_plot(i)




    def buy_sell(signal):
        Buy = []
        Sell = []
        flag = -1

        for i in range(0, len(signal)):
            if signal['MACD'][i] > signal['Signal line'][i]:
                Sell.append(np.nan)
                if flag != 1:
                    Buy.append(signal['Close'][i])
                    flag = 1
                else:
                    Buy.append(np.nan)
            elif signal['MACD'][i] < signal['Signal line'][i]:
                Buy.append(np.nan)
                if flag != 0:
                    Sell.append(signal['Close'][i])
                    flag = 0
                else:
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        return (Buy, Sell)



    
    def MACD(symbol):
        st.header(symbol)
        df = pd.DataFrame(data[symbol].Close)
        df['Date'] = df.index
        #st.write(df)
        # calculate the MACD and signal line indicators
        # calculate the short term exponential moving average
        ShortEMA = df.Close.ewm(span=12, adjust=False).mean()
        #st.write(ShortEMA)
        # calculate the long term exponential moving average
        LongEMA = df.Close.ewm(span=26, adjust=False).mean()
        # calculate the MACD line
        MACD = ShortEMA - LongEMA
        # calculate the singnal line
        signal = MACD.ewm(span=9, adjust=False).mean()

        #fig = plt.figure()
        #plt.plot(df.index, MACD, label='MACD', color='red')
        #plt.plot(df.index, signal, label='Signal Line', color='blue')
        #plt.xticks(rotation = 45)
        #plt.legend(loc='upper left')

        df['MACD'] = MACD
        df['Signal line'] = signal

        a = buy_sell(df)
        df['Buy_Signal_Price'] = a[0]
        df['Sell_Signal_Price'] = a[1]
        st.write(df)
        
        fig = plt.figure()
        plt.scatter(df.index, df['Buy_Signal_Price'], color='green', label='Buy', marker='^', alpha=1)
        plt.scatter(df.index, df['Sell_Signal_Price'], color='red', label='Sell', marker='v', alpha=1)
        plt.plot(df['Close'], label='Close Price', alpha=0.50)
        plt.plot(df.index, MACD, label='MACD', color='red', alpha=0.35)
        plt.plot(df.index, signal, label='Signal Line', color='black', alpha=0.35)
        plt.title('CLose Price Buy & Sell Signals')
        plt.xticks(rotation = 45)
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend(loc = 'best')
        return st.pyplot(fig)     

    if st.button('Show MACD graph'):
        st.header('MACD stock closing price')
        for i in list(df_selected_sector.Symbol)[:-1]:
            MACD(i)


else:
    st.markdown(
        """
    This Section is just for a testing purpose. 
    We have used 2 machine learning technics for predicting the future stock price.
    """
    )

    Stocks = ["AMAZON", "BAJAJFINANCE"]
    choice = st.sidebar.selectbox("Select a stock", Stocks)

    if choice == "AMAZON":
        st.title("AMAZON stock price")
        stock = yf.Ticker("AMZN")
    elif choice == "BAJAJFINANCE":
        st.title("BAJAJ FINANCE stock price")
        stock = yf.Ticker("BAJFINANCE.NS")

    df_stock = stock.history(period="max")

    col1, col2 = st.beta_columns(2)
    col1.write(df_stock)
    col2.line_chart(df_stock["Close"])

    # lets only concentrate on close price
    df = df_stock[["Close"]]
    # create a variable to predict 'x' days out into the future
    future_days = 15
    # create a new column (target) shifted 'x' units/days up
    df["Prediction"] = df[["Close"]].shift(-future_days)
    # st.write(df)

    # create the future data set (X) and convert it to a numpy array and remove the last 'x' rows/days
    X = np.array(df.drop(["Prediction"], 1))[:-future_days]
    # st.write(X)

    # create the target data set(y) and convert it to a numpy array and get all of the target values except the last 'x' rows/days
    y = np.array(df["Prediction"])[:-future_days]
    # st.write(y)

    # split the data into 75% training and 25% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # create the models
    # create the decision tree regressor model
    tree = DecisionTreeRegressor().fit(x_train, y_train)
    # create the LinearRegression model
    lr = LinearRegression().fit(x_train, y_train)

    # get the last 'x' rows of the feature data set
    x_future = df.drop(["Prediction"], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)

    # show the model tree prediction
    tree_prediction = tree.predict(x_future)
    # st.write(tree_prediction)

    # show the model linear regression prediction
    lr_prediction = lr.predict(x_future)
    # st.write(lr_prediction)

    st.info("Decision tree prediction")
    predictions = tree_prediction
    valid = df[X.shape[0] :]
    valid["Prediction"] = predictions
    # st.line_chart(valid)
    plt.figure(figsize=(16, 8))
    plt.title("Model")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.plot(df["Close"])
    plt.plot(valid[["Close", "Prediction"]])
    plt.legend(["Orig", "Val", "Pred"])
    st.pyplot(plt)

    st.info("Linear regression prediction")
    predictions = lr_prediction
    valid = df[X.shape[0] :]
    valid["Prediction"] = predictions
    # st.line_chart(valid)
    plt.figure(figsize=(16, 8))
    plt.title("Model")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.plot(df["Close"])
    plt.plot(valid[["Close", "Prediction"]])
    plt.legend(["Orig", "Val", "Pred"])
    st.pyplot(plt)

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

st.sidebar.title("NIFTY_50")


@st.cache
def load_data():
    url = "https://en.wikipedia.org/wiki/NIFTY_50"
    html = pd.read_html(url, header=0)
    df = html[1]
    return df


df = load_data()


companies = st.beta_expander("Company selection with sectors", expanded=True)
with companies:

    col1, col2 = st.beta_columns(2)
    sector = df.groupby("Sector")

    # Sidebar - Sector selection
    sorted_sector_unique = sorted(df["Sector"].unique())
    # st.write(sorted_sector_unique)
    selected_sector = col1.multiselect(
        "Sector", sorted_sector_unique, sorted_sector_unique
    )

    # Filtering data
    df_selected_sector = df[(df["Sector"].isin(selected_sector))]
    # if st.checkbox("Display Companies in Selected Sector"):
    #    st.header("Display Companies in Selected Sector")
    #    st.dataframe(df_selected_sector)
    col1.write("Number of companies: " + str(df_selected_sector.shape[0]))

    col2.dataframe(df_selected_sector)


stock_history_expander = st.beta_expander(
    "NIFTY50 stock history from yahoo finance", expanded=True
)
with stock_history_expander:

    data = yf.download(
        tickers=list(df_selected_sector[:].Symbol),
        period="ytd",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        prepost=True,
        threads=True,
        proxy=None,
    )
    st.write(data)

# st.info("Algorithmic trading strategy")


def buy_sell(signal):
    Buy = []
    Sell = []
    flag = -1

    for i in range(0, len(signal)):
        if signal["MACD"][i] > signal["Signal line"][i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(signal["Close"][i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif signal["MACD"][i] < signal["Signal line"][i]:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(signal["Close"][i])
                flag = 0
            else:
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)
    return (Buy, Sell)


def MACD(symbol):

    # st.header(symbol)
    df = pd.DataFrame(data[symbol].Close)
    df["Date"] = df.index
    col1, col2 = st.beta_columns(2)
    # col1.write(df["Close"])
    # calculate the MACD and signal line indicators
    # calculate the short term exponential moving average
    ShortEMA = df.Close.ewm(span=12, adjust=False).mean()
    # st.write(ShortEMA)
    # calculate the long term exponential moving average
    LongEMA = df.Close.ewm(span=26, adjust=False).mean()
    # calculate the MACD line
    MACD = ShortEMA - LongEMA
    # calculate the singnal line
    signal = MACD.ewm(span=9, adjust=False).mean()

    df["MACD"] = MACD
    df["Signal line"] = signal

    a = buy_sell(df)
    df["Buy_Signal_Price"] = a[0]
    df["Sell_Signal_Price"] = a[1]
    return df


def MACD_plot(df):
    col1, col2 = st.beta_columns(2)
    col1.write(df)
    fig = plt.figure()
    plt.scatter(
        df.index,
        df["Buy_Signal_Price"],
        color="green",
        label="Buy",
        marker="^",
        alpha=1,
    )
    plt.scatter(
        df.index,
        df["Sell_Signal_Price"],
        color="red",
        label="Sell",
        marker="v",
        alpha=1,
    )
    plt.plot(df["Close"], label="Close Price", alpha=0.50)
    plt.plot(df["MACD"], label="MACD", color="red", alpha=0.35)
    plt.plot(df["Signal line"], label="Signal Line", color="black", alpha=0.35)
    plt.title("CLose Price Buy & Sell Signals")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend(loc="best")
    return col2.pyplot(fig)


algo_trad_strat_radio = st.sidebar.radio(
    "Select the Algorithmic trading strategy", ("MACD", "RSI", "Moving Average")
)
if algo_trad_strat_radio == "MACD":
    st.info("MACD of closing price")
    st.sidebar.write("Short EMA : 12, Long EMA : 26, Signal : 9")
    col1, col2 = st.beta_columns(2)
    col1.success("Buy")
    col2.error("Sell")
    BUY = []
    SELL = []
    for i in list(df_selected_sector.Symbol)[:-1]:
        df_MACD = MACD(i)
        buy = df_MACD["Buy_Signal_Price"][-1]
        sell = df_MACD["Sell_Signal_Price"][-1]
        if not (np.isnan(buy)):
            BUY.append(i)
        elif not (np.isnan(sell)):
            SELL.append(i)
    col1.write(BUY)
    col2.write(SELL)
    for num, i in enumerate(list(df_selected_sector.Symbol)[:-1]):
        text = str(num + 1) + "." + i
        df_MACD = MACD(i)
        buy = df_MACD["Buy_Signal_Price"][-1]
        sell = df_MACD["Sell_Signal_Price"][-1]
        expanded = False
        if not (np.isnan(buy)):
            expanded = True
        elif not (np.isnan(sell)):
            expanded = True

        company_expander = st.beta_expander(text, expanded=expanded)
        with company_expander:
            if not (np.isnan(buy)):
                st.success("BUY")
            elif not (np.isnan(sell)):
                st.error("SELL")
            MACD_plot(df_MACD)

elif algo_trad_strat_radio == "RSI":
    # st.info("RSI of closing price")
    st.write("Coming soon..")
elif algo_trad_strat_radio == "Moving Average":
    # st.info("Moving Average of closing price")
    st.write("Coming soon..")

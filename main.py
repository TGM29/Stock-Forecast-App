#Importing Libs
import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objects as go

DATE_START = '2017-01-01'
DATE_END = date.today().strftime('%Y-%m-%d')

st.title("Stock Analysis")

#Creating the sidebar
st.sidebar.header("Choose your Stock")

n_days = st.slider('Forecast days', 30, 365)

def get_data_stocks():
    path = 'stocks.csv'
    return pd.read_csv(path, delimiter=';')

df = get_data_stocks()


stock = df['snome']
name_stock_choosen = st.sidebar.selectbox('Choose Stock: ', stock)

#filtering the stock data with name
df_stock = df[df['snome'] == name_stock_choosen]
stock_choosen = df_stock.iloc[0]['sigla_acao']

st.write(df_stock.iloc[0]['sigla_acao'])

stock_choosen = stock_choosen + '.SA'

@st.cache
def get_values_online(sigla_acao):
    df = yf.download(sigla_acao,DATE_START,DATE_END)
    df.reset_index(inplace=True)
    return df

df_values = get_values_online(stock_choosen)

st.subheader("Values Table - " + name_stock_choosen)
st.write(df_values.tail(10))

#Price Graphs

st.subheader('Price Graph')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_values['Date'],
                         y = df_values['Close'],
                         name= 'Closing Price',
                         line_color = 'yellow'))

st.plotly_chart(fig)

#Forecast

df_training = df_values[['Date', 'Close']]

#Rename columns
df_training = df_training.rename(columns={'Date': 'ds', 'Close': 'y'})

model = Prophet()
model.fit(df_training)


future = model.make_future_dataframe(periods=n_days, freq= 'B')

forecast = model.predict(future)

st.subheader('Forecast ')
st.write(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(n_days))

graph1 = plot_plotly(model, forecast)
st.plotly_chart(graph1)

graph2 = plot_components_plotly(model, forecast)
st.plotly_chart(graph2)
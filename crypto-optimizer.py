import streamlit as st
from streamlit.elements.arrow import Data
import yfinance as yf
from datetime import date
from plotly import graph_objs as go
import crtools as crt
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame

def compute_cum_returns(df):
    returns = df.pct_change()
    cum_returns = (1 + returns).cumprod() - 1
    # cum_returns.fillna(0, inplace=True)
    return cum_returns

cryptos = ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "TUSD-USD", "BAT-USD", "MANA-USD"]
readable_cryptos = {"BTC-USD": "BTC", "ETH-USD": "ETH", "XRP-USD": "XRP", "LTC-USD": "LTC", "BCH-USD": "BCH", "TUSD-USD": "TUSD", "BAT-USD": "BAT", "MANA-USD": "MANA"}

st.title("CryptOptimizer")
selected_cryptos = st.multiselect("Selecciona las cryptos a incluir en el portafolio (por lo menos 2):", cryptos, format_func=lambda x: readable_cryptos.get(x))
start_date = st.date_input("Inicio:", value=date(2019, 12, 31))
end_date = st.date_input("Final:", value=date.today())

if len(selected_cryptos) > 1:
    crypto_data = yf.download(selected_cryptos, start=start_date, end=end_date)["Adj Close"]
    crypto_returns = crypto_data.pct_change()
    crypto_cum_returns = compute_cum_returns(crypto_data)
    crypto_cum_returns.rename(columns=readable_cryptos, inplace=True)
    st.subheader("Retorno Histórico Acumulado")
    st.line_chart(crypto_cum_returns)

    annual_returns = crt.annualize_rets(crypto_returns.fillna(0), 365)
    annual_returns.rename(index=readable_cryptos, inplace=True)
    st.subheader("Retorno Anualizado")
    st.bar_chart(annual_returns)

    annual_std = crt.annualize_vol(crypto_returns.fillna(0), 365)
    annual_std.rename(index=readable_cryptos, inplace=True)
    st.subheader("Volatilidad Anualizada")
    st.bar_chart(annual_std)

    st.subheader("Modelo de Covarianza")
    cov_models = ["Covarianza de Muestra", "Elton-Gruber", "Ledoit-Wolf"]
    cov_model = st.radio("Selecciona el modelo para la matriz de covarianza:", cov_models)

    if cov_model == "Covarianza de Muestra":
        cov_matrix = crt.sample_cov(crypto_returns)
    elif cov_model == "Elton-Gruber":
        cov_matrix = crt.cc_cov(crypto_returns)
    else:
        user_delta = st.number_input("Delta:", min_value=0.0, max_value=1.0, value=0.5)
        cov_matrix = crt.shrinkage_cov(crypto_returns, user_delta)
    
    fig, ax = plt.subplots()
    sns.heatmap(crt.cov_to_corr(crypto_returns, cov_matrix).rename(index=readable_cryptos, columns=readable_cryptos), ax=ax)
    st.subheader("Correlación entre Activos")
    st.write(fig)

    st.subheader("Portafolio Óptimo")
    optimization_types = ["Mínima Varianza", "Máximo Sharpe", "Risk Parity"]
    optimization_type = st.radio("Selecciona el tipo de optimización:", optimization_types)

    if optimization_type == "Mínima Varianza":
        portfolio = crt.gmv(cov_matrix)
    elif optimization_type == "Máximo Sharpe":
        portfolio = crt.msr(.03, annual_returns, cov_matrix)
    else:
        portfolio = crt.equal_risk_contributions(cov_matrix)
    
    portfolio = DataFrame(portfolio, index=annual_returns.index, columns=["Peso"])
    portfolio = portfolio.round(4)
    portfolio = portfolio.loc[portfolio["Peso"] > 0]
    fig, ax = plt.subplots()
    labels = portfolio.index
    plt.pie(x=portfolio["Peso"], autopct="%.2f%%", explode=[0.05]*len(portfolio), labels=labels, pctdistance=0.5)
    st.write(fig)
from typing import Tuple, List
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def var_histo(yield_series: pd.Series, trust_level: int) -> float:
    """
    Computer Historical VaR

    :param pd.Series yield_series: Serie of the yields of portfolio
    :param int trust_level: alpha % trust level
    :return: VaR
    :rtype: float
    """
    return np.percentile(yield_series, 100 - trust_level)


def cvar(yield_series: pd.Series, var: float) -> float:
    """
    Computes CVaR according to provided VaR value.

    :param pd.Series yield_series: Serie of the yields of portfolio
    :param float var: VaR for a specific trust level alpha %
    :return: CVaR
    :rtype: float
    """
    return yield_series[yield_series <= var].mean()


def ptf_yields_returns(portfolio: dict, start_date: str, end_date: str) -> Tuple[pd.Series, List[str]]:
    """
    Computes the financial returns yields for a portfolio with several products.
    The financial products are tickers used to get market informations from Yahoo Finance.

    :param dict tickers_weights: Dictionary representing the Portfolio of financial products with tickers as key and the amount of each produts as value
    :param str start_date: Start date of the period of market information
    :param str end_date: End date of the period of market information
    :return: Portfolio yields returns and Ticker names
    :rtype: Tuple[pd.Series, List[str]]
    :raises ValueError: if an amount is negative
    """
    amounts = np.array([])
    tickers_names_list = []
    
    for ticker, amount in portfolio.items():
        if(amount < 0):
            raise ValueError("Amount for a ticker cannot be negative")
        amounts = np.append(amounts, amount)
        tickers_names_list.append(ticker)
    
    ptf_amount = np.sum(amounts)

    df_all_tickers = pd.DataFrame()
    
    for ticker, amount in portfolio.items():
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
        df['yield_weight_' + ticker] = df['Close'].pct_change().dropna() * (amount / ptf_amount)
        df_all_tickers['yield_weight_' + ticker] = df['yield_weight_' + ticker]

    df_all_tickers['ptf_yield'] = df_all_tickers.sum(axis=1)
    
    df_sorted = df_all_tickers.sort_values(by='ptf_yield')
    yield_serie = df_sorted['ptf_yield'].dropna()
    
    return (yield_serie, tickers_names_list)


def show_ptf_var_cvar_report(names: List[str], trust_level: int, yield_series: pd.Series, var: float, cvar: float):
    """
    Prints the conclusion within Jupyter Notebook for the Portfolio.
    With graphical representation of the VaR.

    :param List[str] names: Names of the tickers in the Portfolio
    :param int trust_level: alpha % trust level
    :param pd.Series yield_series: Serie of the yields of portfolio
    """
    print("VaR and CVaR for Portfolio: " + " ".join(names))
    print('VaR '+ str(trust_level) +'% is : '+ (var*100).round(2).astype(str) + '%')
    print('CVaR '+ str(trust_level) +'% is : '+ (cvar*100).round(2).astype(str) + '%')
    print('This means there is a probability of '+ str(100 - trust_level) +'% (100 - '+ str(trust_level) +'%) to lose more than '+ (var*100).round(2).astype(str) +'% per day')
    print('Meaning '+ str(trust_level) +'% of the time the loss will be inferior to '+ (var*100).round(2).astype(str) +'%')
    
    sns.histplot(yield_series, bins=100, kde=True, color='lightblue')
    plt.axvline(var, color='green', linestyle='--', label=f'VaR: {var:.2%}')
    plt.axvline(cvar, color='orange', linestyle='--', label=f'CVaR: {cvar:.2%}')

    plt.legend()
    plt.title("Yield distribution for VaR and CVaR "+ str(trust_level) +"%")
    plt.xlabel("Yield Return")
    plt.ylabel("Frequency")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.show()



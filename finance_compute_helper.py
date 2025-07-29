from typing import Tuple, List
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


# --------------
# Historical Var
# --------------

# 1 Asset

def compute_var_histo(yield_series: pd.Series, trust_level: int) -> float:
    return np.percentile(yield_series, 100 - trust_level)


def show_var_histo_report(ticker: str, trust_level: int):
    dat = yf.Ticker(ticker)
    print("VaR for: " + str(dat.info['longName']))
    
    df = dat.history(period='max')
    df['yield_return'] = df['Close'].pct_change().dropna()
    df_sorted = df.sort_values(by='yield_return')
    
    var = compute_var_histo(df_sorted['yield_return'].dropna(), trust_level)
    
    print('Var '+ str(trust_level) +'% is : '+ (var*100).round(2).astype(str) + '%')
    print('This means there is a probability of '+ str(100 - trust_level) +'% (100 - '+ str(trust_level) +'%) to lose more than '+ (var*100).round(2).astype(str) +'% per day')
    print('Meaning '+ str(trust_level) +'% of the time the loss will be inferior to '+ (var*100).round(2).astype(str) +'%')
    
    sns.histplot(df_sorted['yield_return'], bins=100, kde=True, color='lightblue')
    plt.axvline(var, color='red', linestyle='--', label=f'VaR: {var:.2%}')

    plt.legend()
    plt.title("Yield distribution for VaR "+ str(trust_level) +"%")
    plt.xlabel("Yield Return")
    plt.ylabel("Frequency")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.show()

# N Assets

def ptf_yields_returns(tickers_weights: dict, start_date: str, end_date: str) -> Tuple[pd.Series, List[str]]:
    weights = np.array([])
    tickers_names_list = []
    
    for t, w in tickers_weights.items():
        weights = np.append(weights, w)
        tickers_names_list.append(t)
    
    if(np.sum(weights) != 1):
        raise ValueError("Total weights of Asset different from 1")

    df_all_tickers = pd.DataFrame()
    
    for t, w in tickers_weights.items():
        df = yf.download(t, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
        df['yield_weight_' + t] = df['Close'].pct_change().dropna() * w
        df_all_tickers['yield_weight_' + t] = df['yield_weight_' + t]

    df_all_tickers['ptf_yield'] = df_all_tickers.sum(axis=1)
    
    df_sorted = df_all_tickers.sort_values(by='ptf_yield')
    yield_serie = df_sorted['ptf_yield'].dropna()
    
    return (yield_serie, tickers_names_list)


def show_ptf_var_histo_report(names: List[str], trust_level: int, yield_series: pd.Series):
    print("VaR for Portfolio: " + " ".join(names))    
    var = compute_var_histo(yield_series, trust_level)
    
    print('Var '+ str(trust_level) +'% is : '+ (var*100).round(2).astype(str) + '%')
    print('This means there is a probability of '+ str(100 - trust_level) +'% (100 - '+ str(trust_level) +'%) to lose more than '+ (var*100).round(2).astype(str) +'% per day')
    print('Meaning '+ str(trust_level) +'% of the time the loss will be inferior to '+ (var*100).round(2).astype(str) +'%')
    
    sns.histplot(yield_series, bins=100, kde=True, color='lightblue')
    plt.axvline(var, color='red', linestyle='--', label=f'VaR: {var:.2%}')

    plt.legend()
    plt.title("Yield distribution for VaR "+ str(trust_level) +"%")
    plt.xlabel("Yield Return")
    plt.ylabel("Frequency")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.show()
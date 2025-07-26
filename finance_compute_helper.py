import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def compute_var(yield_series: pd.Series, trust_level: int) -> float:
    return np.percentile(yield_series, 100 - trust_level)


def show_var_report(ticker: str, trust_level: int):
    dat = yf.Ticker(ticker)
    print("VaR for: " + str(dat.info['longName']))
    
    df = dat.history(period='max')
    df['Yield'] = df['Close'].pct_change().dropna()
    df_sorted = df.sort_values(by='Yield')
    
    var = compute_var(df_sorted['Yield'].dropna(), trust_level)
    
    print('Var '+ str(trust_level) +'% is : '+ (var*100).round(2).astype(str) + '%')
    print('This means there is a probability of '+ str(100 - trust_level) +'% (100 - '+ str(trust_level) +'%) to lose more than '+ (var*100).round(2).astype(str) +'% per day')
    print('Meaning '+ str(trust_level) +'% of the time the loss will be inferior to '+ (var*100).round(2).astype(str) +'%')
    
    sns.histplot(df_sorted['Yield'], bins=100, kde=True, color='lightblue')
    plt.axvline(var, color='red', linestyle='--', label=f'VaR: {var:.2%}')

    plt.legend()
    plt.title("Yield distribution for VaR "+ str(trust_level) +"%")
    plt.xlabel("Yield")
    plt.ylabel("Frequency")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.show()
import yfinance as yf
ticker = yf.download(tickers='SPY')
print(ticker)
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')


def get_Data(url):
    sp500 = pd.read_html(url)[0]

    sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

    symbols_list = sp500['Symbol'].unique().tolist()

    end_date = '2024-12-10'

    start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)    #Set the start date to 8 years before end date

    df = yf.download(tickers = symbols_list, start = start_date, end = end_date).stack()

    df.index.names = ['date', 'ticker']     # Set multi index to date & ticker

    df.columns = df.columns.str.lower() 

    return df


def calc_atr(stock_data):
    atr = ta.atr(high = stock_data['high'],
                 low = stock_data['low'],
                 close = stock_data['close'],
                 length = 14)
    return atr.sub(atr.mean()).div(atr.std())

def calc_macd(close):
    macd = ta.macd(close = close, length = 20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

def calc_Features(df):
    """
    This function calculates the technical features we will use for machine learning.
    In our case, this is the Garman-Klass Volatility, RSI, bollinger bands, ATR, MACD and dollar volume.
    :return: pandas dataframe
    """
    df['garman_klass_vol'] = ((np.log(df['high'])- np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close']) - np.log(df['open']))**2)

    df['rsi'] = df.groupby(level = 1)['adj close'].transform(lambda x: ta.rsi(close = x, length = 20))

    df['bb_low'] = df.groupby(level = 1)['adj close'].transform(lambda x: ta.bbands(close = np.log1p(x), length = 20).iloc[:,0])

    df['bb_mid'] = df.groupby(level = 1)['adj close'].transform(lambda x: ta.bbands(close = np.log1p(x), length = 20).iloc[:,1])

    df['bb_high'] = df.groupby(level = 1)['adj close'].transform(lambda x: ta.bbands(close = np.log1p(x), length = 20).iloc[:,2])

    df['atr'] = df.groupby(level = 1, group_keys = False).apply(calc_atr)

    df['macd'] = df.groupby(level = 1, group_keys = False)['adj close'].apply(calc_macd)

    df['dollar_volume'] = (df['adj close'] * df['volume'])/1e6

    return df


def agg_indicators(df):

    last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]

    df = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
    df.unstack()[last_cols].resample('M').last().stack('ticker')], axis = 1)).dropna()

    return df

def get_liquid_stocks(df):

    df['dollar_volume'] = (df.loc[:,'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())

    df['dollar_vol_rank'] = (df.groupby('date')['dollar_volume'].rank(ascending=False))
    
    df = df[df['dollar_vol_rank'] < 150].drop(['dollar_volume', 'dollar_vol_rank'], axis = 1)

    return df

def calc_monthly_returns(df):
    outlier_cutoff = 0.05

    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower = x.quantile(outlier_cutoff),
                                                        upper = x.quantile(1-outlier_cutoff)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1))
    return df

def get_monthly_returns(df):
    df = df.groupby(level = 1, group_keys = False).apply(calc_monthly_returns).dropna()
    return df

def calc_RFB(df):

    factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                                 'famafrench',
                                 start = '2010')[0].drop('RF', axis = 1)
    
    factor_data.index = factor_data.index.to_timestamp()

    factor_data = factor_data.resample('M').last().div(100)

    factor_data.index.name = 'date'

    factor_data = factor_data.join(df['return_1m']).sort_index()

    observations = factor_data.groupby(level = 1).size()

    valid_stocks = observations[observations >= 10]
    
    if valid_stocks.empty:
        raise ValueError("No valid stocks found with >= 10 observations.")

    factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

    
    betas = (factor_data.groupby(level = 1, group_keys = False).apply(lambda x: RollingOLS(endog = x['return_1m'],
                                                                                  exog = sm.add_constant(x.drop('return_1m', axis = 1)),
                                                                                  window = min(24, x.shape[0]),
                                                                                  min_nobs = len(x.columns)+1).fit(params_only = True).params.drop('const', axis = 1)))

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    df = (df.join(betas.groupby('ticker').shift()))
    
    df.loc[:, factors] = df.groupby('ticker', group_keys = False)[factors].apply(lambda x: x.fillna(x.mean()))

    df = df.drop('adj close', axis = 1)

    df = df.dropna()        
    return df

def get_clusters(df):

    HYP_K = 4       #The hyperparameter k that I  have chosen is 4.

    from sklearn.cluster import KMeans

    target_rsi_values = [30, 45, 55, 70]

    initial_centroids = np.zeros((len(target_rsi_values), 18))

    initial_centroids[:, 6] = target_rsi_values

    df['cluster'] = KMeans(n_clusters=HYP_K,
                            random_state=0,
                            init=initial_centroids).fit(df).labels_
    return df

def compute_Clusters(df):
    return df.dropna().groupby('date', group_keys = False).apply(get_clusters)

def plot_clusters(prep_df):

    cluster_0 = prep_df[prep_df['cluster'] == 0]
    cluster_1 = prep_df[prep_df['cluster'] == 1]
    cluster_2 = prep_df[prep_df['cluster'] == 2]
    cluster_3 = prep_df[prep_df['cluster'] == 3]

    plt.scatter(cluster_0.iloc[:,0], cluster_0.iloc[:,6], color = 'red', label = 'cluster 0')
    plt.scatter(cluster_1.iloc[:,0], cluster_1.iloc[:,6], color = 'green', label = 'cluster 1')
    plt.scatter(cluster_2.iloc[:,0], cluster_2.iloc[:,6], color = 'black', label = 'cluster 2')
    plt.scatter(cluster_3.iloc[:,0], cluster_3.iloc[:,6], color = 'pink', label = 'cluster 3')

    plt.legend()
    plt.show()

    return

def plots(df):
    plt.style.use('ggplot')
    for i in df.index.get_level_values('date').unique().tolist():
        prep_df = df.xs(i, level = 0)
        
        plt.title(f'Date: {i}')
        plot_clusters(prep_df)
    
    return

def select_assets(df):
    """
    This function selects assets based on the cluster
    
    Return: fixed dates
    """

    filtered_df = df[df['cluster'] == 1].copy()

    filtered_df = filtered_df.reset_index(level = 1)

    filtered_df.index = filtered_df.index+pd.DateOffset(1)

    filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

    dates = filtered_df.index.get_level_values('date').unique().tolist()

    fixed_dates = {}

    for d in dates:

        fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level = 0).index.tolist()
    
    return fixed_dates


def optimize_weights(prices, lower_bound, upper_bound = .1):
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns

    returns = expected_returns.mean_historical_return(prices = prices,
                                                      frequency = 252)
    
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    
    ef = EfficientFrontier(expected_returns= returns,
                           cov_matrix= cov,
                           weight_bounds=(lower_bound,upper_bound),
                           solver='SCS')
    weights = ef.max_sharpe()

    return ef.clean_weights()  
    

def get_prices(df):

    stocks = df.index.get_level_values('ticker').unique().tolist()

    new_df = yf.download(tickers= stocks, start= df.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                         end= df.index.get_level_values('date').unique()[-1])

    return new_df

def calc_returns(new_df, fixed_dates, algo_start_date):

    returns_dataframe = np.log(new_df['Adj Close']).diff()

    portfolio_dataframe = pd.DataFrame()

    for start_date in fixed_dates.keys():
        
        try:
            end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

            cols = fixed_dates[start_date]

            optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')

            optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')

        
            optimization_df = new_df['Adj Close'].loc[optimization_start_date:optimization_end_date, cols]
            
            success = False

            try:
                weights = optimize_weights(prices= optimization_df, lower_bound= round(1/(len(optimization_df.columns)*2),3))

                weights = pd.DataFrame(weights, index= pd.Series(0))


                success = True
            except:
                print(f'Max Sharpe Optimization failed for {start_date}, continuing with Equal-Weights')

            if success==False:
                weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                        index=optimization_df.columns.tolist(),
                                        columns=pd.Series(0)).T
                
            temp_df = returns_dataframe[start_date:end_date]
           

            temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                   .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                          left_index=True,
                          right_index=True)

            temp_df = temp_df.reset_index().set_index(['Date', 'Ticker'])

            temp_df = temp_df.unstack().stack()
            
            temp_df.index.names = ['Date', 'Ticker']
            

            temp_df['weighted_return'] = temp_df['return'] * temp_df['weight']

            temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')

            portfolio_dataframe = pd.concat([portfolio_dataframe, temp_df], axis=0)

        except Exception as e:
            print(f'Exception: {e}')

    portfolio_dataframe = portfolio_dataframe.drop_duplicates()

    spy = yf.download(tickers= 'SPY', start = algo_start_date, end= dt.date.today()).reset_index().set_index(['Date'])
    spy.columns = spy.columns.droplevel(1)
    spy = spy[['Adj Close']]

    spy_ret = np.log(spy).diff().dropna().rename({'Adj Close' : 'SPY Buy & Hold'}, axis = 1)

    spy_ret = spy_ret[['SPY Buy & Hold']]
    
    print(f'Portfolio DF: \n{portfolio_dataframe}')
    print(f'SPY Ret DF: \n{spy_ret}')

    portfolio_dataframe = portfolio_dataframe.merge(spy_ret, left_index= True, right_index= True)

    return portfolio_dataframe


def plot_returns(portfolio_dataframe):

    import matplotlib.ticker as mtick

    plt.style.use('ggplot')

    portfolio_cumulative_return = np.exp(np.log1p(portfolio_dataframe).cumsum()) - 1

    portfolio_cumulative_return[:'2024-12-10'].plot(figsize=(16,6))

    plt.title('Unsupervised Learning Trading Strategy Returns Over Time')

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

    plt.ylabel('Return')

    plt.show()

     

data = get_Data('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
data = calc_Features(data)
data = agg_indicators(data)
data = get_liquid_stocks(data)
data = get_monthly_returns(data)
data = calc_RFB(data)
data = compute_Clusters(data)
#plots(data)
fixed_dates = select_assets(data)
new_df = get_prices(data)

returns = calc_returns(new_df, fixed_dates, '2016-12-10')
plot_returns(returns)




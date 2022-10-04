# # UTOR: FinTech Bootcamp - Project 1: Stock Portfolio Optimizer

# Import libraries
import numpy as np
import pandas as pd
from pandas_datareader import data
import plotly.express as px
import datetime
import plotly.graph_objects as go
import plotly.io as pio
from datetime import timedelta
from MCForecastTools import MCSimulation
from IPython.display import display


# ## User Input

# Yahoo finance API data input
stocks = ['DIS', 'JNJ', 'HD', 'KO', 'NKE']
benchmark = ['^DJI']

start_date = '2008/01/01'
end_date = '2022/09/30'

out_sample_days_held_back = 252

# Plotly graph themes
#Example themes - ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
theme='seaborn'


# Set random seed - for optimal portfolio weight calculation
seed=42
np.random.seed(seed)

# Optimal portfolio calculation
number_opt_porfolios=10000

# Sharpe Calculation
rf = 0.01 # risk factor

# Monte Carlo Simulation
number_simulation = 500
years = 5

init_investment = 10000



# ## Import and Clean Stock Data

# Import Stock data 

df = data.DataReader(stocks, 'yahoo', start=start_date, end=end_date)
display(df.head())
display(df.tail())


# In[ ]:


# Closing price

df = df['Adj Close']
display(df.head())




# Calculate Stocks peercentage change and slice dataframe

stock_returns = pd.DataFrame()

for stock in stocks:
    stock_returns[stock+'_Returns'] = df[stock].pct_change()


stock_returns = stock_returns.dropna()

#Here we slice the dataframe to create backtesting data and out of sample data

count = df.shape[0] - out_sample_days_held_back

last_year_stock_returns = stock_returns.iloc[count:,:]
stock_returns_excl_ly = stock_returns.iloc[0:count,:]
 
display(stock_returns.tail())
display(last_year_stock_returns.head())


# ## Import and Clean Benchmark Data


# Import Benchmark data

bm = data.DataReader(benchmark, 'yahoo', start=start_date, end=end_date)
display(bm.head())
display(bm.tail())

# Closing price

bm = bm['Adj Close']
display(bm.head())



# Calculate Benchmarks peercentage change and slice dataframe

benchmark_returns = pd.DataFrame()
benchmark_returns['BM_Returns'] = bm[benchmark].pct_change()

benchmark_returns = benchmark_returns.dropna()

#Here we slice the dataframe to create backtesting data and out of sample data

count = bm.shape[0] - out_sample_days_held_back

last_year_benchmark_returns = benchmark_returns.iloc[count:,:]
benchmark_returns_excl_ly = benchmark_returns.iloc[0:count,:]

display(benchmark_returns.tail())
display(last_year_benchmark_returns.head())


# # Stock Dataframe graphs/Visualization


# Plot the retruns for each stock

fig = go.Figure()

for stock, col in enumerate(stock_returns.columns.tolist()):
    fig.add_trace(
    go.Scatter(
        x=stock_returns.index,
        y=stock_returns[col],
        name=stocks[stock]
    ))
    
fig.update_layout(
    title={
        'text': "Stock Returns",
    },
    template=theme,
    xaxis=dict(autorange=True,
              title_text='Date'),
    yaxis=dict(autorange=True,
              title_text='Daily Returns')
)

fig.show()



# Plot a Boxplot of each stocks returns

fig = go.Figure()

for stock, col in enumerate(stock_returns.columns.tolist()):
    fig.add_trace(
    go.Box(
        y=stock_returns[col],
        name=stocks[stock]
    ))


fig.update_layout(
    title={
        'text': "Stock Returns Box Plot",
    },
    template=theme,
    xaxis=dict(autorange=True,
              title_text='Stock'),
    yaxis=dict(autorange=True,
              title_text='Return Distribution')
)

fig.show()



# Plot the histogram of the each stocks returns

fig = go.Figure()

for stock, col in enumerate(stock_returns.columns.tolist()):
    fig.add_trace(
    go.Histogram(
        x=stock_returns[col],
        name=stocks[stock]
    ))


fig.update_layout(
    title={
        'text': "Stock Returns Distribuion",
    },
    template=theme,
    xaxis=dict(autorange=True,
              title_text='Daily Returns'),
    yaxis=dict(autorange=True,
              title_text='Count')
)

fig.show()


# Calculate stocks cumulative returns

cumulative_returns = (1 + stock_returns).cumprod()

init_date = cumulative_returns.index[0] - timedelta(days=1)
cumulative_returns.loc[init_date] = 1
cumulative_returns = cumulative_returns.sort_index()


# Plot the stocks cumulative returns

fig = go.Figure()

for stock, col in enumerate(cumulative_returns.columns.tolist()):
    fig.add_trace(
    go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns[col],
        name=stocks[stock]
    ))


fig.update_layout(
    title={
        'text': "Cumulative Stock Returns",
    },
    template=theme,
    xaxis=dict(autorange=True,
              title_text='Date'),
    yaxis=dict(autorange=True,
              title_text='Cumulative Return')
)

fig.show()


# Benchmark cumulative returns

cumulative_benchmark_returns = (1 + benchmark_returns).cumprod()

init_date = cumulative_benchmark_returns.index[0] - timedelta(days=1)
cumulative_benchmark_returns.loc[init_date] = 1
cumulative_benchmark_returns = cumulative_benchmark_returns.sort_index()


# # 

# # Optimal Portfolio Weights Calculation


# Calculate Covariance matrix of the log stock returns. We use log as it produces marginally more accurate results.

cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov() 
cov_matrix


# Covariance matrix heatmap

fig = px.imshow(cov_matrix,text_auto=True, title='Covariance Plot', template=theme)
fig.show()


# Calculate Correlation matrix of the log stock returns. We use log as it produces marginally more accurate results.

corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr() #read up and explain
corr_matrix


# Correlation matrix heatmap

fig = px.imshow(corr_matrix,text_auto=True, title='Correlation Plot', template=theme)
fig.show()


# CAGR for individual companies 

dt = pd.to_datetime(start_date, format='%Y/%m/%d')
dt1 = pd.to_datetime(end_date, format='%Y/%m/%d')

yrs_full = ((dt1-dt).days)/365

bm_annual_returns = bm.pct_change().apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/yrs_full) - 1

annual_returns = df.pct_change().apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/yrs_full) - 1
annual_returns


# Volatility is given by the annual standard deviation. We multiply by 252 because there are 252 trading days/year. Also
# We will use the log of the stock returns in our calculation as it produces mrginally more accurate results.

bm_annual_std_dev = bm.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))

annual_std_dev = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))
annual_std_dev



#Concatenate the annual returns and standard deviation dataframes

risk_return = pd.concat([annual_returns, annual_std_dev], axis=1) # Creating a table for visualising returns and volatility of assets
risk_return.columns = ['Returns', 'Volatility']
risk_return


# Setup lists to hold portfolio weights, returns and volatility

p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = len(df.columns)
num_portfolios = number_opt_porfolios


# Calculate Portfolio weights for num_portfolios

for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, annual_returns) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(252) # Annual standard deviation = volatility
    p_vol.append(ann_sd)


# Insert the stock weights that correspon to the respective portfolio return and volatility

data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(df.columns.tolist()):
    data[symbol+' weight'] = [w[counter] for w in p_weights]


# Create portfolios dataframe to hold the portfolio weights of stocks, and portfolio return, volatility and sharpe ratio

portfolios  = pd.DataFrame(data)

bm_sharpe = (bm_annual_returns-rf)/bm_annual_std_dev

portfolios['Sharpe'] = (portfolios['Returns']-rf)/portfolios['Volatility']

portfolios.head() # Dataframe of the 10000 portfolios created


# Plot efficient frontier

#px.scatter(portfolios, x='Volatility', y='Returns', color='Sharpe', title='Portfolio Efficient Frontier',
#          marginal_y='box')



# Plot Portfolio Return Distribution

px.histogram(portfolios, x='Returns', nbins=50, title='Portfolio Returns Distribution', template=theme)



# Finding the optimal portfolio

optimal_risky_port = portfolios.iloc[(portfolios['Sharpe']).idxmax()]

optimal_risky_port


# Pie Chart of optimal portfolio stock weightings

opt_port_df = pd.DataFrame(data={'Stocks': df.columns.tolist(), 'Weight': optimal_risky_port[2:-1].values})

fig = px.pie(opt_port_df, values='Weight', names='Stocks',
             title='Optimal Portfolio Stock Weighting',
             template=theme
            )
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()



# Optimal Portfolio Stock Weights Table

head = ['<b>Symbol<b>', '<b>Weight<b>']

labels =[]

for stock in opt_port_df['Stocks'].tolist():
    txt='<b>'+stock+'<b>'
    labels.append(txt)
    
wght =[]
    
for weight in opt_port_df['Weight'].tolist():
    w = '{:.2%}'.format(weight)
    wght.append(w)

    
fig = go.Figure(data=[go.Table(
    header=dict(values=head,
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[labels, wght],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(margin=dict(l=5, r=5, b=0,t=10))

fig.show()


# create an optimal portfolio dataframe

optimal_port_df = pd.DataFrame(data={'Returns': optimal_risky_port[0], 'Volatility': optimal_risky_port[1]}, index=[0])
#optimal_port_df


# Plot efficient frontier with optimal portfolio

fig = px.scatter(portfolios, x='Volatility', y='Returns', color='Sharpe', title='Portfolio Efficient Frontier with Optimal Portfolio')
fig.add_trace(go.Scatter(x=optimal_port_df['Volatility'], y=optimal_port_df['Returns'], name='Optimal Portfolio',
            marker=dict(
            color='LightSkyBlue',
            size=20,
            )))

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="top",
    y=1.1,
    xanchor="right",
    x=1
    ),
    template=theme
)
fig.show()


# Calculate Historical Optimal Portfolio Return using optimal weights

weights = optimal_risky_port[2:-1].values

weights_full = weights

portfolio_returns = stock_returns.dot(weights)

# Display sample data
#portfolio_returns.sample(10)



# Convert the historical optimal portfolio returns to a dataframe

port_returns = pd.DataFrame(portfolio_returns)
port_returns.columns = ['Portfolio_returns']



# Calculate the historical cumulative returns for the optimal portfolio

optimal_cumulative_returns = (1 + port_returns).cumprod()

init_date = optimal_cumulative_returns.index[0] - timedelta(days=1)
optimal_cumulative_returns.loc[init_date] = 1
optimal_cumulative_returns = optimal_cumulative_returns.sort_index()



# Concatenate the optimal and benchmark daily returns

historic_daily_returns = pd.concat([benchmark_returns, port_returns], axis=1, join="inner")

historic_daily_returns['Date'] = historic_daily_returns.index
historic_daily_returns['Date'] = historic_daily_returns['Date'].dt.date



#Here we concatenate the historical optimal cumulative portfolio returns with that of the benchmark

historic_returns = pd.concat([cumulative_benchmark_returns, optimal_cumulative_returns], axis=1, join="inner")

historic_returns['Date'] = historic_returns.index
historic_returns['Date'] = historic_returns['Date'].dt.date



# Cumulative Historical Returns: Optimal Porfolio vs. Benchmark

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=historic_returns['Date'],
        y=historic_returns['Portfolio_returns'],
        name="Optimal Portfolio",
        line=dict(color="#33CFA5")
    ))

fig.add_trace(
    go.Scatter(
        x=historic_returns['Date'],
        y=historic_returns['BM_Returns'],
        name='Benchmark',
        line=dict(color="#bf00ff")
    ))


fig.update_layout(
    title={
        'text': "Optimal Cumulative Portfolio Returns vs Benchmark",
    },
    template=theme,
    xaxis=dict(autorange=True,
              title_text='Date'),
    yaxis=dict(autorange=True,
              title_text='Cumulative Returns')
)

fig.show()


# ## Monte Carlo Simulation using full dataset


# Adjust stock dataframe for Monte Carlo Simulation 
df_mc = df.copy()

tickers = df_mc.columns.get_level_values(0).unique()

# Adjust dataframe to use in Monte Carlo Simulation
columns=[]
for ticker in tickers:
    tup = (ticker, 'close')
    columns.append(tup)
    
df_mc.columns=pd.MultiIndex.from_tuples(columns)


# Adjust stock dataframe for Monte Carlo Simulation 
bm_mc = bm.copy()

tickers = bm_mc.columns.get_level_values(0).unique()

# Adjust dataframe to use in Monte Carlo Simulation
columns=[]
for ticker in tickers:
    tup = (ticker, 'close')
    columns.append(tup)
    
bm_mc.columns=pd.MultiIndex.from_tuples(columns)


# Configuring a Monte Carlo simulation to forecast Optimal Portfolio cumulative returns

MC_sim = MCSimulation(
    portfolio_data = df_mc,
    weights = weights_full,
    num_simulation = number_simulation,
    num_trading_days = 252*years
)


# Check Monte Carlo portfolio data head

#MC_sim.portfolio_data.head()



# Check Monte Carlo imulation cumulative returns dataframe

#MC_sim.calc_cumulative_return()



# Plot simulation outcomes
#line_plot = MC_sim.plot_simulation()

# Save the plot for future usage
#line_plot.get_figure().savefig("MC_30year_sim_plot.png", bbox_inches="tight")



# Plot probability distribution and confidence intervals
#dist_plot = MC_sim.plot_distribution()



# Fetch summary statistics from the Monte Carlo simulation results
statistics = MC_sim.summarize_cumulative_return()

# Print summary statistics
#print(statistics)



# Set initial investment
initial_investment = init_investment

# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our initial investment
ci_lower = round(statistics[8]*initial_investment,2)
ci_upper = round(statistics[9]*initial_investment,2)

# Print results
print("There is a 95% chance that an initial investment of ${:0,.2f} in the portfolio"
      " over the next 5 years will end within in the range of"
      " ${:0,.2f} and ${:0,.2f}".format(initial_investment, ci_lower, ci_upper))



# Configuring a Monte Carlo simulation to forecast Benchmark cumulative returns
BM_sim = MCSimulation(
    portfolio_data = bm_mc,
    weights = [1],
    num_simulation = number_simulation,
    num_trading_days = 252*years
)



# Check Monte Carlo portfolio data head

#BM_sim.portfolio_data.head()



# Check Monte Carlo imulation cumulative returns dataframe

#BM_sim.calc_cumulative_return()



# Plot simulation outcomes
#line_plot = BM_sim.plot_simulation()

# Save the plot for future usage
#line_plot.get_figure().savefig("MC_30year_sim_plot.png", bbox_inches="tight")



# Plot probability distribution and confidence intervals
#dist_plot = BM_sim.plot_distribution()



# Fetch summary statistics from the Monte Carlo simulation results
statistics = BM_sim.summarize_cumulative_return()

# Print summary statistics
#print(statistics)



# Set initial investment
initial_investment = init_investment

# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our initial investment
ci_lower = round(statistics[8]*initial_investment,2)
ci_upper = round(statistics[9]*initial_investment,2)

# Print results
print("There is a 95% chance that an initial investment of ${:0,.2f} in the portfolio"
      " over the next 5 years will end within in the range of"
      " ${:0,.2f} and ${:0,.2f}".format(initial_investment, ci_lower, ci_upper))


# ## Full Data Descriptive Statistics



# Descriptive Statistics
# historic_daily_returns
# historic_returns

pm_start = init_investment
bm_start = init_investment

pm_end = round(init_investment * historic_returns['Portfolio_returns'][-1],2)
bm_end = round(init_investment * historic_returns['BM_Returns'][-1],2)

daily_pm_max_return = historic_daily_returns['Portfolio_returns'].max()
daily_bm_max_retrun = historic_daily_returns['BM_Returns'].max()

daily_pm_min_return = historic_daily_returns['Portfolio_returns'].min()
daily_bm_min_retrun = historic_daily_returns['BM_Returns'].min()

pm_return = optimal_risky_port[0]
bm_return = bm_annual_returns[0]

pm_vol = optimal_risky_port[1]
bm_vol = bm_annual_std_dev[0]

pm_sharpe = round(optimal_risky_port[-1],2)
bm_sharpe = round(bm_sharpe,2)

covariance = historic_daily_returns['Portfolio_returns'].cov(historic_daily_returns['BM_Returns'])
variance = historic_daily_returns['BM_Returns'].var()
pm_beta = round((covariance/variance),2)

bm_beta = 1


# Table of Descriptive Statistics

head = ['<b>Statistic<b>', '<b>Optimal Portfolio<b>', '<b>Benchmark<b>']
labels = ['<b>Initial Investment<b>', '<b>Ending Investment<b>', '<b>Max Daily Return<b>',
          '<b>Min Daily Return<b>', '<b>Return<b>', '<b>Volatility<b>', '<b>Sharpe Ratio<b>', '<b>Beta<b>']
pf_stats = ['${:,}'.format(pm_start), '${:,}'.format(pm_end), '{:.2%}'.format(daily_pm_max_return), 
            '{:.2%}'.format(daily_pm_min_return), '{:.2%}'.format(pm_return), '{:.2%}'.format(pm_vol), pm_sharpe, pm_beta]
bm_stats = ['${:,}'.format(bm_start),'${:,}'.format(bm_end), '{:.2%}'.format(daily_bm_max_retrun), 
            '{:.2%}'.format(daily_bm_min_retrun), '{:.2%}'.format(bm_return), '{:.2%}'.format(bm_vol), bm_sharpe, bm_beta]

fig = go.Figure(data=[go.Table(
    header=dict(values=head,
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[labels, pf_stats, bm_stats],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(margin=dict(l=5, r=5, b=0,t=10))

fig.show()



# Plot Daily Returns Optimal Portfolio vs. Benchmark

fig = px.scatter(historic_daily_returns, x='BM_Returns', y='Portfolio_returns', title='Daily Returns Optimal Portfolio vs. Benchmark')

fig.update_layout(template=theme)
fig.show()



# # Calulate Optimal Portfolio weights using sliced dataframe and optimal portfolio performance for out of sample data


# Calculate Covariance matrix of the log stock returns. We use log as it produces marginally more accurate results.

cov_matrix_excl = stock_returns_excl_ly.apply(lambda x: np.log(1+x)).cov()
cov_matrix


# In[ ]:


# Calculate Correlation matrix of the log stock returns. We use log as it produces marginally more accurate results.

corr_matrix = stock_returns_excl_ly.apply(lambda x: np.log(1+x)).corr()
corr_matrix


# CAGR for individual companies 

dt = pd.to_datetime(start_date, format='%Y/%m/%d')
dt2 = pd.to_datetime(stock_returns_excl_ly.index[-1], format='%Y-%m-%d')

yrs = ((dt2-dt).days)/365

bm_annual_returns = benchmark_returns_excl_ly.apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/yrs) - 1

annual_returns = stock_returns_excl_ly.apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/yrs) - 1
annual_returns


# Volatility is given by the annual standard deviation. We multiply by 252 because there are 252 trading days/year. Also
# We will use the log of the stock returns in our calculation as it produces mrginally more accurate results.

bm_annual_std_dev = benchmark_returns_excl_ly.apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))

annual_std_dev = stock_returns_excl_ly.apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))
annual_std_dev


#Concatenate the annual returns and standard deviation dataframes

risk_return = pd.concat([annual_returns, annual_std_dev], axis=1) # Creating a table for visualising returns and volatility of assets
risk_return.columns = ['Returns', 'Volatility']
risk_return


# Setup lists to hold portfolio weights, returns and volatility

p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = len(df.columns)
num_portfolios = number_opt_porfolios



# Calculate Portfolio weights for num_portfolios

for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, annual_returns) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(252) # Annual standard deviation = volatility
    p_vol.append(ann_sd)



# Insert the stock weights that correspon to the respective portfolio return and volatility

data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(df.columns.tolist()):
    data[symbol+' weight'] = [w[counter] for w in p_weights]



# Create portfolios dataframe to hold the portfolio weights of stocks, and portfolio return, volatility and sharpe ratio

portfolios  = pd.DataFrame(data)

bm_sharpe = (bm_annual_returns-rf)/bm_annual_std_dev

portfolios['Sharpe'] = (portfolios['Returns']-rf)/portfolios['Volatility']

portfolios.head() # Dataframe of the 10000 portfolios created


# Plot efficient frontier

#px.scatter(portfolios, x='Volatility', y='Returns', color='Sharpe', title='Portfolio Efficient Frontier',
#          marginal_y='box')



# Plot Portfolio Return Distribution

px.histogram(portfolios, x='Returns', nbins=50, title='Portfolio Returns Distribution', template=theme)


# Finding the optimal portfolio

optimal_risky_port = portfolios.iloc[(portfolios['Sharpe']).idxmax()]

optimal_risky_port


# Pie Chart of optimal portfolio stock weightings

opt_port_df = pd.DataFrame(data={'Stocks': df.columns.tolist(), 'Weight': optimal_risky_port[2:-1].values})

fig = px.pie(opt_port_df, values='Weight', names='Stocks',
             title='Optimal Portfolio Stock Weighting',
             template=theme
            )
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# Optimal Portfolio Stock Weights Table - Sample Data

head = ['<b>Symbol<b>', '<b>Weight<b>']

labels =[]

for stock in opt_port_df['Stocks'].tolist():
    txt='<b>'+stock+'<b>'
    labels.append(txt)
    
wght =[]
    
for weight in opt_port_df['Weight'].tolist():
    w = '{:.2%}'.format(weight)
    wght.append(w)

    
fig = go.Figure(data=[go.Table(
    header=dict(values=head,
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[labels, wght],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(margin=dict(l=5, r=5, b=0,t=10))

fig.show()


# create an optimal portfolio dataframe

optimal_port_df = pd.DataFrame(data={'Returns': optimal_risky_port[0], 'Volatility': optimal_risky_port[1]}, index=[0])
#optimal_port_df


# Plot efficient frontier with optimal portfolio

fig = px.scatter(portfolios, x='Volatility', y='Returns', color='Sharpe', title='Portfolio Efficient Frontier with Optimal Portfolio')
fig.add_trace(go.Scatter(x=optimal_port_df['Volatility'], y=optimal_port_df['Returns'], name='Optimal Portfolio',
            marker=dict(
            color='LightSkyBlue',
            size=20,
            )))

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="top",
    y=1.1,
    xanchor="right",
    x=1
    ),
    template=theme
)
fig.show()



# Calculate Historical Optimal Portfolio Return using optimal weights

weights = optimal_risky_port[2:-1].values

weights_excl_ly = weights

portfolio_returns = stock_returns_excl_ly.dot(weights)

# Display sample data
#portfolio_returns.sample(10)



# Convert the historical optimal portfolio returns to a dataframe

port_returns = pd.DataFrame(portfolio_returns)
port_returns.columns = ['Portfolio_returns']



# Calculate the historical cumulative returns for the optimal portfolio

optimal_cumulative_returns = (1 + port_returns).cumprod()

init_date = optimal_cumulative_returns.index[0] - timedelta(days=1)
optimal_cumulative_returns.loc[init_date] = 1
optimal_cumulative_returns = optimal_cumulative_returns.sort_index()



#Here we concatenate the historical optimal daily portfolio returns with that of the benchmark

sample_daily_historic_returns = pd.concat([benchmark_returns_excl_ly, port_returns], axis=1, join="inner")
sample_daily_historic_returns['Date'] = sample_daily_historic_returns.index
sample_daily_historic_returns['Date'] = sample_daily_historic_returns['Date'].dt.date


#Here we concatenate the historical optimal cumulative portfolio returns with that of the benchmark

sample_cumulative_historic_returns = pd.concat([cumulative_benchmark_returns, optimal_cumulative_returns], axis=1, join="inner")

sample_cumulative_historic_returns['Date'] = sample_cumulative_historic_returns.index
sample_cumulative_historic_returns['Date'] = sample_cumulative_historic_returns['Date'].dt.date



# Cumulative Historical Returns: Optimal Porfolio vs. Benchmark

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=sample_cumulative_historic_returns['Date'],
        y=sample_cumulative_historic_returns['Portfolio_returns'],
        name="Optimal Portfolio",
        line=dict(color="#33CFA5")
    ))

fig.add_trace(
    go.Scatter(
        x=sample_cumulative_historic_returns['Date'],
        y=sample_cumulative_historic_returns['BM_Returns'],
        name='Benchmark',
        line=dict(color="#bf00ff")
    ))


fig.update_layout(
    title={
        'text': "Optimal Cumulative Portfolio Returns vs Benchmark",
    },
    template=theme,
    xaxis=dict(autorange=True,
              title_text='Date'),
    yaxis=dict(autorange=True,
              title_text='Cumulative Returns')
)

fig.show()


# ## Generate animated graph for out of sample data (Optimal Portfolio vs. Benchmark)



#Calculate portfolio returns for last year using updated weights

last_year_portfolio_returns = last_year_stock_returns.dot(weights_excl_ly)

# Display sample data
#last_year_portfolio_returns.sample(10)



# Convert calulated portfolio returns for last year into a dataframe

one_year_port_returns = pd.DataFrame(last_year_portfolio_returns)
one_year_port_returns.columns = ['Portfolio_returns']



# Calculate the cumulative returns for the optimal portfolio for the last year

one_year_cumulative_returns = (1 + one_year_port_returns).cumprod()

init_date = one_year_cumulative_returns.index[0] - timedelta(days=1)
one_year_cumulative_returns.loc[init_date] = 1
one_year_cumulative_returns = one_year_cumulative_returns.sort_index()



# Calculate the cumulative returns for the benchmark for the last year

one_year_benchmark_returns = (1 + last_year_benchmark_returns).cumprod()

init_date = one_year_benchmark_returns.index[0] - timedelta(days=1)
one_year_benchmark_returns.loc[init_date] = 1
one_year_benchmark_returns = one_year_benchmark_returns.sort_index()



# Concatenate the optimal and benchmark daily returns

one_year_historic_daily_returns = pd.concat([last_year_benchmark_returns, one_year_port_returns], axis=1, join="inner")

one_year_historic_daily_returns['Date'] = one_year_historic_daily_returns.index
one_year_historic_daily_returns['Date'] = one_year_historic_daily_returns['Date'].dt.date



# Concatenate the optimal and benchmark cumulative returns

one_year_historic_returns = pd.concat([one_year_benchmark_returns, one_year_cumulative_returns], axis=1, join="inner")

one_year_historic_returns['Date'] = one_year_historic_returns.index
one_year_historic_returns['Date'] = one_year_historic_returns['Date'].dt.date



# Plot An animated graphs of the optimal vs. Benchmark cumulative returns

fig = go.Figure(
    layout=go.Layout(
        updatemenus=[dict(type="buttons", direction="left", x=0.07, y=1.125), ],
        xaxis=dict(autorange=True, 
                   title_text="Date"),
        yaxis=dict(autorange=True,
                   title_text="Returns"),
        title="Out of Sample Optimal Portfolio Returns vs Benchmark",
    ))

# Add traces
init = 0


fig.add_trace(
    go.Scatter(
        x=one_year_historic_returns['Date'][:init],
        y=one_year_historic_returns['Portfolio_returns'][:init],
        name='Optimal Portfolio Returns',
        line=dict(color="#33CFA5")
    ))


fig.add_trace(
    go.Scatter(
        x=one_year_historic_returns['Date'][:init],
        y=one_year_historic_returns['BM_Returns'][:init],
        name='Benchmark Returns',
        line=dict(color="#bf00ff")
    ))


# Animation
fig.update(frames=[
    go.Frame(
        data=[
            go.Scatter(x=one_year_historic_returns['Date'][:k], y=one_year_historic_returns['Portfolio_returns'][:k]),
            go.Scatter(x=one_year_historic_returns['Date'][:k], y=one_year_historic_returns['BM_Returns'][:k])]
            
    )
    for k in range(init, one_year_historic_returns.shape[0]+1)])



# Buttons
fig.update_layout(
    template=theme,
    updatemenus=[
        dict(
            buttons=list([
                dict(label="Play",
                     method="animate",
                    args=[None, {"frame": {"duration": 1}}])
            ]))])


fig.show()


# ## In Sample and Out-of-Sample Descriptive Statistics


# Descriptive Statistics - Sample Data
# sample_daily_historic_returns
# sample_cumulative_historic_returns

pm_start = init_investment
bm_start = init_investment

pm_end = round(init_investment * sample_cumulative_historic_returns['Portfolio_returns'][-1],2)
bm_end = round(init_investment * sample_cumulative_historic_returns['BM_Returns'][-1],2)

daily_pm_max_return = sample_daily_historic_returns['Portfolio_returns'].max()
daily_bm_max_return = sample_daily_historic_returns['BM_Returns'].max()

daily_pm_min_return = sample_daily_historic_returns['Portfolio_returns'].min()
daily_bm_min_return = sample_daily_historic_returns['BM_Returns'].min()

pm_return = optimal_risky_port[0]
bm_return = bm_annual_returns[0]

pm_vol = optimal_risky_port[1]
bm_vol = bm_annual_std_dev[0]

pm_sharpe = round(optimal_risky_port[-1],2)
bm_sharpe = round(bm_sharpe,2)

covariance = sample_daily_historic_returns['Portfolio_returns'].cov(sample_daily_historic_returns['BM_Returns'])
variance = sample_daily_historic_returns['BM_Returns'].var()
pm_beta = round((covariance/variance),2)

bm_beta = 1



# Table of Descriptive Statistics - Sample Data

head = ['<b>Statistic<b>', '<b>Optimal Portfolio<b>', '<b>Benchmark<b>']
labels = ['<b>Initial Investment<b>', '<b>Ending Investment<b>', '<b>Max Daily Return<b>',
          '<b>Min Daily Return<b>', '<b>Return<b>', '<b>Volatility<b>', '<b>Sharpe Ratio<b>', '<b>Beta<b>']
pf_stats = ['${:,}'.format(pm_start), '${:,}'.format(pm_end), '{:.2%}'.format(daily_pm_max_return), 
            '{:.2%}'.format(daily_pm_min_return), '{:.2%}'.format(pm_return), '{:.2%}'.format(pm_vol), pm_sharpe, pm_beta]
bm_stats = ['${:,}'.format(bm_start), '${:,}'.format(bm_end), '{:.2%}'.format(daily_bm_max_return),
            '{:.2%}'.format(daily_bm_min_return), '{:.2%}'.format(bm_return), '{:.2%}'.format(bm_vol), bm_sharpe, bm_beta]

fig = go.Figure(data=[go.Table(
    header=dict(values=head,
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[labels, pf_stats, bm_stats],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(margin=dict(l=5, r=5, b=0,t=10))

fig.show()



# Plot Daily Returns Optimal Portfolio vs. Benchmark - Sample Data

fig = px.scatter(sample_daily_historic_returns, x='BM_Returns', y='Portfolio_returns', 
                 title='Daily Returns Optimal Portfolio vs. Benchmark - Sample Data')

fig.update_layout(template=theme)
fig.show()



# Descriptive Statistics - Out of Sample Data
# one_year_historic_daily_returns
# one_year_historic_returns

pm_start = init_investment
bm_start = init_investment

pm_end = round(init_investment * one_year_historic_returns['Portfolio_returns'][-1],2)
bm_end = round(init_investment * one_year_historic_returns['BM_Returns'][-1],2)

daily_pm_max_return = one_year_historic_daily_returns['Portfolio_returns'].max()
daily_bm_max_retrun = one_year_historic_daily_returns['BM_Returns'].max()

daily_pm_min_return = one_year_historic_daily_returns['Portfolio_returns'].min()
daily_bm_min_retrun = one_year_historic_daily_returns['BM_Returns'].min()

one_yr = yrs_full - yrs

pm_return = one_year_historic_daily_returns['Portfolio_returns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/one_yr) - 1
bm_return = one_year_historic_daily_returns['BM_Returns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/one_yr) - 1

pm_vol = one_year_historic_daily_returns['Portfolio_returns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)
bm_vol = one_year_historic_daily_returns['BM_Returns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)

pm_sharpe = round((pm_return-rf)/pm_vol,2)
bm_sharpe = round((bm_return-rf)/bm_vol,2)

covariance = one_year_historic_daily_returns['Portfolio_returns'].cov(one_year_historic_daily_returns['BM_Returns'])
variance = one_year_historic_daily_returns['BM_Returns'].var()
pm_beta = round((covariance/variance),2)

bm_beta = 1


# Table of Descriptive Statistics - Sample Data

head = ['<b>Statistic<b>', '<b>Optimal Portfolio<b>', '<b>Benchmark<b>']
labels = ['<b>Initial Investment<b>', '<b>Ending Investment<b>', '<b>Max Daily Return<b>',
          '<b>Min Daily Return<b>', '<b>Return<b>', '<b>Volatility<b>', '<b>Sharpe Ratio<b>', '<b>Beta<b>']
pf_stats = ['${:,}'.format(pm_start), '${:,}'.format(pm_end), '{:.2%}'.format(daily_pm_max_return), 
            '{:.2%}'.format(daily_pm_min_return), '{:.2%}'.format(pm_return), '{:.2%}'.format(pm_vol), pm_sharpe, pm_beta]
bm_stats = ['${:,}'.format(bm_start), '${:,}'.format(bm_end), '{:.2%}'.format(daily_bm_max_retrun), 
            '{:.2%}'.format(daily_bm_min_retrun), '{:.2%}'.format(bm_return), '{:.2%}'.format(bm_vol), bm_sharpe, bm_beta]

fig = go.Figure(data=[go.Table(
    header=dict(values=head,
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[labels, pf_stats, bm_stats],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(margin=dict(l=5, r=5, b=0,t=10))

fig.show()


# Plot Daily Returns Optimal Portfolio vs. Benchmark - Out of Sample Data

fig = px.scatter(one_year_historic_daily_returns, x='BM_Returns', y='Portfolio_returns', 
                 title='Daily Returns Optimal Portfolio vs. Benchmark - Sample Data')

fig.update_layout(template=theme)
fig.show()





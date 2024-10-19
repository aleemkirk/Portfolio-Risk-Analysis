import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from operator import itemgetter

# Get investment portfolio metrics

class PortfolioMetrics:

    def __init__(self, securities, weights = None ,market = None, start_date = None, end_date = None, trading_days = None) -> None:

        self.securities = securities
        self.num_securities = len(securities)
        self.weights = weights or [1/self.num_securities for _ in range(self.num_securities)] #assume equal investment if weights are not specified
        self.market = market or 'NASDAQ_COMP'
        self.start_date = start_date or '2014-10-08'
        self.end_date = end_date or '2024-08-27'
        self.trading_days = trading_days or 251
        self.portfolio = self.securities + [self.market]
        self.path = 'data/'
        self.data = pd.DataFrame(data = None)
        self.daily_ROR = pd.DataFrame(data = None)
        self.mean_daily_ROR = pd.Series(data = None)
        self.cov_daily_ROR = pd.DataFrame(data = None)
        self.beta = pd.Series(data = None)

         # Error handling
        if len(securities) != len(weights):
            raise Exception('Investment weights and securities are not the same length')
        
        if np.sum(weights).round(decimals = 1) != 1.0:
            raise Exception('Investment weights must sum to 1')

    # read securities data into a DataFrame
    def getData(self) -> pd.DataFrame:

        self.data = pd.DataFrame(columns=['date'])
        try:
            for ticker in self.portfolio:
                df = pd.read_csv(self.path+ticker+'.csv')
                self.data = pd.merge(self.data, df[['date', 'close']], on='date', how='outer', suffixes=(ticker, ticker))
        except:
            print(f"Invalid security ticker given: {ticker}")
            self.data = pd.DataFrame(data=None)
            return
        else:
            #rename column names with Ticker values
            self.data.columns = ['date'] + [i for i in self.portfolio]

            #convert dates to datetime
            self.data['date'] = pd.to_datetime(self.data['date'])

            #sort dates
            self.data = self.data.sort_values(by='date')

            # Reset the index
            self.data.reset_index(drop=True, inplace=True)

            #drop any rows with NaNs
            #self.data = self.data.dropna()

            #filter data from start and end dates
            self.data = self.data[(self.data['date'] >= self.start_date) & (self.data['date'] <= self.end_date)]

        return self.data 
    
    # compute daily rate of return (ROR)
    def dailyROR(self) -> pd.DataFrame:

        if self.data.empty:
            raise Exception("Securities data does not exist. Try calling getData() first.")

        self.daily_ROR[self.portfolio] = (self.data[self.portfolio].diff()/self.data[self.portfolio].shift(1))*100
        self.daily_ROR['date'] = self.data['date']
        return self.daily_ROR

    # compute mean average return for each security 
    def meanDailyROR(self) -> pd.Series:
        
        if self.data.empty:
            raise Exception("Securities data does not exist. Try calling getData() first.")
        if self.daily_ROR.empty:
            raise Exception("Return information does not exist. Try calling meanDailyROR() first.")
        
        self.mean_daily_ROR = self.daily_ROR[self.portfolio].mean(axis=0)
        return self.mean_daily_ROR
    
    # compute covariance of daily returns
    def covDailyROR(self) -> pd.DataFrame:

        if self.data.empty:
            raise Exception("Securities data does not exist. Try calling getData() first.")
        if self.daily_ROR.empty:
            raise Exception("Return information does not exist. Try calling meanDailyROR() first.")

        self.cov_daily_ROR = self.daily_ROR[self.portfolio].cov()
        return self.cov_daily_ROR
    
    # calculate portfolio beta
    def elasticity(self) -> pd.Series:

        if self.data.empty:
            raise Exception("Securities data does not exist. Try calling getData() first.")
        if self.daily_ROR.empty:
            raise Exception("Return information does not exist. Try calling meanDailyROR() first.")
        if self.cov_daily_ROR.empty:
            raise Exception("Covariance information does not exist. Try calling covDailyROR() first.")
        
        self.beta = (self.cov_daily_ROR.iloc[-1, :self.num_securities])/self.daily_ROR[self.market].var(axis=0)
        self.beta = np.matmul(self.beta.to_list(), np.transpose(self.weights))
        return self.beta
    
    # compute annualized return of portfolio in %
    def annReturn(self) -> np.ndarray:

        if self.mean_daily_ROR.empty:
            raise Exception("Mean daily return does not exist. Try calling meanDailyROR() first.")
        
        self.annual_return = self.trading_days*np.matmul(self.weights, self.mean_daily_ROR[:self.num_securities].to_list()).sum()
        return self.annual_return
    
    # compute annualized risk of portfolio in %
    def annRisk(self) -> np.ndarray:

        if self.cov_daily_ROR.empty:
            raise Exception("Covariance information does not exist. Try calling covDailyROR() first.")
        self.annual_risk = np.sqrt(self.trading_days*np.matmul(np.matmul(self.weights, self.cov_daily_ROR.iloc[:self.num_securities, :self.num_securities].to_numpy()), np.transpose(self.weights)))
        return self.annual_risk

    def divIndex(self):


        if self.daily_ROR.empty:
            raise Exception("Return information does not exist. Try calling meanDailyROR() first.")
        if self.cov_daily_ROR.empty:
            raise Exception("Covariance information does not exist. Try calling covDailyROR() first.")
        
        self.div_index = np.matmul(self.daily_ROR[self.securities].std().to_numpy(), np.transpose(self.weights))/np.sqrt(np.matmul(self.weights, np.matmul(self.cov_daily_ROR.iloc[:self.num_securities, :self.num_securities].to_numpy(), np.transpose(self.weights))))
        return self.div_index

    def getMetrics(self) -> None:

        self.getData() # get data
        self.dailyROR() # compute daily ROR
        self.meanDailyROR() # compute mean daily ROR
        self.covDailyROR() # compute covariance 
        print(f'Portfolio beta: {self.elasticity():.2f}')
        print(f'Portfolio annualized return: {self.annReturn():.2f}%')
        print(f'Portfolio annualized risk: {self.annRisk():.2f}%')
        print(f'Portfolio diversification index: {self.divIndex():.2f}')

# Diversify portfolio using K-means algorithm
class PortfolioDiversifier(PortfolioMetrics):

    def __init__(self, securities, clusters = None, weights = None, market = None, start_date = None, end_date = None, trading_days = None) -> None:
        
        np.random.seed(100)
        super().__init__(securities, weights, market, start_date, end_date, trading_days)
        self.clusters = clusters or 1
        self.stock_clusters = []

    def diversify(self):

        self.getMetrics() #compute all the financial metrics and populate class attributes

        features = np.concatenate([self.mean_daily_ROR[self.securities].to_numpy().reshape(len(self.weights), 1), self.cov_daily_ROR.iloc[:self.num_securities, :self.num_securities].to_numpy()], axis=1)
        cluster = KMeans(algorithm='lloyd', max_iter=100, n_clusters=self.clusters)
        cluster.fit(features)

        self.centroids = cluster.cluster_centers_
        self.labels = cluster.labels_

        return self.labels


    #returns a list of all securities in each cluster where the list index represents the index
    def stockClusters(self) -> list:
        
        for i in range(self.clusters):
            index = np.where(self.labels == i)[0]
            x = itemgetter(*index)(self.securities)
            if isinstance(x, tuple):
                self.stock_clusters.insert(i, [*x])
            else:
                self.stock_clusters.insert(i, [x])
        
        return self.stock_clusters

    def covar(self):

        return self.cov_daily_ROR.iloc[self.labels]


    def showAttr(self) -> None:
        print(self.__dict__)
import pandas as pd
import numpy as np

# Get investment portfolio metrics

class PortfolioMetrics:

    def __init__(self, securities, weights = None ,market=None, start_date = None, end_date = None, trading_days = None) -> None:
        self.securities = securities
        self.num_securities = len(securities)
        self.weights = weights or [1/self.num_securities for _ in range(self.num_securities)] #assume equal investment if weights are not specified
        self.__market = market or 'NASDAQ_COMP'
        self.__start_date = start_date or '2014-10-08'
        self.__end_date = end_date or '2024-08-27'
        self.__trading_days = trading_days or 251
        self.portfolio = self.securities + [self.__market]
        self.__path = 'data/'
        self.data = pd.DataFrame(data = None)
        self.daily_ROR = pd.DataFrame(data = None)
        self.mean_daily_ROR = pd.Series(data = None)
        self.cov_daily_ROR = pd.DataFrame(data = None)
        self.__beta = pd.Series(data = None)

    #read securities data into a DataFrame
    def getData(self) -> pd.DataFrame:

        self.data = pd.DataFrame(columns=['date'])
        try:
            for ticker in self.portfolio:
                df = pd.read_csv(self.__path+ticker+'.csv')
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
            self.data = self.data[(self.data['date'] >= self.__start_date) & (self.data['date'] <= self.__end_date)]

        return self.data 
    
    #compute daily rate of return (ROR)
    def dailyROR(self) -> pd.DataFrame:

        if self.data.empty:
            raise Exception("Securities data does not exist. Try calling getData() first.")

        self.daily_ROR[self.portfolio] = (self.data[self.portfolio].diff()/self.data[self.portfolio].shift(1))*100
        self.daily_ROR['date'] = self.data['date']
        return self.daily_ROR

    #compute mean average return for each security 
    def meanDailyROR(self) -> pd.Series:
        
        if self.data.empty:
            raise Exception("Securities data does not exist. Try calling getData() first.")
        if self.daily_ROR.empty:
            raise Exception("Return information does not exist. Try calling meanDailyROR() first.")
        
        self.mean_daily_ROR = self.daily_ROR[self.portfolio].mean(axis=0)
        return self.mean_daily_ROR
    
    #compute covariance of daily returns
    def covDailyROR(self) -> pd.DataFrame:

        if self.data.empty:
            raise Exception("Securities data does not exist. Try calling getData() first.")
        if self.daily_ROR.empty:
            raise Exception("Return information does not exist. Try calling meanDailyROR() first.")

        self.cov_daily_ROR = self.daily_ROR[self.portfolio].cov()
        return self.cov_daily_ROR
    
    def beta(self) -> pd.Series:

        if self.data.empty:
            raise Exception("Securities data does not exist. Try calling getData() first.")
        if self.daily_ROR.empty:
            raise Exception("Return information does not exist. Try calling meanDailyROR() first.")
        if self.cov_daily_ROR.empty:
            raise Exception("Covariance information does not exist. Try calling covDailyROR() first.")
        
        self.__beta = pd.Series(self.cov_daily_ROR[self.__market])/self.daily_ROR[self.__market].var(axis=0)
        return self.__beta
    
    #return annualized return of portfolio in %
    def annReturn(self) -> np.ndarray:

        if self.mean_daily_ROR.empty:
            raise Exception("Mean daily return does not exist. Try calling meanDailyROR() first.")
        
        return self.__trading_days*np.matmul(self.weights, self.mean_daily_ROR[:self.num_securities].to_list()).sum()


# Diversify portfolio using K-means algorithm
class PortfolioDiversifier:

    def __init__(self, securities) -> None:
        self.securities = securities
        self.metrics = PortfolioMetrics(self.securities) #instance of metrics object to calcualte portfolio metrics
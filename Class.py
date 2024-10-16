import pandas as pd

class PortfolioOptimizer:

    def __init__(self, securities, market='NASDAQ_COMP', start_date = '2014-10-08', end_date = '2024-08-27') -> None:
        self.securities = securities
        self.__market = market
        self.__start_date = start_date
        self.__end_date = end_date
        self.portfolio = self.securities + [self.__market]
        self.__path = 'data/'
        self.num_securities = len(securities)
        self.data = pd.DataFrame(data = None)
        self.daily_ROR = pd.DataFrame(data = None)

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

    def meanDailyROR(self) -> pd.DataFrame:
        
        if self.data.empty:
            raise Exception("Securities data does not exist. Try calling getData() first.")
        if self.daily_ROR.empty:
            raise Exception("Return information does not exist. Try calling meanDailyROR() first.")
        
        self.mean_daily_ROR = self.daily_ROR[self.portfolio].mean(axis=0)
        return self.mean_daily_ROR


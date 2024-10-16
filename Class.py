import pandas as pd

class PortfolioOptimizer:

    def __init__(self, securities, market='NASDAQ_COMP', start_date = '2014-10-08', end_date = '2024-08-27') -> None:
        self.securities = securities
        self.__market = market
        self.__start_date = start_date
        self.__end_date = end_date
        self.portfolio = self.securities + [self.__market]
        self.__path = 'data/'

    #read securities data into a DataFrame
    def getData(self) -> pd.DataFrame:
        self.data = pd.DataFrame(columns=['date'])

        for ticker in self.portfolio:
            data = pd.read_csv(self.__path+ticker+'.csv')
            self.data = pd.merge(self.data, data[['date', 'close']], on='date', how='outer', suffixes=(ticker, ticker))

        #rename column names with Ticker values
        self.data.columns = ['date'] + [i for i in self.portfolio]

        return self.data



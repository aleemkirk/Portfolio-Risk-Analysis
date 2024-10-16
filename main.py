from Class import PortfolioOptimizer


x = ['AAOI', 'AAPL', 'TSLA']
optim = PortfolioOptimizer(securities=x)

optim.getData()
optim.dailyROR()
print(optim.meanDailyROR())


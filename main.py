from Class import PortfolioOptimizer
import pandas as pd
import numpy as np


x = ['AAOI', 'AAPL', 'TSLA', 'MSFT']
optim = PortfolioOptimizer(securities=x)

optim.getData()
optim.dailyROR()
optim.covDailyROR()
print(optim.beta())



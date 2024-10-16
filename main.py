from Class import PortfolioMetrics, PortfolioDiversifier
import pandas as pd
import numpy as np


x = ['AAOI', 'TSLA', 'MSFT', 'AAPL', 'ACAD']
optim = PortfolioMetrics(securities=x)

optim.getData()
optim.dailyROR()
optim.covDailyROR()
optim.meanDailyROR()
print(optim.annReturn())
print(optim.annRisk())
print(optim.divIndex())

print(optim.beta())





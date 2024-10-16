from Class import PortfolioMetrics, PortfolioDiversifier
import pandas as pd
import numpy as np


x = ['AAOI', 'TSLA', 'MSFT', 'AAPL', 'ACAD']
optim = PortfolioMetrics(securities=x, weights=[.1, .5, .2, .1, .1])

optim.getMetrics()





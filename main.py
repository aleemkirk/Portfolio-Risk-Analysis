from Class import PortfolioMetrics, PortfolioDiversifier
import pandas as pd
import numpy as np


w = [.1, .7, .2, .0]
x = ['AAOI', 'TSLA', 'MSFT', 'AAPL']
optim = PortfolioMetrics(securities=x, weights=w)
optim.getMetrics()





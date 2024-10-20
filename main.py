from Class import PortfolioMetrics, PortfolioDiversifier
import pandas as pd
import numpy as np


w = [.1, .5, .2, .2]
x = ['AAOI', 'TSLA', 'MSFT', 'NVDA', 'AMZN', 'META', 'CTSH', 'CRWD', 'TXN', 'GOOGL', 'DASH', 'MDB']
d = PortfolioDiversifier(securities=x, clusters=6)
d.diversify()
d.stockClusters()
d.divPortfolio()
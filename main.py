from Class import PortfolioMetrics, PortfolioDiversifier
import pandas as pd
import numpy as np


w = [.1, .5, .2, .2]
x = ['AAOI', 'TSLA', 'MSFT', 'AAPL']
d = PortfolioDiversifier(securities=x, weights=w, clusters=3)
print(d.diversify())



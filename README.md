# Introduction

This project focuses on using the K-means algorithm to diversifying securities in the NASDAQ securities exchange. Diversification is key to reducing risk and increasing potential returns in investing. The goal is to group securities into different clusters based on their mean daily rate of return and covariance with other securities. By doing so, we can identify distinct asset classes and create a more balanced portfolio.

Using K-means helps investors avoid over-concentration in any single asset class or risk category. This method ensures a broader mix of assets, potentially improving the portfolio's stability and long-term performance.


# Data

NASDAQ securities data was found on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/stock-market-data). 

# Usage

## Prerequisites

- Download the NASDAQ data from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/stock-market-data) and place all securities data in the `/data` directory. 

- Ensure `python 3.12` is installed. Older versions may work but scripts were only tested on `python 3.12`.

- Install all dependencies from `requirements.txt`.

`pip install -r requirements.txt`


## Running Script
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

- Include the ticker symbols of all securities you wish to consider in your portfolio selection in the `securities.csv` file (optional: add the investment weights of each security. Note that all weights mush sum to 1).

- Run the script from the command line 

    `python main.py --path (path to securities.csv) --num (desired portfolio size)`


## Command Line Arguments

`main.py` takes several arguments:


> --path , -p:  path to securities and investment weight data.  
--num, -n:      desired portfolio size.  
--manual, -m:   y/n value indicating if you are entering investment weights in the command line or securities.csv file.  
--weights, -w:  investment weights of each security. Must sum to 1 and length must be the same as the number of securities.


## Example

`python main.py -p securities.csv -n 2`

Assuming that 4 tickers are given, the script will output a portfolio of two securities (one from each cluster). An example output will be: 

```
Original Portfolio Metrics:
['AAOI', 'TSLA', 'MSFT', 'NVDA', 'AMZN', 'META', 'CTSH', 'CRWD', 'TXN', 'GOOGL', 'DASH', 'MDB']
Portfolio beta: 1.28
Portfolio annualized return: 29.07%
Market annualized return: 16.13%
Portfolio annualized risk: 29.06%
Portfolio diversification index: 1.48


Diversified Portfolio Metrics:
['MSFT', 'NVDA']
[0.67, 0.33]
Portfolio beta: 1.27
Portfolio annualized return: 37.47%
Market annualized return: 16.13%
Portfolio annualized risk: 32.08%
Portfolio diversification index: 1.15
```

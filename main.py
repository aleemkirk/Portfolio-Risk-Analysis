from Class import PortfolioDiversifier
import pandas as pd
import argparse


def main():

    parser = argparse.ArgumentParser(description='Add weights')
    parser.add_argument('-p', '--path', required=True,
                        type=str, help='Path to security tickers')
    parser.add_argument('-w', '--weights', nargs='+',  required=False,
                        type=int, default=None, help='Indicate investment weights in order')
    parser.add_argument('-n', '--num',  required=True,
                        type=int, help='Desired portfolio size')

    args = parser.parse_args()

    path = args.path
    weights = args.weights
    clusters = args.num

    securities = pd.read_csv(path)['TICKERS'].tolist()

    d = PortfolioDiversifier(securities=securities,
                             weights=weights, clusters=clusters)
    
    print('\nDefault Portfolio Metrics:')
    d.diversify()
    d.stockClusters()
    d.divPortfolio()


if __name__ == '__main__':
    main()

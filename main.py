from Class import PortfolioDiversifier
import pandas as pd
import argparse


def main():

    parser = argparse.ArgumentParser(description='Add weights')
    parser.add_argument('-p', '--path', required=True,
                        type=str, help='Path to security tickers')
    parser.add_argument('-m', '--manual', required=False,
                        type=str, default='n', help='Enter investment weights manually if True')
    parser.add_argument('-w', '--weights', nargs='+',  required=False,
                        type=int, default=None, help='Indicate investment weights in order')
    parser.add_argument('-n', '--num',  required=True,
                        type=int, help='Desired portfolio size')

    args = parser.parse_args()

    man_weights = args.manual
    path = args.path
    clusters = args.num
    data = pd.read_csv(path)
    securities = data['TICKERS'].tolist()

    if man_weights.lower() == 'y':
        weights = args.weights
    elif man_weights.lower() == 'n':
        weights = data['WEIGHTS'].tolist()
    else:
        raise Exception('enter y/n for manual flag.')

    d = PortfolioDiversifier(securities=securities,
                             weights=weights, clusters=clusters)
    
    print('\n\nOriginal Portfolio Metrics:')
    d.diversify()
    d.stockClusters()

    print('\n\nDiversified Portfolio Metrics:')
    d.divPortfolio()


if __name__ == '__main__':
    main()

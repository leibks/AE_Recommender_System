import os
import sys
path = os.getcwd()
sys.path.append(path)
from src.modules import *


if __name__ == '__main__':
    # set up the original testing data
    beauty = pd.read_csv('resource/cleaned_data/Luxury_Beauty_stock.csv')
    toy = pd.read_csv('resource/cleaned_data/Toys_&_Games_stock.csv')
    fashion = pd.read_csv('resource/cleaned_data/AMAZON_FASHION_stock.csv')
    beauty = beauty.sort_values(by=['Date'])
    beauty = beauty[beauty['overall'] != 0]
    beauty_test = beauty[['overall', 'reviewerID', 'asin']].iloc[-100:].reset_index(drop=True)

    toy = toy.sort_values(by=['Date'])
    toy = toy[toy['overall'] != 0]
    fashion = fashion.sort_values(by=['Date'])
    fashion = fashion[toy['overall'] != 0]
    toy_test = toy[['overall', 'reviewerID', 'asin']].iloc[-1000:].reset_index(drop=True)
    fashion_test = fashion[['overall', 'reviewerID', 'asin']].reset_index(drop=True)
    reviewer = [i for i in Test.reviewerID]
    product = [i for i in Test.asin]
    pair = list(zip(reviewer, product))

    m = SystemModule()

import os
import sys
import matplotlib.pyplot as plt
from collections import Counter

path = os.getcwd()
sys.path.append(path)
from src.modules import *


if __name__ == "__main__":

    # set up the original testing data (draw figures based on generated results)
    beauty = pd.read_csv("resource/cleaned_data/Luxury_Beauty_stock.csv")
    toy = pd.read_csv("resource/cleaned_data/Toys_&_Games_stock.csv")
    fashion = pd.read_csv("resource/cleaned_data/amazon_fashion_stock.csv")

    beauty_item = pd.read_csv("resource/performance_test_data/Beauty_Item_P.csv")
    beauty_user = pd.read_csv("resource/performance_test_data/Beauty_User_P.csv")
    beauty_content = pd.read_csv("resource/performance_test_data/Beauty_Content_P.csv")

    fashion_item = pd.read_csv("resource/performance_test_data/Fashion_Item_P.csv")
    fashion_user = pd.read_csv("resource/performance_test_data/Fashion_User_P.csv")
    fashion_content = pd.read_csv(
        "resource/performance_test_data/Fashion_Content_P.csv"
    )

    def visual(raw, performance):

        asin = list(raw.asin)
        title = list(raw.title)
        length = len(asin)
        product_dict = {asin[i]: title[i] for i in range(length)}
        product_ids = performance["Predicts"]

        products = []
        for i in product_ids:
            temp = []
            for j in i.split(",")[:-2]:
                if len(j) == 10:
                    temp.append(product_dict[j])
            products.append(temp)

        result = {}
        for i in products:
            C = dict(Counter(i))
            for k, v in C.items():
                if k in result:
                    result[k] += v
                else:
                    result[k] = v

        # sort dict by descending order
        result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

        if len(result) <= 10:
            plt.bar(range(len(result)), result.values(), color="g")
            plt.show()
            for i in range(len(result)):
                print(i, ":", list(result.keys())[i])
        else:
            plt.bar(range(10), list(result.values())[:10], color="g")
            plt.show()
            for i in range(10):
                print(i, ":", list(result.keys())[i])

        visual(beauty, beauty_item)
        visual(beauty, beauty_user)
        visual(beauty, beauty_content)
        visual(fashion, fashion_item)
        visual(fashion, fashion_user)
        # visual(fashion, fashion_content)

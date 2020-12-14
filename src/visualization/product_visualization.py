import os
import sys
import matplotlib.pyplot as plt
from collections import Counter

path = os.getcwd()
sys.path.append(path)
from src.modules import *


if __name__ == "__main__":
    m = RMSystemModel()

    # set up the original testing data
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

    def Visual(raw, performance):

        asin = list(raw.asin)
        title = list(raw.title)
        L = len(asin)
        product_dict = {asin[i]: title[i] for i in range(L)}
        product_IDs = performance["Predicts"]

        products = []
        for i in product_IDs:
            temp = []
            for j in i.split(",")[:-2]:
                if len(j) == 10:
                    temp.append(product_dict[j])
            products.append(temp)

        Res = {}
        for i in products:
            C = dict(Counter(i))
            for k, v in C.items():
                if k in Res:
                    Res[k] += v
                else:
                    Res[k] = v

        # sort dict by descending order
        Res = dict(sorted(Res.items(), key=lambda item: item[1], reverse=True))

        if len(Res) <= 10:
            plt.bar(range(len(Res)), Res.values(), color="g")
            plt.show()
            for i in range(len(Res)):
                print(i, ":", list(Res.keys())[i])
        else:
            plt.bar(range(10), list(Res.values())[:10], color="g")
            plt.show()
            for i in range(10):
                print(i, ":", list(Res.keys())[i])

        Visual(beauty, beauty_item)
        Visual(beauty, beauty_user)
        Visual(beauty, beauty_content)
        Visual(fashion, fashion_item)
        Visual(fashion, fashion_user)
        # Visual(fashion, fashion_content)

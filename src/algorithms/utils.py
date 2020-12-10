import numpy as np
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def fetch_users_products(df, algo="user"):
    # if the algo is user-user, dic is the user dictionary
    # if the algo is item-item, dic is the product dictionary
    dic = {}

    user_id_list = df["reviewerID"].tolist()
    product_id_list = df["asin"].tolist()
    users = set()
    products = set()

    for user_id in tqdm(user_id_list, desc="Build User List Loading ...."):
        if user_id not in users:
            users.add(user_id)

    for product in tqdm(product_id_list, desc="Build Product List Loading ...."):
        if product not in products:
            products.add(product)

    # set up related dictionary
    start_index = 0
    if algo == "user":
        for product in tqdm(products, desc="Build Product Dictionary Loading ...."):
            dic[product] = start_index
            start_index += 1
    elif algo == "item":
        for user in tqdm(users, desc="Build User Dictionary Loading ...."):
            dic[user] = start_index
            start_index += 1

    return [list(users), list(products), dic]


# clean up the given format of the price and return the value of the price
def clean_price(price):
    if price[:1] != '$':
        return 0
    if not isinstance(price, float):
        if "-" in price:
            prices = price.split(" - ")
            price = (float(prices[0][1:].replace(",", "")) + float(prices[1][1:].replace(",", ""))) / 2
        else:
            price = float(price[1:].replace(",", ""))
    return price


# Identify the the high/low price for all products based on given rate,
# which means any price that is higher/lower than given percentage of all prices
# will be regarded as the high/low price product::
# Args:
#     prices (A list of floats): a list of prices in the dataset
#     high_rate (float): the percentage that determine the threshold of high-price product
#     low_rate (float): the percentage that determine the threshold of low-price product
# Returns: [identified high value, identified low value]
def identify_price_in_items(prices, high_rate, low_rate):
    unique_prices = []
    seen = set()
    for price in tqdm(prices, desc="Identify prices Loading ...."):
        price = clean_price(price)
        if price != 0 and price not in seen:
            seen.add(price)
            unique_prices.append(price)

    unique_prices.sort()
    high_threshold = math.floor(len(unique_prices) * high_rate)
    low_threshold = math.floor(len(unique_prices) * low_rate)
    high_price = unique_prices[high_threshold]
    low_price = unique_prices[low_threshold]

    return [high_price, low_price]


# calculate the economic factor that affects the utility of the product to the user
def get_economic_factor(stock_rate, price, rate, high_price, low_price):
    if price == 0:
        return 0
    if (price >= high_price and stock_rate < 0) or (price <= low_price and stock_rate > 0):
        return rate * abs(stock_rate) * 10
    else:
        return 0

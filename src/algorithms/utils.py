import numpy as np
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# key: product_id, value: index in each user's product_list
PRODUCT_DICT = {}
# key: user_name, value: index in each product's buyers_list
USER_DICT = {}
# any product's price is higher than it will be regard as the high price product
HIGH_PRICE = 0
# any product's price is lower than it will be regard as the low price product
LOW_PRICE = 0


def fetch_users_products(df, algo="user"):
    global PRODUCT_DICT
    global USER_DICT

    username_list = df["reviewerName"].tolist()
    product_id_list = df["asin"].tolist()
    users = set()
    products = set()

    for user_name in username_list:
        if user_name not in users:
            users.add(user_name)

    for product in product_id_list:
        if product not in products:
            products.add(product)

    start_index = 0
    if algo == "user":
        for product in products:
            PRODUCT_DICT[product] = start_index
            start_index += 1
    elif algo == "item":
        for user in users:
            USER_DICT[user] = start_index
            start_index += 1

    return [list(users), list(products)]


# Identify the the high/low price for all products based on given rate,
# which means any price that is higher/lower than given percentage of all prices
# will be regarded as the high/low price product::
# Args:
#     prices (A list of floats): a list of prices in the dataset
#     high_rate (float): the percentage that determine the threshold of high-price product
#     low_rate (float): the percentage that determine the threshold of low-price product
# Returns:
#     None
def identify_price_in_items(prices, high_rate, low_rate):
    global HIGH_PRICE
    global LOW_PRICE

    unique_prices = []
    seen = set()
    for price in prices:
        if not isinstance(price, float):
            if price[:1] != '$':
                continue
            price = float(price[1:])
        if price not in seen:
            seen.add(price)
            unique_prices.append(price)

    unique_prices.sort()
    high_threshold = math.floor(len(unique_prices) * high_rate)
    low_threshold = math.floor(len(unique_prices) * low_rate)
    HIGH_PRICE = unique_prices[high_threshold]
    LOW_PRICE = unique_prices[low_threshold]

    return [HIGH_PRICE, LOW_PRICE]


def get_economic_factor(stock_rate, price, rate):
    if price == 0:
        return 0
    if (price >= HIGH_PRICE and stock_rate < 0) or (price <= LOW_PRICE and stock_rate > 0):
        return rate * abs(stock_rate) * 10
    else:
        return 0

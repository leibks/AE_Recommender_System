import numpy as np
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from .lsh_for_cosine_similarity import *


def fetch_users_products(df):
    user_id_list = df["reviewerID"].tolist()
    product_id_list = df["asin"].tolist()
    users = set(user_id_list)
    products = set(product_id_list)

    return users, products


def build_dictionary(users, products, algo="user"):
    # if the algo is user-user, dic is the product dictionary
    # if the algo is item-item, dic is the user dictionary
    dic = {}
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
    return dic


# clean up the given format of the price and return the value of the price
def clean_price(price):
    if isinstance(price, float):
        return price
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


# reduce the size of the matrix with help of content-based algorithms
# and locality sensitive hashing for collaborative filtering algorithm
def reduce_matrix(user_ids, product_ids, review_text_dict, user_profiles_dict, feature_size, algo):
    if algo == "user":
        lsh_algo_user = LSH(user_profiles_dict, feature_size, hash_size=3, num_tables=2)
        user_ids = lsh_algo_user.find_big_clusters_items()
    elif algo == "item":
        lsh_algo_item = LSH(review_text_dict, feature_size, hash_size=3, num_tables=1)
        product_ids = lsh_algo_item.find_big_clusters_items()
    return user_ids, product_ids

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


def build_dictionary(users, products):
    user_dic = {}
    product_dic = {}
    # set up related dictionary
    start_index = 0
    for product in tqdm(products, desc="Build Product Dictionary Loading ...."):
        product_dic[product] = start_index
        start_index += 1
    start_index = 0
    for user in tqdm(users, desc="Build User Dictionary Loading ...."):
        user_dic[user] = start_index
        start_index += 1
    return user_dic, product_dic


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
        return rate * abs(stock_rate) * 3
    else:
        return 0


# reduce the size of the matrix with help of content-based algorithms
# and locality sensitive hashing for collaborative filtering algorithm
def reduce_matrix(review_text_dict, feature_size, hash_size, num_tables):
    lsh_algo_item = LSH(review_text_dict, feature_size, hash_size=hash_size, num_tables=num_tables)
    product_ids = lsh_algo_item.find_big_clusters_items()
    return product_ids


def calculate_estimate_rate(review_text_dict, given_item, sim_items, user_id, utility_matrix,
                            product_dic, user_dict, algo):
    sum_rate = 0
    sum_sim = 0
    given_item_vec = np.array([review_text_dict[given_item]])
    for i_id in sim_items:
        compare_product_vec = np.array([review_text_dict[i_id]])
        cos_sim_value = cosine_similarity(given_item_vec, compare_product_vec).item(0)
        utility = 0
        if algo == "user":
            utility = utility_matrix[user_id][product_dic[i_id]]
        elif algo == "item":
            utility = utility_matrix[i_id][user_dict[user_id]]
        sum_rate += cos_sim_value * utility
        sum_sim += cos_sim_value

    if sum_sim == 0:
        return 0
    else:
        return sum_rate / sum_sim


# estimate utilities of users rates to products
# (these products are similar to rated products by users)
def fill_estimated_rates(review_text_dict, feature_size, rated_products, utility_matrix, product_dic,
                         user_dict, algo, hash_size, num_tables):
    lsh_algo_item = LSH(review_text_dict, feature_size, hash_size=hash_size, num_tables=num_tables)
    for user_id in tqdm(rated_products.keys(), desc="Estimate Utilities Loading ...."):
        rated_list = [i for i in rated_products[user_id]]
        if len(rated_list) == 0:
            continue
        for product_id in product_dic.keys():
            if algo == "user":
                product_idx = product_dic[product_id]
                if utility_matrix[user_id][product_idx] == 0:
                    sim_items = lsh_algo_item.filter_similar_items(product_id, rated_list)
                    if len(sim_items) > 0:
                        utility_matrix[user_id][product_idx] = calculate_estimate_rate(
                            review_text_dict, product_id, sim_items, user_id, utility_matrix,
                            product_dic, user_dict, algo)
                        rated_products[user_id].append(product_id)
            elif algo == "item":
                if utility_matrix[product_id][user_dict[user_id]] == 0:
                    sim_items = lsh_algo_item.filter_similar_items(product_id, rated_list)
                    if len(sim_items) > 0:
                        utility_matrix[product_id][user_dict[user_id]] = calculate_estimate_rate(
                            review_text_dict, product_id, sim_items, user_id, utility_matrix,
                            product_dic, user_dict, algo)
                        rated_products[user_id].append(product_id)

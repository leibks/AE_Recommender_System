import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.algorithms.utils import (
    USER_DICT,
    fetch_users_products,
    identify_price_in_items,
    get_economic_factor
)
from src.algorithms.lsh_for_cosine_similarity import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--USER", type=str, default=None, help="the user who is recommended")
parser.add_argument("--TOP_ITEM", type=int, default=10, help="how many items provided for recommendation")
parser.add_argument("--HIGH_VALUE", type=float, default=0.9, help="identify high value products")
parser.add_argument("--LOW_VALUE", type=float, default=0.1, help="identify low value products")
parser.add_argument("--ECO", type=str, default="True", help="consider economic factors")
parser.add_argument("--LSH", type=str, default="True", help="whether use the locality sensitive hashing")

args = parser.parse_args()


# ==================================== Set up matrix ====================================
# key: product_id, value: a list of users
# (index in the list points to the related user name by using USER_DICT)
def build_item_matrix(users, products):
    matrix = {}
    for product in products:
        matrix[product] = []
        for user in users:
            matrix[product].append(0)

    return matrix


# utility (value) consider the rate from 0 to 5 and economic factor
# , each row represent a product, and each column represent a user
def build_item_utility_matrix(utility_matrix, df, consider_economic=False):
    for index in df.index:
        name = df["reviewerName"][index]
        product_id = df["asin"][index]
        rate = df["overall"][index]
        stock_rate = df["stockReturn"][index]
        price = df["price"][index]
        if not isinstance(price, float):
            if price[:1] != '$':
                price = 0
            else:
                price = float(price[1:])
        # if the stock decreased and price of this product was high,
        # It means that the user really likes the product as it brings
        # higher utility on top of the price (economic) effect
        if consider_economic:
            economic_factor = get_economic_factor(stock_rate, price, rate)
        else:
            economic_factor = 0
        utility_matrix[product_id][USER_DICT[name]] = rate + economic_factor


# value is the (rate to the product given by one user - average rate for this product),
# each row represent a product's a list of users
def build_item_similarity_matrix(similarity_matrix, utility_matrix, user_names):
    for product_id in similarity_matrix.keys():
        users = utility_matrix[product_id]
        sum_rates = 0
        length = 0
        for rate in users:
            if rate > 0:
                sum_rates += rate
                length += 1
        average = sum_rates / length
        for user_name in user_names:
            if utility_matrix[product_id][USER_DICT[user_name]] != 0:
                similarity_matrix[product_id][USER_DICT[user_name]] \
                    = utility_matrix[product_id][USER_DICT[user_name]] - average


def set_up_item_matrix():
    df = pd.read_csv('resource/sample_data/joined_sample_electronics.csv')
    users_products = fetch_users_products(df, "item")
    identify_price_in_items(df["price"].tolist(), args.HIGH_VALUE, args.LOW_VALUE)
    users = users_products[0]
    product_ids = users_products[1]
    utility_matrix = build_item_matrix(users, product_ids)
    similarity_matrix = build_item_matrix(users, product_ids)
    build_item_utility_matrix(utility_matrix, df, True if args.ECO == "True" else False)
    build_item_similarity_matrix(similarity_matrix, utility_matrix, users)
    return [utility_matrix, similarity_matrix]
# ==================================== Set up matrix ====================================


# ==================================== Normal method to find similar items ====================================
def find_similar_items(product_id, similarity_matrix):
    similar_products = {}
    given_product_vec = np.array([similarity_matrix[product_id]])
    for product in similarity_matrix.keys():
        if product != product_id:
            compare_product_vec = np.array([similarity_matrix[product]])
            cos_sim_value = cosine_similarity(given_product_vec, compare_product_vec).item(0)
            if cos_sim_value > 0:
                similar_products[product] = cos_sim_value

    similar_res = sorted(similar_products.items(), key=lambda item: item[1], reverse=True)

    return similar_res


def predict_single_product_utility(utility_matrix, similar_products, user_name):
    # ∑_(𝒋∈𝑵(𝒊;𝒙))〖𝒔_𝒊𝒋⋅𝒓_𝒋𝒙 〗, i is the predicted product,
    # x is the predicted user, j is similar product
    sum_weights = 0
    # ∑𝒔_𝒊𝒋  (s_ij --> similarity of all similar products with the selected product)
    sum_similarity = 0
    for similar in similar_products:
        utility = utility_matrix[similar[0]][USER_DICT[user_name]]
        sum_weights += similar[1] * utility
        sum_similarity += similar[1]

    if sum_similarity == 0:
        return 0

    return sum_weights / sum_similarity


def find_recommended_products(user_name, utility_matrix, similarity_matrix):
    idx = USER_DICT[user_name]
    all_product_utilities = {}
    for product_id in similarity_matrix.keys():
        if utility_matrix[product_id][idx] == 0:
            similar_products = find_similar_items(product_id, similarity_matrix)
            utility_matrix[product_id][idx] = \
                predict_single_product_utility(utility_matrix, similar_products, user_name)

        all_product_utilities[product_id] = utility_matrix[product_id][idx]

    sort_products = sorted(all_product_utilities.items(), key=lambda item: item[1], reverse=True)
    recommended_product = []
    for i in sort_products:
        print(i)
        recommended_product.append(i[0])
        if len(recommended_product) > args.TOP_ITEM:
            break

    return recommended_product

# ==================================== Normal method to find similar items ====================================


# ==================================== LSH method to find similar items ====================================
def find_recommended_products_by_lsh(user_name, utility_matrix, similarity_matrix):
    # index of this user in every product's user list
    idx = USER_DICT[user_name]
    all_product_utilities = {}
    lsh_algo = LSH(similarity_matrix, len(USER_DICT))
    for product_id in utility_matrix:
        if utility_matrix[product_id][idx] == 0:
            similarity_dic = lsh_algo.build_similar_dict(product_id)
            # print(similarity_dic)
            # ∑_(𝒋∈𝑵(𝒊;𝒙))〖𝒔_𝒊𝒋⋅𝒓_𝒋𝒙 〗, i is the predicted product,
            # x is the predicted user, j is similar product
            sum_weights = 0
            # ∑𝒔_𝒊𝒋  (s_ij --> similarity of all similar products with the selected product)
            sum_similarity = 0
            for sim_item in similarity_dic.keys():
                sim_val = similarity_dic[sim_item]
                utility = utility_matrix[sim_item][idx]
                sum_weights += sim_val * utility
                sum_similarity += sim_val

            if sum_similarity == 0:
                utility_matrix[product_id][idx] = 0
            else:
                utility_matrix[product_id][idx] = sum_weights / sum_similarity

        all_product_utilities[product_id] = utility_matrix[product_id][idx]

    sort_products = sorted(all_product_utilities.items(), key=lambda item: item[1], reverse=True)
    recommended_product = []
    for i in sort_products:
        print(i)
        recommended_product.append(i[0])
        if len(recommended_product) > args.TOP_ITEM:
            break

    return recommended_product
# ==================================== LSH method to find similar items ====================================


def item_collaborative_filter():
    set_up = set_up_item_matrix()
    utility_matrix = set_up[0]
    similarity_matrix = set_up[1]

    # store the similarity matrix into the file as the model for saving training time

    # find k recommended products
    if args.LSH == "True":
        recommended_products = find_recommended_products_by_lsh(args.USER, utility_matrix, similarity_matrix)
    else:
        recommended_products = find_recommended_products(args.USER, utility_matrix, similarity_matrix)
    print(recommended_products)


item_collaborative_filter()

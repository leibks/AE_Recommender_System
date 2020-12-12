from tqdm import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.algorithms.utils import (
    get_economic_factor,
    clean_price
)
from .lsh_for_cosine_similarity import *


# ==================================== Set up matrix ====================================
# key: user_id, value: a list of values
# (index in the list points to the related product id by using PRODUCT_DICT)
def build_user_matrix(users, products):
    matrix = {}
    for user in tqdm(users, desc="Build Initial User Matrix Loading ...."):
        matrix[user] = []
        for i in products:
            matrix[user].append(0)

    return matrix


# utility (value) consider the rate from 0 to 5 and economic factor
# , each row represent a user, and each column represent a product
def build_user_utility_matrix(utility_matrix, df, product_dic, high_price, low_price, consider_economic=False):
    for index in tqdm(df.index, desc="Build Utility Matrix Loading ...."):
        user_id = df["reviewerID"][index]
        product_id = df["asin"][index]
        if user_id not in utility_matrix:
            continue
        rate = df["overall"][index]
        if not isinstance(rate, float):
            rate = float(rate)
        # if the stock decreased and price of this product was high,
        # It means that the user really likes the product as it brings
        # higher utility on top of the price (economic) effect
        if consider_economic:
            price = clean_price(df["price"][index])
            stock_rate = df["stockReturn"][index]
            economic_factor = get_economic_factor(stock_rate, price, rate, high_price, low_price)
        else:
            economic_factor = 0
        utility_matrix[user_id][product_dic[product_id]] = rate + economic_factor


# value is the (rate to the product for the user - average rate for the user),
# each row represent a user's a list of products
def build_user_similarity_matrix(similarity_matrix, utility_matrix, product_ids, product_dic):
    for user_id in tqdm(similarity_matrix.keys(), desc="Build Sim Matrix Loading ...."):
        products = utility_matrix[user_id]
        sum_rates = 0
        length = 0
        for rate in products:
            if rate > 0:
                sum_rates += rate
                length += 1
        if length == 0:
            continue
        average = sum_rates / length
        for product_id in product_ids:
            if utility_matrix[user_id][product_dic[product_id]] != 0:
                similarity_matrix[user_id][product_dic[product_id]] \
                    = utility_matrix[user_id][product_dic[product_id]] - average
# ==================================== Set up matrix ====================================


# ==================================== Normal method to find similar items ====================================
def find_similar_users(user_id, similarity_matrix):
    print(similarity_matrix)
    similar_users = {}
    given_user_vec = np.array([similarity_matrix[user_id]])
    for user in similarity_matrix.keys():
        if user != user_id:
            compare_user_vec = np.array([similarity_matrix[user]])
            cos_sim_value = cosine_similarity(given_user_vec, compare_user_vec).item(0)
            if cos_sim_value > 0:
                similar_users[user] = cos_sim_value

    similar_res = sorted(similar_users.items(), key=lambda item: item[1], reverse=True)

    return similar_res


def predict_single_product_utility(utility_matrix, similar_res, product_id, product_dic):
    # ∑_(𝑦∈𝑁)〖𝑠_𝑥𝑦⋅𝑟_𝑦𝑖 〗, i is the product,
    # y is every similar user, x is the predicted user
    sum_weights = 0
    # ∑_(𝑦∈𝑁)[𝑠_𝑥𝑦]
    sum_similarity = 0
    for similar in similar_res:
        utility = utility_matrix[similar[0]][product_dic[product_id]]
        sum_weights += similar[1] * utility
        sum_similarity += similar[1]

    if sum_similarity == 0:
        return 0

    return sum_weights / sum_similarity


def find_recommended_products_by_uu(user_id, utility_matrix, similarity_matrix, product_dic, num_recommend):
    # find top k similar users for the given user
    similar_users = find_similar_users(user_id, similarity_matrix)

    all_product_utilities = {}
    for product_id in tqdm(product_dic.keys(), desc="Find Recommendation Loading ...."):
        idx = product_dic[product_id]
        if utility_matrix[user_id][idx] == 0:
            utility_matrix[user_id][idx] \
                = predict_single_product_utility(utility_matrix, similar_users, product_id, product_dic)

        all_product_utilities[product_id] = utility_matrix[user_id][idx]

    sort_products = sorted(all_product_utilities.items(), key=lambda item: item[1], reverse=True)
    recommended_product = []
    for i in sort_products:
        print(i)
        recommended_product.append(i[0])
        if len(recommended_product) > num_recommend:
            break

    return recommended_product
# ==================================== Normal method to find similar items ====================================


# ==================================== LSH method to find similar items ====================================
def find_recommended_products_by_uu_lsh(user_id, utility_matrix, lsh_algo, product_dic, num_recommend):
    all_product_utilities = {}
    similarity_dic = lsh_algo.build_similar_dict(user_id)

    for product_id in tqdm(product_dic.keys(), desc="Find Recommendation(LSH) Loading ...."):
        # index of this product in every user's product list
        idx = product_dic[product_id]
        if utility_matrix[user_id][idx] == 0:
            # ∑_(𝑦∈𝑁)〖𝑠_𝑥𝑦⋅𝑟_𝑦𝑖 〗, i is the product,
            # y is every similar user, x is the predicted user
            sum_weights = 0
            # ∑_(𝑦∈𝑁)[𝑠_𝑥𝑦]
            sum_similarity = 0
            for sim_user in similarity_dic.keys():
                sim_val = similarity_dic[sim_user]
                utility = utility_matrix[sim_user][idx]
                sum_weights += sim_val * utility
                sum_similarity += sim_val

            if sum_similarity == 0:
                utility_matrix[user_id][idx] = 0
            else:
                utility_matrix[user_id][idx] = sum_weights / sum_similarity

        all_product_utilities[product_id] = utility_matrix[user_id][idx]

    sort_products = sorted(all_product_utilities.items(), key=lambda item: item[1], reverse=True)
    recommended_product = []
    for i in sort_products:
        print(i)
        recommended_product.append(i[0])
        if len(recommended_product) > num_recommend:
            break

    return recommended_product
# ==================================== LSH method to find similar items ====================================

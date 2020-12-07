import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.algorithms.utils import (
    TOP_K_USERS,
    TOP_K_PRODUCTS,
    PRODUCT_DICT,
    fetch_users_products
)


# key: user_name, value: a list of values
# (index in the list points to the product id by PRODUCT_DICT)
def build_user_matrix(users, products):
    matrix = {}
    for user in users:
        matrix[user] = []
        for i in products:
            matrix[user].append(0)

    return matrix


# utility (value) consider the rate from 0 to 5 and economic factor
# , each row represent a user, and each column represent a product
def build_user_utility_matrix(utility_matrix, df, consider_economic=False):
    for index in df.index:
        name = df["reviewerName"][index]
        product_id = df["asin"][index]
        rate = df["overall"][index]
        # if the stock decreased and price of this product was high,
        # It means that the user really likes the product as it brings
        # higher utility on top of the price (economic) effect
        if consider_economic:
            economic_factor = 0
        else:
            economic_factor = 0
        utility_matrix[name][PRODUCT_DICT[product_id]] = rate + economic_factor


# value is the (rate to the product for the user - average rate for the user),
# each row represent a user, and each column represent a product
def build_user_similarity_matrix(similarity_matrix, utility_matrix, product_ids):
    for name in similarity_matrix.keys():
        products = utility_matrix[name]
        sum_rates = 0
        length = 0
        for rate in products:
            if rate > 0:
                sum_rates += rate
                length += 1
        average = sum_rates / length
        for product_id in product_ids:
            if utility_matrix[name][PRODUCT_DICT[product_id]] != 0:
                similarity_matrix[name][PRODUCT_DICT[product_id]] \
                    = utility_matrix[name][PRODUCT_DICT[product_id]] - average


def set_up_user_matrix():
    df = pd.read_csv('resource/sample_data/sample_electronics.csv')
    users_products = fetch_users_products(df)
    users = users_products[0]
    product_ids = users_products[1]
    utility_matrix = build_user_matrix(users, product_ids)
    similarity_matrix = build_user_matrix(users, product_ids)
    build_user_utility_matrix(utility_matrix, df)
    build_user_similarity_matrix(similarity_matrix, utility_matrix, product_ids)
    return [utility_matrix, similarity_matrix]


def find_similar_users(user_name, similarity_matrix):
    similar_users = {}
    given_user_vec = np.array([similarity_matrix[user_name]])
    for user in similarity_matrix.keys():
        if user != user_name:
            compare_user_vec = np.array([similarity_matrix[user]])
            cos_sim_value = cosine_similarity(given_user_vec, compare_user_vec).item(0)
            if cos_sim_value > 0:
                similar_users[user] = cos_sim_value

    similar_res = sorted(similar_users.items(), key=lambda item: item[1], reverse=True)

    return similar_res[:TOP_K_USERS] if len(similar_res) > TOP_K_USERS else similar_res


def predict_single_product_utility(utility_matrix, similar_res, product_id):
    # ∑_(𝑦∈𝑁)〖𝑠_𝑥𝑦⋅𝑟_𝑦𝑖 〗, i is the product,
    # y is every similar user, x is the predicted user
    sum_weights = 0
    # ∑_(𝑦∈𝑁)[𝑠_𝑥𝑦]
    sum_similarity = 0
    for similar in similar_res:
        utility = utility_matrix[similar[0]][PRODUCT_DICT[product_id]]
        sum_weights += similar[1] * utility
        sum_similarity += similar[1]

    if sum_similarity == 0:
        return 0

    return sum_weights / sum_similarity


def find_top_k_recommended_products(user_name, utility_matrix, similarity_matrix):
    # find top k similar users for the given user
    similar_users = find_similar_users(user_name, similarity_matrix)

    all_product_utilities = {}
    for product_id in PRODUCT_DICT.keys():
        idx = PRODUCT_DICT[product_id]
        if utility_matrix[user_name][idx] == 0:
            utility_matrix[user_name][idx] \
                = predict_single_product_utility(utility_matrix, similar_users, product_id)

        all_product_utilities[product_id] = utility_matrix[user_name][idx]

    sort_products = sorted(all_product_utilities.items(), key=lambda item: item[1], reverse=True)
    recommended_product = []
    for i in sort_products:
        print(i)
        recommended_product.append(i[0])
        if len(recommended_product) > TOP_K_PRODUCTS:
            break

    return recommended_product


def user_collaborative_filter():
    set_up = set_up_user_matrix()
    utility_matrix = set_up[0]
    similarity_matrix = set_up[1]

    # store the similarity matrix into the file as the model for saving training time

    # find k recommended products
    recommended_products = find_top_k_recommended_products("ESK", utility_matrix, similarity_matrix)
    print(recommended_products)


# test case
user_collaborative_filter()


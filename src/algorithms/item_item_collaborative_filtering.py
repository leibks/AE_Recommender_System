import math
from src.algorithms.utils import (
    get_economic_factor,
    clean_price
)
from .lsh_for_cosine_similarity import *


# ==================================== Set up matrix ====================================
# key: product_id, value: a list of users
# (index in the list points to the related user id by using USER_DICT)
def build_item_matrix(users, products):
    matrix = {}
    for product in products:
        matrix[product] = []
        for i in users:
            matrix[product].append(0)

    return matrix


# utility (value) consider the rate from 0 to 5 and economic factor
# , each row represent a product, and each column represent a user
def build_item_utility_matrix(utility_matrix, df, user_dic, rated_products, high_price, low_price, consider_economic=False):
    test = []
    for index in tqdm(df.index, desc="Build Utility Matrix Loading ...."):
        user_id = df["reviewerID"][index]
        product_id = df["asin"][index]
        if product_id not in utility_matrix:
            continue
        rate = df["overall"][index]
        if not isinstance(rate, float):
            rate = float(rate)
        # test.append(rate)
        # print(f"user_id: {user_id}, product_id: {product_id}, rate: {rate}")
        # if the stock decreased and price of this product was high,
        # It means that the user really likes the product as it brings
        # higher utility on top of the price (economic) effect
        if consider_economic:
            price = clean_price(df["price"][index])
            stock_rate = df["stockReturn"][index]
            economic_factor = get_economic_factor(stock_rate, price, rate, high_price, low_price)
        else:
            economic_factor = 0
        if product_id in utility_matrix:
            utility_matrix[product_id][user_dic[user_id]] = rate + economic_factor
            # record rated products
            if user_id not in rated_products:
                rated_products[user_id] = []
            rated_products[user_id].append(product_id)


# value is the (rate to the product given by one user - average rate for this product),
# each row represent a product's a list of users
def build_item_similarity_matrix(similarity_matrix, utility_matrix, user_ids, user_dic):
    for product_id in tqdm(similarity_matrix.keys(), desc="Build Sim Matrix Loading ...."):
        users = utility_matrix[product_id]
        sum_rates = 0
        length = 0
        for rate in users:
            if rate > 0:
                sum_rates += rate
                length += 1
        if length == 0:
            continue
        average = sum_rates / length
        for user_id in user_ids:
            if utility_matrix[product_id][user_dic[user_id]] != 0:
                similarity_matrix[product_id][user_dic[user_id]] \
                    = utility_matrix[product_id][user_dic[user_id]] - average
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


def predict_single_product_utility_ii(utility_matrix, similarity_matrix, user_id, user_dic, product_id):
    similar_products = find_similar_items(product_id, similarity_matrix)
    # ∑_(𝒋∈𝑵(𝒊;𝒙))〖𝒔_𝒊𝒋⋅𝒓_𝒋𝒙 〗, i is the predicted product,
    # x is the predicted user, j is similar product
    sum_weights = 0
    # ∑𝒔_𝒊𝒋  (s_ij --> similarity of all similar products with the selected product)
    sum_similarity = 0
    for similar in similar_products:
        utility = utility_matrix[similar[0]][user_dic[user_id]]
        sum_weights += similar[1] * utility
        sum_similarity += similar[1]

    if sum_similarity == 0:
        return 0

    return sum_weights / sum_similarity


def find_recommended_products_by_ii(user_id, utility_matrix, similarity_matrix, user_dic, num_recommend):
    idx = user_dic[user_id]
    all_product_utilities = {}
    for product_id in tqdm(similarity_matrix.keys(), desc="Find Recommendation Loading ...."):
        if utility_matrix[product_id][idx] == 0:
            utility_matrix[product_id][idx] = \
                predict_single_product_utility_ii(utility_matrix, similarity_matrix, user_id, user_dic, product_id)

        all_product_utilities[product_id] = utility_matrix[product_id][idx]

    sort_products = sorted(all_product_utilities.items(), key=lambda item: item[1], reverse=True)
    recommended_product = []
    for i in sort_products:
        print(i)
        recommended_product.append(i[0])
        if len(recommended_product) >= num_recommend:
            break

    return recommended_product

# ==================================== Normal method to find similar items ====================================


# ==================================== LSH method to find similar items ====================================
def find_recommended_products_by_ii_lsh(user_id, utility_matrix, lsh_algo, user_dic, num_recommend):
    # index of this user in every product's user list
    idx = user_dic[user_id]
    all_product_utilities = {}
    for product_id in tqdm(utility_matrix.keys(), desc="Find Recommendation(LSH) Loading ...."):
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
        # print(all_product_utilities[product_id])

    sort_products = sorted(all_product_utilities.items(), key=lambda item: item[1], reverse=True)
    recommended_product = []
    for i in sort_products:
        print(i)
        recommended_product.append(i[0])
        if len(recommended_product) >= num_recommend:
            break

    return recommended_product


def predict_single_product_utility_ii_lsh(lsh, product_utility_matrix, user_dict, product_id, user_id):
    lsh_algo = lsh
    similarity_dic = lsh_algo.build_similar_dict(product_id)
    sum_weights = 0
    sum_similarity = 0
    for sim_item in similarity_dic.keys():
        sim_val = similarity_dic[sim_item]
        utility = product_utility_matrix[sim_item][user_dict[user_id]]
        sum_weights += sim_val * utility
        sum_similarity += sim_val
    if sum_similarity == 0:
        return 0
    else:
        return sum_weights / sum_similarity
# ==================================== LSH method to find similar items ====================================

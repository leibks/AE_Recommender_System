import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def fetch_users_products(df):
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

    return [list(users), list(products)]


def build_matrix(users, products):
    matrix = {}
    for user in users:
        matrix[user] = {}
        for product in products:
            matrix[user][product] = 0

    return matrix


# utility (value) is the rate from 0 to 5, each row represent
# a user, and each column represent a product
def build_utility_matrix(utility_matrix, df):
    for index in df.index:
        name = df["reviewerName"][index]
        product_id = df["asin"][index]
        utility_matrix[name][product_id] = df["overall"][index]


# value is the (rate to the product for the user - average rate for the user),
# each row represent a user, and each column represent a product
def build_similarity_matrix(similarity_matrix, utility_matrix, product_ids):
    for name in similarity_matrix.keys():
        sum_rates = 0
        for product_id in product_ids:
            sum_rates += utility_matrix[name][product_id]
        average = sum_rates / len(product_ids)
        for product_id in product_ids:
            if utility_matrix[name][product_id] != 0:
                similarity_matrix[name][product_id] = utility_matrix[name][product_id] - average


def find_similar_users(user_name, similarity_matrix):
    cosine_similarity([], [])
    return


def predict_rate(user_name, product_id, utility_matrix, similarity_matrix):
    return


def user_collaborative_filter():
    df = pd.read_csv('../../resource/sample_data/sample_electronics.csv')
    users_products = fetch_users_products(df)
    users = users_products[0]
    product_ids = users_products[1]
    utility_matrix = build_matrix(users, product_ids)
    similarity_matrix = build_matrix(users, product_ids)
    build_utility_matrix(utility_matrix, df)
    build_similarity_matrix(similarity_matrix, utility_matrix, product_ids)
    # print(utility_matrix)
    print(similarity_matrix)
    # store the similarity matrix into the file as the model


user_collaborative_filter()
# vec1 = np.array([[1,1,0,1,1]])
# vec2 = np.array([[0,1,0,1,1]])
# print(cosine_similarity(vec1, vec2).item(0))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# the number of top similar users for one user
TOP_K_USERS = 5
# the number of recommended products for one user
TOP_K_PRODUCTS = 5
# key: product_id, value: index in each user's product_list
PRODUCT_DICT = {}
# key: user_name, value: index in each product's buyers_list
USER_DICT = {}


def fetch_users_products(df, algo="user"):
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




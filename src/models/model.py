from src.algorithms.user_user_collaborative_filtering import *
from src.algorithms.item_item_collaborative_filtering import *
from src.algorithms.utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--USER", type=str, default=False, help="the user who is recommended")
parser.add_argument("--NUM_REC", type=int, default=10, help="how many items provided for recommendation")
parser.add_argument("--HIGH_RATE", type=float, default=0.9, help="identify rate of determining high value products")
parser.add_argument("--LOW_RATE", type=float, default=0.1, help="identify rate of determining low value products")
parser.add_argument("--ECO", type=str, default="True", help="consider economic factors")
parser.add_argument("--LSH", type=str, default="True", help="whether use the locality sensitive hashing")

args = parser.parse_args()


# Recommender System Model: build required matrix based on dataset and
# select different algorithms to find the recommended products by giving
# any user name.
# (1) data will be imported from the given file
# (2) different algorithms need different matrix structures
class Model:

    def __init__(self, num_recommend=10, high_rate=0.9, low_rate=0.1):
        self.num_recommend = num_recommend
        self.high_rate = high_rate
        self.low_rate = low_rate
        # key: user_name, value: index in each product's buyers_list
        self.user_dict = {}
        # key: product_id, value: index in each user's product_list
        self.product_dict = {}
        # key: user_name, value: a list of utilities (preference) to every product (product_list)
        self.user_utility_matrix = {}
        # key: product_id, value: a list of utilities provided by every user (buyers_list)
        self.product_utility_matrix = {}
        # key: user_name, value: a list of feature value calculated by
        # (rate - average rate for this user) (product_list)
        self.user_sim_matrix = {}
        # key: product_id, value: a list of feature value calculated by
        # (rate - average rate for this product) (buyers_list)
        self.product_sim_matrix = {}

    def set_up_matrix(self, file_path, algo, eco=True):
        df = pd.read_csv(file_path)
        fetch_res = fetch_users_products(df, algo)
        identify_res = identify_price_in_items(df["price"].tolist(), self.high_rate, self.low_rate)
        high_value = identify_res[0]
        low_value = identify_res[1]
        users = fetch_res[0]
        product_ids = fetch_res[1]
        if algo == "user":
            self.product_dict = fetch_res[2]
            self.user_utility_matrix = build_user_matrix(users, product_ids)
            self.user_sim_matrix = build_user_matrix(users, product_ids)
            build_user_utility_matrix(self.user_utility_matrix, df, self.product_dict, high_value, low_value, eco)
            build_user_similarity_matrix(self.user_sim_matrix, self.user_utility_matrix, product_ids, self.product_dict)
        elif algo == "item":
            self.user_dict = fetch_res[2]
            self.product_utility_matrix = build_item_matrix(users, product_ids)
            self.product_sim_matrix = build_item_matrix(users, product_ids)
            build_item_utility_matrix(self.product_utility_matrix, df, self.user_dict, high_value, low_value, eco)
            build_item_similarity_matrix(self.product_sim_matrix, self.product_utility_matrix, users, self.user_dict)

    def find_recommended_products(self, user_name, algo, lsh=True):
        recommended_products = []
        if algo == "user":
            if lsh:
                recommended_products = find_recommended_products_by_uu_lsh(
                    user_name, self.user_utility_matrix, self.user_sim_matrix, self.product_dict, self.num_recommend)
            else:
                recommended_products = find_recommended_products_by_uu(
                    user_name, self.user_utility_matrix, self.user_sim_matrix, self.product_dict, self.num_recommend)
        elif algo == "item":
            if lsh:
                recommended_products = find_recommended_products_by_ii_lsh(
                    user_name, self.product_utility_matrix, self.product_sim_matrix, self.user_dict, self.num_recommend)
            else:
                recommended_products = find_recommended_products_by_ii(
                    user_name, self.product_utility_matrix, self.product_sim_matrix, self.user_dict, self.num_recommend)

        print(recommended_products)


model = Model()
model.set_up_matrix("resource/sample_data/joined_sample_electronics.csv", "user")
model.find_recommended_products("Tazman32", "user")
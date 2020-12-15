import os
import sys
import numpy as np
import json
import pickle
src_path = os.getcwd()
sys.path.append(src_path)
print("Current working path:", src_path)
from src.algorithms.user_user_collaborative_filtering import *
from src.algorithms.item_item_collaborative_filtering import *
from src.algorithms.content_based_filtering import *
from src.algorithms.utils import *
from src.performance.statistics import *

INPUT_PATHS = [
    "resource/cleaned_data/Luxury_Beauty_stock.csv",
    "resource/cleaned_data/amazon_fashion_stock.csv",
    "resource/cleaned_data/Toys_&_Games_stock.csv"
]


# Recommender System Model: build required matrix based on dataset and
# select different algorithms to find the recommended products by giving
# any user id.
# (1) data will be imported from the given file
# (2) different algorithms need different matrix structures
class RMSystemModel:

    def __init__(self, num_recommend=10, high_rate=0.9, low_rate=0.1):
        self.num_recommend = num_recommend
        self.high_rate = high_rate
        self.low_rate = low_rate
        # key: user_id, value: user index in the matrix
        self.user_dict = {}
        # key: product_id, value: product index in the matrix
        self.product_dict = {}
        # key: user_id, value: a list of utilities (preference) to every product (product_list)
        self.user_utility_matrix = {}
        # key: product_id, value: a list of utilities provided by every user (buyers_list)
        self.product_utility_matrix = {}
        # key: user_id, value: a list of feature value calculated by
        # (rate - average rate for this user) (product_list)
        self.user_sim_matrix = {}
        # key: product_id, value: a list of feature value calculated by
        # (rate - average rate for this product) (buyers_list)
        self.product_sim_matrix = {}
        # number of product features
        self.content_feature_size = 0
        # key: product asin, value: features (words in review text)
        self.review_text_dict = {}
        self.review_text = []
        # use TF-IDF on review text
        self.tfidf_review = pd.DataFrame()
        self.product_reviews = None
        self.raw_reviews = None
        self.user_ids = set()
        self.product_ids = set()
        self.lsh = None
        # key: user id, a list of rated product ids
        self.rated_products = {}

    @staticmethod
    def has_trained_matrix(file_path, algo, eco):
        input_name = file_path.split("/")[-1]
        eco = "_eco" if eco else ""
        save_file_name1 = f"{input_name}_utility_matrix_{algo}-algo{eco}.pkl"
        save_file_name2 = f"{input_name}_sim_matrix_{algo}-algo{eco}.pkl"
        return os.path.exists(f"src/trained_matrix/{save_file_name1}") and \
               os.path.exists(f"src/trained_matrix/{save_file_name2}")

    def fetch_trained_matrix(self, file_path, algo, eco, matrix):
        input_name = file_path.split("/")[-1]
        eco = "_eco" if eco else ""
        save_file_name = f"{input_name}_{matrix}_{algo}-algo{eco}.pkl"
        with open(f"src/trained_matrix/{save_file_name}", 'rb') as handle:
            fetch_matrix = pickle.load(handle)
        if matrix == "rated_products":
            for key in fetch_matrix:
                self.rated_products[key] = [i for i in fetch_matrix[key].split(",")]
            return

        for i, key in enumerate(fetch_matrix):
            fetch_matrix[key] = [float(k) for k in fetch_matrix[key].split(",")]

        if algo == "user":
            for user_id in self.user_dict:
                for p_index in range(len(self.product_ids)):
                    if matrix == "utility_matrix":
                        self.user_utility_matrix[user_id][p_index] = fetch_matrix[user_id][p_index]
                    elif matrix == "sim_matrix":
                        self.user_sim_matrix[user_id][p_index] = fetch_matrix[user_id][p_index]
        elif algo == "item":
            for p_index, product_id in enumerate(self.product_ids):
                for u_index, user_id in enumerate(self.user_ids):
                    if matrix == "utility_matrix":
                        self.product_utility_matrix[product_id][u_index] = fetch_matrix[user_id][p_index]
                    elif matrix == "sim_matrix":
                        self.product_sim_matrix[product_id][u_index] = fetch_matrix[user_id][p_index]

    def save_trained_matrix(self, file_path, algo, eco):
        input_name = file_path.split("/")[-1]
        eco = "_eco" if eco else ""
        save_file_name = f"{input_name}_utility_matrix_{algo}-algo{eco}.pkl"
        saved_utility_matrix = {}
        if algo == "user":
            for i, key in enumerate(self.user_utility_matrix):
                saved_utility_matrix[key] = ','.join(str(j) for j in self.user_utility_matrix[key])
        elif algo == "item":
            for user_id in self.user_ids:
                str_res = []
                for product_id in self.product_ids:
                    str_res.append(str(self.product_utility_matrix[product_id][self.user_dict[user_id]]))
                saved_utility_matrix[user_id] = ','.join(str_res)
        with open(f"src/trained_matrix/{save_file_name}", "wb") as handle:
            pickle.dump(saved_utility_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

        save_file_name = f"{input_name}_sim_matrix_{algo}-algo{eco}.pkl"
        saved_sim_matrix = {}
        if algo == "user":
            for i, key in enumerate(self.user_sim_matrix):
                saved_sim_matrix[key] = ','.join(str(j) for j in self.user_sim_matrix[key])
        elif algo == "item":
            for user_id in self.user_ids:
                str_res = []
                for product_id in self.product_ids:
                    str_res.append(str(self.product_sim_matrix[product_id][self.user_dict[user_id]]))
                saved_sim_matrix[user_id] = ','.join(str_res)
        with open(f"src/trained_matrix/{save_file_name}", "wb") as handle:
            pickle.dump(saved_sim_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save the rated products
        save_file_name = f"{input_name}_rated_products_{algo}-algo{eco}.pkl"
        save_rated_products = {}
        for key in self.rated_products:
            save_rated_products[key] = ",".join(self.rated_products[key])
        with open(f"src/trained_matrix/{save_file_name}", "wb") as handle:
            pickle.dump(save_rated_products, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # reduce: determine if reduce the size of the matrix with
    # help of content-based algorithms for collaborative filtering algorithm
    def set_up_matrix(self, file_path, algo, reduce=False, eco=True, hash_size=5, num_tables=3, retrained=False):
        df = pd.read_csv(file_path)
        high_value = 0
        low_value = 0
        if eco:
            identify_res = identify_price_in_items(df["price"].tolist(), self.high_rate, self.low_rate)
            high_value = identify_res[0]
            low_value = identify_res[1]
        # fetch the features of the products (for three algorithms)
        self.product_reviews, self.raw_reviews = build_initial_matrix(eco, df, high_value, low_value)
        self.review_text_dict, self.review_text, self.tfidf_review, self.content_feature_size = review_text_tfidf(
            self.product_reviews)
        print(f"content feature size: {self.content_feature_size}")
        # fetch users and products
        fetch_res = fetch_users_products(df)
        self.user_ids = fetch_res[0]
        self.product_ids = fetch_res[1]
        print(f"number of users: {len(self.user_ids)}")
        print(f"number of products: {len(self.product_ids)}")
        if reduce:
            retrained = False
            print("execute reduce products")
            self.product_ids = reduce_matrix(self.review_text_dict, self.content_feature_size, hash_size, num_tables)
            print(f"After reduce, number of products: {len(self.product_ids)}")
        dict_res = build_dictionary(self.user_ids, self.product_ids)
        self.user_dict = dict_res[0]
        self.product_dict = dict_res[1]

        if algo == "user":
            self.user_utility_matrix = build_user_matrix(self.user_ids, self.product_ids)
            self.user_sim_matrix = build_user_matrix(self.user_ids, self.product_ids)
            print("===================================")
            if not retrained and self.has_trained_matrix(file_path, algo, eco):
                print("fetch the trained matrix")
                self.fetch_trained_matrix(file_path, algo, eco, "utility_matrix")
                self.fetch_trained_matrix(file_path, algo, eco, "sim_matrix")
                self.fetch_trained_matrix(file_path, algo, eco, "rated_products")
            else:
                build_user_utility_matrix(self.user_utility_matrix, df, self.product_dict, self.rated_products,
                                          high_value, low_value, eco)
                print("rate of original utility: ")
                print(calculate_filled_utilities(self.rated_products, len(self.product_ids)))
                fill_estimated_rates(self.review_text_dict, self.content_feature_size, self.rated_products,
                                     self.user_utility_matrix, self.product_dict, self.user_dict, algo,
                                     hash_size, num_tables)
                print("rate of filled utility: ")
                print(calculate_filled_utilities(self.rated_products, len(self.product_ids)))
                build_user_similarity_matrix(self.user_sim_matrix, self.user_utility_matrix, self.rated_products,
                                             self.product_dict)
                self.save_trained_matrix(file_path, algo, eco)
            # print(self.user_utility_matrix)
            print("===================================")
            self.lsh = LSH(self.user_sim_matrix, len(self.product_ids), hash_size=hash_size, num_tables=num_tables)
        elif algo == "item":
            self.product_utility_matrix = build_item_matrix(self.user_ids, self.product_ids)
            self.product_sim_matrix = build_item_matrix(self.user_ids, self.product_ids)
            print("===================================")
            if not retrained and self.has_trained_matrix(file_path, algo, eco):
                print("fetch the trained matrix")
                self.fetch_trained_matrix(file_path, algo, eco, "utility_matrix")
                self.fetch_trained_matrix(file_path, algo, eco, "sim_matrix")
                self.fetch_trained_matrix(file_path, algo, eco, "rated_products")
            else:
                build_item_utility_matrix(self.product_utility_matrix, df, self.user_dict, self.rated_products,
                                          high_value, low_value, eco)
                print("rate of original utility: ")
                print(calculate_filled_utilities(self.rated_products, len(self.product_ids)))
                fill_estimated_rates(self.review_text_dict, self.content_feature_size, self.rated_products,
                                     self.product_utility_matrix, self.product_dict, self.user_dict, algo,
                                     hash_size, num_tables)
                print("rate of filled utility: ")
                print(calculate_filled_utilities(self.rated_products, len(self.product_ids)))
                build_item_similarity_matrix(self.product_sim_matrix, self.product_utility_matrix,
                                             self.user_ids, self.user_dict)
                self.save_trained_matrix(file_path, algo, eco)
            # print(self.product_utility_matrix)
            print("===================================")
            self.lsh = LSH(self.product_sim_matrix, len(self.user_ids), hash_size=hash_size, num_tables=num_tables)
        print(f"Finish set up matrix for {algo} algorithm")

    def find_recommended_products(self, user_id, algo, do_lsh):
        if do_lsh:
            print("using LSH for optimization")
        print(f"run {algo} algorithm")
        recommended_products = []
        if algo == "user":
            if do_lsh:
                recommended_products = find_recommended_products_by_uu_lsh(
                    user_id, self.user_utility_matrix, self.lsh, self.product_dict, self.num_recommend)
            else:
                recommended_products = find_recommended_products_by_uu(
                    user_id, self.user_utility_matrix, self.user_sim_matrix, self.product_dict, self.num_recommend)
        elif algo == "item":
            if do_lsh:
                recommended_products = find_recommended_products_by_ii_lsh(
                    user_id, self.product_utility_matrix, self.lsh, self.user_dict, self.num_recommend)
            else:
                recommended_products = find_recommended_products_by_ii(
                    user_id, self.product_utility_matrix, self.product_sim_matrix, self.user_dict, self.num_recommend)
        elif algo == "content":
            user_profile = build_user_profile(user_id, self.content_feature_size, self.review_text,
                                              self.product_reviews, self.raw_reviews)
            if do_lsh:
                recommended_products = find_recommended_products_by_content_lsh(
                    user_id, self.content_feature_size, self.review_text_dict,
                    user_profile, self.num_recommend)
            else:
                recommended_products = find_recommended_products_by_content(
                    user_profile, self.tfidf_review, self.product_reviews, self.num_recommend)

        print(recommended_products)
        return recommended_products

    # predict the utility of one product to one user by selecting the algorithm
    # before calling the function, we have to call the set up function and input the same algorithm
    def predict_utility(self, user_id, product_id, algo, do_lsh=True):
        res_utility = 0
        if algo == "user":
            if not do_lsh:
                similar_users = find_similar_users(user_id, self.user_sim_matrix)
                res_utility = predict_single_product_utility_uu(self.user_utility_matrix,
                                                                similar_users, product_id, self.product_dict)
            else:
                res_utility = predict_single_product_utility_uu_lsh(
                    self.lsh, self.user_utility_matrix, self.product_dict, user_id, product_id)

        elif algo == "item":
            if not do_lsh:
                res_utility = predict_single_product_utility_ii(self.product_utility_matrix, self.product_sim_matrix,
                                                                user_id, self.user_dict, product_id)
            else:
                res_utility = predict_single_product_utility_ii_lsh(
                    self.lsh, self.product_utility_matrix, self.user_dict, product_id, user_id)
        # if predicted utility is 0, we get the mean from all rated products by this user
        if res_utility == 0:
            sum_ratings = 0
            for p_id in self.rated_products[user_id]:
                if algo == "user":
                    sum_ratings += self.user_utility_matrix[user_id][self.product_dict[p_id]]
                elif algo == "item":
                    sum_ratings += self.product_utility_matrix[p_id][self.user_dict[user_id]]
            res_utility = sum_ratings / len(self.rated_products[user_id])
        return res_utility


# if __name__ == '__main__':
#     # simple test case
#     m = RMSystemModel()
#     m.set_up_matrix("resource/cleaned_data/amazon_fashion_stock.csv", "item", reduce=False,
#                     hash_size=3, num_tables=2, eco=True, retrained=False)
#     res = m.find_recommended_products("A3HX4X3TIABWOV", "item", do_lsh=True)

    # m.set_up_matrix("resource/cleaned_data/Luxury_Beauty_stock.csv", "user", reduce=False,
    #                 hash_size=8, num_tables=2, eco=True)
    # res = m.find_recommended_products("A2HOI48JK8838M", "user", do_lsh=True)

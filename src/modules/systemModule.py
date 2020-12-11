from src.algorithms.user_user_collaborative_filtering import *
from src.algorithms.item_item_collaborative_filtering import *
from src.algorithms.content_based_filtering import *
from src.algorithms.utils import *

import argparse
import platform

parser = argparse.ArgumentParser()
parser.add_argument("--USER", type=str, default=False, help="the user who is recommended")
parser.add_argument("--NUM_REC", type=int, default=10, help="how many items provided for recommendation")
parser.add_argument("--HIGH_RATE", type=float, default=0.9, help="identify rate of determining high value products")
parser.add_argument("--LOW_RATE", type=float, default=0.1, help="identify rate of determining low value products")
parser.add_argument("--ECO", type=str, default="True", help="consider economic factors")
parser.add_argument("--LSH", type=str, default="True", help="whether use the locality sensitive hashing")

args = parser.parse_args()


# Recommender System Module: build required matrix based on dataset and
# select different algorithms to find the recommended products by giving
# any user id.
# (1) data will be imported from the given file
# (2) different algorithms need different matrix structures
class SystemModule:

    def __init__(self, num_recommend=10, high_rate=0.9, low_rate=0.1):
        self.num_recommend = num_recommend
        self.high_rate = high_rate
        self.low_rate = low_rate
        # key: user_id, value: index in each product's buyers_list
        self.user_dict = {}
        # key: product_id, value: index in each user's product_list
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
        # index: reviewerID, columns: features (words in review text)
        self.user_profiles = pd.DataFrame()
        # key: product asin, value: features (words in review text)
        self.review_text_dict = {}
        # key: reviewerID, value: features (words in review text)
        self.user_profiles_dict = {}
        # use TF-IDF on review text
        self.tfidf_review = pd.DataFrame()
        self.product_reviews = None
        self.raw_reviews = None

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
            print("Build the initial matrix")
            build_item_utility_matrix(self.product_utility_matrix, df, self.user_dict, high_value, low_value, eco)
            build_item_similarity_matrix(self.product_sim_matrix, self.product_utility_matrix, users, self.user_dict)
        elif algo == "content":
            self.product_reviews, self.raw_reviews = build_initial_matrix(eco)
            self.review_text_dict, review_text, self.tfidf_review = review_text_tfidf(self.product_reviews)
            self.user_profiles = build_user_profiles(review_text, self.product_reviews, self.raw_reviews)
            self.user_profiles_dict = self.user_profiles.T.to_dict('list')
        print(f"Finish set up matrix for {algo} algorithm")

    def find_recommended_products(self, user_id, algo, lsh):
        recommended_products = []
        if algo == "user":
            if lsh:
                recommended_products = find_recommended_products_by_uu_lsh(
                    user_id, self.user_utility_matrix, self.user_sim_matrix, self.product_dict, self.num_recommend)
            else:
                recommended_products = find_recommended_products_by_uu(
                    user_id, self.user_utility_matrix, self.user_sim_matrix, self.product_dict, self.num_recommend)
        elif algo == "item":
            if lsh:
                recommended_products = find_recommended_products_by_ii_lsh(
                    user_id, self.product_utility_matrix, self.product_sim_matrix, self.user_dict, self.num_recommend)
            else:
                recommended_products = find_recommended_products_by_ii(
                    user_id, self.product_utility_matrix, self.product_sim_matrix, self.user_dict, self.num_recommend)
        elif algo == "content":
            if lsh:
                recommended_products = find_recommended_products_by_content_lsh(
                    user_id, self.user_profiles.shape[1], self.review_text_dict,
                    self.user_profiles_dict[user_id], self.num_recommend)
            else:
                cosine_sim = comp_cosine_similarity(self.user_profiles, self.tfidf_review,
                                                    self.product_reviews["asin"], self.raw_reviews["reviewerID"])
                recommended_products = find_recommended_products_by_content(
                    user_id, cosine_sim, self.product_reviews, self.num_recommend, threshold=0.1)

        print(recommended_products)


m = SystemModule()
windows = platform.system() == 'Windows'
if windows:
    # m.set_up_matrix("resource/cleaned_data/beauty.csv", "content")
    # m.find_recommended_products("A3G5NNV6T6JA8J", "content", lsh=True)
    # m.find_recommended_products("Tazman32", "item", lsh=True)
    # m.set_up_matrix("resource/cleaned_data/beauty.csv", "user")
    m.set_up_matrix("resource/cleaned_data/beauty.csv", "user")
    m.find_recommended_products("A3Z74TDRGD0HU", "user", lsh=True)
    # m.find_recommended_products("S. Ortega", "item", lsh=True)
else:
    # m.set_up_matrix("../../resource/cleaned_data/beauty.csv", "content")
    # m.find_recommended_products("A3G5NNV6T6JA8J", "content", lsh=True)
    # m.find_recommended_products("Tazman32", "item", lsh=True)
    # m.set_up_matrix("../../resource/cleaned_data/beauty.csv", "user")
    m.set_up_matrix("../../resource/cleaned_data/beauty.csv", "user")
    m.find_recommended_products("A3Z74TDRGD0HU", "user", lsh=True)
    # m.find_recommended_products("S. Ortega", "item", lsh=True)


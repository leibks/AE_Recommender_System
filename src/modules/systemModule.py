import argparse
import platform
import os
import sys
from src.algorithms.user_user_collaborative_filtering import *
from src.algorithms.item_item_collaborative_filtering import *
from src.algorithms.content_based_filtering import *
from src.algorithms.utils import *
path = os.getcwd()
sys.path.append(path)

print("Current working path:", path)

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
        # number of product features
        self.CONTENT_FEATURES = 0
        # key: product asin, value: features (words in review text)
        self.review_text_dict = {}
        # key: reviewerID, value: features (words in review text)
        self.user_profiles_dict = {}
        # use TF-IDF on review text
        self.tfidf_review = pd.DataFrame()
        self.product_reviews = None
        self.raw_reviews = None
        self.user_ids = set()
        self.product_ids = set()

    # reduce: determine if reduce the size of the matrix with
    # help of content-based algorithms for collaborative filtering algorithm
    def set_up_matrix(self, file_path, algo, reduce=False, eco=True):
        df = pd.read_csv(file_path)
        identify_res = identify_price_in_items(df["price"].tolist(), self.high_rate, self.low_rate)
        high_value = identify_res[0]
        low_value = identify_res[1]
        if algo == "content":
            self.product_reviews, self.raw_reviews = build_initial_matrix(eco, df, high_value, low_value)
            self.review_text_dict, review_text, self.tfidf_review = review_text_tfidf(self.product_reviews)
            self.user_profiles_dict, self.CONTENT_FEATURES = build_user_profiles(review_text, self.product_reviews, self.raw_reviews)
        else:
            fetch_res = fetch_users_products(df, algo)
            self.user_ids = fetch_res[0]
            self.product_ids = fetch_res[1]
            print(len(self.user_ids))
            print(len(self.product_ids))
            if reduce:
                print("execute reduce")
                # fetch the users profile and products features firstly
                self.product_reviews, self.raw_reviews = build_initial_matrix(eco, df)
                self.review_text_dict, review_text, self.tfidf_review = review_text_tfidf(self.product_reviews)
                self.user_profiles = build_user_profiles(review_text, self.product_reviews, self.raw_reviews)
                self.user_profiles_dict = self.user_profiles.T.to_dict('list')
                self.user_ids, self.product_ids = reduce_matrix(self.user_ids, self.product_ids,
                                                                self.review_text_dict, self.user_profiles_dict,
                                                                self.user_profiles.shape[1], algo)
                print(len(self.user_ids))
                print(len(self.product_ids))
            # using the selected algorithm
            if algo == "user":
                self.product_dict = build_dictionary(self.user_ids, self.product_ids, algo)
                self.user_utility_matrix = build_user_matrix(self.user_ids, self.product_ids)
                self.user_sim_matrix = build_user_matrix(self.user_ids, self.product_ids)
                build_user_utility_matrix(self.user_utility_matrix, df, self.product_dict, high_value, low_value, eco)
                build_user_similarity_matrix(self.user_sim_matrix, self.user_utility_matrix,
                                            self.product_ids, self.product_dict)
            elif algo == "item":
                self.user_dict = build_dictionary(self.user_ids, self.product_ids, algo)
                self.product_utility_matrix = build_item_matrix(self.user_ids, self.product_ids)
                self.product_sim_matrix = build_item_matrix(self.user_ids, self.product_ids)
                build_item_utility_matrix(self.product_utility_matrix, df, self.user_dict, high_value, low_value, eco)
                build_item_similarity_matrix(self.product_sim_matrix, self.product_utility_matrix,
                                            self.user_ids, self.user_dict)
        print(f"Finish set up matrix for {algo} algorithm")

    def find_recommended_products(self, user_id, algo, lsh):
        if user_id not in self.user_ids:
            # we have removed this user from the matrix reduce function
            # so, we have to use the content-based algo
            algo = "content"
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
                    user_id, self.CONTENT_FEATURES, self.review_text_dict,
                    self.user_profiles_dict[user_id], self.num_recommend)
            else:
                cosine_sim = comp_cosine_similarity(self.user_profiles, self.tfidf_review,
                                                    self.product_reviews["asin"], self.raw_reviews["reviewerID"])
                recommended_products = find_recommended_products_by_content(
                    user_id, cosine_sim, self.product_reviews, self.num_recommend, threshold=0.1)

        print(recommended_products)

    # predict the utility of one product to one user by selecting the algorithm
    # before calling the function, we have to call the set up function and input the same algorithm
    def predict_utility(self, user_id, product_id, algo):
        if algo == "user":
            lsh_algo = LSH(self.user_sim_matrix, len(self.product_ids))
            similarity_dic = lsh_algo.build_similar_dict(user_id)
            sum_weights = 0
            sum_similarity = 0
            for sim_user in similarity_dic.keys():
                sim_val = similarity_dic[sim_user]
                utility = self.user_utility_matrix[sim_user][product_id]
                sum_weights += sim_val * utility
                sum_similarity += sim_val
            if sum_similarity == 0:
                return 0
            else:
                return sum_weights / sum_similarity
        elif algo == "item":
            lsh_algo = LSH(self.product_sim_matrix, len(self.user_ids))
            similarity_dic = lsh_algo.build_similar_dict(product_id)
            sum_weights = 0
            sum_similarity = 0
            for sim_item in similarity_dic.keys():
                sim_val = similarity_dic[sim_item]
                utility = self.product_utility_matrix[sim_item][user_id]
                sum_weights += sim_val * utility
                sum_similarity += sim_val

            if sum_similarity == 0:
                return 0
            else:
                return sum_weights / sum_similarity
        return 0


if __name__ == '__main__':
    m = SystemModule()
    m.set_up_matrix("resource/cleaned_data/fashion.csv", "content")
    m.find_recommended_products("A1UVZHFDTI4FPK", "content", lsh=True)
    # m.set_up_matrix("resource/cleaned_data/beauty_demo.csv", "content")
    # m.find_recommended_products("A2EM03F99X3RJZ", "content", lsh=True)
    # m.find_recommended_products("Tazman32", "item", lsh=True)
    # m.set_up_matrix("resource/cleaned_data/beauty.csv", "user")

    # m.set_up_matrix("resource/cleaned_data/beauty.csv", "item", reduce=False)
    # print(m.predict_utility("A3Z74TDRGD0HU", "B00004U9V2", "item"))
    # m.find_recommended_products("A3Z74TDRGD0HU", "user", lsh=False)

    # m.set_up_matrix("resource/sample_data/joined_sample_electronics.csv", "item", reduce=False)
    # m.find_recommended_products("A3G5NNV6T6JA8J", "item", lsh=True)
    # print(m.predict_utility("A3G5NNV6T6JA8J", "106171327X", "item"))

    # m.find_recommended_products("S. Ortega", "item", lsh=True)
    # windows = platform.system() == 'Windows'
    # if windows:
    #     # m.set_up_matrix("resource/cleaned_data/beauty.csv", "content")
    #     # m.find_recommended_products("A3G5NNV6T6JA8J", "content", lsh=True)
    #     # m.find_recommended_products("Tazman32", "item", lsh=True)
    #     # m.set_up_matrix("resource/cleaned_data/beauty.csv", "user")
    #     m.set_up_matrix("resource/cleaned_data/beauty.csv", "user")
    #     m.find_recommended_products("A3Z74TDRGD0HU", "user", lsh=True)
    #     # m.find_recommended_products("S. Ortega", "item", lsh=True)
    # else:
    #     # m.set_up_matrix("../../resource/cleaned_data/beauty.csv", "content")
    #     # m.find_recommended_products("A3G5NNV6T6JA8J", "content", lsh=True)
    #     # m.find_recommended_products("Tazman32", "item", lsh=True)
    #     # m.set_up_matrix("../../resource/cleaned_data/beauty.csv", "user")
    #     m.set_up_matrix("../../resource/cleaned_data/beauty.csv", "user")
    #     m.find_recommended_products("A3Z74TDRGD0HU", "user", lsh=True)
    #     # m.find_recommended_products("S. Ortega", "item", lsh=True)
    #

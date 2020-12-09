
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from src.algorithms.lsh_for_cosine_similarity import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--USER", type=str, default=None, help="the user who is recommended")
parser.add_argument("--TOP_ITEM", type=int, default=10, help="how many items provided for recommendation")
parser.add_argument("--HIGH_RATE", type=float, default=0.9, help="identify rate of determining high value products")
parser.add_argument("--LOW_RATE", type=float, default=0.1, help="iidentify rate of determining low value products")
parser.add_argument("--ECO", type=str, default="True", help="consider economic factors")
parser.add_argument("--LSH", type=str, default="False", help="whether use the locality sensitive hashing")

args = parser.parse_args()

TEST_USER = ""
ECO = "True"


def process_price(row):
    out = {}
    price = row["price"]
    if not isinstance(price, float):
        if price[:1] == '$':
            price = float(price[1:])
        else:
            price = np.NaN
    out["new_price"] = price
    return pd.Series(out)


## combine same data into one column
## stem data e.g. (videos -> video)
## clean data; convert all data to lower case and strip names of spaces
def process_review_text(product_reviews):
    # stem data e.g. (videos -> video)
    sno = nltk.stem.SnowballStemmer('english')

    for i in range(len(product_reviews["reviewText"])):
        sen = []
        words = product_reviews["reviewText"][i].split()
        for w in words:
            sen.append(sno.stem(w))
        product_reviews["reviewText"][i] = ' '.join(sen)

    return product_reviews


# combine stock market data with reviews to do recommendation
def comb_stock(raw_reviews):
    price_temp = raw_reviews.groupby("main_cat", as_index=False).mean()
    cat_avgprice = {}
    for i in range(len(price_temp)):
        cat_avgprice[price_temp["main_cat"][i]] = price_temp["new_price"][i]

    for idx in raw_reviews.index:
        cat = raw_reviews["main_cat"][idx]
        if not np.isnan(raw_reviews["new_price"][idx]):
            new_rate = float(raw_reviews["stockReturn"][idx]) * (cat_avgprice[cat] - raw_reviews["new_price"][idx]) * 1000
        raw_reviews.loc[idx, "overall"] += new_rate

    return raw_reviews


# Function that builds user profiles
def build_user_profiles(features, product_reviews, raw_reviews):
    # Construct a reverse map of product_indices and product asins
    product_indices = pd.Series(product_reviews.index, index=product_reviews['asin'])

    # compute the average scores that users give products they bought
    temp = raw_reviews.groupby("reviewerID", as_index=False).agg(np.array)
    user_avgscore = {}
    for i in range(len(temp)):
        user_avgscore[temp["reviewerID"][i]] = temp["overall"][i].mean()

    user_matrix = []
    for idx in raw_reviews.index:
        user = raw_reviews["reviewerID"][idx]
        asin = raw_reviews["asin"][idx]
        product_idx = product_indices[asin]
        # +1.0 is becuase many users give 5.0 score, which will make the score weight becomes 0
        score_weight = user_avgscore[user] - raw_reviews["overall"][idx] + 1.0 
        user_matrix.append(features[product_indices[asin]] * score_weight)

    user_matrix = pd.DataFrame(user_matrix)
    user_matrix['reviewerID'] = raw_reviews["reviewerID"] 
    # size of user_matrix = user number * number of review words

    user_profile = user_matrix.groupby("reviewerID").mean()
    return user_profile


def review_text_tfidf(product_reviews):
    vectorizer = TfidfVectorizer(stop_words='english')
    X1 = vectorizer.fit_transform(product_reviews["reviewText"])
    review_text = X1.toarray() # shape=(21, 1200)
    # key: product_asin, value: list of features (words)
    review_text_dict = {}
    for i in range(len(review_text)):
        review_text_dict[product_reviews["asin"][i]] = review_text[i]
    # print(X1.shape)  # (21, 1200)
    return review_text_dict, review_text, X1


def build_initial_matrix():
    raw_reviews = pd.read_csv('resource\sample_data\joined_sample_electronics.csv')
    
    raw_reviews['new_price'] = raw_reviews.apply(process_price, axis=1)
    
    # combine same product into one item reviews record
    product_reviews = raw_reviews.groupby("asin", as_index=False).agg(list).eval("reviewText = reviewText.str.join(' ')")

    product_reviews = process_review_text(product_reviews)

    if args.ECO == "True":
        raw_reviews = comb_stock(raw_reviews)

    return product_reviews, raw_reviews


# ==================================== Normal method to find similar items =================================
# Compute the cosine similarity matrix
def comp_cosine_similarity(user_profiles, X1, col, idx):
    cosine_sim = cosine_similarity(user_profiles, X1)
    cosine_sim = pd.DataFrame(cosine_sim)
    cosine_sim.columns = col
    cosine_sim.index = idx
    # print(cosine_sim)
    return cosine_sim


# Function that takes in product title as input and outputs most similar products
def find_recommended_products(reviewerID, cosine_sim, product_reviews, threshold=0.1):
    products = cosine_sim.loc[reviewerID, :]
    # print(products)
    products_value = products.values
    # print(type(products_value))
    sorted_product = -np.sort(-products_value)
    sorted_index = np.argsort(-products_value)
    # print(sorted_product, sorted_index)
    
    # Get the scores of the 10 most similar products, and the result must larger than the threshold
    res_scores = []
    for i in range(min(10, len(sorted_index))):
        if sorted_product[i] > threshold:
            res_scores.append(sorted_index[i])

    recommend_products = []
    for i, idx in enumerate(res_scores):
        print(product_reviews["asin"][idx], sorted_product[i+1])
        recommend_products.append(product_reviews["asin"][idx])

    return recommend_products
# ==================================== Normal method to find similar items =================================


# ==================================== LSH method to find similar items ====================================
def find_recommended_products_by_lsh(user_name, FEATURES_NUM, review_text_dict, user_features):
    all_product_utilities = {}
    review_text_dict[user_name] = np.array(user_features)
    # print("review_text_dict", review_text_dict, len(review_text_dict.keys()))
    lsh_algo = LSH(review_text_dict, FEATURES_NUM)
    similarity_dic = lsh_algo.build_similar_dict(user_name)
    # print("similarity_dic", similarity_dic)
    sorted_similarity_dict = {k: v for k, v in sorted(similarity_dic.items(), key=lambda item: item[1], reverse=True)}

    recommended_product = []
    for key in sorted_similarity_dict.keys():
        if not key == user_name:
            recommended_product.append(key)
            print(key, sorted_similarity_dict[key])
            if len(recommended_product) > args.TOP_ITEM:
                break

    return recommended_product
# ==================================== LSH method to find similar items ====================================


## Review-based filter
# def content_based_filter(user, eco, LSH):
def content_based_filter():
    product_reviews, raw_reviews = build_initial_matrix()
    review_text_dict, review_text, X1 = review_text_tfidf(product_reviews)
    user_profiles = build_user_profiles(review_text, product_reviews, raw_reviews)
    user_profiles_dict = user_profiles.T.to_dict('list')
    print("=== Reviews based Recommender: ===")

    ECO = args.ECO

    if args.USER:
        TEST_USER = args.USER
    else:
        TEST_USER = user
    # find k recommended products
    if args.LSH == "True":
        recommended_products = find_recommended_products_by_lsh(TEST_USER, user_profiles.shape[1], review_text_dict, user_profiles_dict[TEST_USER])
    else:
        cosine_sim = comp_cosine_similarity(user_profiles, X1, product_reviews["asin"], raw_reviews["reviewerID"])
        recommended_products = find_recommended_products(TEST_USER, cosine_sim, product_reviews, threshold=0.1)

    print(recommended_products)
    return recommended_products


recommended_products = content_based_filter()

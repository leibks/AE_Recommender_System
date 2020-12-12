import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from src.algorithms.lsh_for_cosine_similarity import *
from tqdm import *
from src.algorithms.utils import (
    get_economic_factor,
    clean_price
)

# =========================================== Set up matrix ================================================
def process_price(price):
    if not isinstance(price, float):
        if price[:1] == '$':
            if "-" in price:
                prices = price.split(" - ")
                price = (float(prices[0][1:].replace(",", "")) + float(prices[1][1:].replace(",", ""))) / 2
            else:
                price = float(price[1:].replace(",", ""))
        else:
            price = np.NaN
    return price




# combine stock market data with reviews to do recommendation
def comb_stock(raw_reviews, high_price, low_price):
    # print("raw_reviews", raw_reviews)
    # economic_factor = get_economic_factor(stock_rate, price, rate, high_price, low_price)
    # price_temp = raw_reviews.groupby("main_cat", as_index=False).mean()
    # print("price_temp", price_temp)
    # cat_avgprice = {}
    # for i in range(len(price_temp)):
    #     cat_avgprice[price_temp["main_cat"][i]] = price_temp["price"][i]

    for idx in tqdm(raw_reviews.index, desc="Combine Stock Loading ...."):
        price = clean_price(raw_reviews["price"][idx])
        stock_rate = raw_reviews["stockReturn"][idx]
        rate = raw_reviews["overall"][idx]
        economic_factor = get_economic_factor(stock_rate, price, rate, high_price, low_price)
        raw_reviews.loc[idx, "overall"] += economic_factor
        
        # cat = raw_reviews["main_cat"][idx]
        # if not np.isnan(raw_reviews["price"][idx]):
        #     new_rate = float(raw_reviews["stockReturn"][idx]) * (cat_avgprice[cat] - raw_reviews["price"][idx]) * 1000
        # raw_reviews.loc[idx, "overall"] += new_rate

    return raw_reviews


# Function that builds user profiles
def build_user_profile(user, features, product_reviews, raw_reviews):
    # Construct a reverse map of product_indices and product asins
    product_indices = pd.Series(product_reviews.index, index=product_reviews['asin'])

    # compute the average scores that users give products they bought
    temp = raw_reviews.groupby("reviewerID", as_index=False).mean()
    # print("temp", temp)
    user_avgscore = {}
    for i in range(len(temp)):
        user_avgscore[temp["reviewerID"][i]] = temp["overall"][i]

    user_profile = []
    for idx in tqdm(raw_reviews.index, desc="Build User Profile Loading ...."):
        if raw_reviews["reviewerID"][idx] == user:
            asin = raw_reviews["asin"][idx]
            product_idx = product_indices[asin]
            # +1.0 is becuase many users give 5.0 score, which will make the score weight becomes 0
            score_weight = user_avgscore[user] - raw_reviews["overall"][idx] + 1.0
            user_profile = (features[product_indices[asin]] * score_weight).tolist()
            # user_matrix.append(features[product_indices[asin]] * score_weight)

    # user_profiles_dict = {}
    # for i in tqdm(range(len(user_matrix)), desc="Build User Profile Dict Loading ...."):
    #     user_profiles_dict[raw_reviews["reviewerID"][i]] = user_matrix[i].tolist()
    return user_profile


def review_text_tfidf(product_reviews):
    vectorizer = TfidfVectorizer(stop_words='english')
    X1 = vectorizer.fit_transform(product_reviews["reviewText"])
    
    review_text = X1.toarray()  # shape=(21, 1200)
    # key: product_asin, value: list of features (words)
    review_text_dict = {}
    for i in range(len(review_text)):
        review_text_dict[product_reviews["asin"][i]] = review_text[i]
    # print(X1.shape)  # (21, 1200)
    return review_text_dict, review_text, X1, X1.shape[1]


## combine same data into one column
## stem data e.g. (videos -> video)
## clean data; convert all data to lower case and strip names of spaces
def process_review_text(product_reviews):
    # stem data e.g. (videos -> video)
    sno = nltk.stem.SnowballStemmer('english')

    for i in tqdm(range(len(product_reviews["reviewText"])), desc="Process Review Text ...."):
    # for i in range(len(product_reviews["reviewText"])):
        sen = []
        review = product_reviews["reviewText"][i]
        # print("reviewText", product_reviews["reviewText"][i])
        if not pd.isnull(review):
            words = review.split()
            for w in words:
                sen.append(sno.stem(w))
        product_reviews["reviewText"][i] = ' '.join(sen)

    return product_reviews


def build_initial_matrix(eco, raw_reviews, high_value, low_value):
    # raw_reviews = pd.read_csv('resource\sample_data\joined_sample_electronics.csv')
    
    # for idx in tqdm(raw_reviews.index, desc="Clean Price Loading ...."):
    #     raw_reviews.loc[idx, "price"] = process_price(raw_reviews["price"][idx])

    # for idx in tqdm(raw_reviews.index, desc="Build Initial Matrix Loading ...."):
    #     price = process_price(raw_reviews["price"][idx])
    #     if eco == True:
    #         raw_reviews = comb_stock(raw_reviews, high_value, low_value)

    
    # combine same product into one item reviews record
    product_reviews = raw_reviews.groupby("asin", as_index=False).agg(list).eval(
        "reviewText = reviewText.str.join(' ')")

    product_reviews = process_review_text(product_reviews)

    if eco == True:
        raw_reviews = comb_stock(raw_reviews, high_value, low_value)

    return product_reviews, raw_reviews


# =========================================== Set up matrix ================================================


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
def find_recommended_products_by_content(reviewerID, cosine_sim, product_reviews, num_recommend, threshold=0.1):
    products = cosine_sim.loc[reviewerID, :]
    # print(products)
    products_value = products.values
    # print(type(products_value))
    sorted_product = -np.sort(-products_value)
    sorted_index = np.argsort(-products_value)
    # print(sorted_product, sorted_index)

    # Get the scores of the 10 most similar products, and the result must larger than the threshold
    res_scores = []
    for i in range(min(num_recommend, len(sorted_index))):
        if sorted_product[i] > threshold:
            res_scores.append(sorted_index[i])

    recommend_products = []
    for i, idx in enumerate(res_scores):
        print(product_reviews["asin"][idx], sorted_product[i + 1])
        recommend_products.append(product_reviews["asin"][idx])

    return recommend_products


# ==================================== Normal method to find similar items =================================


# ==================================== LSH method to find similar items ====================================
def find_recommended_products_by_content_lsh(user_name, FEATURES_NUM, review_text_dict, user_features, num_recommend):
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
            if len(recommended_product) > num_recommend:
                break

    return recommended_product
# ==================================== LSH method to find similar items ====================================

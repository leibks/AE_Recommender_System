import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
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

    for idx in tqdm(raw_reviews.index, desc="Combine Stock Loading ...."):
        price = clean_price(raw_reviews["price"][idx])
        stock_rate = raw_reviews["stockReturn"][idx]
        rate = raw_reviews["overall"][idx]
        economic_factor = get_economic_factor(stock_rate, price, rate, high_price, low_price)
        raw_reviews.loc[idx, "overall"] += economic_factor

    return raw_reviews


# Function that builds user profiles
def build_user_profile(user, feature_num, features, product_reviews, raw_reviews):
    # Construct a reverse map of product_indices and product asins
    product_indices = pd.Series(product_reviews.index, index=product_reviews['asin'])

    # compute the average scores that users give products they bought
    temp = raw_reviews.groupby("reviewerID", as_index=False).mean()
    # print("temp", temp)
    user_avgscore = {}
    for i in range(len(temp)):
        user_avgscore[temp["reviewerID"][i]] = temp["overall"][i]

    user_profile = [0] * feature_num
    for idx in tqdm(raw_reviews.index, desc="Build User Profile Loading ...."):
        if raw_reviews["reviewerID"][idx] == user:
            asin = raw_reviews["asin"][idx]
            product_idx = product_indices[asin]
            # +1.0 is becuase many users give 5.0 score, which will make the score weight becomes 0
            score_weight = user_avgscore[user] - raw_reviews["overall"][idx] + 1.0
            temp = (features[product_indices[asin]] * score_weight).tolist()
            for i in range(len(temp)):
                user_profile[i] += temp[i]

    return user_profile


def review_text_tfidf(product_reviews):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_review = vectorizer.fit_transform(product_reviews["reviewText"])
    
    review_text = tfidf_review.toarray()  # shape=(21, 1200)
    # key: product_asin, value: list of features (words)
    review_text_dict = {}
    for i in range(len(review_text)):
        review_text_dict[product_reviews["asin"][i]] = review_text[i]
    # print(tfidf_review.shape)  # (21, 1200)
    return review_text_dict, review_text, tfidf_review, tfidf_review.shape[1]


## combine same data into one column
## stem data e.g. (videos -> video)
## clean data; convert all data to lower case and strip names of spaces
def process_review_text(product_reviews):
    # stem data e.g. (videos -> video)
    sno = nltk.stem.SnowballStemmer('english')

    en_stops = set(stopwords.words('english'))
    for i in tqdm(range(len(product_reviews["reviewText"])), desc="Process Review Text ...."):
    # for i in range(len(product_reviews["reviewText"])):
        sen = []
        review = product_reviews["reviewText"][i]
        # print("reviewText", product_reviews["reviewText"][i])
        if not pd.isnull(review):
            words = review.split()
            for w in words:
                if w not in en_stops:
                    sen.append(sno.stem(w))
        product_reviews["reviewText"][i] = ' '.join(sen)

    return product_reviews


def build_initial_matrix(eco, raw_reviews, high_value, low_value):
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
def find_recommended_products_by_content(reviewerID, cosine_sim, product_reviews, num_recommend):
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

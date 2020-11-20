
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import sys

test_user = str(sys.argv[1])

raw_reviews = pd.read_csv('resource\sample_data\sample_electronics.csv')

## Data processing >>
## combine same data into one column
## stem data e.g. (videos -> video)
## clean data; convert all data to lower case and strip names of spaces

## combine same product into one item reviews record
product_reviews = raw_reviews.groupby("asin", as_index=False).agg(list).eval("reviewText = reviewText.str.join(' ')")
# compute the average scores that users give products they bought
temp = raw_reviews.groupby("reviewerID", as_index=False).agg(np.array)
# print(type(temp))
user_avgscore = {}
for i in range(len(temp)):
    user_avgscore[temp["reviewerID"][i]] = temp["overall"][i].mean()

## stem data e.g. (videos -> video)
sno = nltk.stem.SnowballStemmer('english')

for i in range(len(product_reviews["reviewText"])):
    sen = []
    words = product_reviews["reviewText"][i].split()
    for w in words:
        sen.append(sno.stem(w))
    product_reviews["reviewText"][i] = ' '.join(sen)


# Function that builds user profiles
def build_user_profiles(features):
    user_matrix = []
    for idx in raw_reviews.index:
        user = raw_reviews["reviewerID"][idx]
        asin = raw_reviews["asin"][idx]
        product_idx = product_indices[asin]
        score_weight = user_avgscore[user] - raw_reviews["overall"][idx] + 1.0 # +1.0 is becuase many users give 5.0 score, which will make the score weight becomes 0
        user_matrix.append(features[product_indices[asin]] * score_weight)

    # print(len(user_matrix[1]), len(user_matrix))
    user_matrix = pd.DataFrame(user_matrix)
    # user_matrix.index = raw_reviews["reviewerID"]
    user_matrix['reviewerID']=raw_reviews["reviewerID"]
    print(user_matrix)

    user_profile = user_matrix.groupby("reviewerID").mean()
    return user_profile


# Function that takes in product title as input and outputs most similar products
def get_recommendations(reviewerID, cosine_sim, product_reviews=product_reviews, threshold=0.1):
    products = cosine_sim.loc[reviewerID,:]
    # print(products)
    products_value = products.values
    # print(type(products_value))
    sorted_product = -np.sort(-products_value)
    sorted_index = np.argsort(-products_value)
    # print(sorted_index)
    
    # Get the scores of the 10 most similar products, and the result must larger than the threshold
    res_scores = []
    for i in range(1, min(10, len(sorted_index))):
        if sorted_product[i] > threshold:
            res_scores.append(sorted_index[i])

    recommend_products = []
    for idx in res_scores:
        recommend_products.append([product_reviews["asin"][idx], sorted_product[idx]])
    return recommend_products


# Construct a reverse map of product_indices and product asins
product_indices = pd.Series(product_reviews.index, index=product_reviews['asin'])

## Product Reviews based Recommender:
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_reviews["reviewText"])
review_text = X1.toarray()
# print(len(vectorizer.get_feature_names()))
# print(X1.shape)  # (21, 1200)

user_profiles = build_user_profiles(review_text)
# print("build_user_profiles", build_user_profiles(), len(build_user_profiles()))
# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(user_profiles, X1)
cosine_sim = pd.DataFrame(cosine_sim)
cosine_sim.columns = product_reviews["asin"]
cosine_sim.index = raw_reviews["reviewerID"]
# print(cosine_sim)

## UNCOMMENT for the review-based method
# print("Reviews based Recommender:", get_recommendations(test_user, cosine_sim, threshold=0.1))
# exit()

## Product Features Based Recommender
product_features = raw_reviews[["asin", "price", "main_cat"]]
# print(product_features)
product_features = product_features.drop_duplicates(["asin"])
# print("drop_duplicates", product_features)

## Build a feature soup and using IT-IDF to get matrix
def create_soup(x):
    return x['main_cat'] + ' ' + str(x['price'])
product_features['soup'] = product_features.apply(create_soup, axis=1)
# print(product_features["soup"])

count_matrix = vectorizer.fit_transform(product_features['soup'])

# Reset index of our main DataFrame and construct reverse mapping as before
product_features = product_features.reset_index()
product_indices = pd.Series(product_features.index, index=product_features['asin'])

user_profiles = build_user_profiles(count_matrix.toarray())
# Compute the cosine similarity matrix
cosine_sim2 = cosine_similarity(user_profiles, count_matrix)
cosine_sim2 = pd.DataFrame(cosine_sim)
cosine_sim2.columns = product_reviews["asin"]
cosine_sim2.index = raw_reviews["reviewerID"]

# TODO
## Cluster values in each feature to build matrix
# count_matrix = product_reviews[["price", "main_cat"]]
# # print(count_matrix)
# cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# # print(cosine_sim2)


print("Features based Recommender:", get_recommendations(test_user, cosine_sim2))

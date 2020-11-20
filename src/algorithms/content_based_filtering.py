
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
print(len(user_avgscore))

## stem data e.g. (videos -> video)
sno = nltk.stem.SnowballStemmer('english')

for i in range(len(product_reviews["reviewText"])):
    sen = []
    words = product_reviews["reviewText"][i].split()
    for w in words:
        sen.append(sno.stem(w))
    product_reviews["reviewText"][i] = ' '.join(sen)

# Construct a reverse map of product_indices and product asins
product_indices = pd.Series(product_reviews.index, index=product_reviews['asin'])

## Product Reviews based Recommender:
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_reviews["reviewText"])
review_text = X1.toarray()
# print(len(vectorizer.get_feature_names()))
# print(X1.shape)  # (21, 1200)

# Function that builds user profiles
def build_user_profiles():
    user_matrix = []
    for idx in raw_reviews.index:
        user = raw_reviews["reviewerID"][idx]
        asin = raw_reviews["asin"][idx]
        product_idx = product_indices[asin]
        score_weight = user_avgscore[user] - raw_reviews["overall"][idx] + 1.0 # +1.0 is becuase many users give 5.0 score, which will make the score weight becomes 0
        user_matrix.append(review_text[product_indices[asin]] * score_weight)

    # print(len(user_matrix[1]), len(user_matrix))
    user_matrix = pd.DataFrame(user_matrix)
    # user_matrix.index = raw_reviews["reviewerID"]
    user_matrix['reviewerID']=raw_reviews["reviewerID"]
    print(user_matrix)

    # for i in range(len(user_matrix)):
    #     score_sum = 0
    #     for j in range(len(user_matrix[0])):
    #         score_sum += user_matrix[i][j]
    #     user_profile[raw_reviews["reviewerID"][i]].append(score_sum/len(user_matrix))

    user_profile = user_matrix.groupby("reviewerID").mean()
    return user_profile

user_profiles = build_user_profiles()
# print("build_user_profiles", build_user_profiles(), len(build_user_profiles()))
# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(user_profiles, X1)
cosine_sim = pd.DataFrame(cosine_sim)
cosine_sim.columns = product_reviews["asin"]
cosine_sim.index = raw_reviews["reviewerID"]
print(cosine_sim)

# Function that takes in product title as input and outputs most similar products
def get_recommendations(reviewerID, cosine_sim=cosine_sim, product_reviews=product_reviews, threshold=0.1):
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
        if sorted_index[i] > threshold:
            res_scores.append(sorted_index[i])

    recommend_products = []
    for idx in res_scores:
        recommend_products.append([product_reviews["asin"][idx], sorted_product[idx]])
    return recommend_products

# Function that takes in product title as input and outputs most similar products
# def get_recommendations(reviewerID, cosine_sim=cosine_sim, product_reviews=product_reviews, threshold=0.1):
#     # Get the index of the product that matches the asin
#     idx = product_indices[reviewerID]

#     # Get the pairwsie similarity scores of all products with that product
#     sim_scores = list(enumerate(cosine_sim[idx]))

#     # Sort the products based on the similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Get the scores of the 10 most similar products, and the result must larger than the threshold
#     res_scores = []
#     for i in range(1, min(10, len(sim_scores))):
#         if sim_scores[i][1] > threshold:
#             res_scores.append(sim_scores[i])

#     # Get the product product_indices
#     product_indices = [i[0] for i in res_scores]

#     # Return the top 10 most similar products
#     res = product_reviews['asin'].iloc[product_indices]
#     ## TODO: combine with scores
#     return product_reviews['asin'].iloc[product_indices]

## UNCOMMENT for the review-based method
print("Reviews based Recommender:", get_recommendations(test_user, threshold=0.1))
exit()

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

# compute cosine similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# TODO
## Cluster values in each feature to build matrix
# count_matrix = product_reviews[["price", "main_cat"]]
# # print(count_matrix)
# cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# # print(cosine_sim2)


# Reset index of our main DataFrame and construct reverse mapping as before
product_features = product_features.reset_index()
product_indices = pd.Series(product_features.index, index=product_features['asin'])

print(get_recommendations(0, cosine_sim2, product_features), 0.2)

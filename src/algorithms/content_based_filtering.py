
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import linear_kernel

raw_reviews = pd.read_csv('resource\sample_data\sample_electronics.csv')

## Data processing >>
## combine same data into one column
## stem data e.g. (videos -> video)
## clean data; convert all data to lower case and strip names of spaces

## combine same product into one item reviews record
product_reviews = raw_reviews.groupby("asin", as_index=False).agg(list).eval("reviewText = reviewText.str.join(' ')")

## stem data e.g. (videos -> video)
sno = nltk.stem.SnowballStemmer('english')

for i in range(len(product_reviews["reviewText"])):
    sen = []
    words = product_reviews["reviewText"][i].split()
    for w in words:
        sen.append(sno.stem(w))
    product_reviews["reviewText"][i] = ' '.join(sen)


## Product Reviews based Recommender:
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_reviews["reviewText"])
# print(len(vectorizer.get_feature_names()))
# print(X1.shape)

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(X1, X1)

# #Construct a reverse map of indices and product asins
indices = pd.Series(product_reviews.index, index=product_reviews['asin'])
print(indices)

# Function that takes in product title as input and outputs most similar products
def get_recommendations(asin, cosine_sim=cosine_sim, product_reviews=product_reviews, threshold=0.1):
    # Get the index of the product that matches the title
    idx = indices[asin]
    print("idx", idx)

    # Get the pairwsie similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))
    print("sim_scores", sim_scores)

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    print("sim_scores", sim_scores)

    # Get the scores of the 10 most similar products, and the result must larger than the threshold
    res_scores = []
    for i in range(1, min(10, len(sim_scores))):
        if sim_scores[i][1] > threshold:
            res_scores.append(sim_scores[i])

    # Get the product indices
    product_indices = [i[0] for i in res_scores]
    print(product_indices)

    # Return the top 10 most similar products
    res = product_reviews['asin'].iloc[product_indices]
    ## TODO: combine with scores
    return product_reviews['asin'].iloc[product_indices], res_scores

## UNCOMMENT
# print("Reviews based Recommender:", get_recommendations("073530498X"))
# exit()

## Product Features Based Recommender
product_features = raw_reviews[["asin", "price", "main_cat"]]
print(product_features)
product_features = product_features.drop_duplicates(["asin"])
print("drop_duplicates", product_features)

## Build a feature soup and using IT-IDF to get matrix
def create_soup(x):
    return x['main_cat'] + ' ' + str(x['price'])
product_features['soup'] = product_features.apply(create_soup, axis=1)
# print(product_features["soup"])

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(product_features['soup'])

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
indices = pd.Series(product_features.index, index=product_features['asin'])

print(get_recommendations(0, cosine_sim2, product_features), 0.2)

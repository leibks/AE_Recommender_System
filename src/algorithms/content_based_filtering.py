
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import sys
import re
import time
from datasketch import MinHash, MinHashLSHForest


test_user = str(sys.argv[1])

raw_reviews = pd.read_csv('resource\sample_data\joined_sample_electronics.csv')

#Number of Permutations
permutations = 128

#Preprocess will split a string of text into individual tokens/shingles based on whitespace.
def preprocess(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens

def get_forest(data, perms):
    start_time = time.time()
    
    minhash = []
    
    for text in data['reviewText']:
        tokens = preprocess(text)
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)
        
    forest = MinHashLSHForest(num_perm=perms)
    
    for i,m in enumerate(minhash):
        forest.add(i,m)
        
    forest.index()
    
    print('It took %s seconds to build forest.' %(time.time()-start_time))
    
    return forest


## Data processing >>
## combine same data into one column
## stem data e.g. (videos -> video)
## clean data; convert all data to lower case and strip names of spaces

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
raw_reviews['new_price'] = raw_reviews.apply(process_price, axis=1)

# combine same product into one item reviews record
product_reviews = raw_reviews.groupby("asin", as_index=False).agg(list).eval("reviewText = reviewText.str.join(' ')")
# compute the average scores that users give products they bought
temp = raw_reviews.groupby("reviewerID", as_index=False).agg(np.array)
# print(type(temp), temp)
price_temp = raw_reviews.groupby("main_cat", as_index=False).mean()
# print(type(price_temp), price_temp)
user_avgscore = {}
for i in range(len(temp)):
    user_avgscore[temp["reviewerID"][i]] = temp["overall"][i].mean()
cat_avgprice = {}
for i in range(len(price_temp)):
    cat_avgprice[price_temp["main_cat"][i]] = price_temp["new_price"][i]

# stem data e.g. (videos -> video)
sno = nltk.stem.SnowballStemmer('english')

for i in range(len(product_reviews["reviewText"])):
    sen = []
    words = product_reviews["reviewText"][i].split()
    for w in words:
        sen.append(sno.stem(w))
    product_reviews["reviewText"][i] = ' '.join(sen)


# combine stock market data with reviews to do recommendation
def comb_stock():
    for idx in raw_reviews.index:
        cat = raw_reviews["main_cat"][idx]
        if not np.isnan(raw_reviews["new_price"][idx]):
            new_rate = float(raw_reviews["stockReturn"][idx]) * (cat_avgprice[cat] - raw_reviews["new_price"][idx]) * 100
        raw_reviews["overall"][idx] += new_rate


# Function that builds user profiles
def build_user_profiles(features):
    user_matrix = []
    for idx in raw_reviews.index:
        user = raw_reviews["reviewerID"][idx]
        asin = raw_reviews["asin"][idx]
        product_idx = product_indices[asin]
        score_weight = user_avgscore[user] - raw_reviews["overall"][idx] + 1.0 
        # +0.5 is becuase many users give 5.0 score, which will make the score weight becomes 0
        user_matrix.append(features[product_indices[asin]] * score_weight)

    # print(len(user_matrix[1]), len(user_matrix))
    user_matrix = pd.DataFrame(user_matrix)
    # user_matrix.index = raw_reviews["reviewerID"]
    user_matrix['reviewerID'] = raw_reviews["reviewerID"] 
    # size of user_matrix = user number * number of review words
    # print("user_matrix:", user_matrix)

    user_profile = user_matrix.groupby("reviewerID").mean()
    return user_profile


# Function that takes in product title as input and outputs most similar products
def get_recommendations(reviewerID, cosine_sim, product_reviews=product_reviews, threshold=0.1):
    products = cosine_sim.loc[reviewerID, :]
    # print(products)
    products_value = products.values
    # print(type(products_value))
    sorted_product = -np.sort(-products_value)
    sorted_index = np.argsort(-products_value)
    # print(sorted_product, sorted_index)
    
    # Get the scores of the 10 most similar products, and the result must larger than the threshold
    res_scores = []
    for i in range(1, min(10, len(sorted_index))):
        if sorted_product[i] > threshold:
            res_scores.append(sorted_index[i])

    recommend_products = []
    for i, idx in enumerate(res_scores):
        recommend_products.append([product_reviews["asin"][idx], sorted_product[i+1]])
    return recommend_products    


# Construct a reverse map of product_indices and product asins
product_indices = pd.Series(product_reviews.index, index=product_reviews['asin'])

# Product Reviews based Recommender:
comb_stock()

vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_reviews["reviewText"])
review_text = X1.toarray()
# print(len(vectorizer.get_feature_names()))
# print(X1.shape)  # (21, 1200)

user_profiles = build_user_profiles(review_text) # user number * number of review words
# print("build_user_profiles", user_profiles)

# use LSH to compute the similarity, which can 
# reduce complexity and accelerate computing
# forest = get_forest(user_profiles, permutations)

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(user_profiles, X1)
cosine_sim = pd.DataFrame(cosine_sim)
cosine_sim.columns = product_reviews["asin"]
cosine_sim.index = raw_reviews["reviewerID"]
# print(cosine_sim)

## UNCOMMENT for the review-based method
print("Reviews based Recommender:", get_recommendations(test_user, cosine_sim, threshold=0.1))
exit()


# Product Features Based Recommender
product_features = raw_reviews[["asin", "price", "main_cat"]]
# print(product_features)
product_features = product_features.drop_duplicates(["asin"])
# print("drop_duplicates", product_features)


# Build a feature soup and using IT-IDF to get matrix
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

print("Features based Recommender:", get_recommendations(test_user, cosine_sim2))

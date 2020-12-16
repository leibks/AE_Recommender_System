import os
import sys
from tqdm import *
path = os.getcwd()
sys.path.append(path)
from src.modules import *
from sklearn.metrics import mean_squared_error

INPUT_PATHS = [
    "resource/cleaned_data/Luxury_Beauty_stock.csv",
    "resource/cleaned_data/Luxury_Beauty_stock.csv",
    "resource/cleaned_data/amazon_fashion_stock.csv",
    "resource/cleaned_data/amazon_fashion_stock.csv",
    "resource/cleaned_data/Toys_&_Games_stock.csv",
    "resource/cleaned_data/Toys_&_Games_stock.csv",
    "resource/cleaned_data/Luxury_Beauty_stock.csv",
    "resource/cleaned_data/Luxury_Beauty_stock.csv",
    "resource/cleaned_data/amazon_fashion_stock.csv",
    "resource/cleaned_data/amazon_fashion_stock.csv",
    "resource/cleaned_data/Toys_&_Games_stock.csv",
    "resource/cleaned_data/Toys_&_Games_stock.csv"
]
SELECT_TEST = 3  # select different tests you want to run here, indexes are below
OUTPUT_PATHS = [
    "resource/performance_test_data/Performance_Beauty_Item.csv",  # 0
    "resource/performance_test_data/Performance_Beauty_User.csv",  # 1
    "resource/performance_test_data/Performance_Fashion_Item.csv",  # 2
    "resource/performance_test_data/Performance_Fashion_User.csv",  # 3
    "resource/performance_test_data/Performance_Toy_Item.csv",  # 4
    "resource/performance_test_data/Performance_Toy_User.csv",  # 5
    "resource/performance_test_data/Performance_Beauty_Item_NECO.csv",  # 6
    "resource/performance_test_data/Performance_Beauty_User_NECO.csv",  # 7
    "resource/performance_test_data/Performance_Fashion_Item_NECO.csv",  # 8
    "resource/performance_test_data/Performance_Fashion_User_NECO.csv",  # 9
    "resource/performance_test_data/Performance_Toy_Item_NECO.csv",  # 10
    "resource/performance_test_data/Performance_Toy_User_NECO.csv",  # 11
]
STORE_RMSE_PATHS = [
    "resource/performance_test_data/RMSE_Beauty_Item.csv",  # 0
    "resource/performance_test_data/RMSE_Beauty_User.csv",  # 1
    "resource/performance_test_data/RMSE_Fashion_Item.csv",  # 2
    "resource/performance_test_data/RMSE_Fashion_User.csv",  # 3
    "resource/performance_test_data/RMSE_Toy_Item.csv",  # 4
    "resource/performance_test_data/RMSE_Toy_User.csv",  # 5
    "resource/performance_test_data/RMSE_Beauty_Item_NECO.csv",  # 6
    "resource/performance_test_data/RMSE_Beauty_User_NECO.csv",  # 7
    "resource/performance_test_data/RMSE_Fashion_Item_NECO.csv",  # 8
    "resource/performance_test_data/RMSE_Fashion_User_NECO.csv",  # 9
    "resource/performance_test_data/RMSE_Toy_Item_NECO.csv",  # 10
    "resource/performance_test_data/RMSE_Toy_User_NECO.csv",  # 11
]
ALGO = ["item", "user", "item", "user", "item", "user", "item", "user", "item", "user", "item", "user"]
TEST_TYPES = ["beauty_item", "beauty_user", "fashion_item", "fashion_user", "toy_user", "toy_item",
              "beauty_item_NECO", "beauty_user_NECO", "fashion_item_NECO", "fashion_user_NECO",
              "toy_user_NECO", "toy_item_NECO"]
HASH_SIZE = [8, 8, 3, 3, 12, 12, 8, 8, 3, 3, 12, 12]


# run different tests and output the results as the csv files,
# store them in performance_test_data folder
def res_export(test_index):
    print("Start Performance Tests: ")
    reduce = False
    eco = True
    if test_index > 5:
        eco = False
    if INPUT_PATHS[test_index] == "resource/cleaned_data/Toys_&_Games_stock.csv":
        print("reduce the matrix due the huge size")
        reduce = True

    model = RMSystemModel()
    model.set_up_matrix(INPUT_PATHS[test_index], ALGO[test_index], reduce=reduce,
                        hash_size=HASH_SIZE[test_index], num_tables=2, eco=eco)
    r = run_file(tests[test_index], ALGO[test_index], OUTPUT_PATHS[test_index], model, do_reduce=reduce)
    result = pd.DataFrame()
    result["Name"] = [TEST_TYPES[test_index]]
    result["RMSE"] = [r]
    result.to_csv(STORE_RMSE_PATHS[test_index], index=False)


# run the RMSE performance test for different files by giving different parameters
def run_file(test, algo, file_path, model, do_reduce=False):

    predict = []

    reviewer = [i for i in test.reviewerID]
    product = [i for i in test.asin]
    pair = list(zip(reviewer, product))
    valid_count = 0
    for i, v in tqdm(enumerate(pair), desc="Performance Test Loading ...."):
        if do_reduce:
            if v[1] in model.product_dict:
                p = model.predict_utility(v[0], v[1], algo)
                # print(p)
                predict.append(p)
                valid_count += 1
            else:
                predict.append(-1)
        else:
            p = model.predict_utility(v[0], v[1], algo)
            # print(p)
            predict.append(p)
            valid_count += 1

    test["Predict"] = predict
    test = test[test["Predict"] != -1]
    # save the performance result
    test.to_csv(file_path, index=False)

    # test["Res"] = (test["Predict"] != 0.0).astype(int)
    # test_non_zero = test[test["Predict"] != 0.0]
    # test["Predict"].loc[test["Predict"] == 0.0] = test_non_zero["Predict"].mean()
    return (
        mean_squared_error(test["overall"], test["Predict"], squared=False)
    )


if __name__ == "__main__":
    # set up the original testing data
    beauty = pd.read_csv("resource/cleaned_data/Luxury_Beauty_stock.csv")
    toy = pd.read_csv("resource/cleaned_data/Toys_&_Games_stock.csv")
    fashion = pd.read_csv("resource/cleaned_data/amazon_fashion_stock.csv")

    beauty = beauty.sort_values(by=["Date"])
    beauty = beauty[beauty["overall"] != 0]
    beauty_test = (
        beauty[["overall", "reviewerID", "asin"]].iloc[-1000:].reset_index(drop=True)
    )

    fashion = fashion.sort_values(by=["Date"])
    fashion = fashion[fashion["overall"] != 0]
    fashion_test = fashion[["overall", "reviewerID", "asin"]].reset_index(drop=True)

    toy = toy.sort_values(by=["Date"])
    toy = toy[toy["overall"] != 0]
    toy_test = (
        toy[["overall", "reviewerID", "asin"]].iloc[-1000:].reset_index(drop=True)
    )

    tests = [beauty_test, beauty_test, fashion_test, fashion_test, toy_test,
             toy_test, beauty_test, beauty_test, fashion_test, fashion_test,
             toy_test, toy_test]

    # change the test_index to try different tests
    res_export(SELECT_TEST)

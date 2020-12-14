import os
import sys
path = os.getcwd()
sys.path.append(path)
from src.modules import *


if __name__ == '__main__':
    m = SystemModule()

    # set up the original testing data
    beauty = pd.read_csv('resource/cleaned_data/Luxury_Beauty_stock.csv')
    toy = pd.read_csv('resource/cleaned_data/Toys_&_Games_stock.csv')
    fashion = pd.read_csv('resource/cleaned_data/AMAZON_FASHION_stock.csv')

    beauty = beauty.sort_values(by=['Date'])
    beauty = beauty[beauty['overall'] != 0]
    beauty_test = beauty[['overall', 'reviewerID', 'asin']].iloc[-1000:].reset_index(drop=True)

    fashion = fashion.sort_values(by=['Date'])
    fashion = fashion[toy['overall'] != 0]
    fashion_test = fashion[['overall', 'reviewerID', 'asin']].reset_index(drop=True)

    toy = toy.sort_values(by=['Date'])
    toy = toy[toy['overall']!= 0]
    toy_test = toy[['overall', 'reviewerID', 'asin']].iloc[-1000:].reset_index(drop=True)


    def run_file(test, mode, path):
        predict = []

        reviewer = [i for i in test.reviewerID]
        product = [i for i in test.asin]
        pair = list(zip(reviewer, product))

        for i, v in enumerate(pair):
            if v[1] in m.product_dict:
                p = m.predict_utility(v[0], v[1], mode)
                print(p)
                predict.append(p)

        test['Predict'] = predict
        # save the performance result
        test.to_csv(path, index=False)

        test['Res'] = (test['Predict'] != 0.0).astype(int)
        test_nonZero = test[test['Predict'] != 0.0]
        test['Predict'].loc[test['Predict'] == 0.0] = test_nonZero['Predict'].mean()
        return mean_squared_error(test['overall'], test['Predict'], squared=False), sum(test['Res']) / len(test)

    RMSE = []
    none0rate = []

    tests = [beauty_test, beauty_test, fashion_test, fashion_test, toy_test, toy_test]
    paths = ["resource/cleaned_data/Performance_Beauty_Item.csv",
            "resource/cleaned_data/Performance_Beauty_User.csv",
            "resource/cleaned_data/Performance_Fashion_Item.csv",
            "resource/cleaned_data/Performance_Fashion_User.csv",
            "resource/cleaned_data/Performance_Toy_Item.csv",
            "resource/cleaned_data/Performance_Toy_User.csv"]
    modes = ["item", "user", "item", "user", "item", "user"]
    names = ["beauty_item", "beauty_user", "fashion_item", "fashion_user", "toy_user", "toy_item"]
    res_paths = ["resource/cleaned_data/RMSE_Beauty_Item.csv",
             "resource/cleaned_data/RMSE_Beauty_User.csv",
             "resource/cleaned_data/RMSE_Fashion_Item.csv",
             "resource/cleaned_data/RMSE_Fashion_User.csv",
             "resource/cleaned_data/RMSE_Toy_Item.csv",
             "resource/cleaned_data/RMSE_Toy_User.csv"]
    r, n = run_file(tests[0], modes[0], paths[0])
    # r, n = run_file(tests[1], modes[1], paths[1])
    # r, n = run_file(tests[2], modes[2], paths[2])
    # r, n = run_file(tests[3], modes[3], paths[3])
    # r, n = run_file(tests[4], modes[4], paths[4])
    # r, n = run_file(tests[5], modes[5], paths[5])

    def res_export(i):
        r, n = run_file(tests[i], modes[i], paths[i])
        result = pd.DataFrame()
        result['Name'] = [names[i]]
        result['RMSE'] = [r]
        result['None Zero Rate'] = [n]
        result.to_csv(res_paths[i], index=False)


    res_export(0)
    # res_export(1)
    # res_export(2)
    # res_export(3)
    # res_export(4)
    # res_export(5)



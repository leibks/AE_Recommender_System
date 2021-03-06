# Adaptive Economic Recommender System

## Description:

we are developing an adaptive economic recommender system that novelly
takes daily percentage change of S&P 500 price into account to make
recommendations based on both product review and financial condition,
in which product at a comparably low price would have better chance to be
selected (assigned more weight) if index price drops and vise versa.

* time of data: 2018
* amazon review dataset reference: http://deepyeti.ucsd.edu/jianmo/amazon/index.html
* S&P 500 Historical Price reference: https://finance.yahoo.com/quote/%5EGSPC/history?period1=1514592000&period2=1546300800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true

## Setting:

# Installation #

### Requirements

  * Python 3.3+
  * macOS or Windows

### Installing module for the project

  * Install the fundamental module: run command `pip3 install -r requirements.txt`

## How to Run the System:
### Data downloading and processing
```shell script
python3 data_man.py {working directory}
```
Without working directory input parameter, the script will download all raw data files to current working dierectory.

## Run the System
  under the AE_Recommender_System:
  * run command `python src/main.py` to test different algorithms and output recommendations
  * default dataset: Luxury_Beauty_stock (you are free to select file by change file index)
  * required arguments:
    * --USER USER           the user who is recommended, eg: A2HOI48JK8838M
  * optional arguments:
    * -h, --help            show this help message and exit
    * --NUM_REC NUM_REC     how many items provided for recommendation
    * --ALGO ALGO           selected algorithms to use for prediction
    * --HIGH_RATE HIGH_RATE identify rate of determining high value products
    * --LOW_RATE LOW_RATE   identify rate of determining low value products
    * --ECO ECO             consider economic factors
    * --LSH LSH             whether use the locality sensitive hashing
    * --REDUCE REDUCE       whether reduce the matrix
    * --RETRAIN RETRAIN     whether retrain the model

## Run the Performance Test:
  under the AE_Recommender_System:
  * run command `python src/performance/rmse.py` to test different algorithms' performance

## User Interface (WIP)
  under the AE_Recommender_System:
  * run command `python user_interface/app.py` to start web server
  * Go to http://127.0.0.1:5000/ to test the system
  

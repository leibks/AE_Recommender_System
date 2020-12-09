from flask import Flask, render_template, request
from src.algorithms.content_based_filtering import (content_based_filter)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        input = request.form
        print(input)
        # call recommendation functions
        result = content_based_filter(input["customer_id"], input["eco"], input["lsh"])
        return render_template("results.html", result = input, recommendations = result)
    
if __name__ == "__main__":
    app.run(debug=True)
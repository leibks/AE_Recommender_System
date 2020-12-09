from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      input = request.form
      # call recommendation functions
      return render_template("results.html",result = input)
    
if __name__ == "__main__":
    app.run(debug=True)
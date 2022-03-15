from flask import Flask, render_template, request, redirect, url_for
import pickle
# from graphviz import render
import numpy as np


app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/',methods=['POST','GET'])
def index():
    return render_template('index.html')


@app.route('/predict_pay',methods=['POST','GET'])
def predict_pay():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction = model.predict(final)
    output = round(prediction[0], 2)
 
    return render_template('output.html',data=output)


if __name__ == '__main__':
     app.run(debug=True)
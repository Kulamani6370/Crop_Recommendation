from flask import Flask, request, render_template
import numpy as np
import pandas
import sklearn
import pickle

#importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standardscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))
#creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("predict.html")


@app.route('/predict',methods=['POST'])
def predict():
     Nitrogen = request.form['Nitrogen']
     Phosphorus = request.form['Phosporus']
     Potassium = request.form['Potassium']
     Temperature = request.form['Temperature']
     Humidity = request.form['Humidity']
     pH_Value = request.form['Ph']
     Rainfall = request.form['Rainfall']

     feature_list = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH_Value, Rainfall]
     single_pred = np.array(feature_list).reshape(1, -1)

     scaled_features = ms.transform(single_pred)
     final_features = sc.transform(scaled_features)
     prediction = model.predict(final_features)
    #  prediction = model.predict(single_pred)

     crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
      19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

     if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
     else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
     return render_template('predict.html',result =result)

if  __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
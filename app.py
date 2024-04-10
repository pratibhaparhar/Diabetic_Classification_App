from flask import Flask,request,render_template
import pickle


app=Flask(__name__)

@app.route("/")
def home_page():
    return render_template('index.html')

@app.route("/predication",methods=['POST'])
def prediction():

    model=pickle.load(open('navie_bayes_dia_model.pkl','rb'))

    pregnancies=int(request.form['pregnancies'])
    glucose=int(request.form['glucose'])
    blood_pressure=int(request.form['bloodPressure'])
    skin_thickness=int(request.form['skinThickness'])
    insulin=int(request.form['insulin'])
    bmi=float(request.form['bmi'])
    diabetes_pedigree=float(request.form['diabetesPedigree'])
    Age=int(request.form['age'])

    user_input=[[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree,Age]]

    target=[0,1]

    output=model.predict(user_input)

    prediction=target[output[0]]
    
    return render_template('result.html',prediction_result=prediction)

if __name__=="__main__":
    app.run(debug=True)
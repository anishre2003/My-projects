from flask import Flask , render_template,request
import pickle


app=Flask(__name__)

# 1.loading the models
with open("stroke-svc-model.pkl","rb")as f:
    model=pickle.load(f)
with open("smoking-lb-pkl","rb")as f:
    lb_smoking=pickle.load(f)


# 2.defining the function 
def predict_stroke(gender="Male",age=67.00,hypertension="No",heart_disease="Yes",avg_glucose_level=228.69,bmi=36.00,smoking_status="formerly smoked",residence_type="Urban"):
    lst=[]#empty list for storing the value
    if gender=="Female":
        lst=lst+[0]
    elif gender=="Male":
        lst=lst+[1]
    elif gender=="Other":
        lst=lst+[2]
    lst=lst+[age]
    if hypertension=="Yes":
        lst=lst+[1]
    elif hypertension=="No":
        lst=lst+[0]
    if heart_disease=="Yes":
        lst=lst+[1]
    elif heart_disease=="No":
        lst=lst+[0]
    lst=lst+[avg_glucose_level,bmi]
    smoking_status=list(lb_smoking.transform([smoking_status]))
    lst=lst+smoking_status
    if residence_type=="Rural":
        lst=lst+[1,0]
    elif residence_type=="Urban":
        lst=lst+[0,1]  
    result=model.predict([lst])
    # print(result)
    if result==[1]:
        return"person is having stroke"
    elif result==[0]:
        return"person is not having stroke"
    print(result)
    

@app.route("/",methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/about',methods=['GET'])
def about():
    return render_template('about.html')

# contact
@app.route('/contact',methods=['GET'])
def contact():
    return render_template('contact.html')

# methodology
@app.route('/methodology',methods=['GET'])
def methodology():
    return render_template('methodology.html')

# predict

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        gender=request.form.get('gender')
        age=float(request.form.get('age'))
        hypertension=request.form.get('hypertension')
        heart_disease=request.form.get('heart_disease')
        avg_glucose_level=float(request.form.get('avg_glucose_level'))
        bmi=float(request.form.get('bmi'))
        smoking_status=request.form.get('smoking_status')
        Residence_type=request.form.get('Residence_type')
        print(gender,age,hypertension,heart_disease,avg_glucose_level,bmi,smoking_status,Residence_type)
        result=predict_stroke(gender=gender,age=age,hypertension=hypertension,heart_disease=heart_disease,avg_glucose_level=avg_glucose_level,
                       smoking_status=smoking_status,bmi=bmi,residence_type=Residence_type)
        print(result)
        return render_template('index.html',prediction=result)

    return render_template('index.html')



if __name__=='__main__':
    app.run(debug=True)


    
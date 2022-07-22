from flask import Flask,render_template,request
import pickle

app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])


def home():
 if request.method=='POST':
    model=pickle.load(open('randomfs.pkl','rb'))
    user_input_1=request.form.get('Pregnancies')
    #user_input=float(user_input)
    pridiction = model.pridict([[user_input_1]])
    print(pridiction)
                                 
 return render_template('index.html', pridiction="you are Diabetes")

if __name__=='__main__':
    app.run(debug=True)
    
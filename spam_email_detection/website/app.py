from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/", methods=["Get", "POST"])

def index():
    result = "" 

    if request.method == "POST":
        
        message = request.form["message"]
        data = vectorizer.transform([message])
        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "🚫 Spam Message"
        else:
            result = "✅ Not Spam"

    return render_template("index.html", result = result)

if __name__ == "__main__":
    app.run(debug=True)
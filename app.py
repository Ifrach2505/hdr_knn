import json
import base64
from pymongo import MongoClient
from flask import Flask, flash, request, redirect, url_for, render_template, session
import bcrypt
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow as tf
from knn_predict import vector_predict
import PIL


# set app as a Flask instance
app = Flask(__name__)
app.secret_key = "testing"


def MongoDB():
    client = MongoClient('localhost', 27017)
    db = client.get_database('users')
    users = db.register
    return users


users = MongoDB()


@ app.route("/", methods=['post', 'get'])
def index():
    message = ''
    # if method post in index
    if "email" in session:
        return redirect(url_for("logged_in"))
    if request.method == "POST":
        user = request.form.get("fullname")
        email = request.form.get("email")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")
        # if found in database showcase that it's found
        user_found = users.find_one({"name": user})
        email_found = users.find_one({"email": email})
        if user_found:
            message = 'There already is a user by that name'
            return render_template('index.html', message=message)
        if email_found:
            message = 'This email already exists in database'
            return render_template('index.html', message=message)
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('index.html', message=message)
        else:
            # hash the password and encode it
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
            # assing them in a dictionary in key value pairs
            user_input = {'name': user, 'email': email, 'password': hashed}
            # insert it in the record collection
            users.insert_one(user_input)

            # find the new created account and its email
            user_data = users.find_one({"email": email})
            new_email = user_data['email']
            # if registered redirect to logged in as the registered user
            return render_template('logged_in.html', email=new_email)
    return render_template('index.html')


@ app.route("/login", methods=["POST", "GET"])
def login():
    message = 'Please login to your account'
    if "email" in session:
        return redirect(url_for("logged_in"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        email_found = users.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']
            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect(url_for('logged_in'))
            else:
                if "email" in session:
                    return redirect(url_for("logged_in"))
                message = 'Wrong password'
                return render_template('login.html', message=message)
        else:
            message = 'Email not found'
            return render_template('login.html', message=message)
    return render_template('login.html', message=message)


@ app.route('/logged_in')
def logged_in(predcition=None):
    if "email" in session:
        email = session["email"]
        return render_template('logged_in.html', email=email, predcition=predcition)
    else:
        return redirect(url_for("login"))


@ app.route('/predict', methods=['POST'])
def predict():
    if "email" in session:
        email = session["email"]
    if request.method == 'POST':
        init_Base64 = 21
        final_pred = None
        draw = request.form['url']
        draw = draw[init_Base64:]
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        vect = np.asarray(resized, dtype="uint8")
        final_pred = vector_predict(vect)
        print(f'final_pred {final_pred}')
        return render_template('logged_in.html', prediction=final_pred)
    else:
        return redirect(url_for("login"))


@ app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return render_template("signout.html")
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

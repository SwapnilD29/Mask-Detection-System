from flask import Flask, render_template, request, url_for, redirect, flash, Response, session
from flask_sqlalchemy import SQLAlchemy
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
import uuid
import smtplib
#import subprocess
import os

app = Flask(__name__)
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLAlchemy_TRACK_MODIFICATIONS'] = True
app.config['SECRET_KEY'] = uuid.uuid4().hex

# Email
#app.config['MAIL_SERVER'] = "smtp.googlemail.com"
#app.config['MAIL_PORT'] = 587
#app.config['MAIL_USE_TLS'] = True
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = "sender@gmail.com"  # sender  email-id
app.config['MAIL_PASSWORD'] = "password"  # sender email password
mail = Mail(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(20), unique=False)
    lname = db.Column(db.String(20), unique=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), unique=False)
    password = db.Column(db.String(80), nullable=False)

    def get_reset_token(self, expires_sec=1800):
        s = Serializer(app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id': self.id}).decode('utf-8')

    @staticmethod
    def verify_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


# load our serialized face detector model from disk
prototxtPath ="face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

user_mail = []
hr_email = 'hr-mailid@gmail.com'  # receiver hr email-id


def send_email():
    mail_id = user_mail[-1]
    user = User.query.filter_by(email=mail_id).first()
    print("user",user)
    msg = Message('Attention',
                  sender='noreply@demo.com',
                  recipients=[hr_email])
    msg.body = 'Dear HR,\n Your employee,\n Name: '+user.fname+' '+user.lname + ',\n has ' \
        'removed mask in the public please ask them to wear mask otherwise strict action will be taken '
    with app.app_context():
        mail.send(msg)


def gen_frames():
    count = 0
    cap = cv2.VideoCapture(0)

    while True:
        flag, frame = cap.read()
        frame = imutils.resize(frame, width=800)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)
                count += 1
                if count == 60:
                    send_email()
                    count = 0

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/register/', methods=['GET', 'POST'])
def register():
    """Register Form"""
    if request.method == 'POST':
        new_user = User(
            fname=request.form['FirstName'],
            lname=request.form['LastName'],
            email=request.form['email'],
            phone=request.form['PhoneNumber'],
            password=bcrypt.generate_password_hash(request.form['pwd']))

        db.session.add(new_user)
        db.session.commit()
        flash('Successfully created new user!')
        return render_template('login.html')
    return render_template('register.html')


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/mask', methods=['GET'])
def mask():
    user_mail = session['username']
    return render_template('mask.html')


@app.route('/Loginadmin', methods=['GET'])
def Loginadmin():
    return render_template('Loginadmin.html')

@app.route('/admin', methods=['GET'])
def admin():
    return render_template('admin2.html')

@app.route('/notific', methods=['GET'])
def notific():
    return render_template('notific.html',x=notification())

def notification():
    if len(user_mail)==0:
        return "There are no notifications!"
    else:
        mail_id = user_mail[-1]
        user = User.query.filter_by(email=mail_id).first()
        return user.fname+ " "+user.lname+" has remove mask!"
    #return render_template()

@app.route('/video', methods=['GET'])
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["email"]
        passw = request.form["password"]

        login = User.query.filter_by(email=uname).first()
        user_mail.append(login.email)
        session['username'] = login.email
        #print(login)
        #print(session['username'])
        if login:
            if bcrypt.check_password_hash(login.password, passw):
                return redirect(url_for('mask'))

        # if login is not None:
#   return redirect(url_for('home'))
    return render_template("Login.html")

@app.route("/login_ad", methods=["GET", "POST"])
def login_ad():
    if request.method == "POST":
        uname = request.form["email"]
        passw = request.form["password"]

        #login = User.query.filter_by(email=uname).first()
        #session['username'] = login.email
        #print(login)
        #print(session['username'])
        if login:
            if uname=="admin" and passw=="admin":
                return redirect(url_for('admin'))
            return redirect(url_for('Loginadmin'))


@app.route('/login_validation', methods=['POST'])
def loginval():
    email = request.form.get('email')
    password = request.form.get('password')
    return f"{email} {password}"


if __name__ == '__main__':
    app.run(debug=True)

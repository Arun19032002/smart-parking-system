from flask import Flask, render_template, flash, request, session
from flask import render_template, redirect, url_for, request
import mysql.connector
import datetime
import time
import easyocr
import sys

app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/AdminLogin")
def AdminLogin():
    return render_template('AdminLogin.html')


@app.route("/UserLogin")
def UserLogin():
    return render_template('UserLogin.html')


@app.route("/NewUser")
def NewUser():
    return render_template('NewUser.html')


@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb ")
    data = cur.fetchall()

    return render_template('AdminHome.html', data=data)


@app.route("/AdminReport")
def AdminReport():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM entrytb  ")
    data = cur.fetchall()
    return render_template('AdminReport.html', data=data)


@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    error = None
    if request.method == 'POST':

        if request.form['Name'] == 'admin' and request.form['Password'] == 'admin':
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')

            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb ")
            data = cur.fetchall()

            return render_template('AdminHome.html', data=data)

        else:
            data = "UserName or Password Incorrect!"

            return render_template('goback.html', data=data)


@app.route("/newuser", methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':

        name = request.form['t1']

        mobile = request.form['t2']
        email = request.form['t3']
        vno = request.form['t6']
        username = request.form['t4']
        Password = request.form['t5']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' or  VehicleNo='" + vno + "'")
        data = cursor.fetchone()
        if data:
            data = "Already Register  VehicleNo Or UserName!"
            return render_template('goback.html', data=data)

        else:
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
            cursor = conn.cursor()
            cursor.execute(
                "insert into regtb values('','" + name + "','" + mobile + "','" + email + "','" + vno + "','" + username + "','" + Password + "')")
            conn.commit()
            conn.close()
            import LiveRecognition  as liv

            liv.att()
            del sys.modules["LiveRecognition"]
            data = "Record Saved!"

            return render_template('goback.html', data=data)


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':
        username = request.form['Name']
        password = request.form['Password']
        # session['uname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' and Password='" + password + "'")
        data = cursor.fetchone()
        if data is None:

            alert = 'Username or Password is wrong'
            return render_template('goback.html', data=alert)



        else:

            session['uname'] = data[4]
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
            # cursor = conn.cursor()
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb where username='" + username + "' and Password='" + password + "'")
            data = cur.fetchall()

            return render_template('UserHome.html', data=data)


@app.route("/UserHome")
def UserHome():
    username = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb  where username='" + username + "' ")
    data = cur.fetchall()
    return render_template('UserHome.html', data=data)


@app.route("/UserReport")
def UserReport():
    username = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM entrytb  where VehicleNo='" + username + "' ")
    data = cur.fetchall()
    return render_template('UserReport.html', data=data)


@app.route("/Payment")
def Payment():
    username = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM amttb  where VehicleNo='" + username + "' ")
    data = cur.fetchall()
    return render_template('Payment.html', data=data)


@app.route("/PaymentInfo")
def PaymentInfo():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM amttb  ")
    data = cur.fetchall()
    return render_template('PaymentInfo.html', data=data)


@app.route("/Pay")
def Pay():
    session['id'] = request.args.get('id')
    session['amt'] = request.args.get('amt')
    return render_template('Pay.html', amt=session['amt'])


@app.route("/payy", methods=['GET', 'POST'])
def payy():
    if request.method == 'POST':
        conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
        cursor = conn.cursor()
        cursor.execute(
            "update amttb set Status='Paid' where id='" + session['id'] + "'  ")
        conn.commit()
        conn.close()

        alert = 'Payment Successful'
        return render_template('goback.html', data=alert)

def otp1():
    import random

    n = random.randint(1111, 9999)

    sendmsg(session['mob'], "Your OTP " + str(n))
    session['otp'] = n

    mmmsg = "Your OTP" + str(n);

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    fromaddr = "projectmailm@gmail.com"
    toaddr = session['email']

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Alert"

    # string to store the body of the mail
    body = mmmsg

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, "qmgn xecl bkqv musr")

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()

    return render_template('In.html')



@app.route("/checkotp", methods=['GET', 'POST'])
def checkotp():
    if request.method == 'POST':
        vno = request.form['vno']

        if int(vno)== int(session['otp']):
            vno = session['vno']

            import datetime
            ts = time.time()
            date = datetime.datetime.now().strftime('%d-%b-%Y')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

            status = '0'

            conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
            cursor = conn.cursor()
            cursor.execute("SELECT * from regtb where VehicleNo='" + vno + "'")
            data = cursor.fetchone()
            if data is None:

                alert = 'VehicleNo Not Register '
                return render_template('goback.html', data=alert)

            else:

                for x in range(1, 21):
                    print(x)

                    conn = mysql.connector.connect(user='root', password='', host='localhost',
                                                   database='3vehicleparknumdb')
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT * from entrytb where   Date='" + date + "' and Status='in'  and ParkingNo='" + str(
                            x) + "'  ")
                    data = cursor.fetchone()
                    if data is None:
                        conn = mysql.connector.connect(user='root', password='', host='localhost',
                                                       database='3vehicleparknumdb')
                        cursor = conn.cursor()
                        cursor.execute(
                            "insert into entrytb values('','" + vno + "','" + date + "','" + timeStamp + "','','in','" + str(
                                x) + "')")
                        conn.commit()
                        conn.close()
                        status = '1'

                        return render_template('In.html', data=str(x))


                if status == "0":
                    alert = 'Parking lot Full  '
                    return render_template('goback.html', data=alert)

        else:
            alert = 'OTP Incorrect..! '
            return render_template('goback.html', data=alert)



def otp2():
    import random

    n = random.randint(1111, 9999)

    sendmsg(session['mob'], "Your OTP " + str(n))
    session['otp'] = n

    mmmsg = "Your OTP" + str(n);

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    fromaddr = "projectmailm@gmail.com"
    toaddr = session['email']

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Alert"

    # string to store the body of the mail
    body = mmmsg

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, "qmgn xecl bkqv musr")

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()

    return render_template('Out.html')





@app.route("/checkotp1", methods=['GET', 'POST'])
def checkotp1():
    if request.method == 'POST':
        vno = request.form['vno']

        if int(vno)== int(session['otp']):
            vno = session['vno']

            import datetime

            ts = time.time()
            date = datetime.datetime.now().strftime('%d-%b-%Y')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

            conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * from entrytb where vehicleno='" + vno + "' and Date='" + date + "' and Status='in'  ")
            data = cursor.fetchone()
            if data:
                conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
                cursor = conn.cursor()
                cursor.execute(
                    "update entrytb set Status='out',OutTime='" + timeStamp + "' where vehicleno='" + vno + "' and Date='" + date + "' and Status='in'  ")
                conn.commit()
                conn.close()

                #start = datetime.datetime.strptime(str(data[3]), "%H:%M:%S")
                #end = datetime.datetime.strptime(str(data[4]), "%H:%M:%S")

                #difference =  start- end

                #seconds = difference.total_seconds()
                #print('difference in seconds is:', seconds)

                #minutes = seconds / 60
                #print('difference in minutes is:', minutes)

                amt = 100
                uname = session['uname']

                conn = mysql.connector.connect(user='root', password='', host='localhost',
                                               database='3vehicleparknumdb')
                cursor = conn.cursor()
                cursor.execute(
                    "insert into amttb values('','" + uname + "','" + vno + "','" + date + "','" + str(
                        amt) + "','NotPaiid')")
                conn.commit()
                conn.close()

                sendmsg(session['mob'], "Your Parking Amount :" + str(amt))

                alert = ' vehicle out  '
                return render_template('goback.html', data=alert)
            else:
                alert = ' vehicle is not parking  '
                return render_template('goback.html', data=alert)
        else:
            alert = 'OTP Incorrect..! '
            return render_template('goback.html', data=alert)




import cv2
import numpy as np
from skimage.filters import threshold_local
import tensorflow as tf
from skimage import measure
import imutils
import pytesseract
import re
import mysql.connector


def sort_cont(character_contours):
    """
    To sort contours from left to right
    """
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, boundingBoxes),
                                                      key=lambda b: b[1][i], reverse=False))
    return character_contours


def segment_chars(plate_img, fixed_width):
    """
    extract Value channel from the HSV format of image and apply adaptive thresholding
    to reveal the characters on the license plate
    """
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]

    T = threshold_local(V, 29, offset=15, method='gaussian')

    thresh = (V > T).astype('uint8') * 255

    thresh = cv2.bitwise_not(thresh)

    # resize the license plate region to a canoncial size
    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # perform a connected components analysis and initialize the mask to store the locations
    # of the character candidates
    labels = measure.label(thresh, neighbors=8, background=0)

    charCandidates = np.zeros(thresh.shape, dtype='uint8')

    # loop over the unique components
    characters = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255

        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # ensure at least one contour was found in the mask
        if len(cnts) > 0:

            # grab the largest contour which corresponds to the component in the mask, then
            # grab the bounding box for the contour
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # compute the aspect ratio, solodity, and height ration for the component
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])

            # determine if the aspect ratio, solidity, and height of the contour pass
            # the rules tests
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.5 and heightRatio < 0.95

            # check to see if the component passes all the tests
            if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                # compute the convex hull of the contour and draw it on the character
                # candidates mask
                hull = cv2.convexHull(c)

                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    _, contours, hier = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sort_cont(contours)
        addPixel = 4  # value to be added to each dimension of the character
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if y > addPixel:
                y = y - addPixel
            else:
                y = 0
            if x > addPixel:
                x = x - addPixel
            else:
                x = 0
            temp = bgr_thresh[y:y + h + (addPixel * 2), x:x + w + (addPixel * 2)]

            characters.append(temp)
        return characters
    else:
        return None


class PlateFinder:
    def __init__(self):
        self.min_area = 4500  # minimum area of the plate
        self.max_area = 30000  # maximum area of the plate

        self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)  # old window was (5,5)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)  # convert to gray
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)  # sobelX to get the vertical edges
        ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        element = self.element_structure
        morph_n_thresholded_img = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_n_thresholded_img)
        return morph_n_thresholded_img

    def extract_contours(self, after_preprocess):
        _, contours, _ = cv2.findContours(after_preprocess, mode=cv2.RETR_EXTERNAL,
                                          method=cv2.CHAIN_APPROX_NONE)
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)  # index of the largest contour in the area array

            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            rect = cv2.minAreaRect(max_cnt)
            rotatedPlate = plate
            if not self.ratioCheck(max_cntArea, rotatedPlate.shape[1], rotatedPlate.shape[0]):
                return plate, False, None
            return rotatedPlate, True, [x, y, w, h]
        else:
            return plate, False, None

    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y + h, x:x + w]
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img)
            if plateFound:
                characters_on_plate = self.find_characters_on_plate(after_clean_plate_img)
                if (characters_on_plate is not None and len(characters_on_plate) == 10):
                    x1, y1, w1, h1 = coordinates
                    coordinates = x1 + x, y1 + y
                    after_check_plate_img = after_clean_plate_img
                    return after_check_plate_img, characters_on_plate, coordinates
        return None, None, None

    def find_possible_plates(self, input_img):
        """
        Finding all possible contours that can be plates
        """
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []

        self.after_preprocess = self.preprocess(input_img)
        possible_plate_contours = self.extract_contours(self.after_preprocess)

        for cnts in possible_plate_contours:
            plate, characters_on_plate, coordinates = self.check_plate(input_img, cnts)
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(characters_on_plate)
                self.corresponding_area.append(coordinates)

        if (len(plates) > 0):
            return plates
        else:
            return None

    def find_characters_on_plate(self, plate):

        charactersFound = segment_chars(plate, 400)
        if charactersFound:
            return charactersFound

    # PLATE FEATURES
    def ratioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area

        ratioMin = 3
        ratioMax = 6

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

    def preRatioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area

        ratioMin = 2.5
        ratioMax = 7

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect

        if (width > height):
            angle = -rect_angle
        else:
            angle = 90 + rect_angle

        if angle > 15:
            return False
        if (height == 0 or width == 0):
            return False

        area = width * height
        if not self.preRatioCheck(area, width, height):
            return False
        else:
            return True


class NeuralNetwork:
    def __init__(self):
        self.model_file = "./model/binary_128_0.50_ver3.pb"
        self.label_file = "./model/binary_128_0.50_labels_ver2.txt"
        self.label = self.load_label(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.Session(graph=self.graph)

    def load_graph(self, modelFile):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(modelFile, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def load_label(self, labelFile):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(labelFile).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def convert_tensor(self, image, imageSizeOuput):
        """
    takes an image and tranform it in tensor
    """
        image = cv2.resize(image, dsize=(imageSizeOuput, imageSizeOuput), interpolation=cv2.INTER_CUBIC)
        np_image_data = np.asarray(image)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data, axis=0)
        return np_final

    def label_image(self, tensor):

        input_name = "import/input"
        output_name = "import/final_result"

        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        results = self.sess.run(output_operation.outputs[0],
                                {input_operation.outputs[0]: tensor})
        results = np.squeeze(results)
        labels = self.label
        top = results.argsort()[-1:][::-1]
        return labels[top[0]]

    def label_image_list(self, listImages, imageSizeOuput):
        plate = ""
        for img in listImages:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            plate = plate + self.label_image(self.convert_tensor(img, imageSizeOuput))
        return plate, len(plate)

@app.route("/fin")
def fin():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
    cursor = conn.cursor()
    cursor.execute(
        "truncate table temptb   ")
    conn.commit()
    conn.close()


    import LiveRecognition1  as liv1
    liv1.att()
    del sys.modules["LiveRecognition1"]
    return facelogin()





@app.route("/facelogin")
def facelogin():

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
    cursor = conn.cursor()
    cursor.execute("SELECT * from temptb  ")
    data = cursor.fetchone()
    if data is None:

        alert = 'Face  Wrong '
        return render_template('goback.html', data=alert)


    else:
        session['vvno'] = data[2]

        return In()


@app.route("/fout")
def fout():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
    cursor = conn.cursor()
    cursor.execute(
        "truncate table temptb   ")
    conn.commit()
    conn.close()


    import LiveRecognition1  as liv1
    liv1.att()
    del sys.modules["LiveRecognition1"]
    return facelogin1()





@app.route("/facelogin1")
def facelogin1():

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3vehicleparknumdb')
    cursor = conn.cursor()
    cursor.execute("SELECT * from temptb  ")
    data = cursor.fetchone()
    if data is None:

        alert = 'Face  Wrong '
        return render_template('goback.html', data=alert)


    else:
        session['vvno'] = data[2]

        return Out()


@app.route("/In")
def In():
    session['vno'] = ''
    findPlate = PlateFinder()

    # Initialize the Neural Network
    model = NeuralNetwork()
    reader = easyocr.Reader(['en'])

    cap = cv2.VideoCapture(0)
    while (True):
        ret, img = cap.read()
        if ret == True:
            cv2.imshow('original video', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            result = reader.readtext(img)
            for detection in result:
                vno = re.sub(r"[^a-zA-Z0-9]", "", detection[1])
                print(vno)
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                conn = mysql.connector.connect(user='root', password='', host='localhost',
                                               database='3vehicleparknumdb')
                cursor = conn.cursor()
                cursor.execute(
                    "select * from temptb where VehicleNo='" + str(vno) + "'  ")
                data = cursor.fetchone()
                if data is None:
                    print("VehilceNo Not Found")

                else:
                    session['uname'] = data[1]
                    session['mob'] = data[3]
                    session['email'] = data[4]

                    session['vno'] = data[2]
                    cap.release()
                    cv2.destroyAllWindows()
                    return otp1()

            # cv2.waitKey(0)
            possible_plates = findPlate.find_possible_plates(img)
            pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

            if possible_plates is not None:
                for i, p in enumerate(possible_plates):
                    chars_on_plate = findPlate.char_on_plate[i]
                    recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOuput=128)
                    print(recognized_plate)

                    cv2.imshow('plate', p)
                    predicted_result = pytesseract.image_to_string(p, lang='eng',
                                                                   config='--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    print(predicted_result)

                    vno = re.sub(r"[^a-zA-Z0-9]", "", predicted_result)
                    print(vno)

                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    conn = mysql.connector.connect(user='root', password='', host='localhost',
                                                   database='3vehicleparknumdb')
                    cursor = conn.cursor()
                    cursor.execute(
                        "select * from temptb where VehicleNo='" + str(vno) + "'  ")
                    data = cursor.fetchone()
                    if data is None:
                        print("VehilceNo Not Found")

                    else:
                        session['uname'] = data[1]
                        session['mob'] = data[3]
                        session['vno'] = data[2]
                        cap.release()
                        cv2.destroyAllWindows()
                        return otp1()

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break


        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def sendmsg(targetno,message):
    import requests
    requests.post(
        "http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno=" + targetno + "&text=Dear customer your msg is " + message + "  Sent By FSMSG FSSMSS")



@app.route("/Out")
def Out():
    session['vno'] = ''
    findPlate = PlateFinder()

    # Initialize the Neural Network
    model = NeuralNetwork()
    reader = easyocr.Reader(['en'])

    cap = cv2.VideoCapture(0)
    while (True):
        ret, img = cap.read()
        if ret == True:
            cv2.imshow('original video', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            result = reader.readtext(img)
            for detection in result:
                vno = re.sub(r"[^a-zA-Z0-9]", "", detection[1])
                print(vno)
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                conn = mysql.connector.connect(user='root', password='', host='localhost',
                                               database='3vehicleparknumdb')
                cursor = conn.cursor()
                cursor.execute(
                    "select * from temptb where VehicleNo='" + str(vno) + "'  ")
                data = cursor.fetchone()
                if data is None:
                    print("VehilceNo Not Found")

                else:
                    session['uname'] = data[1]
                    session['mob'] = data[3]
                    session['vno'] = data[2]
                    cap.release()
                    cv2.destroyAllWindows()
                    return otp2()

            # cv2.waitKey(0)
            possible_plates = findPlate.find_possible_plates(img)
            pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

            if possible_plates is not None:
                for i, p in enumerate(possible_plates):
                    chars_on_plate = findPlate.char_on_plate[i]
                    recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOuput=128)
                    print(recognized_plate)

                    cv2.imshow('plate', p)
                    predicted_result = pytesseract.image_to_string(p, lang='eng',
                                                                   config='--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    print(predicted_result)

                    vno = re.sub(r"[^a-zA-Z0-9]", "", predicted_result)
                    print(vno)

                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    conn = mysql.connector.connect(user='root', password='', host='localhost',
                                                   database='3vehicleparknumdb')
                    cursor = conn.cursor()
                    cursor.execute(
                        "select * from temptb where VehicleNo='" + str(vno) + "'  ")
                    data = cursor.fetchone()
                    if data is None:
                        print("VehilceNo Not Found")

                    else:
                        session['uname'] = data[1]
                        session['mob'] = data[3]
                        session['vno'] = data[2]
                        cap.release()
                        cv2.destroyAllWindows()
                        return otp2()

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break


        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

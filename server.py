import os
from flask import Flask, request, render_template
import random

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def home():
    return render_template("about.html")
    # return render_template("demo.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/demo")
def perform():
    return render_template("demo.html")

@app.route("/upload", methods=["POST"])
def upload():
    dest=None
    if(request.files.get("file")):
        upload=request.files.get("file")
        filename = upload.filename
        dest='static/input/'+filename
        upload.save(dest)
    else:
        imgfilepath=request.form['imgpath']
        path,filename = os.path.split(imgfilepath)
        dest='static/input/'+filename

    quest=request.form['question']
    print("ImageFile: ",dest)
    print("Question: ",quest)

    str=" python demo.py -image_file_name "+ "'" + dest  +  "' -question '" + quest + "'"

    import subprocess
    process=subprocess.Popen(str, shell=True)
    process.wait()
    print("Program ended")

    labels=[]
    confidence=[]
    lab = open('outputlabel.txt')
    for line in lab:
        print(line)
        labels.append(line)
    lab.close()
    con = open('outputcon.txt')
    for line in con:
        print(line)
        confidence.append(line)
    con.close()
    return render_template("demo.html",inpimg=dest,quest=quest,lab_con=zip(labels,confidence))

if __name__ == "__main__":
    app.run(debug=True)

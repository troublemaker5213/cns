# coding:utf-8
from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
from flask import Flask, jsonify, request
import subprocess
from flask_cors import CORS
import flask
app = Flask(__name__)

CORS(app)  #跨域问题

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        print(file.filename)

        file.save('page/'+file.filename)
        print("done!")
        subprocess.Popen("python test01.py",shell=True)
        #os.system("python test01.py")
    return "done"   #将化成标签json格式

'''
/home/hddb/shc/test_result2/0628/9437c715eb
/home/hddb/shc/test_result2/0628/9437c715eb
/home/hddb/gxd2/pycharm/conv-next-seg/data/youtubevos/test/video/9437c715eb
'''
@app.route('/download',methods=['GET'])
def download():
    no = request.args.get('no')
    print(no)

    return send_from_directory('page',' '*(6-(len(no)-1))+str(no)+'.png')

@app.route('/downloadorien',methods=['GET'])
def downloadorien():
    no = request.args.get('no')
    print(no)
    print('file=','0'*(3 - (len(no)-1))+str(no))
    return send_from_directory('page/orien','0'*(3 - (len(no)-1))+str(no)+'.jpg')

@app.route("/",methods = ['POST','GET'])
def root():
    #return "done"
    return flask.render_template("upload.html")  #测试页面


if __name__ == '__main__':
    app.run(host='localhost',port=5678,debug=True)


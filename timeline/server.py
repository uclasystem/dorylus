from flask import Flask,render_template
import json
import os

app = Flask(__name__,template_folder=".")

@app.route('/',methods=['GET'])
def home():
    return render_template("timeline.html")

def getDataList():
    files=os.listdir(".");
    timestamps=list()
    for f in files:
        if f[-3:]==".ts":
            timestamps.append(f)
    return timestamps


@app.route('/<file>',methods=['GET'])
def renderData(file):
    if file=="datalist":
        return json.dumps(getDataList())
    if file=="default":
        if len(getDataList())!=0:
            file=getDataList()[0]
        else:
            return 'bad request!', 400
    ts=list()
    with open(file) as f:
        cnt=0
        for line in f:
            cnt+=1
            cat=line.split(":")[0]
            t0=(line.split(":")[1]).split(",")[0]
            t1=(line.split(":")[1]).split(",")[1]
            ts.append([cat,t0,t1])
    # sort_by_end(ts)
    return json.dumps(ts)
    

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=7788)
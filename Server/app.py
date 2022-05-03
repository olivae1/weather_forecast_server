from distutils.log import debug
from flask import Flask, send_file
from flask import jsonify
from flask_cors import CORS, cross_origin

import sys
sys.path.append('./WeatherApi')
import weatherausData
import weatherPredict

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#server endpoint, this will only handles GET request which will then call the function for our model
@app.route('/api', methods=['GET'])
@cross_origin()
def index():
    #data2 = weatherausData.funcOne()
    data2 = weatherPredict.funcOne()
    #print(data2)
    #return jsonify(data2)
    return send_file('./image.png', mimetype='image/png', as_attachment=True)

if __name__ == '__main__':
    app.run()
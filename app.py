from flask import Flask, render_template, jsonify, request
from backend.expectimax import expectimax_search
from backend.DQN_test import eval_DQN
import json
import numpy as np


app = Flask(__name__)


@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")


@app.route("/ai", methods=['POST'])
def aiMove():
    data = json.loads(request.data, strict=False)
    grid = np.array(data['grid'])
    algo = data['algorithm']
    if algo == "Expectimax":
        next_move = expectimax_search(grid)
    elif algo == "DQN":
        next_move = eval_DQN(grid)
    return jsonify({'data': int(next_move)})


if __name__ == '__main__':
    app.run(host="0.0.0.0")

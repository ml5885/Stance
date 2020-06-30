from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
from flask import Flask
from flask import jsonify, request
from flask import redirect
from flask import url_for
from flask import render_templategi
import json

import pickle
import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
app.config.from_object(__name__)

catToModel = pickle.load(open("models.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict(message):
    vect_msg = vectorizer.transform([message])
    probs = []
    for cat in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
        pipeline = make_pipeline(vectorizer, catToModel[cat])
        catToModel[cat].predict(vect_msg)
        probs.append(catToModel[cat].predict_proba(vect_msg)[0][1])

        explainer = lime.lime_text.LimeTextExplainer(class_names=[0, 1])
        exp = explainer.explain_instance(message, classifier_fn=pipeline.predict_proba, top_labels=1, num_features=10)
        output_file = 'static/{}_explanation.html'.format(cat)
        exp.save_to_file(output_file)

    return probs

@app.route('/')
def send_js():
  return app.send_static_file('toxicity.html')


@app.route('/flat-ui.css')
def send_css():
  return app.send_static_file('flat-ui.css')


@app.route('/handle_data', methods=['POST', 'GET'])
def handle_data():
  if request.method == 'POST':
    msg = request.form['MSG']
    result = predict(msg)
    result = json.dumps(result)
    return result

@app.route('/toxic')
def toxic():
    return app.send_static_file('toxic_explanation.html')

@app.route('/identity')
def identity():
    return app.send_static_file('identity_hate_explanation.html')

@app.route('/insult')
def insult():
    return app.send_static_file('insult_explanation.html')

@app.route('/obscene')
def obscene():
    return app.send_static_file('obscene_explanation.html')

@app.route('/threat')
def threat():
    return app.send_static_file('threat_explanation.html')

@app.route('/severe')
def severe():
    return app.send_static_file('severe_toxic_explanation.html')

if __name__ == '__main__':
  app.run(port=9875)

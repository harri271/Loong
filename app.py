from flask import Flask, render_template, request
import json

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def home():
    return render_template('BabyLan.html')


@app.route('/LanguageTest')
def another_page():
    return render_template('LanguageTest.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

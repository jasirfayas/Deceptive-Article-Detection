from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import pdb, sys
import worker
app = Flask(__name__, static_url_path='')

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        headline_to_check = request.form['headline']
        body_to_check = request.form['body']
        result = worker.body_scoring(body_to_check)
        result2 = worker.headline_scoring(headline_to_check)
    else:
        result = ''
        result2 = ''
        body_to_check = ''
        headline_to_check = ''
   
    print("headline = " + headline_to_check, file=sys.stderr)
    print("\n")
    print("body = " + body_to_check, file=sys.stderr)
    print("\n")
    print("result = " + str(result), file=sys.stderr)
    print("\n")
    print("result2 = " + str(result2), file=sys.stderr)
    return render_template('index.html',  messages={'result':result, 'sentence':body_to_check, 'result2':result2, 'sentence2':headline_to_check,})

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8083, debug=True)
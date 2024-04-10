from flask import Flask, render_template, send_from_directory


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/hello')
def hello():
    return 'Hello, World!'


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


port = 4002

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)

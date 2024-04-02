from flask import Flask, render_template_string


app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string('''
    <h1>Flask App</h1>     
    <p>Visit <a href="/hello">/hello</a> to see the greeting.</p>
    ''')


@app.route('/hello')
def hello():
    return 'Hello, World!'


port = 4002

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)

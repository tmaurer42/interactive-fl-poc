from flask import Flask, render_template


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


def render_template_by_name(name, **kwargs):
    return render_template(f'{name}/{name}.html', **kwargs)


@app.route('/')
def index():
    return render_template_by_name('index')


@app.route('/hello')
def hello():
    return 'Hello, World!'


port = 4002

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)

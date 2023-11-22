from flask import Flask, request, render_template, redirect, url_for, escape, session


app = Flask(__name__, template_folder='templates')

@app.route('/')
def hello_world():
    return "Hello, world!"

@app.route('/user/<username>')
def user(username):
    return f'User - {username}'


@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'post - {post_id}'

@app.route('/post', methods=['GET', 'POST'])
def post():
    if request.method == 'GET':
        return '/post with GET request'
    elif request.method == 'POST':
        return '/post with POST request'
    else:
        return '123'

@app.route('/hello/<name>')
def hello(name=None):
    print(name)
    return render_template('hello.html', name=name)

@app.route('/redirect_test')
def red_test():
    print('red_test')
    return redirect(url_for('hello', name='test'))


if __name__ == '__main__':
    app.run(debug=True)

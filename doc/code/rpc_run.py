from flask import Flask

from flask_jsonrpc import JSONRPC

# Flask application
app = Flask('application')

# Flask-JSONRPC
jsonrpc = JSONRPC(app, '/api', enable_web_browsable_api=True)


@jsonrpc.method('App.index')
def index() -> str:
    return 'Welcome to Flask JSON-RPC'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
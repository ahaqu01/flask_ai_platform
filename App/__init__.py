from flask import Flask

from App.apis import init_api
from App.ext import init_ext
from App.middleware import load_middleware
from App.settings import envs
from remote_ai_interface import remote_ai_interface_api


def create_app(env):
    app = Flask(__name__)

    app.config.from_object(envs.get(env))

    init_ext(app)

    init_api(app=app)

    remote_ai_interface_api(app=app)  # 远程AI接口

    load_middleware(app)

    return app
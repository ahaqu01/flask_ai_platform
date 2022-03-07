from remote_ai_interface.cv import cv_remote_api


def remote_ai_interface_api(app):
    cv_remote_api.init_app(app)

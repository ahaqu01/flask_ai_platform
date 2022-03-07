
from App.apis.admin_user import admin_user_api
from App.apis.ai_service import ai_service_api
from App.apis.customer_user import customer_user_api
from App.apis.third_developer_user import third_developer_user_api


def init_api(app):
    customer_user_api.init_app(app)
    third_developer_user_api.init_app(app)
    admin_user_api.init_app(app)
    ai_service_api.init_app(app)
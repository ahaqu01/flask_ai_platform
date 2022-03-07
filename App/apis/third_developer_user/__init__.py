from flask_restful import Api

from App.apis.third_developer_user.third_developer_user_api import ThirdDeveloperUsersResource

third_developer_user_api = Api(prefix="/third_developer")

third_developer_user_api.add_resource(ThirdDeveloperUsersResource,'/users/')
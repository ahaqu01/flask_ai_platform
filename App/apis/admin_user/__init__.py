from flask_restful import Api

from App.apis.admin_user.admin_user_api import AdminUsersResource

admin_user_api = Api(prefix = '/admin')

admin_user_api.add_resource(AdminUsersResource, '/adminusers/')
from flask_restful import Api

from App.apis.customer_user.customer_user_api import CustomerUsersResource

customer_user_api = Api(prefix="/customer")

customer_user_api.add_resource(CustomerUsersResource, '/customerusers/')
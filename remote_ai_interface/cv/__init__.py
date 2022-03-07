from flask_restful import Api

from remote_ai_interface.cv.densenet_remote_interface import DensenetRemoteInterface

cv_remote_api = Api(prefix='/cv_remote_api')

cv_remote_api.add_resource(DensenetRemoteInterface, '/densenet/')
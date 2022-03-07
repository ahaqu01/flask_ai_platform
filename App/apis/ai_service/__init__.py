from flask_restful import Api

from App.apis.ai_service.image_classifier_serivce import  ImageClassifierSerivce
from App.apis.ai_service.pytorch_yolov5_service import PytorchYolov5, PytorchYolov5Service

ai_service_api = Api(prefix='/ai_service')

ai_service_api.add_resource(ImageClassifierSerivce, '/predict/')
ai_service_api.add_resource(PytorchYolov5, '/yolov5/')
ai_service_api.add_resource(PytorchYolov5Service, '/yolov5service/')


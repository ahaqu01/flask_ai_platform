from flask import request, jsonify
from flask_restful import Resource, reqparse, fields
from werkzeug.datastructures import FileStorage

from algorithm.densenet.image_classifier import predict_image

parse_base = reqparse.RequestParser()
parse_base.add_argument("image", type=FileStorage, location='files', help="请输入要检测的图片")

image_classifier_fileds = {
    "object_name":fields.String,
}

class ImageClassifierSerivce(Resource):
    def post(self):
        try:
            args = parse_base.parse_args()
            image = args.get("image").read()
            object_name = predict_image(image)
        except Exception as e:
            return jsonify({'err_code': 400})

        return jsonify({'object_name': object_name})  # jsonify 确保 response为 json格式


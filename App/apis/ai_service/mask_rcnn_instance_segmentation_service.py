import json

from flask import jsonify
from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage

from algorithm.mask_rcnn.mask_rcnn_instance_segmentation_by_coco import mask_rcnn_instance_segmentation_api

parse_base = reqparse.RequestParser()
parse_base.add_argument("image", type=FileStorage, location="files", help="请输入要分割的图片")

class InstanceSegmentationService(Resource):
    def post(self):
        try:
            args = parse_base.parse_args()
            image = args.get("image")
            pred_cls = mask_rcnn_instance_segmentation_api(image)

            data_json = json.dumps(pred_cls, ensure_ascii=False, indent=2)

            return data_json

        except Exception as e:
            return jsonify({'err_code': 400})
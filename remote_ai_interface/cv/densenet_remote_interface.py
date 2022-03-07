from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage

parse_base = reqparse.RequestParser()
parse_base.add_argument("image", type=FileStorage, location="files", help="请输入图片")
# parse_base.add_argument("image", type=FileStorage,  help="请输入图片")


class DensenetRemoteInterface(Resource):
    def post(self):
        print("DensenetRemoteInterface" )
        try:
            print("in try")
            args_parse = parse_base.parse_args()
            print("after parse_args")
            image = args_parse.get('image')
            print("after get")
            image.save("image.jpg")
            print("after save")
        except Exception as e:
            pass

        return "data"





import os
from importlib import import_module
from flask import render_template, Response, make_response, Flask
from flask_restful import Resource, Api

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from algorithm.pytorch_yolov5.camera import Camera

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

class PytorchYolov5(Resource):
    def get(self):
        print("in PytorchYolov5")
        """Video streaming home page."""
        # return render_template('index.html')
        # headers = {'Content-Type': 'text/html'}
        # return Response(response=render_template('index.html'))
        return make_response(render_template('index.html'))
        # return make_response(render_template('index.html'), 200, headers)

class PytorchYolov5Service(Resource):
    def get(self):
        """Video streaming route. Put this in the src attribute of an img tag."""
        print("in PytorchYolov5Service" )
        return Response(gen(Camera()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

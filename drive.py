import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import cv2

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.1/self.Ki

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement,steering_angle):
        if(steering_angle > 1):
            steering_angle_for_speed_adj = 1
        elif(steering_angle < -1):
            steering_angle_for_speed_adj = -1
        else:
            steering_angle_for_speed_adj = steering_angle
        # proportional error
        target_speed = self.set_point * (1-(0.8*(np.abs(steering_angle_for_speed_adj))))
        self.error = target_speed - measurement
        if(self.error > 2):
            self.error *= 1
        elif(self.error < 0):
            pass
        elif(self.error < 5):
            self.error *= 0.1

        # integral error
        self.integral += self.error#*np.abs(self.error)
        self.integral = min([1/self.Ki,self.integral])
        
        result = self.Kp * self.error + self.Ki * self.integral
        if(result < 0):
            result *= 0.1
        
        if(measurement < 0.1 and measurement > -0.1):
            result = -1
            steering_angle *= -1
        elif(measurement < 2.5):
            result = 1
        return result,steering_angle,self.integral


controller = SimplePIController(0.1, 0.002)
set_speed =  22
controller.set_desired(set_speed)

class Preprocess():
    use_HLS = True
    image_shape     = (160,320,6)
    
    def __init__(self):
        pass
        
    def preprocess_image(image):
        result          = np.empty(list(Preprocess.image_shape))
        #result[:,:,0:3] = cv2.merge((cv2.equalizeHist(np.uint8(image[:,:,0])), cv2.equalizeHist(np.uint8(image[:,:,1])), cv2.equalizeHist(np.uint8(image[:,:,2])))).astype(float)
        result[:,:,0:3] = np.uint8(image)
        if(Preprocess.use_HLS):
            #result[:,:,0:3]      = cv2.cvtColor((np.uint8(result[:,:,0:3])), cv2.COLOR_RGB2HLS)
            result[:,:,3:6]      = cv2.cvtColor((np.uint8(result[:,:,0:3])), cv2.COLOR_RGB2HSV)
            result[:,:,0:3]      = cv2.cvtColor((np.uint8(result[:,:,0:3])), cv2.COLOR_RGB2HLS)
            #result[:,:,0:3]      = cv2.merge((cv2.equalizeHist(np.uint8(result[:,:,0])), cv2.equalizeHist(np.uint8(result[:,:,1])),cv2.equalizeHist(np.uint8(result[:,:,2])))).astype(float)
            
            for i in range(3):
                result[:,:,i]      = cv2.equalizeHist(np.uint8(result[:,:,i]))

            #result[:,:,3:6]      = result[:,-1::-1,3:6]

        return result
   
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        image_array = Preprocess.preprocess_image(image_array)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle,steering_angle,integral = controller.update(float(speed),steering_angle)

        print(steering_angle, throttle,integral)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        #sio.emit('manual', data={}, skip_sid=True)
        sio.emit('manual', data={}, skip_sid=False)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

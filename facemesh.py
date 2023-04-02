import argparse

import cv2
import mediapipe as mp
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

from utils import add_default_args, get_video_input

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


OSC_ADDRESS = "/wek/inputs"

def send_faces(client: udp_client,
               detections):
    if detections is None:
        return

    # create message and send
    builder = OscMessageBuilder(address=OSC_ADDRESS)
    for detection in detections:
        for landmark in detection.landmark:
            builder.add_arg(landmark.x)
            builder.add_arg(landmark.y)
            builder.add_arg(landmark.z)

    msg = builder.build()
    client.send(msg)


# read arguments
parser = argparse.ArgumentParser()
add_default_args(parser)
args = parser.parse_args()

# create osc client
client = udp_client.SimpleUDPClient(args.ip, args.port)

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    send_faces(client, results.multi_face_landmarks)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
face_mesh.close()
cap.release()

import cv2
import mediapipe as mp
import math
import numpy as np


FONTS = cv2.FONT_HERSHEY_COMPLEX

map_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture('cristiano.mp4')

RightEyeRight = [33]
RightEyeLeft = [133]
LeftEyeRight = [362]
LeftEyeLeft = [263]
LeftIris = [474,475,476,477]
RightIris = [469,470,471,472]


with map_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #  resizing frame
        frame = cv2.flip(frame,1)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        h,w,c = frame.shape
        if results.multi_face_landmarks:
            facepoints =np.array([np.multiply([p.x,p.y],[w,h]).astype(int)
                                 for p in results.multi_face_landmarks[0].landmark])
            (cx,cy),radiusl = cv2.minEnclosingCircle(facepoints[LeftIris])
            (rx,ry), radiusr = cv2.minEnclosingCircle(facepoints[RightIris])
            centerright = np.array([rx,ry],dtype=np.int32)
            cv2.circle(frame,centerright,int(radiusl),(0,0,255),1,cv2.LINE_AA)
            distanceHalf = np.linalg.norm(centerright - facepoints[RightEyeRight])
            distanceAll = np.linalg.norm(facepoints[RightEyeLeft]- facepoints[RightEyeRight])
            ratio = distanceHalf / distanceAll
            if ratio <= 0.4:
                position = 'right'
            if ratio > 0.4 and ratio <= 0.6:
                position= 'Center'
            if ratio > 0.6:
                position = ' left'
        cv2.putText(frame, 'right Eye', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,position,(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break
    cv2.destroyAllWindows()
    cap.release()
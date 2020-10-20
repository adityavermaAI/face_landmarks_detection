import cv2
import dlib


cap = cv2.VideoCapture('sample_720_2x_stable.mp4')
detector =dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    flag, cam_image = cap.read()
    cam_image = cv2.resize(cam_image, (0,0), None, 0.5, 0.5)
    cam_image_gray = cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY)

    faces = detector(cam_image_gray)
    for face in faces:
        print(face)
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()


        cv2.rectangle(cam_image, (x1,y1), (x2,y2), (0,255,0), 2)
        landmarks = predictor(cam_image_gray, face)

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            print(x,y)
            cv2.circle(cam_image, (x,y), 2, (0,255,0), -1)

    
    cv2.imshow('camera', cam_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
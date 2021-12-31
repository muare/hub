import cv2
import paddlehub as hub

model = hub.Module(name='ultra_light_fast_generic_face_detector_1mb_640')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    res = model.face_detection(images=[frame])

    if res and len(res) > 0 and len(res[0]['data']) > 0:        
        for face in res[0]['data']:
            left,right,top,bottom,conf = list(face.values())
            frame = cv2.rectangle(frame,(int(left),int(top)),(int(right),int(bottom)),(0,255,0),2)
            frame = cv2.putText(frame,"conf: %f"%(conf),(int(left),int(top)-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255))
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

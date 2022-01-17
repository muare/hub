import cv2
import paddlehub as hub
import numpy as np
import math

def caculate_euler_angle(rotation_vector, translation_vector):
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    # pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
    pitch, yaw, roll = euler_angles.flatten()
    return pitch, yaw, roll

def draw_pose(im,image_points):
    size = im.shape

    # #2D image points. If you change the image, you need to change vector
    # image_points = np.array([
    #                             (359, 391),     # Nose tip
    #                             (399, 561),     # Chin
    #                             (337, 297),     # Left eye left corner
    #                             (513, 301),     # Right eye right corne
    #                             (345, 465),     # Left Mouth corner
    #                             (453, 469)      # Right mouth corner
    #                         ], dtype="double")

    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner

                            ])

    # Camera internals

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    pitch, yaw, roll = caculate_euler_angle(rotation_vector,translation_vector)
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(1000.0, 0.0, 0.0),(0.0, 1000.0, 0.0),(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # for p in image_points:
    #     cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(im, p1, p2, (255,0,0), 2)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[1][0][0]), int(nose_end_point2D[1][0][1]))
    cv2.line(im, p1, p2, (0,255,0), 2)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[2][0][0]), int(nose_end_point2D[2][0][1]))
    cv2.line(im, p1, p2, (0,0,255), 2)

    if pitch < 0:
        pitch+=180
    else:
        pitch-=180
    cv2.putText(im, "pitch: " + "{:7.2f}".format(pitch), (20, int(size[0]/2 -10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), thickness=2)
    cv2.putText(im, "yaw: " + "{:7.2f}".format(yaw), (20, int(size[0]/2 + 30) ), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), thickness=2)
    cv2.putText(im, "roll: " + "{:7.2f}".format(roll), (20, int(size[0]/2 +70)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), thickness=2)

model = hub.Module(name='face_landmark_localization')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    res = model.keypoint_detection(images=[frame])

    if res and len(res) > 0 and len(res[0]['data']) > 0:        
        for face in res[0]['data']:
            image_points = np.array([
                face[33],
                face[8],
                face[36],
                face[45],
                face[48],
                face[54]
            ]
            ,dtype='double')
            draw_pose(frame,image_points)
            #frame = cv2.rectangle(frame,(int(left),int(top)),(int(right),int(bottom)),(0,255,0),2)
            #frame = cv2.putText(frame,"conf: %f"%(conf),(int(left),int(top)-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255))
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




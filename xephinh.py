import cv2
import mediapipe as mp
import time
import keyinput
lmList = []
w = 640
h = 480
def get_list(hand_landmark):
    for id,landmark in enumerate(hand_landmark):
        x = int(landmark.x*w)
        y = int(landmark.y*h)
        lmList.append([id,x,y])
    return lmList

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

max_num_hands = 1
min_detection_confidence = 0.8
min_tracking_confidence = 0.5
tipIds = [4, 8, 12, 16, 20]
with mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence
) as hands:
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        lmList=[]
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                txt = ''
                # Get the landmarks for gesture
                lmList = get_list(hand_landmarks.landmark)
                if len(lmList)!=0 :
                    fingers = []
                    for id in range(1, 5):
                        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    if fingers[0] == 1 and fingers[1] == fingers[2] == fingers[3] == 0:
                        txt = 'right'
                        keyinput.release_key('a')
                        keyinput.release_key('w')
                        keyinput.press_key('d')
                    elif fingers[3] == 1 and fingers[1] == fingers[2] == fingers[0] == 0:
                        txt = 'left'
                        keyinput.release_key('w')
                        keyinput.release_key('d')
                        keyinput.press_key('a')
                    elif fingers[0] == fingers[3] == 1 and fingers[1] == fingers[2] == 0:
                        txt = 'rotate'
                        keyinput.release_key('a')
                        keyinput.release_key('d')
                        keyinput.press_key('w')
                    else:
                        keyinput.release_key('w')
                        keyinput.release_key('a')
                        keyinput.release_key('d')
                    cv2.putText(image, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    print("FPS:", fps)

cap.release()
cv2.destroyAllWindows()

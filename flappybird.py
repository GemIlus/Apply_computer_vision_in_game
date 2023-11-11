import cv2
import mediapipe as mp
import keyinput
import time

def rock(ip, it, mp_, mt, rp, rt, pp, pt):
    if ip < it and mp_ < mt and rp < rt and pp < pt:
        return True
    return False
THRESHOLD = 0.05
def thumb_index_touch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    distance = ((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)**0.5

    if distance < THRESHOLD:
        return True
    else:
        return False


def paper(ip, it, mp_, mt, rp, rt, pp, pt):
    if ip > it and mp_ > mt and rp > rt and pp > pt:
        return True
    return False

def scissors(ip, it, mp_, mt, rp, rt, pp, pt):
    if ip > it and mp_ > mt and rp < rt and pp < pt:
        return True
    return False

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

max_num_hands = 1
min_detection_confidence = 0.8
min_tracking_confidence = 0.5
jump_threshold = 0.5

with mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence
) as hands:
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
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

                # Get the landmarks for gesture
                ip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                it = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                mp_ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                mt = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                rp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
                rt = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
                pp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
                pt = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
                tp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

                if rock(ip, it, mp_, mt, rp, rt, pp, pt):
                    keyinput.press_key('space')
                    txt = 'rock'
                    # print('jump')
                else:
                    keyinput.release_key('space')
                    txt = 'nothing'
                cv2.putText(image, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    print("FPS:", fps)

cap.release()
cv2.destroyAllWindows()

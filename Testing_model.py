import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dict=pickle.load(open('./model.p','rb'))
model=model_dict['model']
mp_hands = mp.solutions.hands #for all mediapipe operations of hands we use mp.solutions.hands
mp_draw = mp.solutions.drawing_utils #for drawing

hand_mesh = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6) #min detection confidence jitna zyada rakhenge utna mtlb surity rahega that hnn ye hand hi hai jaise agar 0.1 hai to andaaji jisko tisko hand bolne lagega and 0.9 rakhenge to isko badhiya se pura hand dikhega tab hi bolega ki hn hand hai
cap = cv2.VideoCapture(0)
data = []
predicted_char = ""
labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'};
while True:
    ret, frame =cap.read()
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hand_mesh.process(rgb)#Process this image with the type we specified during configuring hand_mesh

    if result.multi_hand_landmarks:
        data_aux = []  # for each image image we'll have separate co-ordinates so everytime hand is detected new array will be created and this will be passed to the model and prediction will be made in runtime
        for i in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,i,mp_hands.HAND_CONNECTIONS)
            for itr in range(len(i.landmark)):
                x = i.landmark[itr].x
                y = i.landmark[itr].y

                data_aux.append(x)
                data_aux.append(y)
        predicted = model.predict([np.asarray(data_aux)]) #we do prediction when all 21 landmarks of images are collected
        predicted_char = labels[int(predicted[0])] #idk y but predicted is a list of 1 elements so we can't access it like predicted but predicted[0] also it's a list of characters so it will be predicted by int(predicted[0])
        print(predicted_char)

    cv2.putText(frame, predicted_char, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (100, 120, 200), 3,
                cv2.LINE_AA)
    cv2.imshow('frame', frame)

    cv2.waitKey(25)
cap.release()
cv2.destroyAllWindows()
import os #to access directories of system

import cv2 #to access and manipulate camera of system


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26 #since we have 26 characters
dataset_size = 100 #Each character will have 100 samples

cap = cv2.VideoCapture(0) #intialize openCV's camera

#iterate over all the folders within ./data and if folders are not present then create the folders and store the alphabets starting from class 0 will contain A,class 1 will contain B......
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read() #it will read whatever is visible in the camera
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA) #it will write in the screen our message with custom font styles
        cv2.imshow('frame', frame) #imshow is used to display whatever .read() function reads
        if cv2.waitKey(1) == ord('q'): #this line means no sooner q is pressed just break out. and yeah jab ye loop break hoga tab hi to neeche wala loop execute hoga and data store hona chalu hoga qki image capture krke store karne ka code to neeche hai
            break

    counter = 0
    while counter < dataset_size: #jab tak 100 samples nai ho jate tab tak loop chalao
        ret, frame = cap.read() #again read whatever is in the screen because above loop have been terminated
        cv2.imshow('frame', frame) #we know this is used to show
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame) #this function is used to save files and the 1st argument is file location,2nd argument is for naming the file and it's type like jpg,jpeg whatsoever and the third argument is the image we want to store here in this case our image is with name as frame.

        counter += 1
# now when the above while loop is executed 100 times now the for loop in the top will move to the next class and do the same process i.e storing next alphabets in their respective classes now again the infinite while loop will execute and the screen will hold untill q is pressed. Honestly that infinite while loop is only for holding the screen and wait for user to be ready and again press q

cap.release()
cv2.destroyAllWindows()
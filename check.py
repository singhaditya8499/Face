import os
import time
import cv2
import numpy as np
import ctypes

subjects = ["Madarchod", "Aditya Singh"]



def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(r'C:\Users\ibm\Desktop\wtf\ww\lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]

    return gray[y:y + w, x:x + h], faces[0]


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []
    for dir_name in dirs:

        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("s", ""))

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        xx = len(os.listdir(subject_dir_path))
        print(xx)
        for image_name in subject_images_names:

            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)

            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data(r"C:\Users\ibm\Desktop\wtf\training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    if face is None:
        return None, 2
    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]

    draw_rectangle(img, rect)
    if confidence>50:
        draw_text(img, "Madarchod!! Sorry", rect[0], rect[1] - 5)

    if confidence>50:
        return img, 0
    else:
        return img, 1

print("Predicting images...")

# load test images
#test_img1 = cv2.imread(r"C:\Users\ibm\Desktop\wtf\test-data\aww.jpg")
#test_img2 = cv2.imread(r"C:\Users\ibm\Desktop\wtf\test-data\test1.jpg")
tt=input("Whether testing?")
cam = cv2.VideoCapture(0)
time.sleep(5)

s = -1
while True:
    ret_val, test_img1 = cam.read()
    predicted_img1, s = predict(test_img1)
    cv2.imshow('my webcam', test_img1)
    if s==0:
        print("Sshhhh!!!! Koi haii..")
        if tt == 'n' or tt == 'N':
            ctypes.windll.user32.LockWorkStation()
            break
        else:
            xx = len(os.listdir(r"C:\Users\ibm\Desktop\wtf\training-data\s1"))
            cv2.imwrite(r"C:\Users\ibm\Desktop\wtf\training-data\s1\\"+str(xx)+".jpg", test_img1)
    else:
        if s==1:
            print("Hello Master Singh!!")
            cv2.imshow(subjects[s], predicted_img1)
            time.sleep(1)
            cv2.destroyAllWindows()
        else:
            print("koii nhii hai!!")
    cv2.waitKey(1000)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
    time.sleep(2)
#predicted_img2, ss = predict(test_img2)
#print("Prediction complete")

# display both images
#cv2.imshow(subjects[s], cv2.resize(predicted_img1, (400, 500)))
#cv2.waitKey(100)
#cv2.imshow(ss, cv2.resize(predicted_img2, (400, 500)))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.waitKey(1)
cv2.destroyAllWindows()






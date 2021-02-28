import cv2

data = 'haarcascade_fullbody.xml'
trined_data = cv2.CascadeClassifier(data)

webcam = cv2.VideoCapture('video.mp4')

while True:
    read_successful, frame = webcam.read()

    if read_successful:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print('video may be end or frame read_unsuccess')
        break

    coordinates = trined_data.detectMultiScale(gray_frame)
    print(coordinates)
    for (x, y, w, h) in coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)

    cv2.imshow('pedestrain_detection', frame)

    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break


print('code complete')

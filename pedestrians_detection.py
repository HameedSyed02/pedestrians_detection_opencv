import cv2

data = 'haarcascade_fullbody.xml'
trined_data = cv2.CascadeClassifier(data)

# image = 'photo.webp'
image = 'photo2.jpg'

img = cv2.imread(image)

image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

coordinates = trined_data.detectMultiScale(image_gray)

print(coordinates)

for (x, y, w, h) in coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 5)

cv2.imshow("Human detection", img)

cv2.waitKey(0)


print("code complete")

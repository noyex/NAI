import cv2


image_path = '/Users/mikolajszechniuk/Library/Mobile Documents/com~apple~CloudDocs/NAI/LAB_6/LAB_6/img.jpg'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

top_left = (150, 100)
bottom_right = (300, 300)
color = (0, 0, 255)
thickness = 1

cv2.rectangle(gray_bgr, top_left, bottom_right, color, thickness)

cv2.imshow('Gray Image with Red Rectangle', gray_bgr)

cv2.imwrite('result1.jpg', gray_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np


logo_path = '/Users/mikolajszechniuk/Library/Mobile Documents/com~apple~CloudDocs/NAI/LAB_6/LAB_6/logo.jpg'
background_path = '/Users/mikolajszechniuk/Library/Mobile Documents/com~apple~CloudDocs/NAI/LAB_6/LAB_6/pjatk.jpg'
logo = cv2.imread(logo_path)
background = cv2.imread(background_path)

gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray_logo, 200, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

logo_height, logo_width = logo.shape[:2]
background = cv2.resize(background, (logo_width, logo_height))

roi = background[0:logo_height, 0:logo_width]

background_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
brightness = 90
background_bg = cv2.add(background_bg, brightness)

logo_fg = cv2.bitwise_and(logo, logo, mask=mask)

combined = cv2.add(background_bg, logo_fg)
background[0:logo_height, 0:logo_width] = combined

cv2.imshow('Overlay Result', background)
cv2.waitKey(0)
cv2.destroyAllWindows()

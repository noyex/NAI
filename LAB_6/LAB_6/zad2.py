import cv2

img_path = '/Users/mikolajszechniuk/Library/Mobile Documents/com~apple~CloudDocs/NAI/LAB_6/LAB_6/noise.jpg'
img = cv2.imread(img_path)
smooth_img = cv2.medianBlur(img, 5)


edge_before = cv2.Canny(img,100,200 )
cv2.imshow('img', edge_before)
edge_after = cv2.Canny(smooth_img,100,200)
cv2.imshow('img1', edge_after)

cv2.waitKey(0)
cv2.destroyAllWindows()
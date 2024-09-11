import cv2

a = [1,2,3,4,5]

for i in a:
    img= cv2.imread(f"./img_{str(i).zfill(3)}")

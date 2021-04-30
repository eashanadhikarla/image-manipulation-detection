import os
import cv2

fileList = os.listdir("/Users/eashan22/Dropbox (LU Student)/Macbook/Desktop/Media Forensics/mini-project-2/Task 1/data/train/")
sorted_fileList = sorted(fileList)

idx = 1
file_ = sorted_fileList[idx]

with open("/Users/eashan22/Dropbox (LU Student)/Macbook/Desktop/Media Forensics/mini-project-2/Task 1/data/train/train_bbox.txt", "r") as reader:
    Lines = reader.readlines()

line = Lines[idx-1]
line = line.split(" ")

print(file_)
# print(line)
print(line[0], line[1], line[2], line[3])
x1, y1, x2, y2 = float(line[0]), float(line[1]), float(line[2]), float(line[3])

# imagePath = "./dataset/coco_synthetic/Tp_513867_476061_362.16_302.48_418.79_343.21_person.png"
image = cv2.imread("/Users/eashan22/Dropbox (LU Student)/Macbook/Desktop/Media Forensics/mini-project-2/Task 1/data/train/"+file_)

start_point = (int(x1), int(y1)) 
end_point = (int(x2), int(y2))

# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
  
# Using cv2.rectangle() method 
# Draw a rectangle with blue line borders of thickness of 2 px 
image = cv2.rectangle(image, start_point, end_point, color, thickness) 

# # show thresh and result    
cv2.imshow("bounding_box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===============================================

# imagePath = "./dataset/coco_synthetic/Tp_513867_476061_362.16_302.48_418.79_343.21_person.png"
# image = cv2.imread(imagePath)

# start_point = (int(3.421600e+02), int(2.745039e+02)) 
# end_point = (int(4.387900e+02), int(3.541598e+02))

# # Blue color in BGR 
# color = (255, 0, 0) 
  
# # Line thickness of 2 px 
# thickness = 2
  
# # Using cv2.rectangle() method 
# # Draw a rectangle with blue line borders of thickness of 2 px 
# image = cv2.rectangle(image, start_point, end_point, color, thickness) 

# # show thresh and result    
# cv2.imshow("bounding_box", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
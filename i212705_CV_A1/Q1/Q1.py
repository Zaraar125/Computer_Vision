import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
folders=os.listdir('coil-20')

for folder in folders:
    temp_path = f'coil-20\\{folder}\\obj{int(folder.split('_')[1])}__0.png'
    
    image_paths = [f"{temp_path[:-5]}{i}.png" for i in range(16)]

    sorted_image_list=[]
    bf = cv2.BFMatcher()
    for i in range(15):
        if i==14:
            break
        if sorted_image_list==[]:
            sorted_image_list.append(image_paths[0])
            first_image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
        else:
            first_image = cv2.imread(sorted_image_list[-1], cv2.IMREAD_GRAYSCALE)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(first_image, None)
        
        best_matche_value = 1000000
        most_similar_image = None

        for image_path in image_paths:
                if image_path not in sorted_image_list:
                    # Read the current image
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    kp2, des2 = sift.detectAndCompute(img, None)
                    
                    matches = bf.match(des1, des2)
                    s=0
                    for i in matches:
                        s=s+i.distance

                    if s<best_matche_value:
                        best_matche_value=s
                        most_similar_image = image_path

        sorted_image_list.append(most_similar_image)
        # print(sorted_image_list[-2],sorted_image_list[-1])
    panaroma=[]
    for i in sorted_image_list:
        panaroma.append(cv2.imread(i,cv2.IMREAD_GRAYSCALE))
    panaroma=np.concatenate(panaroma,1)
    # plt.imshow(panaroma)
    # plt.show()
    cv2.imwrite(f'{folder}.png', panaroma)



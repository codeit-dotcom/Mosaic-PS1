#!/usr/bin/env python
# coding: utf-8

# In[17]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from IPython.display import display as display_dataframe
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from copy import deepcopy
from keras.models import load_model
from collections import deque
from skimage import data, io, filters
import matplotlib.patches as mpatches
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb



def process_image(image):
    
    
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,55,25)
#     cv2.imshow("th",image)

    return image

class Roi:
    def __init__(self, roi, x, y, w, h):
        self.roi = roi
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def display_image(image, title='image'):
    '''
    uses openCV to display 1 image in a new window
    '''
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


def get_rectangle_boundary(segments, min_area=50):
    '''
    returns rect_coordinates, np.array(rect_segments)
    '''
    
    new_segments = np.zeros(segments.shape, dtype=segments.dtype)
    rect_coords = []

    count = 1
    for region in regionprops(segments):
        if region.area >= min_area:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect_coords.append(region.bbox)
            new_segments[minr:maxr, minc:maxc] = count
            count += 1
    
    return rect_coords, new_segments    



def order_in_pandas_dataframe(coordinates):
    '''
    takes a list of rectangular coordinates and returns a pd
    dataframe such that the words represented by the rectangular
    coordinates are in the order that they appear in the sentence
    '''
    df = pd.DataFrame(coordinates, columns=['y-', 'x-', 'y+', 'x+'])

    df['line'] = 0
    count = 1
    for i in range(df.shape[0]):
        if df['line'].iloc[i] == 0:
            #the first unmarked word comes into a new line
            df['line'].iloc[i] = count
            miny, maxy = df[['y-', 'y+']].iloc[i]
            
            #find all the words in the same line
            for j in range(df.shape[0]):
                if df['line'].iloc[j] == 0:
                    midy = (df['y-'].iloc[j] + df['y+'].iloc[j])/2
                    if midy < maxy and midy > miny:
                        df['line'].iloc[j] = count
            count += 1
            
    df = df.sort_values(['line', 'x-'])
            
    return df
    
def fix_margin(image, pixels_in_margin):
    p = pixels_in_margin
    temp = np.lib.pad(np.array(image), ((p, p), (p, p)), 'constant', constant_values=255)
    
    #fixing vertical margin
    horizontal_sum = temp.sum(axis=1)
    all_white = max(horizontal_sum)
    start = -1
    end = 0
    for i in range(horizontal_sum.shape[0]):
        if horizontal_sum[i] != all_white:
            if start == -1:
                start = i
            end = i
    temp = temp[start-p : end+p, :]
                
    
    #fixing horizontal margin
    vertical_sum = temp.sum(axis=0)
    all_white = max(vertical_sum)
    start = -1
    end = 0
    for i in range(vertical_sum.shape[0]):
        if vertical_sum[i] != all_white:
            if start == -1:
                start = i
            end = i
    temp = temp[:, start-p : end+p]
    
    return temp



def keras_predict(model, image):
    processed = keras_process_image(image)
    # print("processed: " + str(processed.shape))
    

    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    
    # print(max(pred_probab))
    return pred_class , max(pred_probab)


def keras_process_image(img):
    image_x = 32
    image_y = 32
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img 

def predict(image):
    model4 = load_model('D:/downloads/mosaic_ps1/model1.h5')

#     pictures_folder = "D:/downloads/images/42.jpg"
    letter_count = {0: 'CHECK', 1: 'क', 2: 'ख', 3: 'ग', 4: 'घ', 5: 'ङ', 6: 'च',
                        7: 'छ', 8: 'ज', 9: 'झ', 10: 'ञ',
                        11: 'ट',
                        12: 'ठ', 13: 'ड', 14: 'ढ', 15: 'ण', 16: 'त', 17: 'थ',
                        18: 'द',

                        19: 'ध', 20: 'न', 21: 'प', 22: 'फ',
                        23: 'ब',
                        24: 'भ', 25: 'म', 26: 'य', 27: 'र', 28: 'ल', 29: 'व', 30: 'श',
                        31: 'ष',32: 'स', 33: 'ह',
                        34: 'क्ष', 35: 'त्र', 36: 'ज्ञ', 37: 'CHECK'}



    
    
#     image = cv2.imread(pictures_folder)
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
#     print(img.shape)

    height, width = img.shape[:2]
    if(height>700): #needs to be tune
        img = cv2.resize(img,(int(0.06*width),int( 0.06*height)), interpolation = cv2.INTER_CUBIC) #needs to be tune
    else:
        img=cv2.resize(img,(200,50))
    # img=cv2.bitwise_not(img)
    image = process_image(img)
#     print("musibat")
#     print(image.shape)

    rect_coords, rect_segments = get_rectangle_boundary(felzenszwalb(image, scale=4000, sigma=1, min_size=20)) #needs to be tune
#     display_image(image=mark_boundaries(image, rect_segments), title='Segmented Image')
#     print("hi")
#     print(rect_coords)


    rect_coords_in_order = order_in_pandas_dataframe(rect_coords)



    words = [image[c['y-']:c['y+'], c['x-']:c['x+']] for _, c in rect_coords_in_order.iterrows()]

#     print("words",words)


    char_count = 0

    listy = []
    seg_img=list()

    for word in words:
        temp = np.array(word) 
        temp_temp = np.array(temp) 
        temp[:word.sum(axis=1).argmin()+1, :] = 255
        temp_temp[:word.sum(axis=1).argmin(), :] = 255
        vertical_sum = temp.sum(axis=0)
        all_white = max(vertical_sum)
        start = -1
    #     end = -1
    #     print(vertical_sum.shape)
        for i in range(vertical_sum.shape[0]):
            if start == -1:
                if vertical_sum[i] != all_white:
                    start = i
            else:
                margin = 3 #ntbt
                if vertical_sum[i] == all_white:

    #                 character_to_save = 
   

#                     display_image(fix_margin(temp_temp[:, start:i], margin))
        #                 print((fix_margin(temp_temp[:, start:i], margin).shape)
                    seg_img.append(fix_margin(temp_temp[:, start:i], margin))
#                     cv2.imwrite(str(char_count)+'.png', fix_margin(temp_temp[:, start:i], margin))
                    char_count += 1
                    start = -1
        char_count += 1
#     print(char_count)
    text=[]
    for i in range(len(seg_img)):
    #     print("shape")
        h,w=seg_img[i].shape
#         print(seg_img[i].shape)
        if (h*w)<300:  #ntbt
            continue
        else:
        #     large = cv2.imread(seg_img[i])
            larges = cv2.resize(seg_img[i], (32, 32))
#             cv2.imshow('larges',larges)
        #     gray_image = cv2.cvtColor(larges,cv2.COLOR_BGR2GRAY)
            im_bw = cv2.bitwise_not(larges)
#             cv2.imshow('im_bw',im_bw)
#             print("bina")
#             print(im_bw.shape)    
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
            classs , prob = keras_predict(model4, im_bw)
            d = str(letter_count[classs])
            text.append(d)
#             print(d)
#             print(prob)
        #     print(classs)
#     print(''.join(text))    
    return text


#  

def test():
    '''
    We will be using a similar template to test your code
    '''
#     image_paths = ['./image1','./image2',',./imagen']
#     correct_answers = [list1,list2,listn]
#     score = 0
#     multiplication_factor=2 #depends on character set size

#     for i,image_path in enumerate(image_paths):
#     image_path = "D:/downloads/images/38.jpg"
    image = cv2.imread(image_path) # This input format wont change
    answer = predict(image) # a list is expected
    print(''.join(answer))# will be the output string

#         n=0
#         for j in range(len(answer)):
#             if correct_answers[i][j] == answer[j]:
#                 n+=1
                
#         if(n==len(correct_answers[i])):
#             score += len(correct_answers[i])*multiplication_factor

#         else:
#             score += n*2
        
    
#     print('The final score of the participant is',score)


if __name__ == "__main__":
    test()







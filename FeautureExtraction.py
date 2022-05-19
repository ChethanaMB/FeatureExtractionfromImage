#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import json
import glob
import numpy as np
import os
import cv2 as cv
import cv2
import pandas as pd
import PIL.Image, PIL
import matplotlib.pyplot as plt
from PIL import ImageDraw


# In[2]:


from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils


# In[3]:


file_ = "/home/chethana/Datasets/HearshDataLeftRigh/20220131_114220/20220131_114220_left_4.json"


# In[4]:


img = cv2.imread(file_.replace(".json", ".png"))


# In[5]:


#To get the perpendiculars
def get_ppoints(p1,p2, N=5):
    import math

    (x1,y1) = p1
    (x2,y2) = p2

    vx = x2-x1
    vy = y2-y1

    length = math.sqrt( vx*vx + vy*vy )

    ux = -vy/length
    uy = vx/length

    x3 = float(x1 + N/2 * ux)
    y3 = float(y1 + N/2 * uy)

    x4 = float(x1 - N/2 * ux)
    y4 = float(y1 - N/2 * uy)

    x5 = float(x2 + N/2 * ux)
    y5 = float(y2 + N/2 * uy)

    x6 = float(x2 - N/2 * ux)
    y6 = float(y2 - N/2 * uy)

    pts = [(x3,y3),(x4,y4), (x5,y5), (x6,y6)]
    return pts


# In[6]:


#To find the distance between two points
import math
def pixel_dist(p1,p2):
    [(x1,y1),(x2,y2)] = [p1,p2]
    vx = x2-x1
    vy = y2-y1
    length = math.sqrt( vx*vx + vy*vy )
    return length


# In[62]:


#to draw polygobns to given pouints
import imutils
def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    #converts the mask to a pillow image
    mask = PIL.Image.fromarray(mask)
    
    xy = list(map(tuple, polygons))
    ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=0)
    mask = np.array(mask, dtype=bool)
    plt.imshow(mask)
    return mask



# In[8]:


def Average(lst):
    return sum(lst) / len(lst)


# In[9]:


#To get the contour points at some step size and getting every point's perpendicular
def drawperp(list):
    cnts = []
    for i in range(0,(len(list)-100), 100):
        p1 = list[i]
        p2 = list[i+1]
        pts = get_ppoints(p1,p2, N=100)
        [(x3,y3),(x4,y4), (x5,y5), (x6,y6)] = pts
        cnts.append([(int(x3),int(y3)),(int(x4),int(y4))])
        cnts.append([(int(x5),int(y5)),(int(x6),int(y6))])
        (x1,y1) = p1
        (x2,y2) = p2
    return cnts


# In[10]:


#To get the middle points
def getmps(points):
    mps = []
    thicknesses = []
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            dist = pixel_dist((x1,y1),(x2,y2))
            thicknesses.append(dist)
            mp = (int((x2+x1)/2), int((y2+y1)/2))
            mps.append(mp)
            cv2.line(common_pts_msk,(x1,y1),(x2,y2),(255,255,255),2)
            mps = sorted(mps , key=lambda k: [k[1], k[0]])
    return mps, thicknesses


# In[11]:


#To Connect all the midpoints
def mpconnect(middlepoints):
    lengths = []
    for i in range(len(mps)-1):
        mp1 = mps[i]
        mp2 = mps[i+1]
        length = pixel_dist(mp1,mp2)
        lengths.append(length)
        cv2.line(common_pts_msk,mp1,mp2,(255,255,255),2)
    plt.figure(figsize=(10,10))
    plt.imshow(common_pts_msk)
    plt.show()
    return lengths


# In[12]:


def get_opmask(ctr):
    #       To draw the perpendicular lines
    # Green color in BGR
    color = (255, 255, 255)

    # Line thickness of 9 px
    thickness = 1
    
    op_mask = np.zeros([height,width])
    for [sp,ep] in cnts: #xys for json points
            op_mask = cv2.line(op_mask, sp, ep, color, thickness)
    return op_mask


# In[85]:


img = cv2.imread(file_.replace(".json", ".png"))
# print(img.shape)
gt_f = open(file_)
gtdata = json.load(gt_f)

height, width = gtdata['imageHeight'], gtdata['imageWidth']
img_id = gtdata['imagePath']
gt_cane_mask = np.zeros([height,width])
gt_spur_mask = np.zeros([height,width])
gt_img_mask = np.zeros([height, width])
gt_trunk_mask = np.zeros([height, width])
gt_maincordon_mask = np.zeros([height, width])

contour_points = []
tcontour_points = []
scontour_points = []
for shape_no, shape in enumerate(gtdata['shapes']):
    label = shape['label']
    points = shape['points']
    if len(points)>2 and label == 'cane':
        print(shape_no, label)
        mask = polygons_to_mask((height,width), points)
        mask = mask.clip(0, 255).astype("uint8")
        cane_contour_pts = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )[0][0]
        area = cv2.contourArea(cane_contour_pts)
        im = cv2.drawContours(mask,cane_contour_pts,-1,(155,155,150),3)
        contour_points.append(cane_contour_pts)
        gt_cane_mask += mask

        bottom = tuple(cane_contour_pts[cane_contour_pts[:, :, 1].argmax()][0])
        p3,p4 = bottom
        
        print("Ditsance between the can and the trunk is: ", pixel_dist(top,bottom))

#         plt.plot([p1,p3], [p2,p4])
#         plt.show()
        
        cnt_points = cane_contour_pts.tolist()
        cnt_list = [item for sublist in cnt_points for item in sublist]
        cnts = drawperp(cnt_list)
        
        cane_filled_mask = cv2.fillPoly(mask, pts = [cane_contour_pts], color =(255,255,255))
        op_mask = get_opmask(cnts)
            
            
        # To get the common region to the canes and the normal        
        common_pts=(cane_filled_mask != 0) & (op_mask!=0)
        common_pts_msk = common_pts.clip(0, 255).astype("uint8")
        
        #To get houghlines
        minLineLength = 5
        maxLineGap = 2
        lines = cv2.HoughLinesP(common_pts_msk,1,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
        mps, thicknesses = getmps(lines)
        lengths = mpconnect(mps)
        print("The Thickness of the cane is : ", Average(thicknesses))
        print("The Length of the cane in pixels is : ", sum(lengths))
        
        
        print("Area of the mask is: ",area)
        
        
        
    if len(points)>2 and label == 'trunk':
        print(shape_no, label)
        mask = polygons_to_mask((height,width), points)
        mask = mask.clip(0, 255).astype("uint8")
        trunk_contour_pts = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )[0][0]
        area = cv2.contourArea(cane_contour_pts)
        im = cv2.drawContours(mask,trunk_contour_pts,-1,(155,155,150),3)
        tcontour_points.append(trunk_contour_pts)
        gt_trunk_mask += mask
        
        top = tuple(trunk_contour_pts[trunk_contour_pts[:, :, 1].argmin()][0])
        p1,p2 = top 
        
        plt.imshow(mask)
        plt.show()
        
        
    if len(points)>2 and label == 'spur':
        print(shape_no, label)
        mask = polygons_to_mask((height,width), points)
        mask = mask.clip(0, 255).astype("uint8")
        spur_contour_pts = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )[0][0]
        area = cv2.contourArea(cane_contour_pts)
        im = cv2.drawContours(mask,spur_contour_pts,-1,(155,155,150),3)
        scontour_points.append(spur_contour_pts)
        gt_spur_mask += mask
        
        x,y,w,h = cv2.boundingRect(spur_contour_pts)
        cv2.rectangle(mask, (x,y), (x+w, y+h), (155,155,150), 1)
        print("Height of spur is : ", h)
        
        plt.imshow(mask)
        plt.show()
        


# In[39]:


top = tuple(cane_contour_pts[cane_contour_pts[:, :, 1].argmin()][0])
print(top)
p1,p2 = top

bottom = tuple(cane_contour_pts[cane_contour_pts[:, :, 1].argmax()][0])
print(bottom)
p3,p4 = bottom


# In[35]:


top[0]


# In[31]:


plt.imshow(mask)


# In[40]:


plt.plot([p1,p3], [p2,p4])


# In[14]:


cnt_points


# In[15]:


len(cnt_points)


# In[16]:


def sum1(lst):
    return sum(lst)


# In[17]:


lst = [2,4,5,6,7,3,70]


# In[18]:


x = sum1(lst)


# In[19]:


cnt_list = [item for sublist in cnt_points for item in sublist]


# In[20]:


cnt_list


# In[21]:


cnts = []
for i in range(0,(len(cnt_list)-50), 50):
    print(cnt_list[i])
    p1 = cnt_list[i]
    p2 = cnt_list[i+1]
    print(p1, p2)
    pts = get_ppoints(p1,p2, N=100)
    [(x3,y3),(x4,y4), (x5,y5), (x6,y6)] = pts
    cnts.append([(int(x3),int(y3)),(int(x4),int(y4))])
    cnts.append([(int(x5),int(y5)),(int(x6),int(y6))])
    (x1,y1) = p1
    (x2,y2) = p2
    plt.plot([x3,x4,x1,x2,x5,x6],[y3,y4,y1,y2,y5,y6])
    plt.show()


# In[22]:


plt.imshow(op_mask)


# In[23]:


plt.imshow(cane_filled_mask)


# In[24]:


common_pts=(cane_filled_mask != 0) & (op_mask!=0)
plt.imshow(common_pts)
plt.show()


# In[25]:


plt.imshow(mask)


# In[26]:


# Green color in BGR
color = (255, 255, 255)
  
# Line thickness of 9 px
thickness = 1

    
cane_filled_mask = cv2.fillPoly(mask, pts = [cane_contour_pts], color =(255,255,255))
op_mask = np.zeros([height,width])
for [sp,ep] in xys:
    op_mask = cv2.line(op_mask, sp, ep, color, thickness)


# In[ ]:


plt.figure(figsize=(25,25))
plt.imshow(op_mask)
plt.show()


# In[ ]:


common_pts=(cane_filled_mask != 0) & (op_mask!=0)


# In[ ]:


plt.figure(figsize=(25,25))
plt.imshow(common_pts)
plt.show()


# In[ ]:


x,y = np.where(common_pts != 0)


# In[ ]:


ppl_pts = []
for pt in zip(x,y):
    ppl_pts.append(pt)


# In[ ]:


import math
def pixel_dist(p1,p2):
    [(x1,y1),(x2,y2)] = [p1,p2]
    vx = x2-x1
    vy = y2-y1
    length = math.sqrt( vx*vx + vy*vy )
    return length


# In[ ]:


common_pts_msk = common_pts.clip(0, 255).astype("uint8")
# edges = cv2.Canny(gray,100,200,apertureSize = 3)
minLineLength = 5
maxLineGap = 2
mps=[]
lines = cv2.HoughLinesP(common_pts_msk,1,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        print((x1,y1),(x2,y2))
        print(pixel_dist((x1,y1),(x2,y2)))
        print(((x2+x1)/2), ((y2+y1)/2))
        mp = (int((x2+x1)/2), int((y2+y1)/2))
        mps.append(mp)
        cv2.line(common_pts_msk,(x1,y1),(x2,y2),(255,255,255),2)


# In[ ]:


for i in range(len(mps)-1):
    mp1 = mps[i]
    mp2 = mps[i+1]
    cv2.line(common_pts_msk,mp1,mp2,(255,255,255),2)


# In[ ]:


plt.figure(figsize=(25,25))
plt.imshow(common_pts_msk)
plt.show()


# In[ ]:


ppl_pts


for i in range(len(ppl_pts)-1):
    fp = ppl_pts[i]
    sp = ppl_pts[i+1]
    dist = pixel_dist(fp,sp)
    print(dist)


# In[ ]:


xys = []
for shape_no, shape in enumerate(gtdata['shapes']):
    label = shape['label']
    points = shape['points']
    for i in range(0,len(points)-10, 10):
#         print(points[i])
        p1 = points[i]
        p2 = points[i+1]
        print(p1,p2)
        pts = get_ppoints(p1,p2, N=100)
        [(x3,y3),(x4,y4), (x5,y5), (x6,y6)] = pts        
        xys.append([(int(x3),int(y3)),(int(x4),int(y4))])
        xys.append([(int(x5),int(y5)),(int(x6),int(y6))])
        (x1,y1) = p1
        (x2,y2) = p2
        plt.plot([x3,x4,x1,x2,x5,x6],[y3,y4,y1,y2,y5,y6])
        plt.show()
print(xys)


# In[ ]:


for i,j in xys:
    print(i,j)


#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import matplotlib.pyplot as plt


# In[14]:


#get_ipython().run_line_magic('matplotlib', 'auto')


# In[15]:


image = cv2.imread('../data/people.jpg')


# In[16]:


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())


# In[24]:


locations, weights = hog.detectMultiScale(image)


# In[18]:


dbg_image = image.copy()
for loc in locations:
    cv2.rectangle(dbg_image, (loc[0], loc[1]), 
                  (loc[0]+loc[2], loc[1]+loc[3]), (0, 255, 0), 2)


# In[22]:


plt.figure(figsize=(12,6))
plt.subplot(121)
plt.title('original')
plt.axis('off')
plt.imshow(image[:,:,[2,1,0]])
plt.subplot(122)
plt.title('detections')
plt.axis('off')
plt.imshow(dbg_image[:,:,[2,1,0]])
plt.tight_layout()
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


def classify(video_src, net, in_layer, out_layer, 
             mean_val, category_names, swap_channels=False):
    cap = cv2.VideoCapture(video_src)

    t = 0
    
    while True:
        status_cap, frame = cap.read()
        if not status_cap:
            break

        if isinstance(mean_val, np.ndarray):
            tensor = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                       1.0, False);
            tensor -= mean_val
        else:
            tensor = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                   mean_val, swap_channels);
        net.setInput(tensor, in_layer);
        prob = net.forward(out_layer);

        prob = prob.flatten()

        r = 1
        for i in np.argsort(prob)[-5:]:
            txt = '"%s"; probability: %.2f' % (category_names[i], prob[i])
            cv2.putText(frame, txt, (0, frame.shape[0] - r*40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2);
            r += 1

        cv2.imshow('classification', frame)
        if cv2.waitKey(1) == 27:
            break
        
    cv2.destroyAllWindows()
    cap.release()


# In[3]:


with open('../data/synset_words.txt') as f:
    class_names = [' '.join(l.split(' ')[1: ]).rstrip() for l in f.readlines()]


# In[4]:


googlenet_caffe = cv2.dnn.readNetFromCaffe('../data/bvlc_googlenet.prototxt', 
                                           '../data/bvlc_googlenet.caffemodel')

classify('../data/shuttle.mp4', googlenet_caffe, 'data', 'prob', (104, 117, 123), class_names)


# In[5]:


resnet_caffe = cv2.dnn.readNetFromCaffe('../data/resnet_50.prototxt', 
                                           '../data/resnet_50.caffemodel')
mean = np.load('../data/resnet_50_mean.npy')

classify('../data/shuttle.mp4', resnet_caffe, 'data', 'prob', mean, class_names)


# In[6]:


with open('../data/imagenet_comp_graph_label_strings.txt') as f:
    class_names = [l.rstrip() for l in f.readlines()]

googlenet_tf = cv2.dnn.readNetFromTensorflow('../data/tensorflow_inception_graph.pb')

classify('../data/shuttle.mp4', googlenet_tf, 
         'input', 'softmax2', 117, class_names, True)


# In[ ]:





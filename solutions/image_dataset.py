import numpy as np
import random
import os
from glob import glob
from skimage import io
from skimage.transform import resize

class ImageDataset(object):
    ###Image dataset using skimage/numpy, slower than tf data api

    def __init__(self, data_dir, h, w, batch_size, crop_proportion, glob_pattern='*/*.jpg'):
        self.data_dir = data_dir
        print(self.data_dir)
        self.h = h
        self.w = w
        self.batch_size = batch_size
        self.idx = 0
        self.crop_proportion = crop_proportion
        self.MEAN = np.reshape(np.array([0.485, 0.458, .407]), [1,1,3])
        if crop_proportion is not None:
            self.load_h = int(self.h/self.crop_proportion)
            self.load_w = int(self.w/self.crop_proportion)
        else:
            self.load_h = self.h
            self.load_w = self.w
        if glob_pattern == None:
            self.do_y = False
            self.filenames = [self.data_dir]
            self.N = 1
            return

        self.do_y = True
        self.filenames = glob(os.path.join(self.data_dir, glob_pattern))
        self.N = len(self.filenames)
        self.label_dict = {'hartebeest':0, 'deer':1, 'sheep':2}
        self.label_list = ['hartebeest', 'deer', 'sheep']
        self.num_labels=len(self.label_list)
        self.labels = [self.label_dict[f.split('/')[-2]] for f in self.filenames]

    def new_epoch(self):
        x_y = list(zip(self.filenames, self.labels))
        random.shuffle(x_y)
        self.filenames, self.labels = zip(*x_y)
        self.idx = 0

    def get_next_batch(self):
        if not self.do_y:
           return np.array([self.load_image(self.filenames[0])])
        if self.idx * self.batch_size >= self.N:
            return None, None
        if (self.idx + 1)  * self.batch_size < self.N:
            batch_files = self.filenames[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
            batch_y = self.labels[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
        else:
            batch_files = self.filenames[self.idx*self.batch_size:]
            batch_y = self.labels[self.idx*self.batch_size:]
        batch = np.array([self.augment_image(self.load_image(f)) for f in batch_files])
        eye = np.eye(self.num_labels)
        batch_y = np.array([eye[label] for label in batch_y])
        self.idx+=1
        return batch, batch_y

    def load_image(self, filename):
        image = io.imread(filename)
        hw = image.shape[:-1]
        
        if hw[0] > hw[1]:
            hd = (hw[0] - hw[1])//2
            crop = image[hd:hd+hw[1],...] 
        elif hw[0] < hw[1]:
            hd = (hw[1] - hw[0])//2
            crop = image[:,hd:hd+hw[0],:]
        elif hw[0] == hw[1]:
            crop = image
        resized = resize(crop, (self.load_h, self.load_w))
        return resized

    def random_crop(self, image):
        ###randomly crops the image to correct size 
        s_h = np.random.randint(image.shape[0]-self.h)
        s_w = np.random.randint(image.shape[1]-self.w)
        cropped = image[s_h:s_h+self.h, s_w:s_w+self.w, :]
        return cropped

    def augment_image(self, image):
        image = image - self.MEAN
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        if self.crop_proportion is not None:
            image = self.random_crop(image)
        return image

        #Add more augmentation if ya want
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
import pydicom
import argparse
import os
import sys
import time
import argparse

# parse the arguments
parser=argparse.ArgumentParser(
    description='load dicom and store it as np-array')
parser.add_argument('--directory', required=True,
                    help='give the name of the directory with dicom files')
args=parser.parse_args()
 
""" gen cube saves a numpy array with [100:500,150:350,150:350] of
    float using the dicom data
"""

class Data():
    """ universal opbject to load and store dicom data """
    def __init__(self,directory):
        self.directory=directory
        self.ld=os.listdir(directory)
        self.ele=self.ld[-1].split('.')[0].split('-')
        self.name="%s-%s"%(self.ele[0],self.ele[1])
        self.quad=None        # define the quad
        self.ci=-1  # i coordinate for selected object 
        self.cj=-1  # j coordinate for selected object 
        self.ck=-1  # k coordinate for selected object 

    def select_pict(self,nr):
        """ select and load one dicom slice nr """
        s="%s/%s-%04d.dcm" % (self.directory,self.name,nr)
        # print(s)
        try:
            ds1=pydicom.dcmread(s)
            im1=ds1.pixel_array[150:350,150:350]
            im1[im1<(-1000)]=-1000
            im1+=1000
            im1[im1>4096]=4096
        except:
            print('error: could not open:',s)
            sys.exit(1)
        return im1

    def MinMax(self):
        """ 
        reads the dicom data and returns the min and max values of each layer
        """
        Min=[]
        Max=[]
        for i in range(100,500):
            s="%s/%s-%04d.dcm" % (self.directory,self.name,i)
            ds1=pydicom.dcmread(s)
            im1=ds1.pixel_array[150:350,150:350]
            Min.append(np.amin(im1))
            Max.append(np.amax(im1))
        return Min, Max
    
    def show(self,nr):
        """ loads and shows a slice as an image """
        im1=self.select_pict(nr)
        plt.ioff()
        plt.clf()
        plt.subplot(111)
        cmap = plt.get_cmap('jet', 500)
        cmap.set_under('white')
        img=plt.imshow(im1,cmap=cmap)
        img.set_clim(0,4096)
        plt.colorbar(cmap=cmap)
        title=self.directory+' slice: '+str(nr)
        plt.title(title)
        plt.show()
        
    def show3(self,k):
        """ selects the image directly from the 3D space """
        if(self.quad is None):
            print('error: load the 3D data firstly')
            return False
        im1=self.quad[k,:,:]
        plt.ioff()
        plt.clf()
        plt.subplot(111)
        cmap = plt.get_cmap('jet', 500)
        cmap.set_under('white')
        img=plt.imshow(im1,cmap=cmap)
        img.set_clim(0,4096)
        plt.colorbar(cmap=cmap)
        title=self.directory+' slice: '+str(k)
        plt.title(title)
        plt.show()
        
    def save(self):
        """ collect and save the data into directory npy """
        filename=self.directory.split('/')[-1] # select the filename
        if('npy' not in os.listdir()):
            os.mkdir('npy')
        t0=time.time()
        quad=np.zeros((400,200,200)) # [100:500,150:350,150:350]
        for i in range(100,500):
            quad[i-100]=self.select_pict(i)
        f=open('npy/%s.npy' % filename,'wb')
        np.save(f,quad)
        self.quad=quad
        print('time:',time.time()-t0,'s')
        return True
    
data=Data(args.directory)
data.save()

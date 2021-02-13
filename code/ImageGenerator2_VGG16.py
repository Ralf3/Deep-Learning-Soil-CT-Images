import numpy as np
import os
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.models import Sequential, model_from_yaml
from tensorflow.keras.layers import Dense, Input, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.keras.models import Model
import joblib
import argparse

# parse the arguments
parser=argparse.ArgumentParser(
    description='load binary npy file and visualize it')
parser.add_argument('--file', required=True,
                    help='give the name of the npy file for training')
parser.add_argument('--object', required=True, type=int,
                    help='use one of the objects as y: 1,2,3,4,5')
args=parser.parse_args()

# read one selected volume
V=np.load('npy/%s' % args.file)

def build_objects(V):
    objects=[1,2,3,4,5,6]
    hist,edges=np.histogram(V,bins=6)
    res={}
    for i in objects:
        V1=np.zeros((V.shape[0],V.shape[1],V.shape[2]))
        V1=V1.astype('int')
        V1[(V>=edges[i-1])&(V<edges[i])]=1       # solid 
        res[i]=V1
    return res

res= build_objects(V)

def porosity(sel):
    global res
    por=np.zeros(res[sel].shape[0])
    for i in range(res[sel].shape[0]):
        por[i]=np.sum(res[sel][i])
    return 1.0-por/(V.shape[1]*V.shape[2])

por={1:porosity(1), 2:porosity(2), 3: porosity(3), 4: porosity(4),
     5: np.minimum(porosity(5),porosity(6))}
del res # save space
# prepare the images as RGB

def RGB(V):
    # build a 4D volume
    V4=np.zeros((V.shape[0],V.shape[1],V.shape[2],3))
    objects=[1,2,3]
    hist,edges=np.histogram(V,bins=3)
    for i in range(V4.shape[0]):
        im1=np.copy(V[i])
        im2=np.copy(V[i])
        im3=np.copy(V[i])
        im1[im1>edges[1]]=0
        im2[(im2<=edges[1])&(im2>edges[2])]=0
        im3[im3<=edges[2]]=0
        V4[i,:,:,0]=im1
        V4[i,:,:,1]=im2
        V4[i,:,:,2]=im3
    V4[:,:,:,0]/=np.max(V4[:,:,:,0])
    V4[:,:,:,1]/=np.max(V4[:,:,:,1])
    V4[:,:,:,2]/=np.max(V4[:,:,:,2])
    return V4

V4=RGB(V)


##################################################################

#CNN---VGG16
#example from github
#URL: https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py

import os

inputs = Input(shape=(200, 200, 3))

# Block 1
x = Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv1')(inputs)
x = Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv1')(x)
x = Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv1')(x)
x = Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv2')(x)
x = Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv1')(x)
x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv2')(x)
x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv1')(x)
x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv2')(x)
x = Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

# regression block
x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='regression1')(x)
x = Dense(4096, activation='relu', name='regression2')(x)
outputs = Dense(1,activation='linear', name='predictions')(x)

# Create model.
model = Model(inputs, outputs, name='vgg16')
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

x1=V4
y1=por[int(args.object)]
x_train,x_test,y_train,y_test=train_test_split(x1, y1, test_size=0.3)

time1=time.time()
res=model.fit(x_train, y_train, batch_size=64, epochs=10,
              verbose=1,validation_split=0.2, shuffle=True)
time2=time.time()

print('elapsed time', time2-time1)
file=args.file
file=file.split('.')[0]
if('models' not in os.listdir()):
    os.mkdir('models')
#save the model and weight
model_yaml=model.to_yaml()
with open("models/%s_%d.yaml" % (file,int(args.object)), "w") as yaml_file:
     yaml_file.write(model_yaml)

model.save_weights("models/%s_%d.h5" % (file,int(args.object)))
print("saved model to disk")

# save the the res as joblib
if('res' not in os.listdir()):
    os.mkdir('res')
joblib.dump(res.history,'res/%s_%d.dat' % (file,int(args.object)))

scores = model.evaluate(x_test, y_test, verbose=0)
print("train_error(mse):", res.history['mse'][-1])
print("test_error(mse):", scores[1])

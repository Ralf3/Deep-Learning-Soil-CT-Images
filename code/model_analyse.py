import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import joblib
import pylab as plt
import numpy as np
import cv2
import argparse

# parse the arguments
parser=argparse.ArgumentParser(
    description='visualize the trained network with the real dataset')
parser.add_argument('--method', default='heatmap',
                    help='select between heatmap and activation')
parser.add_argument('--file', required=True,
                    help='give the name of the npy file for testing')
parser.add_argument('--model', required=True,
                    help='select the model without file extension')
parser.add_argument('--layer', default='block1_conv1',
                    help='select one of: block1_conv1, block1_conv2, block2_conv1, block2_conv2, block3_conv1 block3_conv2, block3_conv3, block4_conv1,block4_conv2,block4_conv3, block5_conv1,block5_conv2, block5_conv3')
parser.add_argument('--slice', default=100,
                    help='select one of the silce [0:399]')
parser.add_argument('--thresh', default=100, type=int,
                    help='adapt the thresh for activation')


args=parser.parse_args()

file=args.file
model=args.model

def plt_history():
    global file,model
    hist=joblib.load('res/%s.dat' % model)
    fig, ax = plt.subplots()
    ax.plot(hist['mse'][5:],label='train_mse')
    ax.plot(hist['val_mse'][5:],label='val_mse')
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.show()
    return hist

def load_model():
    global model
    with open('models/%s.yaml' % model,'rb') as yaml:
        modell=tf.keras.models.model_from_yaml(yaml)
    return modell

def load_weights(modell):
    global model
    modell.load_weights('models/%s.h5' % model)
    return True

modell=load_model()
load_weights(modell)

def describe_conv_layers(modell):
    for i in range(len(modell.layers)):
        layer = modell.layers[i]
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # summarize output shape
        print(i, layer.name, layer.output.shape)
    return

# load the data used for training
datapath='npy/%s' % file
def load_data(path1=datapath):
    # read one selected volume
    V=np.load(path1)
    # build objects
    objects=[1,2,3,4,5,6]
    hist,edges=np.histogram(V,bins=6)
    res={}
    for i in objects:
        V1=np.zeros((V.shape[0],V.shape[1],V.shape[2]))
        V1=V1.astype('int')
        V1[(V>=edges[i-1])&(V<edges[i])]=1       # solid 
        res[i]=V1
    return res,V

res, V = load_data()

# define the porosity of a selected object (1,2,3,4,5,6)
def porosity(sel):
    global res
    por=np.zeros(res[sel].shape[0])
    for i in range(res[sel].shape[0]):
        por[i]=np.sum(res[sel][i])
    return 1.0-por/(res[sel].shape[1]*res[sel].shape[2])

# example calc por1 for bio and pore 56 for stones
por={1:porosity(1), 2:porosity(2), 3:porosity(3), 4:porosity(4),
     5:np.minimum(porosity(5),porosity(6))}

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

def generate_pattern(layer_name,filter_index,slx=200,size=200):
    epochs=100
    step=1.
    submodel = tf.keras.models.Model([modell.inputs[0]],
                                     [modell.get_layer(layer_name).output])

    
    #input_img_data = np.random.random((1, size, size, 3))
    input_img_data = V4[slx].reshape(1,200,200,3)
    input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

    # Iterate gradient ascents
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            outputs = submodel(input_img_data)
            loss_value = tf.reduce_mean(outputs[:, :, :, filter_index])
        grads = tape.gradient(loss_value, input_img_data)
        normalized_grads = grads /\
                           (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
        input_img_data.assign_add(normalized_grads * step)
    img=input_img_data.numpy().reshape(size,size,3)
    img=(img-np.min(img))/\
          (np.max(img)-np.min(img))
    return img

def show_pattern(layer_name,filter_index,slx=200):
    """
    layer_name: use describe_conv_layers(modell) to select one conv
    filter_index = depends on the number of conv fitlers
    """
    img=generate_pattern(layer_name,filter_index,slx=slx)
    fig, ax = plt.subplots(figsize=(10, 10), ncols=1)
    pos=ax.imshow(img)
    # fig.colorbar(pos,ax=ax)
    # return img
    
def display_activation_useful(soil_layer,layer_name,thresh=args.thresh):
    """
    soil_layer: Select the layer from V4. The range is 0 to 399.
    layer_frame: Select the layer name from model.summary.
    thresh: Define threshold value for extract useful images from 
    convolution layers.
    """
    global modell,V4
    # adapt the modell according to the outputs
    layer_outputs = [layer.output for layer in modell.layers]
    model1 = Model(inputs=modell.input, outputs=layer_outputs)
    # list all layers
    layer_names = []
    for layer in model1.layers:
        layer_names.append(layer.name)
    # calc the activation of the adapted model1
    activations = model1.predict(V4[soil_layer].reshape(1,200,200,3))
    act_index=layer_names.index(layer_name)
    activation = activations[act_index]
    
    global filters
    filters=[]
    i=0
    while i < model1.layers[act_index].output.shape[3]:
        if(np.sum(activation[:,:,:,i])>thresh):
            image = activation[0,:,:,i]
            filters.append(image)
            i+=1
        else:
            i+=1
    print('the number of filters = ', len(filters))
    if(len(filters)==0):
        print('No filter is available. Change the value of threshold.')
        return False
            
    if not len(filters)==0:
        row_size=np.sqrt(len(filters)).astype(int)
        col_size=np.sqrt(len(filters)).astype(int)+1

    count=0
    # while count < len(filters):
    fig,ax=plt.subplots(row_size, col_size,
                        figsize=(row_size*2.5,col_size*1.5))
    plt.axis('off')
    for row in range(0,row_size):
        for col in range(0,col_size):
            if(count>=len(filters)):
                return True
            ax[row][col].imshow(filters[count], cmap='copper')
            count += 1
    plt.show()
    
if(args.method=='activation'):
    display_activation_useful(int(args.slice),args.layer)

def make_heatmap(layer,slx,index=1):
    """
    layer: the name of a conv layer 
    slx: the index of V4 [0,399]
    index: index of the CNN layer
    stores the heatmap and the original image in temp
    """
    global modell,V4
    grad_model = Model([modell.input], [modell.get_layer(layer).output, modell.output])
    # Get the score for target class
    with tf.GradientTape() as tape:
        image=V4[slx,:,:,:]
        conv_outputs, predictions = grad_model(np.array([image]))
    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(predictions, conv_outputs)[0]

    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # Build a ponderated map of filters according to gradients importance
    cam = np.zeros(output.shape[0:2], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:, :, index]

    cam=cam.numpy() # remove the TF
    np.save('./cam.npy', cam)
    np.save('./img.npy', V4[slx])
    img=V4[slx]
    cam = cv2.resize(cam, (200, 200))
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    img = cv2.applyColorMap(np.uint8(255*img), cv2.COLORMAP_BONE)
    output_image = cv2.addWeighted(cv2.cvtColor(255*img.astype('uint8'),
                                                cv2.COLOR_RGB2BGR), 0.7,
                                   cam, 0.3, 0)

    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(12, 4), ncols=3)
    ax1.set_title('heatmap')
    ax1.imshow(cam)
    ax2.set_title('original')
    ax2.imshow(img)
    ax3.set_title('overwrite')
    ax3.imshow(output_image)
    plt.show()
    
if(args.method=='heatmap'):
    make_heatmap(args.layer,int(args.slice))


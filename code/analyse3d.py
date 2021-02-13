from mayavi import mlab
import numpy as np
import pylab as plt
import argparse

# parse the arguments
parser=argparse.ArgumentParser(
    description='load binary npy file and visualize it')
parser.add_argument('--file', required=True,
                    help='give the name of the npy file for visualization')
parser.add_argument('--object', required=True, nargs='+', type=int,
                    help='3D visualisation of an object list [1,2,3,4,5,6]')
args=parser.parse_args()

#load the 3D data from the directory npy
V=np.load('npy/%s' % args.file)
V=V.astype('int')
print(V.shape)

# calc a histogram of soil probe
# V1=np.copy(V)
plt.hist(V[200],bins=5)
hist,edges=np.histogram(V,bins=6)
plt.show()

def sub_soil(objects=[1,2,3,4,5,6]):
    global edges,V
    res={}
    for i in objects:
        V1=np.zeros((V.shape[0],V.shape[1],V.shape[2]))
        V1[(V>=edges[i-1])&(V<edges[i])]= V[(V>=edges[i-1])&(V<edges[i])]/\
                                          V[(V>=edges[i-1])&(V<edges[i])].max()
        res[i]=V1
    return res

res=sub_soil([1,2,3,4,5,6])

def show_object(i):
    mlab.contour3d(res[i],contours=[0.5],colormap='copper',
                   vmin=0,vmax=1,name='object%d' % i)


def show_colorX(i,r,g,b):
    """ 
    show a RGB image of th comportment i at three layers in z direction:
    r,g,b 
    for example r=300,g=320,b=340
    """
    im_r=res[i][r,:,:]
    im_g=res[i][g,:,:]
    im_b=res[i][b,:,:]
    im=np.zeros((im_r.shape[0],im_r.shape[1],3)) # three layers three colors
    im[:,:,0]=im_r
    im[:,:,1]=im_g
    im[:,:,2]=im_b
    plt.imshow(im)
    # return im

for i in args.object:
    show_object(int(i))
mlab.show()

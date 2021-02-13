# Deep-Learning-Soil-CT-Images
Same scripts to handle CT Images using deep learning

The Python programs are used to investigate soil structures from CT images.
They are part of the paper:

Use of Deep Learning for structural analysis of CT-images of soil samples

All the data and programs used here are open source and were 
created with the help of open source software. A detailed summary of the installation steps can be found in Install.txt.

To make it easier for the reader to understand, the programs have been designed to build the data and the model step by step from an original CT data set. We proceed as follows:

Please download the *.tgz and unpack them (tar -zxvf 3086.tgz). Thne you have directories with the dataset in DICOM format.
To read this data and to transfer it into a structure suitable for training neural networks the program gen_cube.py must be called. This is done by passing parameters on the command line. To get an overview of the commands, type:

python gen_cube.py -h

gen_cube.py creates the directory 'npy' and writes the generated data "Deutschland_3086.npy" into the directory.

In the second step the 3D analysis of the data is performed by analyse3d.py

python analyse3d.py -h 

explains the parameters. It needs as --file the Deutschland_3086.npy and with the help of --object 1,2,3,4,5,6 or combinations of these like 1 2 the objects for the visualization can be chosen. mayavi is interactive and very interesting insights into the data can be created. 

The third step is the training of deep CNN. Here Tensorflow and opencv must be installed first (see Install.txt). The training takes some time without a GPU this can be more than one hour. The program is used:

python ImageGenerator2_VGG16.py -h

It needs the  Deutschland_3086.npy (or another file) created in the first step and an object from [1,2,3,4,5], where 5 combines the objects 5 and 6. The number of epochs was reduced to 10 due to time constraints, in the paper 100 epochs were used for the training. The model parameters (Deutschland_3086_1.h5) and its structure (Deutschland_3086_1.yaml) are located in the newly created models directory. The results of the training are stored in the directory res (Deutschland_3086_1.dat), which was also created. It should be mentioned that regularization like early stoping was not used in the example due to the limited number of epochs. It can be easily added when a GPU is available.

Perhaps the most interesting program "model_analyse.py" evaluates the results of the training and visualizes the heatmap or the partial images for the CNN. This can be controlled with --method heatmap or --method activation. Otherwise the program needs the --file Deutschland_3086.npy the --modell Deutschland_3086_1 (without file extension) Furthermore the model layer can be created from the block2_conv2, block3_conv1 block3_conv2,... can be selected. With --slice a slice from 0..399 is selected in the spatial structure Deutschland_3086.npy. The parameter --thresh allows the adjustment of the threshold value when using the method activation. 

Happy hacking,

Ralf Wieland.



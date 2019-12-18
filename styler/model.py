import random
from keras.applications import vgg16, vgg19

zoo = {
'vgg16':{
'layers':[
'block1_conv1',
'block1_conv2',

'block2_conv1',
'block2_conv2',

'block3_conv1',
'block3_conv2',
'block3_conv3',

'block4_conv1',
'block4_conv2',
'block4_conv3',

'block5_conv1',
'block5_conv2',
'block5_conv3',
],
'model' :lambda x=None:vgg16.VGG16(input_tensor=x, weights='imagenet', include_top=False)
},

'vgg19':{
'layers':[
'block1_conv1',
'block1_conv2',

'block2_conv1',
'block2_conv2',

'block3_conv1',
'block3_conv2',
'block3_conv3',
'block3_conv4',

'block4_conv1',
'block4_conv2',
'block4_conv3',
'block4_conv4',

'block5_conv1',
'block5_conv2',
'block5_conv3',
'block5_conv4',
],
'model' :lambda x=None:vgg19.VGG19(input_tensor=x, weights='imagenet', include_top=False)
}}

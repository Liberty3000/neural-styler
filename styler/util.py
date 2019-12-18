import numpy as np, keras.backend as K
from keras.applications import vgg16, vgg19
from keras.preprocessing.image import load_img, save_img, img_to_array

# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss(style, combination, shape):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    size = shape[0] * shape[1]
    return K.sum(K.square(S - C)) / (4. * (3 ** 2) * (size ** 2))

# an auxiliary loss function, designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
class total_variation_loss(object):
    def __init__(self, shape, weight):
        self.rows = shape[0]
        self.cols = shape[1]
        self.weight = weight

    def __call__(self, x):
        assert K.ndim(x) == 4
        if K.image_data_format() == 'channels_first':
            a = K.square(x[:, :, :self.rows - 1, :self.cols - 1] - x[:, :, 1:, :self.cols - 1])
            b = K.square(x[:, :, :self.rows - 1, :self.cols - 1] - x[:, :, :self.rows - 1, 1:])
        else:
            a = K.square(x[:, :self.rows - 1, :self.cols - 1, :] - x[:, 1:, :self.cols - 1, :])
            b = K.square(x[:, :self.rows - 1, :self.cols - 1, :] - x[:, :self.rows - 1, 1:, :])
        return K.sum(K.pow(a + b, self.weight))

# util function to open, resize and format images into appropriate tensors
def preprocess(image_path, h,w):
    img = load_img(image_path, target_size=(h, w))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# util function to convert a tensor into a valid image
def deprocess(x, h, w):
    x = x.reshape((h, w, 3))

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

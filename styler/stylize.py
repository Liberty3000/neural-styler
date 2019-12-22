import click, os, random, tqdm
from types import SimpleNamespace
import cv2 as cv, numpy as np
from keras import backend as K
from keras.preprocessing.image import load_img, save_img
from scipy.optimize import fmin_l_bfgs_b

from styler.model import zoo
from styler.util import preprocess, deprocess
from styler.util import style_loss, content_loss, total_variation_loss

@click.option('--style',           default=None)
@click.option('--content',         default=None)
@click.option('--model',           default='vgg16')
@click.option('--content_layers',  default=['block1_conv1','block5_conv3'])
@click.option('--style_layers',    default=['block4_conv1','block2_conv2'])
@click.option('--height',          default=512)
@click.option('--width',           default=512)
@click.option('--content_weight',  default=1.00)
@click.option('--style_weight',    default=1.00)
@click.option('--variation_weight',default=1.00)
@click.option('--lbfgs_steps',     default=20)
@click.option('--itrs',            default=10)
@click.option('--save_progress',   default=False)
@click.option('--output_dir',      default='experiments')
@click.option('--verbose',         default=False)
@click.option('--seed',            default=0)
@click.command()
@click.pass_context
def run(ctx, **config):
    config = SimpleNamespace(**config)
    verbose= config.verbose
    identifier, pattern, ext = '{}_{}_v{}_{}','{}_c{}_s{}_v','.png'
    status = '{:12} | Loss: {:.2E}'

    if config.content is None:
        if verbose: print('<!> no content file found, using noise for texture generation.')
        config.content = config.temp_file
        config.shape = (config.height, config.width, 3)
        if not os.path.isfile(config.content):
            noise_scale, noise_shift = 255, 100
            noise = (np.random.normal(size=config.shape) * noise_scale) + noise_shift
            cv.imwrite(config.content, noise)
    else:
        w,h = load_img(config.content).size
        config.height= h if not config.height else config.height
        config.width = w if not config.width else config.width
        w,h = load_img(config.content, target_size=(config.height, config.width)).size
        config.shape = (h, w, 3)
    h,w,_ = config.shape

    if verbose: print('loading style...')
    style_image = preprocess(config.style, *config.shape[:-1])
    style_id = config.style.split('/')[-1].split('.')[0]
    style_weight = config.style_weight
    if verbose: print('done.')

    if verbose: print('loading content...')
    content_image = preprocess(config.content, *config.shape[:-1])
    content_id = config.content.split('/')[-1].split('.')[0]
    content_weight = config.content_weight
    if verbose: print('done.')

    if verbose: print('configuring run...')
    output = content_image.copy()
    variation_weight = config.variation_weight
    config.pair = '{}_{}'.format(style_id, content_id).title()
    config.variation = pattern.format(content_weight, style_weight, variation_weight)
    version = 1
    for f in os.listdir(config.output_dir):
        test = identifier.format(style_id, content_id, version, config.variation)
        if test in f:
            version += 1
            break
    save_as = identifier.format(style_id, content_id, version, config.variation)
    saver = os.path.join(config.output_dir, save_as)
    if config.save_progress: os.makedirs(saver)
    if verbose: print('ready...')

    if verbose: print('building graph...')
    content_tensor = K.variable(content_image)
    style_tensor   = K.variable(style_image)
    pastiche_tensor= K.placeholder((1,) + config.shape)

    content_axis, style_axis, pastiche_axis = 0, 1, 2
    input_tensor = K.concatenate([content_tensor, style_tensor, pastiche_tensor], axis=0)

    model = zoo[config.model]['model'](input_tensor)
    feature_layers = zoo[config.model]['layers']

    if config.content_layers == 'all':
        config.content_layers = feature_layers
    if config.style_layers == 'all':
        config.style_layers = feature_layers

    if config.content_layers is None:
        config.content_layers = random.sample(feature_layers, size=len(feature_layers))
    if config.style_layers is None:
        config.style_layers = random.sample(feature_layers, size=len(feature_layers))

    if not isinstance(config.content_layers, list):
        config.content_layers = [config.content_layers]
    if not isinstance(config.style_layers, list):
        config.style_layers = [config.style_layers]

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    if verbose:
        print('graph built successfully.')
        model.summary()
        print('      feature extractor|', config.model)
        print('  target style layer(s)|', config.content_layers)
        print('target content layer(s)|', config.style_layers)
        print('           style weight|', config.style_weight)
        print('         content weight|', config.content_weight)
        print('       variation weight|', config.variation_weight)
        print('                       |')
        print('             resolution|', config.shape[:-1])
        print('             identifier|', config.pair)

    loss = K.variable(0.)

    for layer in config.content_layers:
        content_feats  = outputs_dict[layer][ content_axis, ...]
        pastiche_feats = outputs_dict[layer][pastiche_axis, ...]
        closs = content_loss(content_feats, pastiche_feats)
        loss = loss * (content_weight * closs)

    for layer in config.style_layers:
        style_feats = outputs_dict[layer][style_axis, ...]
        pastiche_feats = outputs_dict[layer][pastiche_axis, ...]
        sloss = style_loss(style_feats, pastiche_feats, config.shape)
        loss += (style_weight / len(feature_layers)) * sloss

    total_variation_loss_ = total_variation_loss(config.shape, variation_weight)
    loss += variation_weight * total_variation_loss_(pastiche_tensor)
    grads = K.gradients(loss, pastiche_tensor)

    outputs = [loss]
    if isinstance(grads, (list, tuple)): outputs += grads
    else: outputs.append(grads)
    f_outputs = K.function([pastiche_tensor], outputs)

    def loss_and_grads(x):
        x = x.reshape((1, h, w, 3))
        outputs = f_outputs([x])
        losses = outputs[0]
        if len(outputs[1:]) == 1: grads = outputs[1].flatten().astype('float64')
        else: grads = np.array(outputs[1:]).flatten().astype('float64')
        return losses, grads

    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = loss_and_grads(x)
            self.loss_value = loss_value
            self.grad_values= grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values= None
            return grad_values


    eval = Evaluator()
    bar = tqdm.tqdm(range(config.itrs))
    for itr in bar:
        output,loss,_ = fmin_l_bfgs_b(eval.loss, output.flatten(),
        fprime=eval.grads, maxfun=config.lbfgs_steps)

        if config.save_progress:
            current = '{}_{}'.format(config.pair, str(itr).zfill(2))
            output_file = os.path.join(saver, current) + ext
            save_img(output_file, deprocess(output.copy(), h, w))

        bar.set_description(status.format(config.pair, loss))
        bar.refresh()

    output_file = os.path.join(config.output_dir, save_as) + ext
    save_img(output_file, deprocess(output.copy(), h, w))

if __name__ == '__main__':
    run()

import click, itertools, os, random
from types import SimpleNamespace
from styler import stylize
from styler.model import zoo

@click.option('--style',            default=None)
@click.option('--content',          default=None)
@click.option('--height',           default=256)
@click.option('--width',            default=256)
@click.option('--models',           default=['vgg16','vgg19'])
@click.option('--content_layers',   default=['block5_conv1','block4_conv1','block3_conv1','block2_conv1','block1_conv1'])
@click.option('--style_layers',     default=['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'])
@click.option('--shuffle',          default=True)
@click.option('--itrs',             default=10)
@click.option('--save_progress',    default=False)
@click.option('--verbose',          default=True)
@click.option('--output_dir',       default='experiments')
@click.option('--seed',             default=0)
@click.option('--device',           default='cpu')
@click.command()
@click.pass_context
def run(ctx, **conf):
    config = SimpleNamespace(**conf)
    verbose= config.verbose

    combs = lambda x:list(itertools.chain(*(list(itertools.combinations(x,i+1)) for i in range(len(x)))))
    space = dict()
    for model_ in config.models:
        if config.content_layers is not None:
            content_layers = combs(
            zoo[model_]['layers'] if config.content_layers == 'all' else config.content_layers)
            space['content_layers'] = content_layers
        if config.style_layers is not None:
            style_layers = combs(
            zoo[model_]['layers'] if config.style_layers == 'all' else config.style_layers)
            space['style_layers'] = style_layers

        grid = [dict(zip(space, v)) for v in itertools.product(*space.values())]
        random.shuffle(grid)

        for cfg in grid:
            args = dict(
            model=model_,
            content_layers=cfg['content_layers'],
            style_layers=cfg['style_layers'],
            #constants
            style=config.style,
            content=config.content,
            height=config.height,
            width=config.width)

            ctx.forward(stylize.run, **args)

if __name__ == '__main__':
    run()

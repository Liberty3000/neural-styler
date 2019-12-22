import click, glob, itertools, os, random
from types import SimpleNamespace
from styler import stylize

@click.option('--style',            default=None)
@click.option('--content',          default=None)
@click.option('--height',           default=512)
@click.option('--width',            default=512)
@click.option('--models',           default=['vgg16','vgg19'])
@click.option('--content_layers',   default=['block1_conv1','block3_conv1'])
@click.option('--style_layers',     default=['block1_conv1','block2_conv1','block4_conv1'])
@click.option('--shuffle',          default=True)
@click.option('--itrs',             default=10)
@click.option('--save_progress',    default=False)
@click.option('--verbose',          default=False)
@click.option('--output_dir',       default='experiments')
@click.option('--seed',             default=0)
@click.option('--device',           default='cpu')
@click.command()
@click.pass_context
def run(ctx, **conf):
    config = SimpleNamespace(**conf)
    verbose= config.verbose

    queries = config.style
    if os.path.isfile(config.style):
        queries = [config.style]
    if os.path.isdir(config.style):
        queries = list(glob.glob(config.style + '/*'))

    iterator = list(itertools.combinations(queries,2))
    if config.shuffle: random.shuffle(iterator)

    for i in iterator:
        args = dict(
        style=i[0],content=i[1],
        height=config.height,
        width=config.width)
        ctx.forward(stylize.run, **args)
        args['style'],args['content']= i[1],i[0]
        ctx.forward(stylize.run, **args)

if __name__ == '__main__':
    run()

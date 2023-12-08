import os
import cv2
import yaml
import argparse
import subprocess

from llm.genetic import *
from llm.pl.functool import get_ext
from llm.genetic.loader import load_prog
from envs import MultiToolEnv


def html2prog(line):
    n = len(line)
    dep, lasr = 0, -1
    ret = []
    for i in range(n):
        if line[i] == '<':
            if dep == 0 and lasr + 1 <= i -1: #  and line[lasr+1] not in [' ']:
                ret.append(line[lasr+1:i].removeprefix('&#39;').removesuffix('&#39;'))
            dep += 1
        elif line[i] == '>':
            lasr = i
            dep -= 1
    return ret


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Program file path')

    args = parser.parse_args()

    prog_file = args.path

    # convert python program to html
    os.system("pygmentize -f html -O full,style=solarized-light,lineonos=1 -l python -o test.html {}".format(prog_file))

    # copied from train_genome.py, to render subgoal image 
    from llm.genetic.frankwolfe import FrankWolfe
    from llm.genetic.utils import SOLPATH
    true_prog = load_prog(prog_file)[0]
    if true_prog.doc is not None:
        optim_cfg = yaml.load(true_prog.doc, Loader=yaml.FullLoader)
    else:
        optim_cfg = {}
    TRAINER = FrankWolfe
    optim_cfg['n_stages'] = 2
    trainer = TRAINER.parse(true_prog, env=None, parser=parser, **optim_cfg)
    trainer.env.render('0.png')

    # replace .th to image
    from tools.utils import totensor
    from llm.envs.plb import MultiToolEnv
    img_cnt = 0
    outp = ''
    with open('test.html', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'subgoal' in line:    
                prog = html2prog(line)   # parse html back to python code 
                nhtml = ''
                for itm in prog:
                    if '.th' in itm:  # render .th image and insert it into html
                        img_cnt += 1
                        state = torch.load(itm)
                        img = trainer.env.render_state_rgb(state)
                        cv2.imwrite('{}.png'.format(img_cnt), img[..., ::-1])
                        nhtml += '<img src=\"{}.png\" width=\"128\" height=\"128\" style=\"vertical-align:middle\">'.format(img_cnt)
                    else:
                        nhtml += '<span class="n">' + itm + '</span>'
                outp += nhtml + '\n'
            else:
                outp += line
    with open('test.html', 'w') as f:
        f.write(outp)

        
def format_code(code):
    cmd = 'autopep8 --ignore=E731 - '
    code = subprocess.check_output(cmd, shell=True, input=code.encode('utf-8')).decode('utf-8')
    return code


def render_prog(prog, images=None, path=None):#env: MultiToolEnv):
    import os
    import inspect

    if isinstance(prog, str) and os.path.exists(prog):
        with open(prog, 'r') as f:
            prog = ''.join(f.readlines())
    elif inspect.isfunction(prog):
        prog = inspect.getsource(prog)

    code = format_code(prog)

    cmd = "pygmentize -f html -O full,style=solarized-light,lineonos=1 -l python"
    output = subprocess.check_output(cmd, shell=True, input=code.encode('utf-8')).decode('utf-8').strip()

    from tools.utils import logger
    if logger.get_run():
        run = logger.get_run()
        # import wandb
        # artifact = wandb.Artifact('images', 'images')

    outputs = []
    for i in output.split('\n'):
        if '<span class="o">=</span>' in i and images is not None:
            assert path is not None
            import bs4
            soup = bs4.BeautifulSoup(i)
            varname = soup.find_all('span', class_='n')[0].contents[0]
            if varname in images:
                img_path = os.path.join(path, varname+'.png')
                cv2.imwrite(img_path, images[varname][..., ::-1])

                html = f'<span>&nbsp</span><img src=\"{varname}.png\" width=\"32\" height=\"32\" style=\"vertical-align:middle\"><span class="n"> =</span>'

                if logger.get_run():
                    run.save(img_path)
                    # print('saving..', img_path)
                    # artifact.add_file(img_path, name=varname+'.png')

                soup.insert(4, bs4.BeautifulSoup(html))
                outputs.append('   ' + str(soup))
            else:
                outputs.append(i)
        else:
            outputs.append(i)
    # if logger.get_run():
    #     run.log(artifact)

    return '\n'.join(outputs)


if __name__ == '__main__':
    main()
    #def main():
    #    print(123)
    #    f(a(b(3, 4, 5), 6), 7)
    # import torch
    # images = torch.load('images.th') 
    # with open('test.html', 'w') as f:
    #     f.write(render_prog('dataset/genome/wind_cpdeform.py', images, '.'))
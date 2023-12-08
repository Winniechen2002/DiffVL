# scripts for plot single stage images
import os
import numpy as np
import cv2
import textwrap
from diffsolver.utils.rendering import create_image_with_titles, text_to_image
tasks = ['5', '37', '10', '11']
taskid = {'11': 'Winding: first pick up the back end of the yellow rope and then move it across the orange rope.', '5': 'moveover: lift the white above the black wall', '10': 'cover: incline and pick up the front of the top left mat and use it to cover the brown core below', '37': 'cut: use the knife to cut out the left half of the white mat'}
mode = ['badinit', 'emdonly', 'ours']

folder = 'examples/output/images'


results = {}
for k in tasks:
    k = f'task{k}'
    results[k] = {}
    results[k]['start'] = cv2.imread(f'{folder}/ours_{k}/000.png')[:,:,::-1]
    for m in mode:
        print(f'{folder}/{m}_{k}/final.png')
        results[k][m] = cv2.imread(f'{folder}/{m}_{k}/final.png')[:,:,::-1]
    results[k]['goal'] = cv2.imread(f'{folder}/ours_{k}/goal.png')[:,:,::-1]


os.makedirs('single_images', exist_ok=True)

wrapper = textwrap.TextWrapper(width=40)

output = []
#for mode, mode_result in results.items(): 
for t in tasks:
    todo = results['task'+t]
    #for p, exp_outs in mode_result.items():
        # for out in exp_outs:

        #     todo={}
        #     for k, v in out.texts.items():
        #         todo['lang: ' + '\n'.join(wrapper.wrap(out.labels['tool_lang']))+'FONT:20'] = np.uint8(np.asarray(text_to_image(v, (512, 512))))
        #     for k, v in out.images.items():
        #         todo[k] = np.uint8(cv2.resize(plt.imread(out.images[k]) * 255, (512, 512)))  # type: ignore

    for k, v in todo.items():
        cv2.imwrite(f'single_images/{k}_{t}.png', v[:,:,::-1])
    outs = create_image_with_titles(list(todo.values()), list(todo.keys()), dir='above')
    output.append(create_image_with_titles([outs], [taskid[t]], dir='below'))

#max_width = max([o.shape[1] for o in output])
# make all images have the same width
#output = [cv2.copyMakeBorder(o, 0, 0, 0, max_width - o.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for o in output]
cv2.imwrite('images.png', np.concatenate(output, axis=0)[:,:,::-1])
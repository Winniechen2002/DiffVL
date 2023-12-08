#!/usr/bin/env python3
import argparse
import os
from omegaconf import OmegaConf
from diffsolver.launch.utils import keys, PathCompleter, prompt
from diffsolver.paths import get_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default=None, choices=['tool', 'phys'])
    parser.add_argument('--code', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    #parser.add_argument('--task', type=str, data)
    #parser.add_argument('--start', action='store_true', help='run the command')
    args, unknown = parser.parse_known_args()

    #if args.code is None:
    if args.mode == 'tool':
        from diffsolver.toolguide.prompts.get_prompts import answer
        while True:
            code = args.code or input('Enter code: ')
            if code == 'exit':
                break
            print(answer(code, verbose=True))
            if args.code is not None:
                break
    else:
        #from diffsolver.program.scenes import load_scene_with_envs
        #from diffsolver.utils import MultiToolEnv
        #env = MultiToolEnv(sim_cfg=dict(max_steps=10))
        PATH = args.config

        while True:
            path_completer = PathCompleter(only_directories=False)
            if PATH is not None:
                print("Current config: ", PATH)
                out = (keys('select new path?', ['yes', 'no', 'exit']))
                redo = out == 'yes'
                if out == 'exit':
                    break
            else:
                redo = True
            if redo:
                PATH = prompt('Enter path: ', completer=path_completer, default='examples/single_stage_dev')

            if not PATH.endswith('.yaml') and not PATH.endswith('.yml'):
                break
            cfg = OmegaConf.load(PATH) 
            naming_dict = OmegaConf.load(os.path.join(get_path("VISION_TASK_PATH"), 'naming.yml'))
            task_id = cfg.scene.path
            task_id = task_id.split('.')[0].split('_')[0]

            objects = naming_dict.get(int(task_id), None)
            print('existing objects', objects)
            scene_description = ', '.join(objects)

            code = args.code or input('Enter code: ')
            if code == 'exit':
                break
            
            from diffsolver.program.clause.translator import translate_program_from_scene
            from diffsolver.config import TranslatorConfig
            from diffsolver.program import progs

            tool_lang = cfg.tool_sampler.lang

            scene_description = f"Objects: {objects}\nInput: "


            output = translate_program_from_scene(code, None, TranslatorConfig('v3', verify_scene=False), scene=scene_description) # type: ignore
            print(output)

            if args.config is not None:
                break




if __name__ == '__main__':
    main()
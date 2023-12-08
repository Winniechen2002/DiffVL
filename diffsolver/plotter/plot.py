# scripts for plotting and summarizing the results
import pandas as pd
import textwrap
import cv2
import matplotlib.pyplot as plt
import argparse
import enum
import os
import numpy as np
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple, Dict, Any, Optional, Union, cast
from numpy.typing import NDArray
from diffsolver.plotter.renderer import HTMLRenderer, HTMLRenderConfig
from diffsolver.plotter.exp_results import ExperimentOutput, Task, Curve
from diffsolver.plotter.drawer import Drawer, DrawConfig
from diffsolver.launch.utils import sort_by_task

FILEPATH = os.path.dirname(os.path.abspath(__file__))


class LoadFailureException(Exception):
    pass



class StageMode(enum.Enum):
    single_stage="single_stage"
    multistage="multistage"

class RenderMode(enum.Enum):
    rgb_array="rgb_array"
    html="html"
    figure="figure"


@dataclass
class GroupBy:
    # group by method
    method: str = 'mode'
    env: str = 'name'


@dataclass
class PlotConifg:
    root_path: str
    stage_mode: StageMode = StageMode.single_stage
    modes: List[str] = field(default_factory=lambda: [])
    pattern: str = ""

    latest: int = 1
    find_exist: bool = False

    load_task: bool=False

    load_tool: bool = True
    load_phys: bool = True
    load_intermediate: bool = False
    load_ending: bool = True
    load_curves: bool = True
    load_video: bool = True

    load_tool_lang: bool = False
    render_mode: RenderMode = RenderMode.html
    html: HTMLRenderConfig = field(default_factory=HTMLRenderConfig)

    groupby: GroupBy = field(default_factory=GroupBy)

    
    drawer: DrawConfig = field(default_factory=DrawConfig)

    plot_task: bool = False


def match_regex(s: str, regex: str):
    import re
    match = re.search(regex, s)
    if match:
        return True
    return False

def load_exist_abspath(d: Dict[str, str], key, path, not_exist_ok=False):
    if not os.path.exists(path):
        if not not_exist_ok:
            raise LoadFailureException(f"{path} does not exist")
    else:
        d[key] = os.path.abspath(path)


class Plotter:
    def __init__(self, cfg: PlotConifg) -> None:
        self.cfg = cfg
        if cfg.plot_task:
            print('qwewqe')
            self.plot_task = OmegaConf.load(os.path.join(FILEPATH, 'plot_task.yaml'))

        self.no_break = OmegaConf.load(os.path.join(FILEPATH, 'no_break.yaml'))

        root_path = None
        if not OmegaConf.is_missing(cfg, "root_path"):
            root_path = cfg.root_path
        elif "MODEL_DIR" in os.environ:
            root_path = os.environ["MODEL_DIR"]


        self.folders = {}
        self.summary = {}

        runs: List[ExperimentOutput] = []

        for m in cfg.modes:
            paths = [root_path, cfg.stage_mode.value, m]
            base_path = os.path.join(*paths)
            if not os.path.exists(base_path):
                continue
            pattern = cfg.pattern
            #print(sorted(os.listdir(base_path)))
            folders = [
                f for f in sort_by_task(os.listdir(base_path)) if os.path.isdir(os.path.join(*paths, f)) and match_regex(f, pattern)
            ]
            self.folders[m] = folders
            #self.results[m] = {}
            for f in folders:
                loaded = self.load_by_timestamp(os.path.join(base_path, f))
                runs += loaded
            
            self.summary[m] = {
                'base_path': base_path,
            }
            self.summary['pattern'] = pattern
                #self.results[m][f] = loaded
        self.results = self.group_by(runs)

        self.drawer = Drawer(cfg.drawer)
        self.summary['tables'] = self.drawer.get_tables(self.results)



        
    def group_by(self, runs: List[ExperimentOutput]):
        #return results
        group_by = self.cfg.groupby
        results: Dict[str, Dict[str, List[ExperimentOutput]]] = {}
        for e in runs:
            method = e.labels[group_by.method]
            env = e.labels[group_by.env]
            results[method] = results.get(method, {})
            results[method][env] = results[method].get(env, [])
            results[method][env].append(e)

        def sort_dict(d):
            return {k: d[k] for k in sort_by_task(list(d.keys()))}
        results = sort_dict({k: sort_dict(v) for k, v in results.items()})
        return results


    def render(self):
        results = self.results
        if self.cfg.render_mode is RenderMode.rgb_array:
            wrapper = textwrap.TextWrapper(width=40)
            from diffsolver.utils.rendering import create_image_with_titles, text_to_image
            output = []
            for mode, mode_result in results.items(): 
                for p, exp_outs in mode_result.items():
                    for out in exp_outs:

                        todo={}
                        for k, v in out.texts.items():
                            todo['lang: ' + '\n'.join(wrapper.wrap(out.labels['tool_lang']))+'FONT:20'] = np.uint8(np.asarray(text_to_image(v, (512, 512))))
                        for k, v in out.images.items():
                            todo[k] = np.uint8(cv2.resize(plt.imread(out.images[k]) * 255, (512, 512)))  # type: ignore
                        outs = create_image_with_titles(list(todo.values()), list(todo.keys()))
                        output.append(create_image_with_titles([outs], [p]))

            max_width = max([o.shape[1] for o in output])
            # make all images have the same width
            output = [cv2.copyMakeBorder(o, 0, 0, 0, max_width - o.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for o in output]
            cv2.imwrite('images.png', np.concatenate(output, axis=0)[:,:,::-1])
        elif self.cfg.render_mode is RenderMode.html:
            from diffsolver.plotter.renderer import HTMLRenderer
            renderer = HTMLRenderer(self.cfg.html)
            renderer.run(results, self.summary)

        elif self.cfg.render_mode is RenderMode.figure:
            self.drawer.draw(results)
        else:
            raise NotImplementedError(f"Render mode {self.cfg.render_mode} is not implemented")

    def load_rl(self, path):
        if not os.path.isdir(path):
            raise LoadFailureException(f"{path} is not a folder")
        exp_out = ExperimentOutput()
        mode, name, date = path.split('/')[-3:]
        exp_out.labels['path'] = path
        exp_out.labels['mode'] = mode
        exp_out.labels['name'] = name


        try:
            data_frame = pd.read_csv(os.path.join(path, 'progress.csv'))
            iou_final = Curve(np.array(data_frame['eval/iou_final'].dropna()))
            iou_increase = Curve(np.array(data_frame['eval/iou_increase'].dropna()))
            if self.cfg.load_curves:
                exp_out.curves['iou.final'] = iou_final
                exp_out.curves['iou.increase'] = iou_increase
            exp_out.scalars['iou'] = iou_final.mean.max()
            exp_out.labels['iou'] = f"{iou_final.mean.max():.4f}"
        except Exception:
            pass

        if os.path.exists(os.path.join(path, 'video.mp4')):
            video_path = os.path.join(path, 'video.mp4')
            exp_out.videos['video'] = video_path

        return exp_out
            
    def load_single(self, path: str) -> ExperimentOutput:
        if not os.path.isdir(path):
            raise LoadFailureException(f"{path} is not a folder")

        exp_out = ExperimentOutput()

        # parse meta data
        mode, name, date = path.split('/')[-3:]
        exp_out.labels['path'] = path
        exp_out.labels['mode'] = mode
        exp_out.labels['name'] = name
        
        # exp_out.labels['date'] = date

        from diffsolver.config import DefaultConfig
        try:
            cfg = cast(DefaultConfig, OmegaConf.load(os.path.join(path, 'new_config.yaml')))
        except FileNotFoundError:
            raise LoadFailureException(f"{path} does not contain new_config.yaml")
        if self.cfg.load_tool_lang:
            if os.path.exists(os.path.join(path, 'tool_lang.txt')):
                exp_out.texts['tool_lang'] = open(os.path.join(path, 'tool_lang.txt')).read()
            if os.path.exists(os.path.join(path, 'phys_lang.txt')):
                exp_out.texts['phys_lang'] = open(os.path.join(path, 'tool_lang.txt')).read()

        def add_tool_code():
            exp_out.labels['Tool: '] = str(cfg.tool_sampler.lang)
            exp_out.texts['tool_code'] = str(OmegaConf.to_yaml(
                OmegaConf.create({
                    'eq': cfg.tool_sampler.equations, 'cons': cfg.tool_sampler.constraints,
                })
            ))

        if self.cfg.load_task:
            exp_out.labels['Phys: '] = str(cfg.prog.lang)
            exp_out.texts['phys_code'] = str(cfg.prog.code)
            add_tool_code()


        if self.cfg.load_tool:
            load_exist_abspath(exp_out.images, 'start', os.path.join(path, 'start.png'))
            if not self.cfg.load_task:
                add_tool_code()



        if self.cfg.load_phys:
            load_exist_abspath(exp_out.images, 'ending', os.path.join(path, 'ending.png'), not_exist_ok=True)
            #raise NotImplementedError
            if self.cfg.load_video:
                video_key = None
                video_path = ''
                for i in range(50, 1000, 50):
                    if os.path.exists(os.path.join(path, f'iter_{i}.mp4')):
                        video_key = f'{i}'
                        video_path = os.path.join(path, f'iter_{i}.mp4')
                        if self.cfg.load_intermediate:
                            exp_out.videos[video_key] = video_path
                if os.path.exists(os.path.join(path, 'best.mp4')) or video_key is None:
                    video_key = 'best'
                    video_path = os.path.join(path, 'best.mp4')
                
                exp_out.videos[video_key] = video_path
            
            try:
                data_frame = pd.read_csv(os.path.join(path, 'progress.csv'))
                iou_final = Curve(np.array(data_frame['iou.final'].dropna()))
                iou_increase = Curve(np.array(data_frame['iou.increase'].dropna()))
                if self.cfg.load_curves:
                    exp_out.curves['iou.final'] = iou_final
                    exp_out.curves['iou.increase'] = iou_increase

                exp_out.scalars['iou'] = iou_final.mean.max()
                exp_out.labels['iou'] = f"{iou_final.mean.max():.4f}"
            except Exception :
                pass


        load_exist_abspath(exp_out.images, 'goal', os.path.join(path, 'goal.png'), not_exist_ok=True)
        # load_exist_abspath(exp_out.images, 'sols', os.path.join(path, 'sols.png'), not_exist_ok=True)

        return exp_out

    def load_multi(self, path: str) -> ExperimentOutput:
        exp_out = ExperimentOutput()

        mode, name, date = path.split('/')[-3:]
        exp_out.labels['path'] = path
        exp_out.labels['mode'] = mode
        exp_out.labels['name'] = name
        finals = []
        task = name
        task_id = int(task.replace('task', ''))

        if self.cfg.plot_task and task_id not in self.plot_task:
            print(task_id)
            return

        data_frame = None
        try:
            data_frame = pd.read_csv(os.path.join(path, 'progress.csv'))
        except Exception:
            pass

        #for idx, f in enumerate(os.listdir(path)):
        for idx in range(20):
            if os.path.exists(os.path.join(path, f'stage_{idx}')):
                exp_out.labels[f'stage_{idx}'] = os.path.join(path, f'stage_{idx}')
                try:
                    sub_exp = self.load_single(os.path.join(path, f'stage_{idx}'))

                    if idx == 0 and 'start' in sub_exp.images:
                        exp_out.images['start'] = sub_exp.images['start']
                    if 'ending' in sub_exp.images:
                        exp_out.images[f'ending_{idx}'] = sub_exp.images['ending']
                    if 'goal' in sub_exp.images:
                        exp_out.images[f'goal_{idx}'] = sub_exp.images['goal']

                    for k, v in sub_exp.videos.items():
                        exp_out.videos[f'stage_{idx}_{k}'] = v
                    if self.cfg.load_curves:
                        name = f'stage_{idx}/final.final'
                        if data_frame is not None and name in data_frame:
                            finals.append(np.array(data_frame[name].dropna()))

                except LoadFailureException:
                    continue
        try:
            task_id = int(task.replace('task', ''))
            if task_id in self.no_break:
                def find_max_stage(dir_path):
                    subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
                    stage_dirs = [d for d in subdirs if d.startswith('stage_')]
                    stage_numbers = [int(d.replace('stage_', '')) for d in stage_dirs]
                    max_stage_dir = 'stage_' + str(max(stage_numbers))
                    return max(stage_numbers), os.path.join(dir_path, max_stage_dir)

                max_stage, max_stage_path = find_max_stage(path)

                import torch
                # print(max_stage_path)
                data_frame = torch.load(os.path.join(max_stage_path, 'trajs.pt'))
                states: Sequence[WorldState] = data_frame['states']

                final_state = states[len(states) - 1]

                pcd = final_state.X

                from diffsolver.program.common.pcd import near_pairs
                pcd = torch.from_numpy(pcd).cuda()
                ind = near_pairs(pcd)
                assert ind is not None
                mx, _ = torch.linalg.norm(
                    pcd[ind[0].long()] - pcd[ind[1].long()], axis=-1
                ).max(axis=-1)
                assert isinstance(mx, torch.Tensor)

                no_break_result = torch.relu(mx)
                no_break_result = no_break_result.cpu().numpy()

                if no_break_result < self.no_break[task_id]:
                    exp_out.scalars['no_break'] = 1
                    exp_out.labels['no_break'] = f"{no_break_result:.4f}"
                else:
                    exp_out.scalars['no_break'] = 0
                    exp_out.labels['no_break'] = f"{no_break_result:.4f}"
        except Exception:
            task_id = int(task.replace('task', ''))
            if task_id in self.no_break:
                print('no_break_false')
                exp_out.scalars['no_break'] = 0
                exp_out.labels['no_break'] = 'nan'

        if len(finals) > 0:
            exp_out.curves['iou.increase'] = Curve(np.concatenate(finals))
            exp_out.scalars['iou'] = exp_out.curves['iou.increase'].mean.max()
            exp_out.labels['iou'] = f"{exp_out.curves['iou.increase'].mean.max():.4f}"
        
        return exp_out


    def load_by_timestamp(self, folder):
        outs: List[ExperimentOutput] = []
        for f in sort_by_task(os.listdir(folder))[::-1]:
            p = os.path.join(folder, f)
            if os.path.isdir(p):
                try:
                    if 'sac' in p or 'ppo' in p:
                        outs.append(self.load_rl(p))
                    else:
                        if self.cfg.stage_mode == StageMode.single_stage:
                            outs.append(self.load_single(p))
                        else:
                            out = self.load_multi(p)
                            if out is not None:
                                outs.append(out)

                except LoadFailureException as e:
                    if not self.cfg.find_exist:
                        raise e
                    else:
                        print('load failed', p)
                if len(outs) > self.cfg.latest:
                    break
        return outs


def get_plotter(_task: Task|None = None):

    _cfg: DictConfig = OmegaConf.structured(PlotConifg)

    if _task is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("task", type=Task, default=Task.tool_sample)
        parser.add_argument("--public", action="store_true")
        args, unknown = parser.parse_known_args()
        task: Task = args.task
        from diffsolver.plotter import renderer
        if args.public:
            renderer.SUDO = False
        else:
            renderer.SUDO = True
        renderer.TASK = task.value
    else:
        task = _task
        unknown = []

    _cfg.merge_with(OmegaConf.load(os.path.join(FILEPATH, f"{task.value}_plotter.yaml")))
    _cfg.merge_with(OmegaConf.from_dotlist(unknown))

    cfg = cast(PlotConifg, _cfg)
    return Plotter(cfg)


def main():
    plotter = get_plotter()
    plotter.render()



if __name__ == "__main__":
    main()
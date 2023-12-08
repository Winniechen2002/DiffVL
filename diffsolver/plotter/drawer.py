# given a group of runs
# draw a figure for each run
#   grouped by (method, env_name) for each run
# library for drawing paper-oriented figures
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import Dict, List, cast, Tuple
import matplotlib.pyplot as plt
from .exp_results import ExperimentOutput, Curve
import numpy as np
from numpy.typing import NDArray




def smooth(y, smoothingWeight=0.95):
    y_smooth = []
    last = y[0]
    for i in range(len(y)):
        y_smooth.append(last * smoothingWeight + (1 - smoothingWeight) * y[i])
        last = y_smooth[-1]
    return np.array(y_smooth)




@dataclass
class DrawConfig:
    curves: List[str] = field(default_factory=list)
    bin_width: int = 10
    max_steps: int|None = None
    curve_scale: float = 1
    width: int = 5
    colors: List[str] = field(default_factory=lambda: ['C3', 'C0', 'C1', 'C2', 'C4', 'C5'])
    smooth_weight: float = 0.3
    linewidth: float = 3.
    metric: str = 'metric.yaml'
    use_no_break: bool = False

    font_size: int = 20
    figsize: Tuple[int, int] = (8, 6)

    plot_success: bool = True
    plot_task: bool = False

    
    scalars: List[str] = field(default_factory=lambda: [])

    dump_final_iou: str = ''

import os
FILEPATH = os.path.dirname(os.path.abspath(__file__))

class Drawer:
    def __init__(self, config: DrawConfig) -> None:
        self.config = config
        self.metric = OmegaConf.load(os.path.join(FILEPATH, config.metric))
        if self.config.plot_task:
            self.plot = OmegaConf.load(os.path.join(FILEPATH, 'plot_task.yaml'))

    def draw(self, runs: Dict[str, Dict[str, List[ExperimentOutput]]]):
        print('123')
        self.runs = runs
        self.sort_runs()
        for curve_name in self.config.curves:
            self.plot_curves(curve_name)

    def get_tables(self, runs: Dict[str, Dict[str, List[ExperimentOutput]]]):
        self.runs = runs
        outs = []
        if self.config.use_no_break:
            for val_name in self.config.scalars:
                if val_name != 'no_break':
                    outs.append([val_name, self.summarize(val_name)])
        else:
            for val_name in self.config.scalars:
                outs.append([val_name, self.summarize(val_name)])
        return outs

    def sort_runs(self):
        category=dict(
            Deform=[7, 38, 11, 24],
            Move=[10, 22, 5, 13],
            Wind=[2, 36, 57, 50],
            Fold=[18, 65, 62, 28],
            Cut=[39, 37, 70, 25],
        )

        #for k, v in category.items():
        rank = {}
        idx = 0
        rename = {}
        for i in range(4):
            for k, v in category.items():
                print(idx, v[i])
                rank[v[i]] = idx
                rename[idx] = f"{k}-v{i}"
                idx += 1

        #print(category)
        #print(self.runs.keys())
        output = [''] * len(rank)
        labels = list(self.runs.keys())
        for i in range(len(self.runs)):
            taskid = int(labels[i].split('_')[0][4:])
            if taskid in rank:
                output[rank[taskid]] = labels[i]
                rename[labels[i]] = rename[rank[taskid]]
            else:
                print(idx)
        print(output, len(output))
        
        self.runs = {rename[k]: self.runs[k] for k in output}
        print(len(self.runs))




    def summarize(self, val_name):
        outs = {}
        env_names = set()
        success_num = {}
        total_num = {}
        sum_iou = {}
        for method in self.runs:
            for env_name, exps in self.runs[method].items():
                env_names.add(env_name)

                alls = []
                for exp in exps:
                    if val_name in exp.scalars:
                        alls.append(exp.scalars[val_name])
                no_breaks = []
                for exp in exps:
                    if 'no_break' in exp.scalars:
                        no_breaks.append(exp.scalars['no_break'])
                outs[env_name] = outs.get(env_name, {})

                n = len(alls)
                if self.config.plot_success:
                    task_id = method.split('_')[0].replace('task', '')
                    if self.config.plot_task and int(task_id) not in self.plot:
                        continue
                    if int(task_id) in self.metric:
                        threshold = self.metric[int(task_id)]
                        if self.config.use_no_break and len(no_breaks) == len(alls):
                            p = len([i for i,j in zip(alls, no_breaks) if i >= threshold and j == 1])
                        else:
                            p = len([i for i in alls if i >= threshold])
                        n = f'{p}/{len(alls)}'
                        if env_name not in total_num:
                            total_num[env_name] = 0
                            success_num[env_name] = 0
                            sum_iou[env_name] = 0
                        total_num[env_name] += len(alls)
                        success_num[env_name] += p
                        sum_iou[env_name] += np.sum(alls)
                outs[env_name][method] = {'mean': np.mean(alls), 'std': np.std(alls), 'n': n}


        print(total_num)
        print(success_num)
        print(sum_iou)
        
        if len(self.config.dump_final_iou) > 0:
            #df = pd.DataFrame(outs)
            #df.to_csv(self.config.dump_final_iou)
            import pickle
            with open(self.config.dump_final_iou, 'wb') as pickle_file:
                pickle.dump(outs, pickle_file)


        html = "<table border='1'>"
        html += "<tr><th>EnvName</th>"
        env_names = sorted(list(env_names))
        for env_name in env_names:
            html += f"<th>{env_name}</th>"
        html += "</tr>"

        for idx, method in enumerate(self.runs):
            html += f"<tr><td>({idx+1:d}) {method}</td>"

            best = None
            best_env = None
            for env_name in env_names:
                if method in outs[env_name]:
                    m = outs[env_name][method]['mean']
                    if best is None or m > best:
                        best = m
                        best_env = env_name

            for env_name in env_names:
                if method in outs[env_name]:
                    mean = f"{outs[env_name][method]['mean']:.3f} &plusmn; {outs[env_name][method]['std']:.3f} ({outs[env_name][method]['n']})"
                    if env_name == best_env:
                        mean = f"<b>{mean}</b>"
                else:
                    mean = "N/A"
                html += f"<td>{mean}</td>"
            html += "</tr>"

        html += "</table>"
        return html
                


    def plot_curves(self, curve_name: str):

        font = {'size': self.config.font_size}
        import matplotlib
        import os
        matplotlib.rc('font', **font)

        curves: Dict[str, Dict[str, Curve]] = {}
        for method in self.runs:
            for env_name, exps in self.runs[method].items():
                xs, ys = [], []
                for exp in exps:
                    if curve_name in exp.curves:
                        curve = exp.curves[curve_name]
                        xs.append(curve.xs)
                        ys.append(curve.mean)

                curves[method] = curves.get(method, {})
                if len(xs) > 0:
                    if env_name == 'ppo':
                        xs = [i*20 for i in xs]
                    curves[method][env_name] = self.merge_curves(xs, ys)

        

        width = min(self.config.width, len(curves))
        n_rows = (len(curves) + width - 1)//width

        # https://github.com/hzaskywalker/RPG/blob/rpg/rpg/scripts/plot.py

        fig, _axs = plt.subplots(n_rows, width, figsize=(self.config.figsize[0] * width, self.config.figsize[1] * n_rows))

        if isinstance(_axs[0], np.ndarray):
            axs: List[plt.Axes] = sum([list(x) for x in _axs], [])
        else:
            axs = cast(List[plt.Axes], _axs)

        id = ord('A')

        label2color = {}
        color_idx = 0
        handles = {}

        for ax, env_name in zip(axs, curves):
            #out = plot_env(ax, env_name, chr(id))

            for method, curve in curves[env_name].items():
                _idx = label2color.get(method, None)
                if _idx is None:
                    _idx = color_idx
                    color_idx = (color_idx + 1) % len(self.config.colors)
                
                label2color[method] = _idx

                handles[method], = self.plot_curve_with_shade(
                    ax,  curve.xs, curve.mean,  curve.stds, method, 
                    color=self.config.colors[_idx]
                )
            ax.set_title("("+ chr(id) +") " + env_name.capitalize())
            ax.set_xlabel("# Episodes"); 
            ax.set_ylabel('IOU')
            ax.grid(True)
            id += 1

        # hind axes unused
        for ax in axs[len(curves):]:
            ax.axis('off')
        fig.subplots_adjust(bottom=0.25)
        fig.tight_layout()

        fig.legend(list(handles.values()), ['Adam', 'PPO', 'SAC'], loc='upper center', ncols=3, prop={'size': 30}) 
        fig.subplots_adjust(top=0.95)


        # fig.subplots_adjust(top=0.9, left=0.155, right=0.99, bottom=0.2)
        plt.savefig(f'{curve_name}.png')


    def plot_curve_with_shade(self, ax: plt.Axes, x: NDArray, mean: NDArray, std: NDArray, label, color='green', **kwargs):
        smoothingWeight = self.config.smooth_weight
        linewidth = self.config.linewidth
        y_smooth = smooth(mean, smoothingWeight)
        std_smooth = smooth(std, smoothingWeight) * 0.3
        ax.fill_between(x, (y_smooth - std_smooth).clip(0, np.inf), y_smooth + std_smooth, facecolor=color, alpha=0.2) # type: ignore
        return ax.plot(x, y_smooth, color=color, label=label, linewidth=linewidth, **kwargs)
    

    def merge_curves(self, x_list, y_list):
        bin_width = self.config.bin_width

        x = np.concatenate(x_list)
        y = np.concatenate(y_list)

        idx = x.argsort()
        x = x[idx]
        y = y[idx]

        if self.config.max_steps is not None:
            idx = x <= self.config.max_steps
            x = x[idx]
            y = y[idx]
        assert (x >= 0).all()
        nbins = int(x.max() // bin_width + 1)
            
        n, _ = np.histogram(x, bins=nbins)
        sy, _ = np.histogram(x, bins=nbins, weights=y)
        sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
        xx, _ = np.histogram(x, bins=nbins, weights=x)

        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)
        xx = xx / n
        idx = xx>0
        return Curve(mean[idx], std[idx], xs=xx[idx]/self.config.curve_scale)

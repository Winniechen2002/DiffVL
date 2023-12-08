from typing import List, Dict, Any
import threading
import subprocess
import os
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify
from dataclasses import dataclass
from diffsolver.plotter.exp_results import ExperimentOutput, Task

@dataclass
class HTMLRenderConfig:
    port: int = 8000                                    # port for the renderer
    mode: str = 'all'                                   # all, finihsed, time, failed

SUDO = False


TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'templates', 'template.html')


app = Flask(__name__, template_folder=os.path.dirname(TEMPLATE_PATH), static_folder=None)
DATA = None
TASK = None
EXTRA = None
current_dir = os.getcwd()

OUTPUT = None

def run_command(command):
    print('running command', command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    global OUTPUT
    OUTPUT = output.decode('utf-8')
    return output, error

@app.route('/run_command/<command>/<parameter>')
def command(command, parameter):
    if SUDO:
        parameter = parameter.replace('>>', '/')
        if command == 'ls':
            cmd = command + ' ' + parameter
            thread = threading.Thread(target=run_command, args=(cmd ,))
            thread.start()
        elif command == 'sync':
            from diffsolver.launch.utils import get_sync_cmd
            cmd = get_sync_cmd(parameter)
            thread = threading.Thread(target=run_command, args=(cmd,))
            thread.start()
        else:
            raise NotImplementedError
        return jsonify({'message': f'Command {cmd} is being executed'}), 200
    else:
        return jsonify({'message': f'You can not run command without SUDO'}), 200


@app.route('/get_output')
def get_output():
    global OUTPUT
    result = OUTPUT
    # OUTPUT = ""  # reset the output
    return jsonify({'output': str(result)})




@app.route('/<path:filename>')
def custom_static(filename):
    # Get the current directory
    global current_dir
    return send_from_directory(current_dir, filename)

def reload():
    global DATA, TASK, EXTRA
    from diffsolver.plotter.plot import get_plotter
    plotter = get_plotter(Task(TASK))
    DATA = HTMLRenderer(plotter.cfg.html).load_results(plotter.results)
    EXTRA = plotter.summary


@app.route('/delete_file', methods=['POST'])
def delete_file():
    filename = str(request.form.get('filename'))
    try:
        if SUDO:
            os.system("rm -r " + filename)
        # print('deleteing..' + filename)
        reload()
    except FileNotFoundError:
        return "There was an error while deleting the file"
    return redirect(request.referrer or url_for('home'))

@app.route('/taskview')
def taskview():
    from diffsolver.plotter.task_viewer import collect_tasks
    return render_template('taskview.html', data=collect_tasks())

# @app.after_request
# def add_no_cache(response):
#     response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
#     response.headers["Pragma"] = "no-cache"
#     response.headers["Expires"] = "-1"
#     return response


@app.route('/', methods=['GET', 'POST'])
def index():
    global DATA, TASK
    if request.method == 'POST':
        TASK = request.form.get("task")
        reload()
    return render_template('template.html', variations=DATA, tasks=[t.value for t in Task], cur_task=TASK, extra=EXTRA)


class HTMLRenderer:
    def __init__(self, config: HTMLRenderConfig) -> None:
        self.config = config

    def get_run_infos(self, exp: ExperimentOutput):
        elements = []

        for k, v in exp.texts.items():
            text = {
                'type': 'code',
                'title': k,
                'content': v,
            }
            elements.append(text)

        for k, v in exp.images.items():
            image_path = os.path.relpath(v, self.root_path)
            image = {
                'path': image_path,
                'type': 'image', 
                'title': k,
            }
            elements.append(image)
        
        for k, v in exp.videos.items():
            video_path = os.path.relpath(v, self.root_path)
            videos = {
                'path': video_path,
                'type': 'video',
                'title': k,
                'extension': video_path.split('.')[-1],
            }
            elements.append(videos)


        for k, v in exp.curves.items():
            curves = {
                'type': 'curve',
                'title': k,
                'content': {
                    'x_values': v.xs.tolist(),
                    'y_values': v.mean.tolist(),
                    'y_uppers': (v.mean + v.stds).tolist(),
                    'y_lowers': (v.mean - v.stds).tolist(),
                }
            }
            elements.append(curves)

        for idx, p in enumerate(elements):
            p['eid'] = idx
        return elements

    def load_results(self, results: Dict[str, Dict[str, List[ExperimentOutput]]]):
        variations = []
        self.root_path = os.getcwd()
        print(self.root_path)

        #TODO: summary later
        runid = 0
        for mode in results:
            runs = []
            for name, exps in results[mode].items():
                for e in exps:
                    # TODO: make labels into table

                    run = {'elements': self.get_run_infos(e), 'title': name, 'id': runid, 'labels': [{'key': key, 'value': e.labels[key]} for key in e.labels]}
                    runs.append(run)
                    runid += 1

            variations.append({'runs': runs, 'name': mode})
        return variations


    def run(self, results: Dict[str, Dict[str, List[ExperimentOutput]]], extra: Dict[str, Any]):
        global DATA, EXTRA
        EXTRA = extra
        DATA = self.load_results(results)
        app.run('0.0.0.0' if not SUDO else None, debug=True, port=self.config.port)
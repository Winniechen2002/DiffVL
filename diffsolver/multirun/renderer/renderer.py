from typing import List, Mapping
import re
import os
from ..multirun_config import RenderConfig, BaseConfigType
from jinja2 import Environment, FileSystemLoader
from glob import glob
import http.server
import socketserver


TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'multirun_template.html')

class HTMLRenderer:

    def __init__(self, config: RenderConfig, path: str, exps: Mapping[str, BaseConfigType], extra_info=None) -> None:
        self.config = config
        self.exps = exps
        self.path = path
        self.extra_info = extra_info

        assert self.config.mode == 'all'

    def get_run_infos(self, name: str, exp: BaseConfigType):
        rel_path = os.path.relpath(exp.saver.path, self.path)

        match = re.match(r'(.*)_seed_(\d+)', name)
        if match is not None:
            group_name = match.group(1)
            seed = match.group(2)
        else:
            group_name = rel_path
            seed = None


        outputs = []
        for path in glob(os.path.join(exp.saver.path, 'config.yaml')):
            #TODO: only visualize the latest
            path = os.path.dirname(path)
            rel_path = os.path.relpath(path, self.path)


            n_intemediate = exp.trainer.nsteps // exp.saver.render_interval
            videos = []
            filenames = ['best.mp4'] + [f'iter_{(k+1) * exp.saver.render_interval}.mp4' for k in range(n_intemediate)] 
            for f in filenames:
                f = os.path.join(rel_path, f)
                if os.path.exists(os.path.join(self.path, f)):
                    videos.append({
                        'path': f,
                        'extension': 'mp4',
                        'title': f.split('/')[-1].split('.')[0],
                    })

            output = {
                'path': path,
                'videos': videos,
            }
            for image_names in ['curve', 'start', 'goal', 'ending']:
                if os.path.exists(os.path.join(self.path, rel_path, image_names+'.png')):
                    output[image_names] = os.path.join(rel_path, image_names + '.png')
            print(output)
            outputs.append(output)
        return {
            'name': group_name,
            'seed': seed,
            'runs': outputs,
        }

    def run(self):
        env = Environment(loader=FileSystemLoader(os.path.dirname(TEMPLATE_PATH)))

        template = env.get_template(os.path.basename(TEMPLATE_PATH))

        variations = [self.get_run_infos(k, v) for k, v in self.exps.items()]

        html_content = template.render(variations=variations, extra_info=self.extra_info)
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, 'summary.html'), 'w') as output_file:
            output_file.write(html_content)

        # open a port and serve the html, 
        PORT = self.config.port

        root_path = self.path
        class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.directory = root_path
                super().__init__(*args, directory=self.directory, **kwargs)

            def do_GET(self):
                if self.path == '/':
                    self.path = '/summary.html'
                return super().do_GET()


        with socketserver.TCPServer(("", PORT), MyRequestHandler) as httpd:
            try:
                print("serving at port", PORT)
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nShutting down the server...")
                httpd.shutdown()
                httpd.server_close()
                print("Server has been shut down.")
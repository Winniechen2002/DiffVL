import subprocess
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import PromptSession, CompleteStyle
from typing import List, Tuple
from prompt_toolkit.completion import WordCompleter, PathCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.application.current import get_app
import os

class NautilusError(Exception):
    pass

def sort_by_task(tasks: List[str]):
    def task_id(task):
        task = task.split('.')[0]
        val = task.split('_')[0].replace('task', '')
        try:
            index = int(val)
            return ("{:04d}".format(index), task)
        except:
            return (task, task)
    return sorted(tasks, key=task_id)

def get_http_pod():
    # get the output of kubeclt get pods|grep hza|grep Running
    cmd = "kubectl get pods|grep hza|grep Running|awk '{print $1}'"
    output = subprocess.getoutput(cmd)
    for k in output.split('\n'):
        if k.startswith('hza-http-'):
            return k.strip().split(' ')[0]
    raise NautilusError('http pod not found')

def get_MODEL_DIR():
    if 'MODEL_DIR' not in os.environ:
        raise NautilusError('MODEL_DIRS not set')
    return os.environ['MODEL_DIR']


REMOTE_MODEL_PATH  = '/cephfs/hza/diffsolver/models/'
def get_sync_cmd(dir):
    http = get_http_pod()
    dir = os.path.relpath(dir, get_MODEL_DIR())

    remote_dir = os.path.join(REMOTE_MODEL_PATH, dir)
    local_dir = os.path.join(get_MODEL_DIR(), dir)

    cmd = f"kubectl cp {http}:{remote_dir} {local_dir}"
    return cmd

def delete_remote_cmd(dir: str):
    http = get_http_pod()
    dir = os.path.join(REMOTE_MODEL_PATH, os.path.relpath(dir, get_MODEL_DIR()))
    return f"kubectl exec {http} -- bash -c 'rm -rf {dir}'"



def interactive_input(prompt, options):
    option_completer = WordCompleter(options, ignore_case=True, sentence=True)

    while True:
        session = PromptSession(prompt, completer=option_completer,
                                complete_style=CompleteStyle.MULTI_COLUMN)
        out = session.prompt()
        if out in options:
            return out



def keys(prompttext, options):
    selected_index = 0
    start = 0
    end = None

    confirmed = False

    bindings = KeyBindings()

    @bindings.add(Keys.Down)
    def _(event):
        nonlocal selected_index
        selected_index = (selected_index + 1) % len(options)
        event.app.layout.current_buffer.reset()

    @bindings.add(Keys.Up)
    def _(event):
        nonlocal selected_index
        selected_index = (selected_index - 1) % len(options)
        event.app.layout.current_buffer.reset()


    @bindings.add(Keys.Any)
    def _(event):
        nonlocal selected_index, confirmed
        key = event.key_sequence[0].key

        if '0' <= key <= '9':
            index = int(key) - 1
            if 0 <= index < len(options):
                selected_index = index
                confirmed = True
                event.app.exit()


    def get_prompt_text():
        height = get_app().output.get_size().rows
        nonlocal start, end
        end = start + height - 2
        
        while selected_index >= end:
            start += 1
            end += 1
        while selected_index < start:
            start -= 1
            end -= 1

        formatted_options: List[Tuple[str, str]] = [('default', f'Use arrow keys or numbers to select an option {prompttext} (~{len(options)}/{height}):\n')]
        for i, option in enumerate(options):
            if i >= start and i < end:
                if i == selected_index:
                    formatted_options.append(('green', f"> {i+1}. {option}\n"))
                else:
                    formatted_options.append(('default', f"  {i+1}. {option}\n"))

        # height_max = height - 2
        # if len(options) > height_max:
        #     mid = selected_index
        #     low = max(0, mid - height_max // 2)
        #     high = min(len(options), low + height_max)
        #     formatted_options = [formatted_options[0], *formatted_options[low+1:high+1]]

        return FormattedText(formatted_options)

    prompt(get_prompt_text, key_bindings=bindings)
    return options[selected_index]



if __name__ == '__main__':
    #print(get_http_pod())
    print(get_sync_cmd('multistage/dev/'))
import os
import termcolor
import time
import termcolor
from diffsolver.program.clause.prompt import get_prompt
from diffsolver.program.scenes import SceneTuple
from diffsolver.program.types import SceneSpec

from diffsolver.config import TranslatorConfig

def query(content: str, model='gpt-4'):
    start = time.time()
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo",
        model=model,
        messages=[
                {"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": content},
            ]
        # Please give me three different answers for the provided questions and separate them with single line of single '%'. 
    )

    result = ''
    for choice in response.choices: # type: ignore
        result += choice.message.content

    end = time.time()
    return end - start, result

    
def translate(task, config: TranslatorConfig):
    prompts = get_prompt(task, config)
    if config.version == 'v3': 
        print(prompts)
        t, answer = query(prompts)
        print(answer)
        termcolor.cprint(f"Passing ... {float(t)} seconds", color='green')
        clauses = []
        for line in answer.split('\n'):
            if line.startswith('-'):
                clauses.append(line[1:].strip())
        return 'tand(\n  ' + ',\n  '.join(clauses) + '\n)'
    else:
        t, answer = query(prompts)
        print(answer)
        termcolor.cprint(f"Passing ... {float(t)} seconds", color='green')

        values = []
        clauses = []
        for line in answer.split('\n'):
            line = line.strip()
            if line.startswith('Objects'):
                values.append('\n'.join(eval(line[9:].strip())))
            elif line.startswith('Prog:'):
                clauses.append(line[6:].strip())
            
        out = '\n'.join(values) + '\n\ntand(\n  ' + ',\n  '.join(clauses) + '\n)'
        return out

    
def translate_program_from_scene(lang: str, scene_tuple: SceneTuple, config: TranslatorConfig, max_retry: int=10, scene: SceneSpec|str|None = None, tool_lang=None):
    history = ''
    while max_retry:
        if config.version == 'precode':
            return config.code
        from .scene_description import get_scene_description
        if isinstance(scene, str):
            scene_description = scene
        else:
            if config.version != 'v3':
                scene_description = get_scene_description(scene_tuple)
            else:
                assert tool_lang is not None
                scene_description = f"Objects: {list(scene_tuple.names.keys())}\nInput: " # \nTool: {tool_lang}


        if config.version != 'v3':
            prompts = "Please generate clauses for the following task. First generate the scene descriptions and then generate the clauses. Please do not use functions not mentioned before."
            prompts += "The scene contains the following  objects: [" + scene_description + "]" + ". The task is: " + lang
        else:
            prompts = "Please translate the input into a program. The generated programs satisfy several requirements: Do not use functions not mentioned before. The program should be concise. Do not use no_break for 'all'. Only use 'touch' at most once and if we need to manipulate multiple objects, please use touch('all'). Use 'fix_place' at most once and it should always have the form of fix_place(others(obj_name)).\n Make sure that 'away', 'no_break', or 'fix_place' only appears in the program if it or its synonyms appear in the input.\nHere is the input to translate:\n"
            prompts += scene_description + lang

        if config.version == 'emdonly':
            return """obj0 = get_iobj('all')
goal = get_goal('all')
tand(
    keep(touch(obj0)),
    last(emd(obj0, goal))
)"""

        code =  translate(prompts, config)

        history = history + '\n\n' + code

        try:
            print("generate code: ")
            print(code)

            if code.count('fix_place') > 1:
                raise Exception('fix_place should be used at most once')

            if code.count('touch') > 1:
                raise Exception('touch should be used at most once')

            from ..prog_registry import _PROG_LIBS
            fn = _PROG_LIBS.parse(code)

            if isinstance(scene, SceneSpec) and config.verify_scene:
                fn(scene)

            return code
        except Exception as e:
            termcolor.cprint(f"Failed to parse the code, retrying ... {e}", color='red')

            if max_retry > 0:
                termcolor.cprint("Failed to parse the code, retrying ...", color='red')
            else:
                termcolor.cprint("Failed to parse the code, aborting ...", color='red')
        max_retry -= 1

    return 'HISTORY:\n' + history + '\n\n'
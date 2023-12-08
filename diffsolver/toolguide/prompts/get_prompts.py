import json
import os

def get():
    FILEPATH = os.path.dirname(os.path.abspath(__file__))

    inputs = {}
    for filename in ['clause', 'dsl', 'tool_type', 'examples']:
        with open(f'{FILEPATH}/{filename}.json') as f:
            inputs[filename] = json.load(f)


    output = inputs['dsl']['explanation']
    output += "It contains the following functions:\n"
    for op in inputs['dsl']['DSL']:
        output += op['code'] + '   # ' + op['explanation'] + '\n'

    output += 'Here are examples of using those functions:\n'
    for op in inputs['clause']['examples']:
        output += op['code'] +  '   # ' + op['explanation'] + '\n'

    output += 'Notice that you should set the tool based on the text description. Below are tools and their example text description\n'
    for tool in ['knife', 'gripper', 'board', 'large_fingers', 'single_finger']:
        #output += op['code'] + '   # ' + op['explanation'] + '\n'
        output += f'{tool}:   ' + ', '.join(inputs['tool_type']['examples'][tool]) + '\n'
        #print(inputs['tool_type']['examples'][tool])

    #print(output)
    output += 'Combinig the tools and various clauses, we can now translate the text description into a program. Here are some examples:\n\n'
    output += "Notice that we must set_tool in the beginning. There can be only one locate clause for each text input." 

    for ex in inputs['examples']:
        output += "Text input: " + ex['text'] + "\n- "
        output += '\n- '.join(ex['codes'])
        output += '\n'

    
    return output

def get_input(task):
    return get() + "\nNow please try to translate the following description into codes of the aforementioned form.\n" + task

def answer(task: str, verbose=False) -> str:
    from diffsolver.program.clause.translator import query
    from tools.utils import logger
    print("Parsing text input: " + task + '...', end='')
    inp = get_input(task)
    if verbose:
        print(inp)
    seconds, result = query(inp)
    print('Done. It takes ' + str(seconds) + ' seconds.')
    #logger.log('Parsing text input: ' + task + '...', 'Done. It takes ' + str(seconds) + ' seconds.\n' + result + '\n')
    print(result)
    return result

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--run', action='store_true')
    # args = parser.parse_args()

    # """1. split the ball into two parts\n 2. grasp the object horizontally and put the tool on the right of the object\n 3. Deform the object. Put the tool on the left of the object. 
    # """

    # input = get() + "\nNow please try to translate the following text inputs into codes of the aforementioned form.\n Grasp the object horizontally and ensure the tool on the right of the object." 
    # print(input)
    # if args.run:
    #     from diffsolver.program.clause.translator import query
    #     print(query(input)[1])
    #answer("touch the backend of obj vertically using a single finger.")
    task = "grasp the object horizontally and ensure the tool on the right of the object."
    inp = get_input(task)
    print(inp)
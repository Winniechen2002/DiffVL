from omegaconf import OmegaConf, DictConfig
class ParseError(Exception):
    pass

def parse_function(s: str):
    start = s.find("(")
    if start == -1:
        return None, None

    end = s.rfind(")")
    if end == -1:
        return None, None

    func_name = s[:start].strip()
    args_str = s[start + 1:end].strip()

    args = []
    open_parens = 0
    arg_start = 0
    for i, c in enumerate(args_str):
        if c == "(":
            open_parens += 1
        elif c == ")":
            open_parens -= 1
        elif c == "," and open_parens == 0:
            args.append(args_str[arg_start:i].strip())
            arg_start = i + 1

    if arg_start < len(args_str):
        args.append(args_str[arg_start:].strip())

    return func_name, args

def parse_function_calls(text: str):
    lines = text.split("\n")
    function_calls = []

    for line in lines:
        if line.startswith("- "):
            function_call = line[2:].strip()
            function_calls.append(function_call)

    return function_calls
    

def parse2yaml(text_input: str):
    functions = parse_function_calls(text_input)

    output = {
        "sampler": {
            "use_tips": False,
            'optimize_loss': True,
            "equations": [],
            "constraints": ['collision_free'],
        }
    }

    USE_CPDFORM = False
    OBJ_NAME = None

    for func_str in functions:
        func_name, args = parse_function(func_str)
        if func_name is None or args is None:
            print(f"Error parsing function: {func_str}")
            continue
        print(f"Function name: {func_name}")
        print(f"Arguments: {args}")
        print()

        if func_name == 'set_tool':
            assert len(args) == 1

            tool_name = args[0]
            if tool_name == 'knife':
                config = {
                    "tool_name": "Pusher",
                    "qpos": [0.5, 0.25, 0.5, 0., 0., 0.],
                    "size": (0.02, 0.1, 0.2),
                    "friction": 0.0,
                }
                output["sampler"]['use_tips'] = True
            elif tool_name == 'gripper':
                config = {
                    "tool_name": "Gripper",
                    "size": (0.03, 0.08, 0.03),
                    "friction": 1.
                }
            elif tool_name == 'board':
                config = dict(
                    tool_name="Pusher",
                    qpos= [0.5, 0.25, 0.5, 0., 0., 0.],
                    size=(0.15, 0.03, 0.15),
                    friction=5.
                )
                output["sampler"]['use_tips'] = True
            elif tool_name == 'single_finger':
                config = dict(
                    tool_name="Pusher",
                    size=(0.03, 0.08, 0.03),
                    friction=1.
                )
                output["sampler"]['use_tips'] = True
            elif tool_name == 'large fingers' or tool_name == "large_fingers":
                config = dict(
                    tool_name='DoublePushers',
                    qpos=[0.3, 0.1, 0.5, 0., 0., 0., 0.7, 0.1, 0.5, 0., 0., 0.],
                    size=(0.02, 0.2, 0.2)
                )
            else:
                raise ParseError(f"Tool {tool_name} not implemented")


            output['scene'] = dict(Tool=config)
        elif func_name.startswith('cpdeform'):
            output['sampler']['constraints'].append(func_str)
            USE_CPDFORM = True
        
        else:
            if func_name == 'set_coord':
                OBJ_NAME = args[0]

            output['sampler']['equations'].append(func_str)

    if not USE_CPDFORM and OBJ_NAME is not None:
        output['sampler']['constraints'].append(f'touch_pcd({OBJ_NAME})')


    return OmegaConf.create(output)



if __name__ == "__main__":
    text_input = """
What is the cuda visible devices? NOT SET
- set_tool(gripper)
- locate(get_iobj('object'))
- horizontal()
- isright(get_iobj('object'))
"""
    print(OmegaConf.to_yaml(parse2yaml(text_input)))
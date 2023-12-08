from .mydsl import *
from .libpcd import *
from .tool import tool_name_type

class SceneAssertError(Exception):
    """Error in type inference"""

@dsl.as_primitive
def require(f: mybool) -> none:
    if not f:
        executor = dsl.default_executor
        outs = []
        from llm.pl.program import FuncCall
        for i in executor.trace:
            if isinstance(i, list):
                op = i[0]
                if isinstance(op.value, Program):
                    op.pretty_print()
                outs += [f'of {op.token}\n']
            else:
                if isinstance(i, FuncCall):
                    outs += [f'{i.lineval[:70]} in line {i.lineno}  ']
        raise SceneAssertError(''.join(outs[::-1]))

def get_env():
    return dsl.solver.env
        
@dsl.as_primitive
def use_tool() -> none:
    dsl.solver.worm_scale[:] = 0
    dsl.solver.actor_scale[:] = 1


@dsl.as_primitive
def use_soft_body(id: myint, scale:List(myfloat)) -> none:
    dsl.solver.worm_scale[:] = 0
    dsl.solver.actor_scale[:] = 0
    from tools.utils import totensor
    dsl.solver.worm_scale[
        dsl.solver.scene.obj(id).indices] = torch.stack(scale)


@dsl.as_primitive
def switch_tool(tool_name: tool_name_type, qpos: List(myfloat)) -> none:
    env = get_env()
    state = env.get_state()

    if len(qpos) == 0:
        qpos = None

    state = state.switch_tools(tool_name, qpos)
    env.set_state(state)
    use_tool()

    
def set_initial_action() -> none:
    raise NotImplementedError
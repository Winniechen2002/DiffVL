from ..scenes import load_scene_with_envs, SceneTuple


def get_scene_description(scene_tuple: SceneTuple) -> str:
    return ', '.join(list(map(str, scene_tuple.names.keys())))
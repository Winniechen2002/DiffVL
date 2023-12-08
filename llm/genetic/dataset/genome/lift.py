from llm.genetic import *

def lift(scene: Scene):
    """
    scene:  box_pick.yml
    """
    #TODO: add slowly to control the box moving speed ..
    obj = scene.obj(0)
    pickup(scene, obj, [0., 0., 0.], 25, 0.2)
    make_stable(scene, 25)
    return scene
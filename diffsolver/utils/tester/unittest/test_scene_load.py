def test():
    from diffsolver.utils import  load_scene 
    
    scene = load_scene("wind_gripper.yml", clear_cache=True).state
    scene2 = load_scene("wind_gripper.yml").state

    assert scene.allclose(scene2)

    
if __name__ == "__main__":
    test()
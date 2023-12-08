import os
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import sqlite3
from diffsolver.utils import rendering_objects, get_path, WorldState, MultiToolEnv
from omegaconf import OmegaConf
from typing import Tuple


class TaskSeq:
    def __init__(self, task_id) -> None:


        # reload pid means that the particles id are reloaded from the beginning
        data_path = get_path('VISION_TASK_PATH', 'data', f'task_{task_id}')
        image_path = get_path('VISION_TASK_PATH', 'static', f'task_{task_id}')
        naming_dict = OmegaConf.load(os.path.join(get_path("VISION_TASK_PATH"), 'naming.yml'))


        self.naming = naming_dict.get(int(task_id), None)
        #if naming 
        # print(f"{self.naming=}")

        self.task_id = task_id
        # with open(os.path.join(data_path, 'name.txt'), 'r') as f:
        #     self.name = f.readline().strip()


        self.image_paths = sorted(glob.glob(os.path.join(image_path, 'scene_[0-9].png')))
        self.scene_paths = sorted(glob.glob(os.path.join(data_path, 'scene_[0-9].pkl')))
        # print('task: ', self.name)
        if len(self.image_paths) == 0:
            raise Exception(f'no image found in {image_path}')
        print([i.split('/')[-1] for i in self.image_paths])
        print([i.split('/')[-1] for i in self.scene_paths])
        self.num_stages = min(len(self.image_paths), len(self.scene_paths))
        if len(self.image_paths) != len(self.scene_paths):
            logging.warning(f'image and scene number mismatch {len(self.image_paths)} != {len(self.scene_paths)}, {self.image_paths} != {self.scene_paths}')
        # self.db = self.load_db(os.path.join(data_path, 'anno.db'))

        self.use_stage0_id = True and self.task_id != 9

        if self.use_stage0_id:
            state, self.obj_id = self.fetch_stage(0)
            self.stage0_ids = state.ids


    def load_db(self, db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM post')
        rows = c.fetchall()
        conn.close()
        return rows


    def fetch_scene_images(self) -> np.ndarray:
        return np.concatenate([plt.imread(i) for i in self.image_paths], axis=1)
    
    def render_state_images(self, env: MultiToolEnv):
        images = []
        for i in range(self.num_stages):
            state, _ = self.fetch_stage(i)
            env.set_state(state)
            image = env.render('rgb_array')
            assert image is not None
            images.append(image.astype(np.uint8))
        return np.concatenate(images, axis=1) 



    def fetch_stage(self, idx):
        with open(self.scene_paths[idx], 'rb') as f:
            state: WorldState = pickle.load(f).state

            state.tool_cfg = None
            state.tool_name = 'Gripper'
            state.qpos = np.zeros(7).astype(np.float32)
            state.rigid_bodies = None

            object_id = np.unique(state.ids)

            if self.naming:
                title = self.naming
                object_dict = {str(i): np.where(state.ids ==j)[0] for i, j in zip(title, object_id)}
            else:
                title = object_id
                object_dict = {str(int(i)): np.where(state.ids ==j)[0] for i, j in zip(title, object_id)}

            if self.use_stage0_id and idx != 0:
                state.ids = self.stage0_ids
                object_dict = self.obj_id

            return state, object_dict
        return None, None

    def render_stage_objects(self, env: MultiToolEnv, idx):
        state, objects = self.fetch_stage(idx)
        env.set_state(state)
        return rendering_objects(env, objects, scene_title=f'scene_{self.task_id}')

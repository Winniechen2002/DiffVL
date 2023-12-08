import os
from omegaconf import OmegaConf
from moviepy.editor import VideoFileClip, concatenate_videoclips
config_str = """path: None
taskname: task2_mp2

goal_transparency: 0.75
reverse: True

render_mp: True
set_tool_pos: False
# frames: [0, 20, 40, 60, 80]
lookat:
  radius: 1.5
  center: [0.5, 0.5, 0.0]
"""

import os

path = 'Task48'

config = OmegaConf.create(config_str)

os.makedirs('task48', exist_ok=True)
files = []

for i in range(100):
    p = f'{path}/stage_{i}'
    print(p)
    if os.path.exists(p):
        if os.path.exists(f'{p}/goal.png'):
            config.path = p
            config.taskname = f'task48_{i}'
        print(config.path)

        config.output=f'task48/{i}'
        OmegaConf.save(config, f'task48_config.yml')
        
        ok = os.system(f'python plotter/render_images.py task48_config.yml')
        files.append(f'task48/{i}/video.mp4')
        if ok!=0:
            break


def merge_mp4_files(file_paths, output_path):
    clips = []
    
    for file_path in file_paths:
        clip = VideoFileClip(file_path)
        clips.append(clip)
    
    final_clip = concatenate_videoclips(clips)
    
    # Write the final merged clip to a file
    final_clip.write_videofile(output_path)
    
    # Close the clips to free up resources
    for clip in clips:
        clip.close()
    
    final_clip.close()


merge_mp4_files(files, 'task48/merged.mp4')
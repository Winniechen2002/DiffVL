import os
import glob

# Get all yaml/yml files in the examples/single_stage directory
# config_files = glob.glob("examples/single_stage_dev/*.yaml") + glob.glob("examples/single_stage_dev/*.yml")

tasks = ['task39_cut.yaml', 'task25_pick_place.yaml', 'task57_1.yaml', 'task37_cut.yaml', 'task22_stage1.yaml', 'task38_deform.yaml', 'task33_pick_place.yaml', 'task11_wind.yaml', 'task50_3.yaml', 'task2_wind.yaml', 'task62_2.yaml', 'task65_1.yaml', 'task18.yaml', 'task28_wrap.yaml', 'task13_pick_place.yml', 'task10_stage1.yml', 'task24_carving.yml', 'task7_press.yml']

# Loop through each file and execute remote_run.py with the appropriate config file and job name
for file in tasks:
    file = os.path.join("examples/single_stage_dev", file)
    job_name = os.path.splitext(os.path.basename(file))[0].replace("_", "-")
    command = f"python3 run_single_stage.py debug_phys --config {file}"
    out =os.system(command)
    # if out != 0:
    #     print(f"Failed to run {command}")
    #     break
import os
import glob

# Get all yaml/yml files in the examples/single_stage directory
config_files = glob.glob("examples/single_stage/*.yaml") + glob.glob("examples/single_stage/*.yml")

# Loop through each file and execute remote_run.py with the appropriate config file and job name
for file in config_files:
    job_name = os.path.splitext(os.path.basename(file))[0].replace("_", "-")
    command = f"remote_run.py run_single_stage.py prog --config {file} --job_name {job_name} --run"
    os.system(command)
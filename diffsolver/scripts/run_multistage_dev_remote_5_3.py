import os
import glob

# Get all yaml/yml files in the examples/single_stage directory
config_files = glob.glob("examples/multistage_dev/*/total.y*ml")

# Loop through each file and execute remote_run.py with the appropriate config file and job name
for file in config_files:
    job_name = os.path.dirname(file).split('/')[-1].replace("_", "-")
    command = f"remote_run.py run_multistage.py dev --config {file} --job_name {job_name}-multidev --run"
    print(command)
    os.system(command)
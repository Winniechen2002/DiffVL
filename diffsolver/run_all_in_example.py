import os
import subprocess

def main():
    base_command = 'python run_multistage.py prog --config'
    base_directory = 'examples/Tasks'

    for folder_name in os.listdir(base_directory):
        if folder_name.startswith('task'):
            yaml_path = os.path.join(base_directory, folder_name, 'total.yaml')
            yml_path = os.path.join(base_directory, folder_name, 'total.yml')
            
            if os.path.isfile(yaml_path):
                command = f'{base_command} {yaml_path}'
                print(f'Running: {command}')
                subprocess.run(command, shell=True)
            elif os.path.isfile(yml_path): 
                command = f'{base_command} {yml_path}'
                print(f'Running: {command}')
                subprocess.run(command, shell=True)
            else:
                print(f'Skipping {folder_name} as total.yaml was not found.')

if __name__ == '__main__':
    main()

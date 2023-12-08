import os
import subprocess
import argparse

def main():
    # 设置基础命令和目录
    base_command = 'python run_multistage.py mid --config'
    base_directory = 'examples/multistage_show'

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = str, default='task')
    parser.add_argument('--saver', type = str, default = None)
    args, unknown = parser.parse_known_args()

    # 遍历目录
    for folder_name in os.listdir(base_directory):
        # 检查是否以"task"开头
        if folder_name.startswith(args.task):
            # 构建yaml配置文件路径
            yaml_path = os.path.join(base_directory, folder_name, 'total.yaml')
            yml_path = os.path.join(base_directory, folder_name, 'total.yml')
            
            # 检查文件是否存在
            if os.path.isfile(yaml_path):
                # 构建命令并运行
                if args.saver is None:
                    command = f'{base_command} {yaml_path}'
                else:
                    command = f'{base_command} {yaml_path} --saver {args.saver}'
                print(f'Running: {command}')
                subprocess.run(command, shell=True)
            elif os.path.isfile(yml_path): 
                if args.saver is None:
                    command = f'{base_command} {yml_path}'
                else:
                    command = f'{base_command} {yml_path} --saver {args.saver}'
                print(f'Running: {command}')
                subprocess.run(command, shell=True)
            else:
                print(f'Skipping {folder_name} as total.yaml was not found.')

if __name__ == '__main__':
    main()

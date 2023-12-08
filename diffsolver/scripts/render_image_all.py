import os

f = os.listdir("examples/task_images")

for i in f:
    #if i.startswith('task10') or i.startswith('task11'):
    if '37' in i:
        os.system("python3 plotter/render_images.py examples/task_images/{}".format(i))
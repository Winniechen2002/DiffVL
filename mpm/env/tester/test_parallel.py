from multiprocessing import Process


def work():
    import os
    os.system("python3 test3.py")

outs = []
for i in range(3):
    p = Process(target=work, args=())
    p.start()
    outs.append(p)

for i in outs:
    i.join()
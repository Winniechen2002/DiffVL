import pickle
import numpy as np
import tabulate


category=dict(
    deformation=[7, 38, 11, 24],
    move=[10, 22, 5, 13],
    wind=[2, 36, 57, 50],
    folding=[18, 65, 62, 28],
    cut=[39, 37, 70],
)

def group(x):
    mean = np.mean([i['mean'] for i in x if not np.isnan(i['mean'])])
    count = len(x)

    rates = []
    for i in x:
        assert '/' in i['n']
        if int(i['n'].split('/')[-1]) == 0:
            continue
        rate = eval(i['n'])
        rates.append(rate)

    success = np.mean(rates)
    #return f"{success*100:.3f}% ({mean:.2f})" # ({count})"
    return {
        "success": success,
        "mean": mean,
    }
    # return {'mean': mean, 'n': count}


def get_table():
    with open('score.pkl', 'rb') as f:
        tables = pickle.load(f)


    tables['lang']['task2_wind'] = {'mean': 0.445, 'std': 0.009, 'n': '3/3'}
    tables['lang']['task7_press'] = {'mean': 0.649, 'std': 0.045, 'n': '3/3'}
    

    env_types = {}
    for k, v in category.items():
        for i in v:
            env_types[i] = k

    count = {k: {cats: [] for cats in category} for k in tables}
    task_set = set()

    for method in tables:
        for k, v in tables[method].items():
            task = int(k.split('_')[0].replace('task', ''))
            if task in env_types:
                count[method][env_types[task]].append(v)

    keys = list(filter(lambda x: x!='oracle', tables.keys()))
    keys = ['sac', 'ppo', 'cpdeform', 'visiononly', 'badinit', 'emdonly', 'lang']


    #print(count['oracle']['deformation'])
    count2 = [[' '] + keys]

    avg_success = {}
    avg_mean = {}

    for env_type in category:
        count2.append([env_type])

        vals = []
        best = 0.
        for k2 in keys:
            avg_success[k2] = avg_success.get(k2, [])
            avg_mean[k2] = avg_mean.get(k2, [])
            #count2[k][k2] = group(count[k][k2])
            p = group(count[k2][env_type])

            avg_success[k2].append(p['success'])
            avg_mean[k2].append(p['mean'])

            success = p['success']
            vals.append(p)
            if success > best:
                best = success
        for k2, v in zip(keys, vals):
            text = f"{v['success']*100:.3f}% ({v['mean']:.3f})"
            if v['success'] >= best:
                text = f"\\textbf" + "{" + text + "}"
            count2[-1].append(text)

    # get the mean of each row
    tot = ['tot']
    for idx, k in enumerate(keys):
        tot.append(f"{np.mean(avg_success[k])*100:.3f}% ({np.mean(avg_mean[k]):.3f})")
    count2.append(tot)

    # transpose the table
    count2 = list(map(list, zip(*count2)))


    # print table of count2
    # headers = ['RandTool', 'OnlyToolLang', 'Ours', 'SAC']
    headers = list(category.keys()) + ['total']
    # headers = ()
    print(tabulate.tabulate(count2[1:], headers=headers, tablefmt='latex_raw').replace('%', '\\%'))
    



if __name__ == '__main__':
    get_table()
# scene manager
import os
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from .auth import login_required
from .db import get_db, init_db

def fetch_posts(id):
    db = get_db()
    posts = db.execute(
        'SELECT p.id, title, body, created, author_id, image_path, username'
        f' FROM post p JOIN user u ON p.author_id = {id}'
        ' ORDER BY created DESC'
    ).fetchall()
    pp = []
    for i in posts:
        if i['title'] == 'goal':
            scene = get_subgoal(id, i['id'])
            i = dict(i)
            i['image_path'] = scene['image_path']
        pp.append(i)
    return pp

bp = Blueprint('scene', __name__, url_prefix='/')


SCENES = {}
TASKS = dict()
def get_scene(id):
    import os, pickle
    global SCENES
    if id in SCENES:
        return SCENES[id]

    from frontend import DATA_PATH
    path = os.path.join(DATA_PATH, 'scene_{}.pkl'.format(id))
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                print('load scene from', path)
                scene = pickle.load(f)
                scene.state.qpos = scene.state.get_empty_state_qpos()
        

            out = {}
            out['id'] = id
            out['scene_config'] = scene

            import cv2
            img_path = 'scene_{}.png'.format(id)
            from frontend import FLASK_PATH
            cv2.imwrite(os.path.join(FLASK_PATH, 'static', img_path), scene.views['screen'][:, :, [2, 1, 0]])
            out['image_path'] = img_path

            SCENES[id] = out
            return out


        except Exception as e:
            flash(str(e) + f" when loading scene {id}")


def get_all_scenes():
    # TODO: write scene images and add image path to the scene

    import os
    from frontend import DATA_PATH
    files = os.listdir(DATA_PATH)
    ids = []
    for i in files:
        if i.startswith('scene_') and i.endswith('.pkl') and 'subgoal' not in i:
            id = i.split('_')[1].split('.')[0]
            get_scene(id) # update if not exists 
            ids.append(id)
    #k = sorted(SCENES.keys())
    k = sorted(ids)
    return [SCENES[i] for i in k]

def get_all_tasks():
    global TASKS
    TASKS = dict()
    import os
    from frontend import DATA_PATH
    for item in os.listdir(DATA_PATH):
        if '.' in item: continue
        with open(os.path.join(DATA_PATH, item, 'name.txt'), 'r') as fi:
            # print(fi.readlines())
            nam = fi.readlines()
            if len(nam) == 0:
                nam = ''
            else:
                nam = nam[0]
            TASKS[int(item[5:])] = nam
    return TASKS

@bp.route('/', methods=('POST','GET'))
def start():
    TASKS = get_all_tasks()
    if request.method == 'POST':
        if 'refresh' in request.form:
            global SCENES
            SCENES = {}
        elif 'done' in request.form:
            N_TASKS = len(TASKS.keys()) + 1
            print(N_TASKS)
            title = request.form['title']
            import os
            from frontend import DATA_PATH, FLASK_PATH
            pkl_path = os.path.join(DATA_PATH, 'task_{}'.format(N_TASKS))
            img_path = os.path.join(FLASK_PATH, 'static', 'task_{}'.format(N_TASKS))
            dtb_path = os.path.join(DATA_PATH, 'task_{}'.format(N_TASKS), 'anno.db')
            os.mkdir(pkl_path)
            os.mkdir(img_path)
            for fil in os.listdir(DATA_PATH):
                if fil.endswith('.pkl'):
                    os.system('mv {} {}/'.format(os.path.join(DATA_PATH, fil), pkl_path))
            for fil in os.listdir(os.path.join(FLASK_PATH, 'static')):
                if fil.endswith('.png'):
                    os.system('mv {} {}/'.format(os.path.join(FLASK_PATH, 'static', fil), img_path))
            db = get_db()
            import sqlite3
            bkp = sqlite3.connect(dtb_path)
            db.backup(bkp)
            with open(os.path.join(pkl_path, 'name.txt'), 'w') as fo:
                fo.write(title)
            TASKS[N_TASKS] = title
            init_db()
            db = get_db()

            sc = bkp.execute('SELECT * FROM user')
            for row in sc.fetchall():
                db.execute(
                    "INSERT INTO user (username, password) VALUES (?, ?)",
                    (row[1], row[2]),
                )

        else:    
            # print(request.form, TASKS, TASKS.keys())
            cur = None
            for ky in TASKS.keys():
                if str(ky) in request.form:
                    cur = str(ky)
                    break
            if cur is not None:
                # return render_template('scene/start_sim.html', scenes=get_all_scenes(), tasks=get_all_tasks())
                import os
                from frontend import DATA_PATH, FLASK_PATH
                for file in os.listdir(DATA_PATH):
                    if file.endswith('.pkl'):
                        os.system('rm '+os.path.join(DATA_PATH, file))
                for file in os.listdir(os.path.join(FLASK_PATH, 'static')):
                    if file.endswith('.png'):
                        os.system('rm '+os.path.join(FLASK_PATH, 'static', file))
                os.system('cp '+os.path.join(DATA_PATH, 'task_'+cur, '*')+' '+DATA_PATH)
                os.system('cp '+os.path.join(FLASK_PATH, 'static', 'task_'+cur, '*')+' '+os.path.join(FLASK_PATH, 'static'))
            import sqlite3
            db = get_db()
            # print(os.path.join(DATA_PATH, 'task_'+cur, 'anno.db'))
            bkp = sqlite3.connect(os.path.join(DATA_PATH, 'task_'+cur, 'anno.db'))
            print(bkp)
            bkp.backup(db)
            
    return render_template('scene/start_sim.html', scenes=get_all_scenes(), tasks=get_all_tasks())

    
@bp.route('/create_sim', methods=('POST',))
def manage_sim():
    if request.method == 'POST':
        from frontend.backend import get_gui
        if 'start' in request.form:
            print('created sim')
            backend = get_gui()
        elif 'quit' in request.form:
            print('quit')
            backend = get_gui()
            backend.quit()
        else:
            raise NotImplementedError('Unknown request value {}'.format(request.values))

        return redirect(url_for("scene.start"))
    
    return render_template('scene/start_sim.html', scenes=get_all_scenes(), tasks=get_all_tasks)


@bp.route('/<string:id>/delete_scene', methods=('GET',))
def delete_scene(id):
    import os
    from frontend import DATA_PATH
    path = os.path.join(DATA_PATH, 'scene_{}.pkl'.format(id))
    os.system("rm {}".format(path))
    if id in SCENES:
        del SCENES[id]

    # TODO: delete tokens from the database ..
    return redirect(url_for('scene.start'))


def update_scene_config(id, scene, path=None):
    # update scene to path..
    import pickle
    assert path is None

    from frontend import DATA_PATH
    path = os.path.join(DATA_PATH, 'scene_{}.pkl'.format(id))
    with open(path, 'wb') as f:
        pickle.dump(scene, f)

@bp.route('/<string:id>/add_names')
def add_names(id):
    from .tagger import configs
    scene = get_scene(id)
    scene_config = scene['scene_config']

    for a in configs['LABELS']:
        if a['name'] not in scene_config.names:
            scene_config.names[a['name']] = {'bbox': a}
            # extract masks??
    
    update_scene_config(id, scene_config)
    return redirect(url_for('scene.view_scene', id=id))

    
@bp.route('/<string:id>/delete_name')
def delete_name(id):
    name = request.args.get("name")
    scene_config = get_scene(id)['scene_config']
    if name is not None:
        del scene_config.names[name]
        update_scene_config(id, scene_config)
    return redirect(url_for('scene.view_scene', id=id))


def edit_scene(scene):
    from frontend.backend import get_gui
    #from frontend.backend import backend
    id = scene['id']
    backend = get_gui(create=False)
    if backend is not None:
        while not backend.output_queue.empty():
            p = backend.output_queue.get()
            assert p[0] == 'refreshed'
            assert p[1] == id, f"{p[1]} != {id}"
            del SCENES[id]
            print('update...')
            assert id not in SCENES
            scene = get_scene(id) # update scene

    if 'edit' in request.form:
        backend = get_gui()
        backend.upload(scene['scene_config'], id)

    return None

@bp.route('/<string:id>/view_scene', methods=('GET', 'POST'))
def view_scene(id):
    # pass
    scene = get_scene(id)

    if scene is None:
        abort(404, f"Post id {id} doesn't exist.")
    else:
        if 'select_name' in request.form:
            print(request.form['select_name'])

        ret = edit_scene(scene)

        if 'tag' in request.form:
            from .tagger import init_config_by_scene
            init_config_by_scene(scene)
            return redirect(url_for('label.tagger'))

        if ret is not None:
            return ret

        #if 'names' not in scene:
        seg = scene['scene_config'].views['seg']
        scene['names'] = []
        img = scene['scene_config'].views['screen'][..., [2, 1, 0]]
        for idx, (k, v) in enumerate(scene['scene_config'].names.items()):
            scene['names'].append(k)
            print(idx, k, v.keys())
            if 'id' in v:
                #print(k, v)
                import cv2
                from frontend import FLASK_PATH
                mask = (seg & (1<<idx)) > 0.
                import numpy as np
                mask = (mask * 255).astype(np.uint8)

                xx = img.copy()
                xx[mask > 0] = [255, 255, 255]
                cv2.imwrite(os.path.join(FLASK_PATH, 'static', f'scene_mask_{k}.png'), xx)
            else:
                import cv2
                print(v['bbox'])
                xx = img.copy()

                def tt(x, y):
                    return int(float(x)), int(float(y))
                cv2.rectangle(xx, tt(v['bbox']['xMin'], v['bbox']['yMin']), tt(v['bbox']['xMax'], v['bbox']['yMax']), (0, 0, 255), 2)

                cv2.imwrite(os.path.join(FLASK_PATH, 'static', f'scene_mask_{k}.png'), xx)


        return render_template('scene/view_scene.html', posts=fetch_posts(id), scene=scene)

def get_subgoal(scene_id, stage):
    stage_id_name = f'subgoal_{scene_id}_{stage}'
    scene = get_scene(stage_id_name)
    if scene is None:
        import copy
        scene = copy.deepcopy(get_scene(scene_id))
    scene['id'] = stage_id_name
    return scene

@bp.route('/<string:stage>/add_subgoal', methods=('GET', 'POST'))
def edit_subgoal(stage):
    scene_id = request.args.get("id")
    if scene_id.startswith('subgoal'): 
        _, scene_id, stage = scene_id.split('_')
    scene = get_subgoal(scene_id, stage)
    print(scene['id'])
    ret = edit_scene(scene)
    #return render_template('scene/create_tokens.html', scene=scene, scene_id=scene_id)
    return redirect(url_for('scene.create_token', id=scene_id))


@bp.route('/create_token', methods=('GET', 'POST'))
@login_required
def create_token():
    id = request.args.get("id")
    assert id is not None
    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        body2 = request.form['body2']
        print(request.data)
        print(title, body, body2)
        if 'image_path' in request.form:
            image_path = request.form['image_path']
        else:
            image_path = None
        error = None

        if not title:
            error = 'Title is required.'

        if error is not None:
            flash(error)
        else:
            db = get_db()
            out = db.execute(
                'INSERT INTO post (title, body, body2, author_id, image_path)'
                ' VALUES (?, ?, ?, ?, ?)',
                (title, body, body2, id, image_path)
            )
            db.commit()
            pid = out.lastrowid

            scene_xx = get_subgoal(id, 'new')
            edit_scene(scene_xx)
            print('#' * 100, '\n', title)

            if title == 'goal':
                scene_xx['id'] = f'subgoal_{id}_{pid}'

                print('save...', scene_xx['id'])
                update_scene_config(scene_xx['id'], scene_xx['scene_config'])
            return redirect(url_for('scene.view_scene', id=id))

    scene = get_scene(id)
    return render_template('scene/create_tokens.html', scene_id=id, scene=scene)

    

def get_post(id):
    post = get_db().execute(
        'SELECT p.id, title, body, body2, created, author_id, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' WHERE p.id = ?',
        (id,)
    ).fetchone()

    if post is None:
        abort(404, f"Post id {id} doesn't exist.")

    return post

    
@bp.route('/<int:id>/update_token', methods=('GET', 'POST'))
def update_token(id):
    post = get_post(id)

    if request.method == 'POST':
        if 'title' in request.form:
            title = request.form['title']
        else:
            title = post['title']
        body = request.form['body']
        body2 = request.form['body2']
        error = None

        if not title:
            error = 'Title is required.'

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                'UPDATE post SET title = ?, body = ?, body2 = ?'
                ' WHERE id = ?',
                (title, body, body2, id)
            )
            db.commit()
            return redirect(url_for('scene.view_scene', id=post['author_id']))

    scene_id = post['author_id']
    return render_template('scene/update_tokens.html', post=post, posts=fetch_posts(scene_id))

    
@bp.route('/<int:id>/delete_token', methods=('POST',))
def delete_token(id):
    post = get_post(id)
    scene_id = post['author_id']
    db = get_db()
    db.execute('DELETE FROM post WHERE id = ?', (id,))
    db.commit()
    return redirect(url_for('scene.view_scene', id=scene_id))
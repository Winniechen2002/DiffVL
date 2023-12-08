import sys
from os import walk
import imghdr
import csv
import argparse

from flask import Flask, redirect, url_for, request
from flask import render_template
from flask import send_file

# scene manager
import os
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from .auth import login_required
from .db import get_db
from .blog import fetch_posts

bp = Blueprint('label', __name__, url_prefix='/label')



# bp = Flask(__name__)
# bp.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
configs = {
    'image': 'move.png',
    'LABELS': [],
    'HEADER': 'head',
    'scene_id': 0,
    'EXIT_NAME': [],
}


def init_config_by_scene(scene):
    global configs
    configs['LABELS'] = []
    configs['image'] = scene['image_path']
    configs['scene_id'] = scene['id']

    for k, v in scene['scene_config'].names.items():
        if v.get('bbox', None) is not None:
            configs['LABELS'].append(v['bbox'])

    configs['EXIT_NAME'] = list(scene['scene_config'].names)
    configs['HEADER'] = 'Scene {}'.format(scene['id']) + " ".join(configs['EXIT_NAME'])
    print(configs['EXIT_NAME'])

@bp.route('/tagger')
def tagger():
    #directory = bp.config['IMAGES']
    directory = '.'
    #image = bp.config["FILES"][bp.config["HEAD"]]
    return render_template(
        'scene/tagger.html',
                           directory=directory,
                           image=configs['image'],
                           labels=configs['LABELS'],
                           head=configs['HEADER'],
                           )

@bp.route('/next')
def next():
    # image = bp.config["FILES"][bp.config["HEAD"]]
    # bp.config["HEAD"] = bp.config["HEAD"] + 1
    # with open(bp.config["OUT"],'a') as f:
    #     for label in bp.config["LABELS"]:
    #         f.write(image + "," +
    #         label["id"] + "," +
    #         label["name"] + "," +
    #         str(round(float(label["xMin"]))) + "," +
    #         str(round(float(label["xMax"]))) + "," +
    #         str(round(float(label["yMin"]))) + "," +
    #         str(round(float(label["yMax"]))) + "\n")
    # bp.config["LABELS"] = []
    return redirect(url_for('scene.add_names', id=configs['scene_id']))

@bp.route('/add/<id>')
def add(id):
    xMin = request.args.get("xMin")
    xMax = request.args.get("xMax")
    yMin = request.args.get("yMin")
    yMax = request.args.get("yMax")
    configs["LABELS"].append({"id":id, "name":"", "xMin":xMin, "xMax":xMax, "yMin":yMin, "yMax":yMax})
    return redirect(url_for('label.tagger'))

@bp.route('/remove/<id>')
def remove(id):
    index = int(id) - 1
    del configs["LABELS"][index]
    for label in configs["LABELS"][index:]:
        label["id"] = str(int(label["id"]) - 1)
    return redirect(url_for('label.tagger'))

@bp.route('/label/<id>')
def label(id):
    name = request.args.get("name")
    if name in configs['EXIT_NAME']:
        abort(403, f'name {name} has exists.. ')
    else:
        configs["LABELS"][int(id) - 1]["name"] = name
        return redirect(url_for('label.tagger'))

# @bp.route('/prev')
# def prev():
#     bp.config["HEAD"] = bp.config["HEAD"] - 1
#     return redirect(url_for('tagger'))

@bp.route('/image/<f>')
def images(f):
    images = bp.config['IMAGES']
    return send_file(images + f)

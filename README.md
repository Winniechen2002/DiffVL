# DiffVL

12/08/2023 Release the code of UI-system and DiffVL algorithm.

## Prerequsite

### Docker

Install Nvidia-docker. Then:
```bash
cd DiffVL/docker
sudo bash build.sh
sudo bash run.sh
sudo bash start.sh
```

### Conda (Linux only)

```bash
conda create -n anno python
conda activate anno
cd TaskAnnotator-cf
pip install -e .
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Test the environment:

For UI system:
```bash
python ./ui/tester/test_viewer.py 0
flask --app ./frontend/flaskr init-db
```

For DiffVL algorithm:
```
python diffsolver/main.py
```

### Download Dataset:

Download the [Dataset](https://drive.google.com/file/d/1DReTiVe8uoqts80qTXU3ERfPqhpa3yVi/view?usp=sharing) on the google drive.
Put the dataset in the folder: `DiffVL/diffsolver/assets/`

Make sure the path of every single data follows the format: `DiffVL/Diffsolver/assets/Task/Task/data/task_x`

## Run:

### UI-system:

```bash
flask --app ./frontend/flaskr init-db
flask --app ./frontend/flaskr --debug run --host 0.0.0.0
```
Then open http://127.0.0.1:5000 in your browser to see the annotator.

### DiffVL solver:

Single stage Tasks
```
python diffsolver/run_single_stage.py lang --config diffsolver/examples/singlestage_dev/task2_wind.yaml 
```
You could choose the config YAML file in the `examples/` folder.

Multistage Tasks
```
python diffsolver/run_multistage.py lang --config diffsolver/examples/multistage/task1/total.yml
```
You could choose the config YAML file in the `examples/` folder.

## Cite:

If you find this codebase useful in your research, please consider citing:

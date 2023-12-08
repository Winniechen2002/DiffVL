import os
# rsync -av -e ssh --exclude='*.wandb' /path/to/source/

os.system("rsync -av -e ssh --exclude='*.wandb' bayes:TaskAnnotator/diffsolver/single_stage/sac/* ./examples/output/single_stage/sac/")
os.system("rsync -av -e ssh --exclude='*.wandb' descartes:TaskAnnotator/diffsolver/single_stage/sac/* ./examples/output/single_stage/sac/")
os.system("rsync -av -e ssh --exclude='*.wandb' bayes:TaskAnnotator/diffsolver/single_stage/ppo/* ./examples/output/single_stage/ppo/")
os.system("rsync -av -e ssh --exclude='*.wandb' descartes:TaskAnnotator/diffsolver/single_stage/ppo/* ./examples/output/single_stage/ppo/")
#os.system("scp -r descartes:TaskAnnotator/diffsolver/single_stage/sac/* ./examples/output/single_stage/sac/")
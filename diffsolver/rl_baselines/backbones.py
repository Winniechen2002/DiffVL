from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)

from .pointnet import PointNet

BACKBONES = {
    #'PointNet': PointNet,
    'CNN': NatureCNN,
    'PointNet': PointNet,
}
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass

def test():
    @dataclass
    class U:
        name: str|int = 1




    x = OmegaConf.structured(U)
    x.merge_with(OmegaConf.structured(U(name='x')))
    print(x)
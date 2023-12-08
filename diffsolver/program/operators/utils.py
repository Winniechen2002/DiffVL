import torch

__chamfer = None
def chamfer(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    global __chamfer
    #from chamferdist import ChamferDistance
    #__chamfer = __chamfer or ChamferDistance()

    #return __chamfer(x[None,:].clone(), y[None,:].clone())/len(x)
    from pytorch3d.loss import chamfer_distance
    __chamfer = __chamfer or chamfer_distance

    return __chamfer(x[None,:].clone(), y[None,:].clone())[0]/len(x)

_emd_fn = None
def emd(a, b):
    global _emd_fn
    from geomloss import SamplesLoss
    _emd_fn = _emd_fn or SamplesLoss(loss='sinkhorn', p=2, blur=0.01)
    p=1; blur=0.01; #, *args, **kwargs
    _emd_fn.p = p
    _emd_fn.blur = blur
    return _emd_fn(a.clone(), b.clone())

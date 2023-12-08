import torch

def test_friction():
    friction = 5.
    grid_v_t = torch.nn.Parameter(torch.tensor([ 2.181014, -29.669395, -5.723454], dtype=torch.float64), requires_grad=True)

    normal_component = torch.nn.Parameter(torch.tensor([  -4.555081], dtype=torch.float64), requires_grad=True)

    
    grid_v_t_norm = torch.sqrt(grid_v_t.dot(grid_v_t) + 1e-30)
    # print(grid_v)
    out = grid_v_t * (1. / grid_v_t_norm) * torch.clamp(grid_v_t_norm + normal_component[0] * friction, 0.)
    print(out)

    grad_out = torch.tensor([ -0.014337, -0.052902, -0.200059])

    (grad_out *out).sum().backward()
    print(grid_v_t.grad)
    print(normal_component.grad)

#test_friction()
def test_normalized():
    inp = torch.nn.Parameter(torch.tensor([1, 2, 3], dtype=torch.float64), requires_grad=True)
    grad_out = torch.tensor([4., 5., 6.])

    normed = inp/torch.linalg.norm(inp)
    (normed *grad_out).sum().backward()
    print(inp.grad)

#test_normalized()

def test_cross():
    inp1 = torch.nn.Parameter(torch.tensor([1, 2, 3], dtype=torch.float64), requires_grad=True)
    inp2 = torch.nn.Parameter(torch.tensor([4, 5, 6], dtype=torch.float64), requires_grad=True)

    out = torch.cross(inp1, inp2)

    k = torch.zeros(3).double()
    k[0] = 34
    k[1] = 50
    k[2] = -10
    inp1.grad = None
    inp2.grad = None
    (out * k).sum().backward()
    print(inp1.grad)
    print(inp2.grad)

    print(torch.cross(inp2, k))
    print(torch.cross(k, inp1))


def qmul(q, v):
    qvec = q[1:]
    uv = qvec.cross(v)
    uuv = qvec.cross(uv)
    out = v + 2 * (q[0] * uv + uuv)
    return out

def qinv(q):
    x = q.clone()
    x[1:] = -x[1:]
    return x

def test_qmul():
    # 12.303394 29.159481 -94.458145 56.899994
    # 0.785979 2.760870 -8.292155
    q = torch.nn.Parameter(torch.tensor([-0.347315, -0.068429, 0.926209, -0.129719], dtype=torch.float64), requires_grad=True)
    v = torch.nn.Parameter(torch.tensor([1., 2., 3.]).double(), requires_grad=True)

    (qmul(q, v) * torch.tensor([4., 5., 6.])).sum().backward()
    print(q.grad)
    print(v.grad)

def test_inv_spatial():
    p = torch.nn.Parameter(torch.tensor([ 0.207442, 0.276186, 0.776365 ], dtype=torch.float64), requires_grad=True)
    q = torch.nn.Parameter(torch.tensor([ -0.598229, -0.612276, 0.417175, 0.305294], dtype=torch.float64), requires_grad=True)
    v = torch.nn.Parameter(torch.tensor([0.546875, 0.421875, 0.703125]).double(), requires_grad=True)

    out = qmul(qinv(q), v-p)
    (out * torch.tensor([-678867.437500, -1452.505859, -112671.820312])).sum().backward()
    print(q.grad)
    print(p.grad)
    
#test_qmul()


#test_inv_spatial()
#test_friction()

def test_full_model():
    friction = 5.
    dist = torch.nn.Parameter(torch.tensor([ 0.002958], dtype=torch.float64), requires_grad=True)
    v_out = torch.nn.Parameter(torch.tensor([1.587825, -30.541767, -1.401250], dtype=torch.float64), requires_grad=True)
    bv = torch.nn.Parameter(torch.tensor([ -0.225119, 0.013681, -0.130683], dtype=torch.float64), requires_grad=True)
    bq = torch.nn.Parameter(torch.tensor([-0.344138, 0.649593, -0.194354, -0.649480], dtype=torch.float64), requires_grad=True)
    old_normal = torch.nn.Parameter(torch.tensor([1., 0., 0.], dtype=torch.float64), requires_grad=True)


    influence = torch.exp(-dist[0] * 666.)
    rel_v = v_out - bv
    normal = qmul(bq, old_normal)
    normal.register_hook(lambda x: print('normal grad', x))
    normal_component = torch.dot(rel_v, normal)

    assert normal_component < 0
    grid_v_t = rel_v - normal * normal_component
    grid_v_t.register_hook(lambda x:print('grid_v_t grad', grid_v_t))

    grid_v_t_norm = torch.linalg.norm(grid_v_t)

    grid_v_t = grid_v_t * (1.0 / grid_v_t_norm) * torch.relu(grid_v_t_norm + normal_component * friction) # max(0, x)

    out = bv + rel_v * (1-influence) + grid_v_t * influence
    print(out, influence)
    print('rel v', rel_v)
    print('grid v t', grid_v_t)

    (out * torch.tensor([-0.102845, -0.379476, -1.435053])).sum().backward()

    print(dist.grad)
    print('bv_grad', bv.grad)
    print('bq grad', bq.grad)
    print('old_normal grad', old_normal.grad)

# test_full_model()
def test_dw():
    fx = torch.nn.Parameter(torch.tensor([1.020372, 1.170355, 1.346863], dtype=torch.float64), requires_grad=True)
    pow2 = lambda x: x*x
    w = torch.stack((0.5 * pow2(1.5 - fx), 0.75 - pow2(fx - 1.), 0.5 * pow2(fx - 0.5)), axis=1)
    i=2
    j=2
    k=2
    weight = w[0][i] * w[1][j] * w[2][k]
    print(weight)
    weight.sum().backward()
    print(fx.grad*64)

test_dw()
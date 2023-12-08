import torch

sigma = torch.nn.Parameter(torch.tensor([10.736234, 0.132587, 0.003376], dtype=torch.float64), requires_grad=True)

s_new = torch.clamp(sigma, 0.05, torch.inf)
epsilon = torch.log(s_new)
epsilon_hat = epsilon - (epsilon[0] + epsilon[1] + epsilon[2]) / 3.


epsilon_hat_norm = torch.sqrt(epsilon_hat.dot(epsilon_hat))
delta_gamma = epsilon_hat_norm - 10000. / (2 * 2083.3333333333333333)

exp_eps = torch.exp(epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat)
print(exp_eps)

U = torch.tensor([
0.469347,0.026447,0.882617,
-0.143307,-0.984019,0.105691,
0.871307,-0.176091,-0.458057], dtype=torch.float64).reshape(3, 3)

V = torch.tensor([
0.469367,0.026658,0.882601,
-0.143079,-0.984039,0.105811,
0.871334,-0.175946,-0.458061,], dtype=torch.float64).reshape(3, 3)

grad_J = -0.000235

grad_r = torch.tensor([-0.750059,-0.642795,-0.933756,
0.804710,0.421160,-0.977095,
0.198846,-0.170647,0.903322], dtype=torch.float64).reshape(3, 3)

mat = torch.zeros(3, 3, dtype=torch.float64)
mat[[0,1,2], [0,1,2]] = exp_eps

F = U@mat@V.T 
j = exp_eps[0] * exp_eps[1] * exp_eps[2]

(j*grad_J + (F*grad_r)).sum().backward()
print(sigma.grad.data.detach().cpu().numpy())
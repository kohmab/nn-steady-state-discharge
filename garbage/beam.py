import numpy as np
import scipy.interpolate
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt

from pyDOE import lhs
from torch import nn
from tqdm import tqdm

from helpers import beam_field
from networks import MLP, FourierFeatTransform, AnalyticNet

print(torch.cuda.is_available())
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = "cpu"

torch.manual_seed(1234)

lb = np.asarray([-4., 0.])  # z,r
ub = np.asarray([8., 4.])


def uniform(N, l, r, dim=1):
    return l + (r - l) * lhs(dim, N)


# %%

# %%


Npoints = 1024 * 192
Nbcpoints = 1024 * 16
Nicpoints = 1024 * 16


class Data(torch.utils.data.Dataset):
    def __init__(self, Npoints, Nbcpoints, Nicpoints, lb, ub):
        self.Npoints = Npoints
        self.Nbcpoints = min(Nbcpoints, Npoints)
        self.Nicpoints = min(Nicpoints, Npoints)

        self.x = uniform(Npoints, lb, ub, dim=2)

        ones = np.ones(Nbcpoints)[:, None]
        tmp = uniform(self.Nbcpoints, lb[0], ub[0])
        x_bottom = np.hstack((tmp, ones * lb[1]))

        tmp = uniform(self.Nbcpoints, lb[0], ub[0])
        x_top = np.hstack((tmp, ones * ub[1]))
        y_top = beam_field(x_top)

        ones = np.ones(Nicpoints)[:, None]
        tmp = uniform(self.Nicpoints, lb[1], ub[1])
        x_ic = np.hstack((ones * lb[0], tmp))
        y_ic = beam_field(x_ic)

        self.x_bc_bottom = x_bottom

        self.x_bc_top = x_top
        self.y_bc_top = y_top

        self.x_ic = x_ic
        self.y_ic = y_ic

    def __len__(self):
        return self.Npoints

    def __getitem__(self, idx):
        idx_bc = idx % self.Nbcpoints
        idx_ic = idx % self.Nicpoints
        return (self.x[idx],
                self.x_bc_bottom[idx_bc],
                self.x_bc_top[idx_bc], self.y_bc_top[idx_bc],
                self.x_ic[idx_ic], self.y_ic[idx_ic])


# %%


def grad(y, x):
    return torch.autograd.grad(y, x, to_gpu(torch.ones_like(y), False), create_graph=True)[0]


def to_gpu(x, requires_grad=True):
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    return x.to(device=device).float().requires_grad_(requires_grad)


# %%
dataset = Data(Npoints, Nbcpoints, Nicpoints, lb, ub)
test_dataset = Data(10026, 1024, 1024, lb, ub)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# %%

feat_trans = FourierFeatTransform(lb, ub).to(device=device, dtype=torch.float32)
mlp = MLP(nhid=192, nlay=4, nin=4, nout=2, activator=torch.nn.Tanh).to(device=device, dtype=torch.float32)
model = nn.Sequential(feat_trans, mlp).to(device=device)
# %%
optimizerA = torch.optim.Adam(model.parameters(), lr=1.e-5)
# %%
optimizer = torch.optim.LBFGS(model.parameters(), lr=.001, max_iter=20000, max_eval=50000,
                              history_size=2, tolerance_grad=1e-05,
                              tolerance_change=0.5 * np.finfo(float).eps,
                              line_search_fn="strong_wolfe")


# %%
def loss_fun(model, x, xbb, xbt, ybt, xic, yic):
    mse = torch.nn.functional.mse_loss

    # equation losses
    z = to_gpu(x[:, [0]])
    r = to_gpu(x[:, [1]])
    out = model(torch.hstack((z, r)))

    Er, Ei = out[:, [0]], out[:, [1]]

    dEr_dz, dEi_dz = grad(Er, z), grad(Ei, z)

    rdEr_dr, rdEi_dr = r * grad(Er, r), r * grad(Ei, r)

    drdEr_dr, drdEi_dr = grad(rdEr_dr, r), grad(rdEi_dr, r)

    res_r, res_i = 2 * r * dEr_dz + drdEi_dr, -2 * r * dEi_dz + drdEr_dr
    zero = torch.zeros_like(res_r, device=device, dtype=torch.float32)

    loss_eq = mse(res_r, zero) + mse(res_i, zero)

    # bottom bc losses
    zbb = to_gpu(xbb[:, [0]])
    rbb = to_gpu(xbb[:, [1]])

    out_bb = model(torch.hstack((zbb, rbb)))

    Er_bb, Ei_bb = out_bb[:, [0]], out_bb[:, [1]]

    dEr_dr_bb, dEi_dr_bb = grad(Er_bb, rbb), grad(Ei_bb, rbb)

    zero = torch.zeros_like(dEr_dr_bb, device=device, dtype=torch.float32)
    loss_bc_bottom = mse(dEr_dr_bb, zero) + mse(dEi_dr_bb, zero)

    # top bc losses
    zbt = to_gpu(xbt[:, [0]])
    rbt = to_gpu(xbt[:, [1]])

    out_bt = model(torch.hstack((zbt, rbt)))
    loss_bc_top = mse(out_bt, to_gpu(ybt))

    # ic losses

    zic = to_gpu(xic[:, [0]])
    ric = to_gpu(xic[:, [1]])

    out_ic = model(torch.hstack((zic, ric)))
    loss_ic = mse(out_ic, to_gpu(yic))
    ##
    loss = loss_eq + loss_bc_bottom + loss_bc_top + loss_ic
    return loss


# %%
test = AnalyticNet()
args = next(iter(dataloader))
print(loss_fun(test, *args))


# %%
# try :
#     model.load_state_dict(torch.load("/home/baho/Projects/pidon/garbage/beam_model.pth", weights_only=True))
#     print("loaded")
# except:
#     print("not_loaded")
#     pass
#     ##model.eval()
# %%
def closure():
    optimizer.zero_grad()
    losses = 0.
    for data in dataloader:
        losses += loss_fun(model, *data)
    losses = losses / len(dataloader)
    losses.backward()
    print(losses)
    return losses


# %%
for e in range(1100256):
    print(f"Epoch {e}")
    optimizer.step(closure)
    tqdm.write(f"test_losses  = {loss_fun(model, *next(iter(test_dataloader)))}")
    torch.save(model.state_dict(), f'beam_model_l.pth')

# %%

# %%
for e in range(1100256):
    for data in tqdm(dataloader):
        optimizerA.zero_grad()
        losses = loss_fun(model, *data)
        losses.backward()
        optimizerA.step()

    tqdm.write(f"Epoch {e}")
    tqdm.write(f"train_losses = {losses}")
    tqdm.write(f"test_losses  = {loss_fun(model, *next(iter(test_dataloader)))}")
    torch.save(model.state_dict(), f'beam_model_ae-5.pth')

#%%
    torch.save(model, f'beam_model_g.pth')
#%%
model = torch.load("beam_model_g-6.pth")

# %%
z, r = np.meshgrid(np.linspace(lb[0], ub[0], 256), np.linspace(lb[1], ub[1], 256))
inp = np.hstack((z.flatten()[:, None], r.flatten()[:, None]))
out = model(to_gpu(inp, False)).cpu().detach().numpy()

Er_net = scipy.interpolate.griddata(inp, out[:, 0], (z, r), method='linear')
Ei_net = scipy.interpolate.griddata(inp, out[:, 1], (z, r), method='linear')

exact = beam_field(inp)

Er_ex, Ei_ex = exact[:, [0]].reshape(z.shape), exact[:, [1]].reshape(z.shape)

# %%
mpl.use('gtk4agg')

fig = plt.figure()
ax = fig.add_subplot(221)
cf = plt.contourf(z, r, Er_net)
fig.colorbar(cf)
ax = fig.add_subplot(222)
cf = plt.contourf(z, r, Ei_net)
fig.colorbar(cf)
ax = fig.add_subplot(223)
cf = plt.contourf(z, r, Er_ex)
fig.colorbar(cf)
ax = fig.add_subplot(224)
cf = plt.contourf(z, r, Ei_ex)
fig.colorbar(cf)
plt.show()
#%%
fig = plt.figure()
ax = fig.add_subplot(121)
cf = plt.contourf(z, r, Er_net-Er_ex)
fig.colorbar(cf)
ax = fig.add_subplot(122)
cf = plt.contourf(z, r, Ei_net-Ei_ex)
fig.colorbar(cf)
plt.show()

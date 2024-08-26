import numpy as np
import scipy.interpolate
import torch
from matplotlib import pyplot as plt
from pyDOE import lhs
from tqdm import tqdm

from networks import MLP

# %%

Npoints = 400000
torch.manual_seed(777)

lb = np.asarray([0., 0.])
ub = np.asarray([1., 1.])


def uniform(N, l, r, dim=1):
    return l + (r - l) * lhs(dim, N)


class Data(torch.utils.data.Dataset):
    def __init__(self, Npoints, lb, ub):
        self.Npoints = Npoints
        self.x_data = uniform(Npoints, lb, ub, dim=2)
        self.y_data = np.zeros([Npoints, 1])

        nbc = Npoints // 4
        ones = np.ones(nbc)[:, None]
        tmp = uniform(nbc, lb[0], ub[0])
        x_bottom = np.hstack((tmp, ones * lb[1]))
        y_bottom = np.sin(np.pi / (ub[0] - lb[0]) * tmp)
        tmp = uniform(nbc, lb[0], ub[0])
        x_top = np.hstack((tmp, ones * ub[1]))
        y_top = -np.sin(np.pi / (ub[0] - lb[0]) * tmp)
        tmp = uniform(nbc, lb[1], ub[1])
        x_left = np.hstack((ones * lb[0], tmp))
        y_left = np.sin(np.pi / (ub[1] - lb[1]) * tmp)
        tmp = uniform(nbc, lb[1], ub[1])
        x_right = np.hstack((ones * ub[0], tmp))
        y_right = -np.sin(np.pi / (ub[1] - lb[1]) * tmp)

        self.x_data_bc = np.vstack((x_bottom, x_top, x_left, x_right))
        self.y_data_bc = np.vstack((y_bottom, y_top, y_left, y_right))

    def __len__(self):
        return self.Npoints

    def __getitem__(self, idx):
        return self.x_data[idx], self.x_data_bc[idx], self.y_data[idx], self.y_data_bc[idx]


dataset = Data(Npoints=Npoints, lb=lb, ub=ub)
test_dataset = Data(Npoints=1024, lb=lb, ub=ub)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=True)

model = MLP(nhid=64, nlay=4, nout=1).to(device='cuda:0')
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)


# %%


def loss_fun(model, x, xb, y, yb):
    mse = torch.nn.functional.mse_loss

    x1 = x[:, [0]].to(device='cuda:0').float().requires_grad_()
    x2 = x[:, [1]].to(device='cuda:0').float().requires_grad_()
    phi = model(torch.hstack((x1, x2)))

    dx1_phi = torch.autograd.grad(phi, x1, torch.ones_like(phi), create_graph=True)[0]
    dx2_phi = torch.autograd.grad(phi, x2, torch.ones_like(phi), create_graph=True)[0]
    ddx1_phi = torch.autograd.grad(dx1_phi, x1, torch.ones_like(phi), create_graph=True)[0]
    ddx2_phi = torch.autograd.grad(dx2_phi, x2, torch.ones_like(phi), create_graph=True)[0]

    res = ddx1_phi + ddx2_phi

    losseq = mse(res, y.to(device='cuda').float())

    lossbc = mse(model(xb.to(device='cuda').float()), yb.to(device='cuda').float())

    return torch.mean(losseq + lossbc)


# %%
for e in range(200):
    for x, xb, y, yb in tqdm(dataloader):
        optimizer.zero_grad()
        losses = loss_fun(model, x, xb, y, yb)
        losses.backward()
        optimizer.step()

    tqdm.write(f"Epoch {e}")
    tqdm.write(f"train_losses = {losses}")
    tqdm.write(f"test_losses  = {loss_fun(model, *next(iter(test_dataloader)))}")
    torch.save(model.state_dict(), f'poison_model.pth')
# %%
x, y = np.meshgrid(np.linspace(lb[0], ub[0], 256), np.linspace(lb[1], ub[1], 256))
phi = model(torch.from_numpy(dataset.x_data).to(device='cuda:0').float()).cpu().detach().numpy()
dat = scipy.interpolate.griddata(dataset.x_data, phi, (x, y), method='linear')
dat = np.squeeze(dat)
# %%
fig = plt.figure(1)
cf = plt.contourf(x, y, dat)
fig.colorbar(cf)
plt.show()

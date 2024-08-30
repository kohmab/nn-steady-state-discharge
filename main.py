import torch
import numpy as np

# %%
from parametersholder import ParametersHolder

torch.manual_seed(1234)
conf = {}
conf["z_limits"] = np.asarray([-4., 8.])
conf["r_limits"] = np.asarray([0., 4.])

conf["max_field_limits"] = np.asarray([1., 2.])

conf["power_limits"] = np.asarray([1., 1.])
conf["coefficient_K0_limits"] = np.asarray([0.1, 10.])
conf["collision_frequency_limits"] = np.asarray([0.0, 0.0])

conf["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf["dtype"] = torch.float32

conf["Npoints"] = 1024 * 1024
conf["Npoints_for_boundary_conditions"] = 1024 * 256
conf["Npoints_for_initial_condition"] = 1024 * 256

conf["train_batch_size"] = 512
conf["test_batch_size"] = 512

conf["trunk_hidden_layers"] = 3
conf["branch_hidden_layers"] = 3
conf["trunk_hidden_features"] = 96
conf["branch_hidden_features"] = 96
conf["branch_and_trunk_out_features"] = 96

ParametersHolder(conf)
# %%
from data import *
from helpers import prepare_tensor, beam_field
from networks import Pidonet
from tqdm import tqdm

from losses import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy

# %%
model = Pidonet().to(device=ParametersHolder().device, dtype=ParametersHolder().dtype)

s = 0
for p in model.parameters():
    s += p.numel()
print(s)
# %%

optimizerA = torch.optim.Adam(model.parameters(), lr=1.e-4)
dataloaders = JointDataLoaders(ParametersHolder().train_batch_size,
                               DomainData(),
                               {"ic": IcData(), "bc_axis": BcAtAxisData(), "bc_top": BcAtUpperRLimitData()})
# %%

for e in range(100500):
    for i in tqdm(dataloaders):
        optimizerA.zero_grad()

        l_eq = eq_losses(model, *dataloaders.data["main"])
        l_bc_u = upper_r_bc_losses(model, *dataloaders.data["bc_top"])
        l_bc_a = axis_bc_losses(model, *dataloaders.data["bc_axis"])
        l_ic = ic_losses(model, *dataloaders.data["ic"])

        losses = l_eq + l_bc_u + l_bc_a + l_ic

        losses.backward()
        optimizerA.step()

    print(f"\n Epoch {e}")
    print(f"train_losses = {losses}\n")

    # TODO test losses
    torch.save(model, f'beam_pidon_model.pth')

# %%
model = torch.load("beam_pidon_model.pth", weights_only=False)

# %%
zmin = ParametersHolder().get_lb()[0]
zmax = ParametersHolder().get_ub()[0]
rmin = ParametersHolder().get_lb()[1]
rmax = ParametersHolder().get_ub()[1]

z, r = np.meshgrid(np.linspace(zmin, zmax, 256), np.linspace(rmin, rmax, 256))

zt = prepare_tensor(z.flatten()[:, None])
rt = prepare_tensor(r.flatten()[:, None])
params = torch.tensor([1, 1, 1., 0], dtype=ParametersHolder().dtype, device=ParametersHolder().device)

points = np.hstack((z.flatten()[:, None], r.flatten()[:, None]))

out = model(zt, rt, params).cpu().detach().numpy()

Er_net = scipy.interpolate.griddata(points, out[:, 0], (z, r), method='linear')
Ei_net = scipy.interpolate.griddata(points, out[:, 1], (z, r), method='linear')

exact = beam_field(points)

Er_ex, Ei_ex = exact[:, [0]].reshape(z.shape), exact[:, [1]].reshape(z.shape)

# %%
mpl.use("WebAgg")
# plt.close('all')

fig = plt.figure(1)
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

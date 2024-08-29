import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 10)
        self.parameter_i_dont_want_to_train = torch.nn.Parameter(torch.empty(2), requires_grad=False)

    def forward(self, x):
        return self.fc1(x + self.parameter_i_dont_want_to_train)


m = Model()
print(m.parameter_i_dont_want_to_train)
print(m.fc1.weight.data)
m.to(device="cuda")
target = torch.ones(10, requires_grad=False, device="cuda")
input = torch.ones(2, requires_grad=True, device="cuda")
opt = torch.optim.SGD(m.parameters(), lr=1)
print(m(input))
l = torch.nn.functional.mse_loss(m(input), target)
l.backward()
opt.step()
print(m)
print(m.parameter_i_dont_want_to_train)
print(m.fc1.weight.data)
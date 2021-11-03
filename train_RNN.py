import torch
from torch.optim import Adam
from torch import nn
import numpy as np
from tqdm import tqdm

#-----------------------------------------------------------------------------
#LOAD THE DATA
#-----------------------------------------------------------------------------

#Load the muscle activity
muscle_activity = np.load('./muscle_activity.npy')
#Load the Wsoc and the initial condition for the dynamics
with open('initial_dynamics.npy', 'rb') as f: 
    Wsoc = np.load(f)
    x0 = np.load(f)

#-----------------------------------------------------------------------------
#TENSORIZE THE MUSCLE ACTIVITY
#-----------------------------------------------------------------------------

muscle_activity_tensor = torch.tensor(muscle_activity, dtype=torch.float32).cuda()

#-----------------------------------------------------------------------------
#DEFINE THE RNN
#-----------------------------------------------------------------------------

class RNNDynamics(nn.Module):
    def __init__(self, 
                 input_dim, # number of neurons * number of conditions
                 output_dim, # z dimension?
                 r_0 = 20,
                 r_max = 100,
                 tau = 200): 

        super(RNNDynamics, self).__init__()

        # create connectivity matrix with absolute random values
        self.W = torch.tensor(Wsoc, dtype=torch.float32).cuda()

        # creating g parameters
        # g = gain parameters for the slow version
        # g0 = gain parameters for the fast version constrained to be one

        self.g = nn.Parameter(torch.rand(1,400, 
                                         dtype=torch.float32), 
                              requires_grad=True)
        # self.g_group = nn.Parameter(torch.rand(1,40, 
                                        #  dtype=torch.float32), 
                              # requires_grad=True)
        self.g0 = nn.Parameter(torch.ones(1,400, 
                                         dtype=torch.float32), 
                              requires_grad=False)
        
        self.tau = tau
        self.r_0 = r_0
        self.r_max = r_max
        self.W_out = nn.Linear(400//2, 
                               1, 
                               bias=True) # `m` and `b` weights

    def f(self, x, g):
        mask = x < 0
        R = torch.empty_like(x).cuda() #[1,N]
        R[mask] = self.r_0
        R[~mask] = self.r_max-self.r_0

        return R * torch.tanh(x*g*(1./R)) #[1,N]

    def f_dynamics(self, x, g):
        F = torch.matmul(self.W, self.f(x,g).T).T
        return (1/self.tau)*(-x + F)
    
    def muscle_output(self,x,g):
        f_rate = self.f(x,g)
        mask = self.W.sum(0) > 0
        f_rate_exc = f_rate[mask.unsqueeze(0)].unsqueeze(0)
        return self.W_out(f_rate_exc)

    def RK4(self, x,g):
        
        h = 5

        k1 = self.f_dynamics(x , g)
        k2 = self.f_dynamics(x + (h/2)*k1, g)
        k3 = self.f_dynamics(x + (h/2)*k2, g)
        k4 = self.f_dynamics(x + h*k3, g)
        
        return x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    def forward(self,x_slow, x_fast):

        # W_out already includes m^T and b
        z_current_fast = self.muscle_output(x_fast, self.g0)
        z_current_slow = self.muscle_output(x_slow, self.g)

        x_new_fast = self.RK4(x_fast, self.g0, 5)
        x_new_slow = self.RK4(x_slow, self.g, 5)

        return z_current_slow, z_current_fast, x_new_slow, x_new_fast

#-----------------------------------------------------------------------------
#PERFORM TRAINING
#-----------------------------------------------------------------------------

N = 400
y_index = -1
n_epochs = 500
T,output_dim = muscle_activity_tensor.shape #[T,1]

rnn_dynamic = RNNDynamics(input_dim=2*N, # number of neurons * 2 conditions
                          output_dim=1*2, # z dimension * 2 conditions
                          r_0=20,
                          r_max=100).cuda()

#opt = Adam(rnn_dynamic.parameters(), lr = 1e-3, weight_decay = 1 )

opt = torch.optim.Adam(
[{'params': rnn_dynamic.W_out.parameters(), 'lr':1e-3, 'weight_decay':1}] +\
[{'params': rnn_dynamic.g, 'lr': 1e-3, 'weight_decay':0}]
)

x_slow_seqs = []
z_slow_seqs = []

x_fast_seqs = []
z_fast_seqs = []

losses = []
losses_slow = []
losses_fast = []

with torch.autograd.detect_anomaly():
  for ep in tqdm(range(n_epochs)):

    x_slow_seq = []
    z_slow_seq = []

    x_fast_seq = []
    z_fast_seq = []


    # Initial activity
    x_slow = torch.tensor(x0.T, dtype=torch.float32).cuda()
    x_fast = torch.tensor(x0.T, dtype=torch.float32).cuda()
    
    for t in range(T):
        x_slow_seq.append(x_slow) #When t = 0 it stores the initial condition
        x_fast_seq.append(x_fast) #When t = 0 it stores the initial condition

        z_current_slow, z_current_fast, x_slow, x_fast = rnn_dynamic(x_slow, x_fast)

        z_slow_seq.append(z_current_slow)
        z_fast_seq.append(z_current_fast)

    x_slow_seq = torch.cat(x_slow_seq, dim=0)
    x_fast_seq = torch.cat(x_fast_seq, dim=0)

    z_slow_seq = torch.cat(z_slow_seq, dim=0)
    z_fast_seq = torch.cat(z_fast_seq, dim=0)

    loss_slow = torch.pow(torch.norm(z_slow_seq.squeeze() - muscle_activity_tensor[:,-1]), 2)
    loss_fast = torch.pow(torch.norm(z_fast_seq.squeeze() - muscle_activity_tensor[:,0]), 2)
    loss = loss_slow + loss_fast
    
    opt.zero_grad()
    loss.backward() 
    opt.step()

    losses.append(loss.item())
    losses_slow.append(loss_slow.item())
    losses_fast.append(loss_fast.item())

    x_slow_seqs.append(x_slow_seq.detach().cpu().numpy())
    x_fast_seqs.append(x_fast_seq.detach().cpu().numpy())

    z_slow_seqs.append(z_slow_seq.detach().cpu().numpy())
    z_fast_seqs.append(z_fast_seq.detach().cpu().numpy())

x_slow_seqs = np.array(x_slow_seqs)
z_slow_seqs = np.array(z_slow_seqs)

x_fast_seqs = np.array(x_fast_seqs)
z_fast_seqs = np.array(z_fast_seqs)

#-----------------------------------------------------------------------------
#EXTRACT AND STORE THE DATA
#-----------------------------------------------------------------------------

g_slow = rnn_dynamic.g.detach().cpu().numpy()
readout = rnn_dynamic.W_out.state_dict()
w = readout['weight'].detach().cpu().numpy()
b = readout['bias'].detach().cpu().numpy()

with open('trainRNN_3.npy', 'wb') as f:
        np.save(f,g_slow)
        np.save(f,w)
        np.save(f,b)
        np.save(f,losses)
        np.save(f,losses_slow)
        np.save(f,losses_fast)
        np.save(f,x_fast_seqs)
        np.save(f,x_slow_seqs)
        np.save(f,z_fast_seqs)
        np.save(f,z_slow_seqs)

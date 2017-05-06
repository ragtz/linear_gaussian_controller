from linear_gaussian_controller.src.lgc import *
import matplotlib.pyplot as plt
import numpy as np

red_grasp = np.load('../../sorttasksim/data/red_demos_grasp.npy')

g_demos = [demo['gripper'][:,:-1] for demo in red_grasp]
b_demos = [demo['block'][:,:-3] for demo in red_grasp]
t_demos = [demo['t'] for demo in red_grasp]

N = len(t_demos)
order = 5

state = []
action = []

for i in range(N):
    g = g_demos[i]
    b = b_demos[i]
    t = t_demos[i]

    for j in range(order-1,len(t)-1):
        state_j = []
        for k in range(order):
            state_j.append(np.concatenate((g[j-k], b[j-k])))
        state.append(np.concatenate(state_j))
        
        dx = (g[j+1]-g[j])
        action.append(dx)

state = np.array(state)
action = np.array(action)

lgc = LinearGaussianController()
lgc.train(state, action)    

steps = 1000
g_start = np.array([400,100])
b_start = np.array([400,500])

g_state = order*[g_start]
for i in range(steps):
    state = []
    for j in range(order):
        state.append(np.concatenate((g_state[-(j+1)], b_start)))
    state = np.concatenate(state)
    action = lgc.step(state) 
    g_state.append(g_state[-1] + action)
g_state = np.array(g_state)

plt.plot(g_state[:,0], g_state[:,1])
plt.xlim([0,1000])
plt.ylim([0,1000])

plt.figure()
plt.plot(range(len(g_state)), g_state[:,0])
plt.xlabel('T')
plt.ylabel('X')
plt.ylim([0,1000])

plt.figure()
plt.plot(range(len(g_state)), g_state[:,1])
plt.xlabel('T')
plt.ylabel('Y')
plt.ylim([0,1000])

plt.show()


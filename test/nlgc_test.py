from linear_gaussian_controller.src.lgc import *
import matplotlib.pyplot as plt
import numpy as np

class IncludeDistFeatures(object):
    def __init__(self, distIdxs, distPoints):
        self.distIdxs = distIdxs
        self.distPoints = distPoints
        
    def project(self, X):
        projX = []
        
        for x in X:
            projx = x.tolist()
            
            for d in self.distIdxs:
                i, j = d
                projx.append(self._dist(x[i], x[j]))
                
            for d in self.distPoints:
                i, p = d
                projx.append(self._dist(x[i], np.array(p)))
                
            projX.append(projx)
            
        return np.array(projX)

    def _dist(self, a, b):
        return np.linalg.norm(a-b)

red_grasp = np.load('/home/ragtz/si-machines/sandbox/sorttasksim/data/red_demos_grasp.npy')

g_demos = [demo['gripper'][:,:-1] for demo in red_grasp]
b_demos = [demo['block'][:,:-3] for demo in red_grasp]
t_demos = [demo['t'] for demo in red_grasp]

N = len(t_demos)
order = 5

#f = IncludeDistFeatures([([0,1], [2,3])], [([2,3], [200,400]), ([2,3], [475,200]), ([2,3], [750,400])])
f = IncludeDistFeatures([([0,1], [2,3])], [])

state = np.array([np.concatenate((g_demos[i][:-1], b_demos[i][:-1]), axis=1) for i in range(N)])
action = np.array([np.array([g[j+1]-g[j] for j in range(len(g)-1)]) for g in g_demos])

state = np.array([f.project(s) for s in state])

lgc = NthOrderLinearGaussianController(order)
lgc.train(state, action)    

steps = 1000
#g_start = np.array([800,200])
#b_start = np.array([300,300])
#g_start = np.array([850,800])
#b_start = np.array([300,700])
g_start = np.array([434.0, 788.0])
b_start = np.array([438, 837])

g_state = [g_start]
d_state = [f._dist(g_start, b_start)]
for i in range(steps):
    state = f.project([np.concatenate((g_state[-1], b_start))])[0]
    #state = np.concatenate((g_state[-1], b_start))
    action = lgc.step(state)
    #action = lgc.sample_step(state)
    g_state.append(g_state[-1] + action)
    d_state.append(f._dist(g_state[-1], b_start))
g_state = np.array(g_state)
d_state = np.array(d_state)

#plt.plot(g_demos[0][:,0], g_demos[0][:,1])

plt.plot(b_start[0], b_start[1], 'x')
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

plt.figure()
plt.plot(range(len(d_state)), d_state)
plt.xlabel('T')
plt.ylabel('Dist')
plt.ylim([0,1000])

plt.show()


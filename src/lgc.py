import numpy as np

class LinearGaussianController(object):
    def __init__(self):
        self.n = None
        self.m = None
        self.A = None
        self.E = None
        self.detE = None
        self.invE = None

    def train(self, states, actions):
        self.n = states[0][0].shape[0]
        self.m = actions[0][0].shape[0]

        states = np.matrix(np.concatenate(states))
        actions = np.matrix(np.concatenate(actions))

        self.A = (np.linalg.pinv(states)*actions).T
        self.E = np.cov(actions.T - self.A*states.T)
        self.detE = np.linalg.det(self.E)
        self.invE = np.linalg.inv(self.E)

    def likelihood(self, state, action):
        d = np.matrix(action).T - self.A*np.matrix(state).T
        return ((((2*np.pi)**self.n)*self.detE)*np.exp(-0.5*d.T*self.invE*d))[0,0]

    def sample(self, state):
        m = np.squeeze(np.asarray(self.A*np.matrix(state).T))
        return np.random.multivariate_normal(m, self.E)

    def step(self, state):
        m = np.squeeze(np.asarray(self.A*np.matrix(state).T))
        return m

class NthOrderLinearGaussianController(LinearGaussianController):
    def __init__(self, N):
        self.N = N
        self.prev_states = None
        self.started = False
        super(NthOrderLinearGaussianController, self).__init__()

    def _expand_states(self, states, actions):
        e_states = []

        D = len(states)
        for i in range(D):
            ss = []
            s = states[i]
            for j in range(self.N-1, len(s)):
                s_j = []
                for k in range(self.N-1,-1,-1):
                    s_j.append(s[j-k])
                ss.append(np.concatenate(s_j)) 
            e_states.append(np.array(ss))

        return np.array(e_states), np.array([actions[i][self.N-1:] for i in range(D)])

    def _start(self, state):
        if self.started:
            self.prev_states = (self.N-1)*[state]
            self.started = False

    def _step(self, f, state):
        if self.started:
            self.prev_states = (self.N-1)*[state]
            self.started = False

        self.prev_states.append(state)
        state = np.concatenate(self.prev_states) 
        step = f(state)
        self.prev_states = self.prev_states[1:]

        return step
        
    def train(self, states, actions):
        if self.N > 1:
            states, actions = self._expand_states(states, actions)
        super(NthOrderLinearGaussianController, self).train(states, actions)
        self.reset()

    def likelihood(self, state, action):
        if self.N > 1:
            state = np.concatenate(self.prev_states + [state])
        return super(NthOrderLinearGaussianController, self).likelihood(state, action)

    def sample(self, state):
        if self.N > 1:
            state = np.concatenate(self.prev_states + [state])
        return super(NthOrderLinearGaussianController, self).sample(state)

    def sample_step(self, state):
        self._start(state)
        f = super(NthOrderLinearGaussianController, self).sample

        if self.N > 1:
            return self._step(f, state)
        else:
            return f(state)

    def step(self, state):
        self._start(state)
        f = super(NthOrderLinearGaussianController, self).step

        if self.N > 1:
            return self._step(f, state)
        else:
            return f(state)
    
    def reset(self):
        self.started = True


import numpy as np
import math


class function_class():
    def __init__(self):
        pass

    def triangle(self, x, peak = 0.5, intercept = 0.9, slope = 0.95, exp = 1):
        return intercept-slope*(abs(x-peak)**exp)

    def sine(self, x, peak = 1/3):
        return 2/(3*math.pi)*math.sin(3*math.pi/2*(x-peak+1/3))


class simulation:
    def __init__(self, T, sigma=math.sqrt(0.1), peak = 0.5, step = 0.05, prob = 1/300, seed_start=0, func_type = 'triangle'):
        self.T = T
        self.step = step
        self.restart = False
        self.peaks = [peak]
        if step > 0:
            self.restart = True
            self.points = []
            for t in range(1,T):
                if np.random.sample() < prob:
                    self.points.append(t)
                    v = self.peaks[-1] + np.random.binomial(1,0.5)*step*2-step
                    if v>1-step:
                        v = 1-2*step
                    elif v < step:
                        v = 2*step
                    self.peaks.append(v)
            self.points = np.array(self.points)
            self.peaks = np.array(self.peaks)
        self.func = getattr(function_class(), func_type)
        self.seed_start = seed_start
        np.random.seed(1)
        self.sigma = sigma
        np.random.seed(seed_start)
        self.noise = np.random.normal(size=T+5)*self.sigma
    def value(self, t, x):
        if not self.restart:
            regret = self.func(self.peaks[0],self.peaks[0]) - self.func(x,self.peaks[0])
            val = self.func(x,self.peaks[0])+self.noise[t]
        else:
            ind = sum(t>=self.points)
            regret = self.func(self.peaks[ind],self.peaks[ind]) - self.func(x,self.peaks[ind])
            val = self.func(x,self.peaks[ind])+self.noise[t]
        if isinstance(regret, (list,np.ndarray)):
            regret = regret[0]
        if isinstance(val, (list,np.ndarray)):
            val = val[0]
        return regret, val

def check_1d(cen, rad, inte = [0,1]):
    intervals = [[cen[i]-rad[i], cen[i]+rad[i]] for i in range(len(cen))]
    val = max([i[-1] for i in intervals])
    intervals.sort(key = lambda x: x[0])
    if intervals[0][0] > inte[0]:
        return False, (intervals[0][0]+inte[0])/2
    if val < inte[1]:
        return False, (val+inte[1])/2
    output = intervals[0]
    for start, end in intervals[1:]:
        lastEnd = output[1]
        if start <= lastEnd:
            output[1] = max(lastEnd,end)
        else:
            return False, (lastEnd+start)/2
    return True, None

def zooming(sigma, T, simulator, points = [], inte = [0,1]):
    regret = []
    cum_regret = []
    c = 13/2*sigma**1.5
    cen = [(inte[1]-inte[0])*np.random.random_sample(1)+inte[0]]
    time = [1]
    rad = [math.sqrt(c * math.log(T))]
    result = simulator.value(1,cen[0])
    regret.append(result[0])
    cum_regret.append(result[0])
    mu = [result[1]]
    for t in range(2,T+1):
        if t in points:
            # cen = [0.5 * inte[0] + 0.5 * inte[1]]
            cen = [(inte[1]-inte[0])*np.random.random_sample(1)+inte[0]]
            time = [1]
            rad = [math.sqrt(c * math.log(T))]
            result = simulator.value(t, cen[0])
            mu = [result[1]]
        else:
            res = check_1d(cen, rad, inte)
            if res[0]:
                tilde_mu = [mu[i]+2*rad[i] for i in range(len(mu))]
                ind = np.argmax(tilde_mu)
                time[ind] += 1
                rad[ind] *= math.sqrt((time[ind] - 1) / time[ind])
                result = simulator.value(t,cen[ind])
                mu[ind] = (mu[ind]*(time[ind]-1) + result[1])/time[ind]
            else:
                cen = np.concatenate((cen, [res[1]]), axis=0)
                time = np.concatenate((time, [1]), axis=0)
                rad = np.concatenate((rad, [math.sqrt(c * math.log(T))]), axis=0)
                result = simulator.value(t, cen[-1])
                mu = np.concatenate((mu, [result[1]]), axis=0)
        regret.append(result[0])
        cum_regret.append(cum_regret[-1]+result[0])
    return regret, cum_regret



def zooming_tsr(sigma, T, simulator, H, inte = [0,1], prior = False):
    if not prior:
        points = [H * i for i in range(1,math.ceil(T / H))]
    else:
        points = H
    regret = []
    cum_regret = []
    c = 13/2*sigma**1.5
    cen = [(inte[1]-inte[0])*np.random.random_sample(1)+inte[0]]
    time = [1]
    rad = [math.sqrt(c * math.log(T))]
    sd = [math.sqrt(math.pi*sigma**2*math.log(T))]
    result = simulator.value(1,cen[0])
    regret.append(result[0])
    cum_regret.append(result[0])
    mu = [result[1]]
    for t in range(2,T+1):
        if t in points:
            cen = [(inte[1]-inte[0])*np.random.random_sample(1)+inte[0]]
            time = [1]
            rad = [math.sqrt(c * math.log(T))]
            sd = [math.sqrt(math.pi * sigma ** 2 * math.log(T))]
            result = simulator.value(t, cen[0])
            mu = [result[1]]
        else:
            res = check_1d(cen, rad, inte)
            if res[0]:
                samp = np.random.normal(size=len(mu))
                samp = np.maximum(samp, 1/math.sqrt(2*math.pi))
                tilde_mu = samp*sd + mu
                ind = np.argmax(tilde_mu)
                time[ind] += 1
                rad[ind] *= math.sqrt((time[ind] - 1) / time[ind])
                sd[ind] *= math.sqrt((time[ind] - 1) / time[ind])
                result = simulator.value(t,cen[ind])
                mu[ind] = (mu[ind]*(time[ind]-1) + result[1])/time[ind]
            else:
                cen = np.concatenate((cen, [res[1]]), axis=0)
                time = np.concatenate((time, [1]), axis=0)
                rad = np.concatenate((rad, [math.sqrt(c * math.log(T))]), axis=0)
                sd = np.concatenate((sd, [math.sqrt(math.pi * sigma ** 2 * math.log(T))]), axis=0)
                result = simulator.value(t, cen[-1])
                mu = np.concatenate((mu, [result[1]]), axis=0)
        regret.append(result[0])
        cum_regret.append(cum_regret[-1]+result[0])
    return regret, cum_regret

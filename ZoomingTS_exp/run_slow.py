from alg1_codes_slowtv import *

import os
import pandas as pd
import concurrent.futures
import warnings

warnings.filterwarnings("ignore")
import argparse

# T = 40000
# cT = 5
# points = [3012,8063,14979,23136,30598]
# H = 8459
# sigma = math.sqrt(0.1)
# peak_candidate = [1/12,1/3,1/2,2/3,11/12]
# seed_start = 0

parser = argparse.ArgumentParser(description='Zooming TS algorithm with Restarts')
parser.add_argument('-rep', '--rep', type=int, default=20, help='repeat times')
# parser.add_argument('-cT', '--cT', type=int, default=3, help='number of change points')
parser.add_argument('-T', '--T', type=int, default=90000, help='total time')
parser.add_argument('-seed', '--seed', type=int, default=0, help='seed starting point')
parser.add_argument('-sigma', '--sigma', type=float, default=math.sqrt(0.1), help='noise sub gaussian parameter')
parser.add_argument('-funtype', '--funtype', type=str, default='triangle',
                    help='lipschitz function type: triangle or sine')
parser.add_argument('-peak', '--peak', type=float, default=0.5, help='peak')
parser.add_argument('-prob', '--prob', type=float, default=1/300, help='probability of moving')
parser.add_argument('-step', '--step', type=float, default=0.05, help='stepsize of moving')
parser.add_argument('-save', '--save', type=str, default='False', help='save the data or not')
parser.add_argument('-para', '--para', type=str, default='False', help='parallel computing')
parser.add_argument('-inte', '--inte', nargs='+', default=[0, 1], help='exploration intervals')
parser.add_argument('-H', '--H', type=int, default=45000, help='epoch size for restarts')
args = parser.parse_args()

T = args.T
rep = args.rep
seed = args.seed
sigma = args.sigma
funtype = args.funtype
if funtype != 'sine' and funtype != 'triangle':
    print("function type is set to default (triangle).")
    funtype = 'triangle'
peak = args.peak
prob = args.prob
step = args.step
if args.save != 'False':
    save = True
else:
    save = False
if args.para != 'False':
    para = True
else:
    para = False
inte = args.inte
inte = [float(x) for x in inte]
H = args.H

if para:
    def func(n0):
        data = simulation(T, sigma, peak, step, prob, seed_start=n0 + seed, func_type=funtype)
        zoomreg, zoomcul = zooming(sigma, T, data, points=[], inte=inte)
        tsreg, tscul = zooming_tsr(sigma, T, data, H, inte=inte)
        return zoomreg, zoomcul, tsreg, tscul


    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [nn for nn in range(rep)]
        results = executor.map(func, secs)
        results = list(results)
    res1 = np.array([i[0] for i in list(results)])
    res1cul = np.array([i[1] for i in list(results)])
    res2 = np.array([i[2] for i in list(results)])
    res2cul = np.array([i[3] for i in list(results)])


    print(funtype)
    print('mean')
    print('reg_zooming: {0}'.format(res1cul.mean(axis=0)[-5:]))
    print('reg_zoomingts: {0}'.format(res2cul.mean(axis=0)[-5:]))


    print('\n')
    print('std')
    print('std_zooming: {0}'.format(res1cul.std(axis=0)[-5:]))
    print('std_zoomingts: {0}'.format(res2cul.std(axis=0)[-5:]))



else:
    res1, res1cul, res2, res2cul = np.empty((0, T)), np.empty((0, T)), np.empty((0, T)), np.empty((0, T))
    for i in range(0, rep):
        data = simulation(T, sigma, peak, step, prob, seed_start=i + seed, func_type=funtype)
        zoomreg, zoomcul = zooming(sigma, T, data, points=[], inte=inte)
        tsreg, tscul = zooming_tsr(sigma, T, data, H, inte=inte)
        res1 = np.concatenate((res1, np.array(zoomreg).reshape(1, -1)))
        res1cul = np.concatenate((res1cul, np.array(zoomcul).reshape(1, -1)))
        res2 = np.concatenate((res2, np.array(tsreg).reshape(1, -1)))
        res2cul = np.concatenate((res2cul, np.array(tscul).reshape(1, -1)))


    print(funtype)
    print('mean')
    print('reg_zooming: {0}'.format(res1cul.mean(axis=0)[-5:]))
    print('reg_zoomingts: {0}'.format(res2cul.mean(axis=0)[-5:]))

    print('\n')
    print('std')
    print('std_zooming: {0}'.format(res1cul.std(axis=0)[-5:]))
    print('std_zoomingts: {0}'.format(res2cul.std(axis=0)[-5:]))

if save:
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + funtype + '/'):
        os.mkdir('results/' + funtype + '/')
    path = 'results/' + funtype + '/'
    file = path + str(T) + '_' + str(peak) + '_' + str(step) + '_' + str(prob)
    file = file + '.csv'

    df = pd.DataFrame(
        {'zooming': res1cul.mean(axis=0), 'zts': res2cul.mean(axis=0),
         'std_zooming': res1cul.std(axis=0),
         'std_zts': res2cul.std(axis=0)})
    df.to_csv(file)




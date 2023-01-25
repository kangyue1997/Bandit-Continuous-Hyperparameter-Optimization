from alg1_codes import *

import os
import pandas as pd
import concurrent.futures
import warnings
warnings.filterwarnings("ignore")
import argparse



parser = argparse.ArgumentParser(description='Zooming TS algorithm with Restarts')
parser.add_argument('-rep', '--rep', type=int, default=20, help='repeat times')
# parser.add_argument('-cT', '--cT', type=int, default=3, help='number of change points')
parser.add_argument('-T', '--T', type=int, default=90000, help='total time')
parser.add_argument('-seed', '--seed', type=int, default=275, help='seed starting point')
parser.add_argument('-H', '--H', type=int, default=24000, help='epoch size for restarts')
parser.add_argument('-sigma', '--sigma', type=float, default=math.sqrt(0.1), help='noise sub gaussian parameter')
parser.add_argument('-funtype', '--funtype', type=str, default='triangle', help='lipschitz function type: triangle or sine')
parser.add_argument('-peakcan', '--peakcan', nargs='+', default=[0.05,0.45,0.7,0.95], help='peak candidates')
parser.add_argument('-points', '--points', nargs='+', default=[22861, 47734, 74817], help='change points for data generation, type 0 for no change point')
parser.add_argument('-save', '--save', type=str, default='False', help='save the data or not')
parser.add_argument('-para', '--para', type=str, default='True', help='parallel computing')
parser.add_argument('-inte', '--inte', nargs = '+', default = [0,1], help = 'exploration intervals')
args = parser.parse_args()

T = args.T
rep = args.rep
seed = args.seed
H = args.H
sigma = args.sigma
funtype = args.funtype
if funtype != 'sine' and funtype != 'triangle':
    print("function type is set to default (triangle).")
    funtype = 'triangle'
peak_candidate = args.peakcan
peak_candidate = [float(x) for x in peak_candidate]
points = args.points
points = [int(float(x)) for x in points]
cT = len(points)
if points == [0]:
    points = []
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

if para:
    def func(n0):
        data = simulation(T, sigma, peak_candidate, points, seed_start=n0+seed, func_type=funtype)
        zoomreg, zoomcul = zooming(sigma, T, data, points = [], inte = inte)
        tsreg, tscul = zooming_tsr(sigma, T, data, H, inte = inte)
        oraclereg, oraclecul = zooming_tsr(sigma, T, data, H = points, inte = inte, prior = True)
        return zoomreg, zoomcul, tsreg, tscul, oraclereg, oraclecul

    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [nn for nn in range(rep)]
        results = executor.map(func, secs)
        results = list(results)
    res1 = np.array([i[0] for i in list(results)])
    res1cul = np.array([i[1] for i in list(results)])
    res2 = np.array([i[2] for i in list(results)])
    res2cul = np.array([i[3] for i in list(results)])
    res3 = np.array([i[4] for i in list(results)])
    res3cul = np.array([i[5] for i in list(results)])

    print(funtype)
    print('mean')
    print('reg_zooming: {0}'.format(res1cul.mean(axis=0)[-5:]))
    print('reg_zoomingts: {0}'.format(res2cul.mean(axis=0)[-5:]))
    print('reg_oracle: {0}'.format(res3cul.mean(axis=0)[-5:]))


    print('\n')
    print('std')
    print('std_zooming: {0}'.format(res1cul.std(axis=0)[-5:]))
    print('std_zoomingts: {0}'.format(res2cul.std(axis=0)[-5:]))
    print('std_oracle: {0}'.format(res3cul.std(axis=0)[-5:]))
    

else:
    res1, res1cul, res2, res2cul, res3, res3cul = np.empty((0,T)),np.empty((0,T)),np.empty((0,T)),np.empty((0,T)),np.empty((0,T)),np.empty((0,T))
    for i in range(0,rep):
        data = simulation(T, sigma, peak_candidate, points, seed_start=i+seed, func_type=funtype)
        zoomreg, zoomcul = zooming(sigma, T, data, points = [], inte = inte)
        tsreg, tscul = zooming_tsr(sigma, T, data, H, inte = inte)
        oraclereg, oraclecul = zooming_tsr(sigma, T, data, H = points, inte = inte, prior = True)
        res1 = np.concatenate((res1, np.array(zoomreg).reshape(1,-1)))
        res1cul = np.concatenate((res1, np.array(zoomcul).reshape(1, -1)))
        res2 = np.concatenate((res1, np.array(tsreg).reshape(1, -1)))
        res2cul = np.concatenate((res1, np.array(tscul).reshape(1, -1)))
        res3 = np.concatenate((res1, np.array(oraclereg).reshape(1, -1)))
        res3cul = np.concatenate((res1, np.array(oraclecul).reshape(1, -1)))


    print(funtype)
    print('mean')
    print('reg_zooming: {0}'.format(res1cul.mean(axis=0)[-5:]))
    print('reg_zoomingts: {0}'.format(res2cul.mean(axis=0)[-5:]))
    print('reg_oracle: {0}'.format(res3cul.mean(axis=0)[-5:]))


    print('\n')
    print('std')
    print('std_zooming: {0}'.format(res1cul.std(axis=0)[-5:]))
    print('std_zoomingts: {0}'.format(res2cul.std(axis=0)[-5:]))
    print('std_oracle: {0}'.format(res3cul.std(axis=0)[-5:]))
    
if save:
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + funtype + '/'):
        os.mkdir('results/' + funtype + '/')
    path = 'results/' + funtype + '/'
    file = path + str(T) + '_' + str(H) + '_' + str(cT) + '_'
    for v in peak_candidate:
        file += str(v) + '_'
    file = file[:-1] + '.csv'

    df = pd.DataFrame(
        {'zooming': res1cul.mean(axis=0), 'zts': res2cul.mean(axis=0), 'oracle': res3cul.mean(axis=0), 'std_zooming': res1cul.std(axis=0),
         'std_zts': res2cul.std(axis=0), 'std_op': res3cul.std(axis=0)})
    df.to_csv(file)
    print('\n')
    print('Save successful.')




# -*- coding: utf-8 -*-
"""NumPy replay of seven January-frozen predictors; no target or pair files are read."""
import argparse,json
from pathlib import Path
import numpy as np
NAMES=['LastValue','HistoricalMean','LinearDrift','RidgeCenter','STID_seed1','STID_seed2','STID_seed3']
def infer(x,weights,scale):
    x=np.asarray(x,dtype='float64')
    if x.ndim!=2 or x.shape[1]!=12 or not np.isfinite(x).all(): raise ValueError('Expected finite [origins,12] histories')
    out=[np.tile(x[:,-1,None],(1,12)),np.tile(x.mean(axis=1)[:,None],(1,12)),
         x[:,-1,None]+np.arange(1,13)[None,:]*(x[:,-1,None]-x[:,-4,None])/3,
         x@weights['ridge_coef']+weights['ridge_bias']]
    for seed in [1,2,3]:
        prefix='seed%d/'%seed
        def affine(a,key):
            return a@weights[prefix+key+'.weight'][:,:,0,0].T+weights[prefix+key+'.bias']
        v=np.concatenate([affine((x-scale['mean'])/scale['std'],'time_series_emb_layer'),
                          np.tile(weights[prefix+'node_emb'],(len(x),1))],axis=1)
        for k in range(3):
            key='encoder.%d.'%k
            v=v+affine(np.maximum(affine(v,key+'fc1'),0),key+'fc2')
        out.append(affine(v,'regression_layer')*scale['std']+scale['mean'])
    return np.maximum(np.stack(out),0)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--scaler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite prediction output')
    z=np.load(a.inputs,allow_pickle=False)
    weights=np.load(a.weights,allow_pickle=False)
    scale=json.loads(a.scaler.read_text(encoding='utf-8'))
    p=infer(z['x'],weights,scale)
    np.savez_compressed(a.output,predictions=p,ids=z['ids'],models=np.asarray(NAMES))
    print(json.dumps({'shape':list(p.shape),'targets_read':False,'new_fits':0,'replayed_configurations':7}))
if __name__=='__main__': main()


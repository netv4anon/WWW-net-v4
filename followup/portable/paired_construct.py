"""Fully reconstructible multisequence diagnostic, conditional on retained profiles."""
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
ARMS=('B','P','R','F','C','N')
TIMINGS=((8,4,'history_completed'),(10,8,'cutoff_spanning'),(14,4,'future_early'),(18,6,'future_late'))
SEED=2026100907

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def seed(*parts):return int.from_bytes(hashlib.sha256(('|'.join(map(str,(SEED,)+parts))).encode()).digest()[:4],'little')

def generate(profile_catalog=None, fixed_neighbor_path=None, legacy_ids=False):
    for record in json.loads((ROOT/'PINS.json').read_text(encoding='utf-8'))['files']:
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256']
    spec=importlib.util.spec_from_file_location('mv_legacy',ROOT/'pins/legacy_generator.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if profile_catalog is None:
        contexts=json.loads((ROOT/'pins/contexts.json').read_text(encoding='utf-8'))
        with np.load(ROOT/'pins/profiles.npz',allow_pickle=False) as z:original=z['profiles'].copy()
        groups={}
        for row,context in enumerate(contexts):
            name=context['profile_id']
            if name in groups:assert np.array_equal(original[groups[name]],original[row])
            else:groups[name]=row
        names=sorted(groups);profiles=np.stack([original[groups[name]] for name in names])
    else:
        with np.load(profile_catalog,allow_pickle=False) as z:
            names=z['profile_id'].tolist();profiles=z['profile'].copy()
        assert names==sorted(names) and len(names)>=8
        assert profiles.shape==(len(names),7,96)
        assert np.isfinite(profiles).all() and (profiles.mean(axis=(1,2))>0).all()

    means=profiles.mean(axis=(1,2))
    features=np.concatenate([(profiles/means[:,None,None]).reshape(len(names),-1),np.log(means)[:,None]],axis=1)
    distances=((features[:,None,:]-features[None,:,:])**2).mean(axis=2)
    neighbors=np.stack([np.argsort(distances[i],kind='stable')[np.argsort(distances[i],kind='stable')!=i][:7] for i in range(len(names))])
    if fixed_neighbor_path is not None:
        with np.load(fixed_neighbor_path,allow_pickle=False) as fixed:
            assert np.array_equal(fixed['profile_id'],np.array(names))
            neighbors=fixed['neighbor_indices'].copy()
        assert neighbors.shape==(len(names),7)
    n=len(names)*7*4*4
    arrays=np.empty((6,n,24,9),dtype=np.float32)
    residual=np.empty((n,12,9),dtype=np.float32)
    context_profile=[];background_id=[];timing=[];starts=[];ends=[];dates=[];catalog_indices=[];severity_values=[]
    base_id=0;case=0
    for index,name in enumerate(names):
        assignment=np.r_[index,index,neighbors[index]]
        for day in range(7):
            for hour in (3,9,15,21):
                date=datetime(2021,12,20,hour)+timedelta(days=day)
                b=np.empty((24,9));mu=np.empty((24,9))
                for slot in (0,2,3,4,5,6,7,8):
                    np.random.seed(seed('background',name,day,hour,slot))
                    b[:,slot],mu[:,slot]=module.generate_ar1_timeseries(profiles[assignment[slot]],date,noise_scale=.025,year=2021)
                b[:,1]=b[:,0];mu[:,1]=mu[:,0]
                severity=1+(index+day+hour//3)%5
                for start,duration,tag in TIMINGS:
                    f=b.copy();p=b.copy();r=b.copy();mask=np.zeros(24,dtype=bool)
                    for slot in (0,2,3,4,5,6,7,8):
                        attenuation=.003*(1 if slot==0 else slot)
                        arguments=dict(distance_deg=attenuation,severity_level=severity,dispatch_sec=300.,response_sec=600.,ev_start=start,ev_duration=duration)
                        event_seed=seed('amplitude',name,day,hour)
                        np.random.seed(event_seed)
                        f[:,slot],marks=module.apply_ev_perturbation_v4(b[:,slot],mu[:,slot],**arguments)
                        # Pulse-only arm shares the same stochastic amplitude and pulse values.
                        p[:,slot]=b[:,slot];p[start:start+duration,slot]=f[start:start+duration,slot]
                        p[:,slot]=np.clip(p[:,slot],3,b[:,slot].max()*1.3)
                        np.random.seed(event_seed)
                        r[:,slot],_=module.apply_ev_perturbation_v4(b[:,slot],mu[:,slot],**{**arguments,'severity_level':0})
                        alternative=f[:,slot].copy();end=start+duration
                        if end<24:
                            # Recover the unclipped final pulse value using the same fixed amplitude draw.
                            np.random.seed(event_seed)
                            alpha_values={0:0.,1:np.random.uniform(.06,.15),2:np.random.uniform(.14,.25),3:np.random.uniform(.22,.36),4:np.random.uniform(.32,.48),5:np.random.uniform(.42,.58)}
                            np.random.uniform(.08,.18)
                            impact=alpha_values[severity]*np.exp(-attenuation/.015)
                            progress=(duration-1)/duration
                            envelope=.6*(1-(2*progress-1)**2)+.4*np.exp(-abs(progress-.4)*duration*1.5)
                            anchor=b[end-1,slot]-impact*mu[end-1,slot]*envelope
                            positions=np.arange(end,24)
                            alternative[positions]=b[positions,slot]+(anchor-b[end-1,slot])*np.exp(-(positions-end)/2.)
                            alternative=np.clip(alternative,3,b[:,slot].max()*1.3)
                        residual[case,:,slot]=alternative[12:]
                        mask=marks.astype(bool)
                    for v in (f,p,r):v[:,1]=v[:,0]
                    residual[case,:,1]=residual[case,:,0]
                    c=b.copy();c[:12,:2]=f[:12,:2];c[12:]=f[12:]
                    other=b.copy();other[:12,2:]=f[:12,2:];other[12:]=f[12:]
                    arrays[:,case]=np.stack([b,p,r,f,c,other]).astype(np.float32)
                    context_profile.append(index);background_id.append(base_id);timing.append(tag);starts.append(start);ends.append(start+duration)
                    dates.append(date.isoformat());catalog_indices.append(assignment);severity_values.append(severity)
                    case+=1
                base_id+=1
    assert case==n and base_id==len(names)*28
    assert np.array_equal(arrays[:,:,:,0],arrays[:,:,:,1])
    assert np.array_equal(arrays[3,:,:12,:2],arrays[4,:,:12,:2])
    assert np.array_equal(arrays[0,:,:12,:2],arrays[5,:,:12,:2])
    ids=np.array([hashlib.sha256((('NETV4-MV|' if profile_catalog is None or legacy_ids else 'NETV4-confirmation|'+hashlib.sha256(Path(profile_catalog).read_bytes()).hexdigest()+'|')+str(SEED)+'|'+str(a)+'|'+str(j)).encode()).hexdigest()[:32] for a in range(6) for j in range(n)])
    order=np.random.RandomState(SEED).permutation(len(ids))
    future_coords=np.arange(12,24)[None,:]
    start_values=np.array(starts);end_values=np.array(ends)
    masks=(future_coords>=start_values[:,None])&(future_coords<end_values[:,None])
    return ({'inputs':{'sample_id':ids[order],'history':arrays[:,:,:12].reshape(-1,12,9)[order]},
            'evaluation':{'sample_id':ids,'context_id':np.tile(np.arange(n),6),'arm':np.repeat(np.arange(6),n),
                'target':arrays[:,:,12:].reshape(-1,12,9),'residual_full_target':residual,
                'profile_index':np.array(context_profile),'background_id':np.array(background_id),
                'timing':np.array(timing),'start':start_values,'end':end_values,'future_pulse_mask':masks,
                'calendar_coordinate':np.array(dates),'profile_assignment':np.array(catalog_indices),
                'severity':np.array(severity_values)},
            'catalog':{'profile_id':np.array(names),'profile':profiles,'neighbor_indices':neighbors}},
            {'contexts':n,'backgrounds':base_id,'profiles':len(names),'input_rows':len(ids),'arms':list(ARMS),
             'pulse_counts_per_horizon':masks.sum(axis=0).tolist(),'numeric_slots':9,'unique_profile_slots':8,
             'actual_spatial_adjacency_established':False,'old_raw_source_pool_recovered':False,
             'new_observed_traffic_records':0,'new_generated_12_to_12_rows':len(ids),'seed':SEED})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'task');parser.add_argument('--verify',type=Path);args=parser.parse_args()
    objects,description=generate()
    if args.verify:
        compared=0
        for name,values in objects.items():
            with np.load(args.verify/(name+'.npz'),allow_pickle=False) as z:
                assert set(z.files)==set(values)
                for key,value in values.items():assert np.array_equal(z[key],value),(name,key);compared+=value.size
        print(json.dumps({'exact_regeneration':True,'array_elements_checked':compared,**description}))
        return
    args.out.mkdir(parents=True,exist_ok=False)
    for name,values in objects.items():np.savez_compressed(args.out/(name+'.npz'),**values)
    dump(args.out/'DESCRIPTION.json',description)
    dump(args.out/'BINDING.json',{'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL_ZH.md').read_bytes()).hexdigest(),
        'constructor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files':[{'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(args.out.glob('*.npz'))]})
    print(json.dumps(description),flush=True)

if __name__=='__main__':main()

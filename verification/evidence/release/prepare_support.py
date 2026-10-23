# -*- coding: utf-8 -*-
"""Construct fixed support interventions from byte-verified numerical traffic states."""
import collections,hashlib,json,mmap
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;B=HERE.parent;AUDIT=B/'incident_split_feasibility_2026-10-03'
N=5585;NT=3909;Q0=4746
def read(p):return json.loads(p.read_text('utf-8-sig'))
def dump(name,obj):(HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for a in iter(lambda:f.read(4194304),b''):h.update(a)
 return h.hexdigest()
def support_metrics(states,rows):
 q=states[Q0:,12:];xx=set(states[rows,:12].ravel().tolist());yy=set(states[rows,12:].ravel().tolist())
 mx=np.isin(q,list(xx));my=np.isin(q,list(yy));both=mx|my
 whole={states[i].tobytes() for i in rows}
 return dict(train_rows=len(rows),test_rows=839,
  test_target_positions_in_train_history=int(mx.sum()),test_target_positions_in_train_target=int(my.sum()),
  test_target_positions_in_either=int(both.sum()),test_rows_with_any_target_in_train_history=int(mx.any(1).sum()),
  test_rows_with_any_target_in_train_target=int(my.any(1).sum()),test_rows_with_any_target_in_either=int(both.any(1).sum()),
  test_whole_traffic_matches=sum(states[i].tobytes() in whole for i in range(Q0,N)),
  distinct_training_history_states=len(xx),distinct_training_target_states=len(yy))
def main():
 assert not (HERE/'PREPARATION_SEAL.json').exists()
 start=read(HERE/'START.json')
 for rel,d in start['bindings'].items():assert sha(B/rel)==d
 source=AUDIT/'sources/contra_complete.npy'
 assert sha(source)=='bc6c5608457f88864625d683f9a66d17bc0fc31fe5acba366b6033708a099279'
 dump('PREPARATION_SEAL.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(HERE/'PROTOCOL_ZH.md'),code_sha256=sha(Path(__file__)),source_sha256=sha(source)))
 records=read(AUDIT/'ARRAY_OFFSETS.json')
 arrays={(r['row'],r['role']):r for r in records if r['role'] in ['x_data','y_data']}
 assert len(arrays)==N*2
 states=np.empty((N,24),dtype=np.int64)
 X=np.lib.format.open_memmap(HERE/'HISTORY_FLOW.npy',mode='w+',dtype='<f4',shape=(N,12,496))
 Y=np.lib.format.open_memmap(HERE/'TARGET_FLOW.npy',mode='w+',dtype='<f4',shape=(N,12,496))
 keymap={};representatives=[];hits=0
 with source.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
  for i in range(N):
   for role,j,flow in [('x_data',0,X),('y_data',12,Y)]:
    a=arrays[i,role];raw=mm[a['offset']:a['offset']+a['bytes']]
    assert hashlib.sha256(raw).hexdigest()==a['sha256']
    v=np.frombuffer(raw,dtype='<f4').reshape(12,496,3)
    flow[i]=v[:,:,0]
    for t in range(12):
     off=a['offset']+t*496*3*4;vraw=mm[off:off+5952];digest=hashlib.sha256(vraw).digest()
     if digest in keymap:
      k=keymap[digest];prior=representatives[k]
      assert mm[prior:prior+5952]==vraw
      hits+=1
     else:
      k=len(representatives);keymap[digest]=k;representatives.append(off)
     states[i,j+t]=k
 X.flush();Y.flush();del X,Y
 np.save(HERE/'STATE_IDS.npy',states,allow_pickle=False)
 np.save(HERE/'STATE_BYTE_OFFSETS.npy',np.array(representatives,dtype=np.int64),allow_pickle=False)
 whole={states[i].tobytes() for i in range(Q0,N)}
 forbidden=np.unique(states[Q0:,12:])
 masks={'Original':np.arange(NT),'WholeTraffic':np.array([i for i in range(NT) if states[i].tobytes() not in whole],dtype=int),
        'TargetSupport':np.flatnonzero(~np.isin(states[:NT],forbidden).any(1))}
 for seed in [4101,4102,4103]:
  masks['Random%d'%seed]=np.sort(np.random.Generator(np.random.PCG64(seed)).choice(NT,len(masks['TargetSupport']),replace=False))
 np.savez_compressed(HERE/'TRAIN_ROWS.npz',**masks)
 old=read(AUDIT/'PAIRED_RECORD_REFINEMENT.json')
 paired=set(old['train_test_shared_history_and_target_test_rows'])
 yy=set(states[:NT,12:].ravel().tolist())
 partial={i for i in range(Q0,N) if np.isin(states[i,12:],list(yy)).any()}-paired
 untouched=set(range(Q0,N))-paired-partial
 assert (len(paired),len(partial),len(untouched))==(93,466,280)
 groups={'all':np.arange(839),'whole_traffic_reuse':np.array(sorted(paired))-Q0,
         'other_target_reuse':np.array(sorted(partial))-Q0,'no_original_target_reuse':np.array(sorted(untouched))-Q0}
 np.savez_compressed(HERE/'TEST_GROUPS.npz',**groups)
 metrics={name:support_metrics(states,rows) for name,rows in masks.items()}
 assert metrics['TargetSupport']['test_target_positions_in_either']==0
 metadata=read(AUDIT/'SAMPLE_METADATA.json')
 distributions={}
 for name,rows in masks.items():
  d={}
  for field in ['Type','Description','Holiday']:
   d[field]=dict(sorted(collections.Counter(str(metadata[i]['incident_features'][field]) for i in rows).items()))
  d['weekday']=dict(sorted(collections.Counter(str(metadata[i]['last_history_week_phase']//288) for i in rows).items()))
  d['hour']=dict(sorted(collections.Counter(str((metadata[i]['last_history_week_phase']%288)//12) for i in rows).items()))
  distributions[name]=d
 dump('TRAINING_DISTRIBUTIONS.json',distributions)
 dump('SUPPORT_RESULTS.json',dict(status='prepared_pending_independent_verification',row_count=N,distinct_full_network_states=len(representatives),
      state_byte_comparisons=hits,conditions=metrics,test_group_sizes={k:len(v) for k,v in groups.items()},
      training_feasibility_passed=len(masks['TargetSupport'])>=500,
      test_targets_used_for_mask_construction=True,absolute_dates_recovered=False,
      future_generalization_established=False,new_fits=0,new_predictions=0))
 dump('PREPARED_INPUT_BINDINGS.json',{name:sha(HERE/name) for name in ['HISTORY_FLOW.npy','TARGET_FLOW.npy','STATE_IDS.npy','STATE_BYTE_OFFSETS.npy','TRAIN_ROWS.npz','TEST_GROUPS.npz','SUPPORT_RESULTS.json','TRAINING_DISTRIBUTIONS.json']})
 print(json.dumps(metrics,ensure_ascii=False))
if __name__=='__main__':main()

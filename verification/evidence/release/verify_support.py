# -*- coding: utf-8 -*-
"""Independent raw-byte test of masks and overlap counts, without primary state IDs."""
import collections,hashlib,json,mmap
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;B=HERE.parent;A=B/'incident_split_feasibility_2026-10-03'
def read(p):return json.loads(p.read_text('utf-8-sig'))
def dump(name,o):(HERE/name).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 offsets={(r['row'],r['role']):r for r in read(A/'ARRAY_OFFSETS.json') if r['role'] in ['x_data','y_data']}
 masks=np.load(HERE/'TRAIN_ROWS.npz',allow_pickle=False)
 results=read(HERE/'SUPPORT_RESULTS.json')['conditions']
 history=np.load(HERE/'HISTORY_FLOW.npy',mmap_mode='r',allow_pickle=False)
 targets=np.load(HERE/'TARGET_FLOW.npy',mmap_mode='r',allow_pickle=False)
 qkeys=collections.defaultdict(list);fullqueries=collections.defaultdict(list)
 with (A/'sources/contra_complete.npy').open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
  def raw(i,role):
   a=offsets[i,role];data=mm[a['offset']:a['offset']+a['bytes']]
   assert hashlib.sha256(data).hexdigest()==a['sha256']
   return data
  for i in range(4746,5585):
   x=raw(i,'x_data');y=raw(i,'y_data')
   fullqueries[x+y].append(i-4746)
   for k in range(12):qkeys[y[k*5952:(k+1)*5952]].append((i-4746)*12+k)
  overlaps_x=[];overlaps_y=[];whole=[];witnesses=[]
  for i in range(5585):
   x=raw(i,'x_data');y=raw(i,'y_data')
   assert np.array_equal(np.frombuffer(x,dtype='<f4').reshape(12,496,3)[:,:,0],history[i])
   assert np.array_equal(np.frombuffer(y,dtype='<f4').reshape(12,496,3)[:,:,0],targets[i])
   if i>=3909:continue
   rx,ry=set(),set();first=None
   for role,data,storage in [('x',x,rx),('y',y,ry)]:
    for k in range(12):
     block=data[k*5952:(k+1)*5952]
     if block in qkeys:
      storage.update(qkeys[block])
      if first is None:
       pos=qkeys[block][0];first=dict(train_row=i,train_role=role,train_step=k,test_row=4746+pos//12,test_step=pos%12,byte_count=5952,sha256=hashlib.sha256(block).hexdigest())
   overlaps_x.append(rx);overlaps_y.append(ry);whole.append(fullqueries.get(x+y,[]))
   if first is not None:witnesses.append(first)
 kept=np.array([i for i in range(3909) if not (overlaps_x[i] or overlaps_y[i])])
 wholekept=np.array([i for i in range(3909) if not whole[i]])
 assert np.array_equal(kept,masks['TargetSupport']) and np.array_equal(wholekept,masks['WholeTraffic'])
 checks={}
 for name in masks.files:
  rows=masks[name];sx=set().union(*(overlaps_x[i] for i in rows));sy=set().union(*(overlaps_y[i] for i in rows))
  values=dict(train_rows=len(rows),test_target_positions_in_train_history=len(sx),test_target_positions_in_train_target=len(sy),
   test_target_positions_in_either=len(sx|sy),test_rows_with_any_target_in_train_history=len({v//12 for v in sx}),
   test_rows_with_any_target_in_train_target=len({v//12 for v in sy}),
   test_rows_with_any_target_in_either=len({v//12 for v in sx|sy}),test_whole_traffic_matches=len(set().union(*(set(whole[i]) for i in rows))))
  for k,v in values.items():assert results[name][k]==v,(name,k,v,results[name][k])
  checks[name]=values
 assert not set(masks['TargetSupport']).intersection(w['train_row'] for w in witnesses)
 assert len(witnesses)==3909-len(kept)
 for seed in [4101,4102,4103]:
  expected=np.sort(np.random.Generator(np.random.PCG64(seed)).choice(3909,len(kept),replace=False))
  assert np.array_equal(expected,masks['Random%d'%seed])
 dump('REMOVAL_WITNESSES.json',witnesses)
 dump('SUPPORT_VERIFICATION.json',dict(status='passed',method='Complete source bytes as dictionary keys; no primary state ID/hash grouping reused.',
      input_arrays_compared=11170,training_removal_witnesses=len(witnesses),conditions=checks,
      retained_rows=2665,minimum_required=500,ready_for_fixed_training=True,
      chronological_repair_claimed=False,physical_date_identity_established=False))
 print(json.dumps({'status':'passed','retained':len(kept),'removed':len(witnesses),'conditions_verified':len(checks)}))
if __name__=='__main__':main()

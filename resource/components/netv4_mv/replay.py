"""Regenerate arrays and replay saved forecasts without parent project or GPU."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    output=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'construct.py'),'--verify',str(ROOT/'task')],cwd=ROOT,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True).stdout
    (args.out/'REGEN_VERIFICATION.json').write_bytes(output)
    subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'score.py'),'--out',str(args.out/'scores')],cwd=ROOT,check=True)
    result=json.loads((args.out/'scores/SUMMARY.json').read_text(encoding='utf-8'))
    expected=json.loads((ROOT/'scores/SUMMARY.json').read_text(encoding='utf-8'))
    assert result==expected
    subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'verify.py'),'--out',str(args.out/'DIRECT_VERIFICATION.json')],cwd=ROOT,check=True)
    (args.out/'REPLAY_VERIFICATION.json').write_text(json.dumps({'exact_array_regeneration':True,
        'saved_forecast_summary_identical':True,'direct_loss_check_completed':True,
        'new_model_fits':0,'new_independent_evidence':False,'network_parent_project_or_gpu_required':False},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()

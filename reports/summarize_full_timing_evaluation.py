import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path('src').resolve()))
from chess_self.timing_metrics import compare_scorecards
root=Path('reports/timing-full-evaluation-v1')
arms=['reference','e03-position','e04-window','e05-squares']
rows=[json.loads(s) for s in Path('data/timing-v2/val.jsonl').read_text().splitlines()]
summary={'test_evaluated':False,'evaluation_positions':len(rows),'conditions':{},'limitations':['Development checkpoints were selected on this validation set; intervals do not remove selection bias.','Matched refit control and extra seeds are not trained.']}
for points in (256,1024):
 for support in ('feasible','raw'):
  condition=f'{support}-{points}';reports={a:json.loads((root/condition/f'{a}.json').read_text()) for a in arms}
  result={}
  for a,r in reports.items():
   result[a]={'overall':r['groups']['all'],'move_policy':r['move_policy'],'joint_move_time_interval_nll':r.get('joint_move_time_interval_nll'),'groups':{k:v for k,v in r['groups'].items() if k.startswith(('control/','clock/','position/'))}}
   if a!='reference':result[a]['vs_reference']=json.loads((root/condition/f'{a}-vs-reference.json').read_text())
  result['e05-squares']['vs_e04']=json.loads((root/condition/'e05-squares-vs-e04-window.json').read_text())
  subsets={'under10':[i for i,r in enumerate(rows) if r['remaining_seconds']<10],'bullet':[i for i,r in enumerate(rows) if r['initial_seconds']<=60]}
  result['e05-squares']['subgroup_vs_reference']={}
  for name,ids in subsets.items():
   pair=[dict(reports[a],per_row=[reports[a]['per_row'][i] for i in ids]) for a in ('e05-squares','reference')]
   result['e05-squares']['subgroup_vs_reference'][name]=compare_scorecards(*pair)
  summary['conditions'][condition]=result
summary['quadrature_change']={a:summary['conditions']['feasible-1024'][a]['overall']['crps_seconds']-summary['conditions']['feasible-256'][a]['overall']['crps_seconds'] for a in arms}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'overall':{a:summary['conditions']['feasible-1024'][a]['overall']['crps_seconds'] for a in arms},'best_ci':summary['conditions']['feasible-1024']['e05-squares']['vs_reference']['crps_seconds'],'quadrature_change':summary['quadrature_change']},indent=2))

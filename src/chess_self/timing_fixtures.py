"""Native reference tensors for actual browser WASM/WebGPU parity and latency runs."""
import math
import chess
import torch
from .model import encode, load_model
from .timing_model import MaiaFeatures,load_timing_head,selected_move_inputs,history_tensor
from .timing_cache import digest,read_rows,write_json
from .timing_experiments import side_inputs


def select_fixture_rows(rows):
    if not rows:
        raise ValueError('Need reference positions')
    by_color = {'white': [], 'black': []}
    for row in rows:
        color = 'white' if chess.Board(row['fen']).turn else 'black'
        if row.get('player_color') != color:
            raise ValueError('Fixture player_color must match the FEN side to move')
        clock = row.get('remaining_seconds')
        if not isinstance(clock, (int, float)) or not math.isfinite(clock) or clock < 0:
            raise ValueError('Fixture clocks must be finite and nonnegative')
        by_color[color].append(row)
    for color, candidates in by_color.items():
        if not candidates:
            raise ValueError(f'Need an explicit {color} development position for browser fixtures')
    low = min(rows, key=lambda row: row['remaining_seconds'])
    # Keep low-clock coverage distinct too when each color has other positions.
    selected = [next((row for row in by_color[color] if row['fen'] != low['fen']),
                     by_color[color][0]) for color in ('white', 'black')]
    return selected + [low]


def export_browser_fixtures(checkpoint,head_path,positions,output):
    from pathlib import Path
    output=Path(output)
    if output.exists():raise ValueError('Choose a new browser fixture output')
    if Path(positions).name=='test.jsonl':raise ValueError('Use development positions for runtime fixtures')
    rows=read_rows(positions)
    selected=select_fixture_rows(rows)
    head,meta=load_timing_head(head_path,base_checkpoint_sha256=digest(checkpoint))
    from .timing_data import feature_names
    if feature_names(meta['extra']['feature_groups'])!=meta['feature_names']:
        raise ValueError('Checkpoint feature ordering mismatch')
    extractor=MaiaFeatures(load_model(checkpoint)).eval()
    def pack(t):
        return {'data':t.detach().cpu().flatten().tolist(),'shape':list(t.shape),
                'dtype':'int64' if t.dtype==torch.int64 else 'float32'}
    cases=[]
    with torch.inference_mode():
        for name,batch in [('white',[selected[0]]),('black',[selected[1]]),('low-clock',[selected[2]]),('batch-two',selected[:2])]:
            enc=[encode(r) for r in batch]
            inputs={k:torch.stack([e[k] for e in enc]) for k in ('tokens','self_elos','oppo_elos')}
            features=extractor(**inputs)
            legal=torch.stack([e['legal'] for e in enc])
            probs=features['move_logits'].masked_fill(~legal,-torch.inf).softmax(-1)
            entropy=-(probs*probs.clamp_min(1e-30).log()).sum(-1)
            top=probs.topk(2,-1).values
            for i,row in enumerate(batch):
                row['policy_entropy']=float(entropy[i]);row['top_two_gap']=float(top[i,0]-top[i,1])
            from .timing_cache import add_opponent_surprisal
            add_opponent_surprisal(extractor,batch)
            indices,attrs=selected_move_inputs(batch,features['move_logits'])
            side={'side_features':side_inputs(batch,meta['extra']['feature_groups'],'cpu'),
                  'move_indices':indices,'move_attributes':attrs,
                  'history':history_tensor(batch,head.config.history_length)}
            raw=head(features['pooled'],features['base_logits'],side['side_features'],features['squares'],indices,attrs,side['history'])
            expected={**features,'timing_raw':raw}
            cases.append({'name':name,
                          'positions':[{key:row[key] for key in ('game_id','ply','fen','player_color','move_uci','remaining_seconds')} for row in batch],
                          'inputs':{k:pack(v) for k,v in inputs.items()},
                          'side_inputs':{k:pack(v) for k,v in side.items()},
                          'expected':{k:pack(v) for k,v in expected.items()}})
    report={'schema_version':1,'cases':cases,'tolerances':{'atol':.0002,'rtol':.001},
            'config':{'architecture_id':meta['architecture'],'head_config':meta['config'],
                      'base_checkpoint_sha256':digest(checkpoint),'head_checkpoint_sha256':digest(head_path),
                      'positions_sha256':digest(positions),'smoke_only':meta['extra'].get('smoke_only',False)},
            'scheduler':{'case_name':'white','position_key':selected[0]['fen'],
                         'move_uci':selected[0]['move_uci'],
                         'legal_moves':[m.uci() for m in __import__('chess').Board(selected[0]['fen']).legal_moves],
                         'remaining_seconds':selected[0]['remaining_seconds'], 'target_seconds':1.,'reserve_seconds':0.},
            'browser_execution_verified':False}
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,report)
    return {'cases':len(cases),'path':str(output),'browser_execution_verified':False}

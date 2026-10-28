"""CLI for the timing experiment plan; no command implicitly evaluates test games."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from .timing_cache import (FeatureCache, build_cache, read_rows, fresh_directory,
                           write_json, digest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Enrich immutable benchmark train/val splits')
    prepare.add_argument('--pgn', nargs='+', type=Path, required=True)
    prepare.add_argument('--players', nargs='+', required=True)
    prepare.add_argument('--reference', type=Path, default=Path('data/processed'))
    prepare.add_argument('--output', type=Path, required=True)
    cache = sub.add_parser('cache', help='Cache exact frozen backbone features')
    cache.add_argument('--positions', type=Path, required=True)
    cache.add_argument('--checkpoint', type=Path, required=True)
    cache.add_argument('--output', type=Path, required=True)
    cache.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    cache.add_argument('--batch-size', type=int, default=32)
    cache.add_argument('--limit-games', type=int)
    cache.add_argument('--predictor-training-ids', type=Path, help='JSON list of all games fitted by the checkpoint; required for calibration provenance')
    training = sub.add_parser('train')
    training.add_argument('--train-cache', type=Path, required=True)
    training.add_argument('--val-cache', type=Path, required=True)
    training.add_argument('--config', type=Path, required=True)
    training.add_argument('--output', type=Path, required=True)
    training.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    training.add_argument('--epochs', type=int, default=8)
    training.add_argument('--batch-size', type=int, default=32)
    training.add_argument('--learning-rate', type=float, default=.001)
    training.add_argument('--backbone-lr', type=float, default=.000001)
    training.add_argument('--weight-decay', type=float, default=.01)
    training.add_argument('--patience', type=int, default=2)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--initialize', type=Path)
    training.add_argument('--checkpoint', type=Path, help='Exact backbone required for E13')
    training.add_argument('--final-blocks', type=int, default=0, choices=[0,1,2])
    training.add_argument('--smoke', action='store_true')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--cache', type=Path, required=True)
    evaluate.add_argument('--head', type=Path)
    evaluate.add_argument('--backbone', type=Path, help='Adapted E13 checkpoint; recomputes features online')
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--conditional', action='store_true', help='Human move diagnostic, not equal-information comparison')
    evaluate.add_argument('--unconstrained', action='store_true')
    evaluate.add_argument('--batch-size', type=int, default=32)
    evaluate.add_argument('--crps-points', type=int, default=256)
    evaluate.add_argument('--training-positions', type=Path, help='Training-only repetition index for scorecard groups')
    calibrate = sub.add_parser('calibrate')
    calibrate.add_argument('--fit-cache', type=Path, required=True)
    calibrate.add_argument('--val-cache', type=Path, required=True)
    calibrate.add_argument('--output', type=Path, required=True)
    calibrate.add_argument('--mode', choices=['temperature','bias','regime','histogram'], default='temperature')
    calibrate.add_argument('--epochs', type=int, default=30)
    calibrate.add_argument('--seed', type=int, default=42)
    playback = sub.add_parser('playback')
    playback.add_argument('--checkpoint', type=Path, required=True)
    playback.add_argument('--head', type=Path)
    playback.add_argument('--calibration', type=Path)
    playback.add_argument('--event', type=Path)
    playback.add_argument('--positions', type=Path, help='Recorded-state replay; omitted means generated full games')
    playback.add_argument('--mode', choices=['sample','mean','median'], default='sample')
    playback.add_argument('--unconstrained', action='store_true')
    playback.add_argument('--reserve-seconds', type=float, default=0.)
    playback.add_argument('--base', type=float, default=300.)
    playback.add_argument('--increment', type=float, default=0.)
    playback.add_argument('--max-plies', type=int, default=200)
    playback.add_argument('--games', type=int, default=10)
    playback.add_argument('--first-move-rule',choices=['charge_all','free_first_turn'],default='charge_all')
    playback.add_argument('--clock-charge-mode',choices=['target','wall'],default='target')
    playback.add_argument('--seed', type=int, default=42)
    playback.add_argument('--output', type=Path, required=True)
    export = sub.add_parser('export')
    export.add_argument('--head', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    backbone_export = sub.add_parser('export-backbone')
    backbone_export.add_argument('--checkpoint',type=Path,required=True)
    backbone_export.add_argument('--output',type=Path,required=True)
    fixtures = sub.add_parser('browser-fixtures')
    fixtures.add_argument('--checkpoint',type=Path,required=True)
    fixtures.add_argument('--head',type=Path,required=True)
    fixtures.add_argument('--positions',type=Path,required=True)
    fixtures.add_argument('--output',type=Path,required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--data', type=Path, required=True)
    audit.add_argument('--output', type=Path, required=True)
    partition = sub.add_parser('partition', help='New timestamp/session-safe development cutoff')
    partition.add_argument('--rows', type=Path, required=True)
    partition.add_argument('--cutoff', required=True)
    partition.add_argument('--reserved-audit', type=Path, required=True)
    partition.add_argument('--output', type=Path, required=True)
    partition.add_argument('--retire-validation',action='store_true',help='Explicitly retire old validation scores if those games enter training')
    public = sub.add_parser('public-check')
    public.add_argument('--rows', type=Path, required=True)
    public.add_argument('--target-accounts', nargs='+', required=True)
    public.add_argument('--cutoff', required=True)
    public.add_argument('--reserved-audit', type=Path, required=True)
    public.add_argument('--output', type=Path, required=True)
    augment = sub.add_parser('augment', help='Causal embedding retrieval or offline engine features')
    augment.add_argument('--cache', type=Path, required=True)
    augment.add_argument('--training-cache', type=Path)
    augment.add_argument('--engine', type=Path)
    augment.add_argument('--shallow-depth', type=int, default=4)
    augment.add_argument('--deep-depth', type=int, default=8)
    augment.add_argument('--output', type=Path, required=True)
    events = sub.add_parser('events')
    events.add_argument('--training', type=Path, required=True)
    events.add_argument('--validation', type=Path, required=True)
    events.add_argument('--output', type=Path, required=True)
    events.add_argument('--epochs', type=int, default=100)
    compare = sub.add_parser('compare')
    compare.add_argument('--left', type=Path, required=True)
    compare.add_argument('--right', type=Path, required=True)
    compare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.threads < 1: parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    for key in ('positions','rows','training','validation'):
        value=getattr(args,key,None)
        if value and value.name=='test.jsonl':
            parser.error('Reserved test data is not accepted by development commands')
    if args.command == 'prepare':
        from .timing_data import prepare_timing_data
        report = prepare_timing_data(args.pgn,args.players,args.reference,args.output)
        print(f'Prepared {args.output}; original test games reserved')
    elif args.command == 'cache':
        ids = json.loads(args.predictor_training_ids.read_text()) if args.predictor_training_ids else None
        if ids is not None and (not isinstance(ids,list) or not all(isinstance(x,str) for x in ids)):
            parser.error('Predictor training IDs must be a JSON list of strings')
        build_cache(args.positions,args.checkpoint,args.output,device=args.device,batch_size=args.batch_size,
                    limit_games=args.limit_games,predictor_training_ids=ids)
    elif args.command == 'train':
        from .timing_experiments import train_experiment
        options = vars(args).copy()
        for key in ('threads','command'): options.pop(key)
        options['config'] = json.loads(args.config.read_text())
        report = train_experiment(**options)
        print(json.dumps({'selected_epoch':report['selected_epoch'],'accepted_candidate':report['accepted_candidate'],
                          'smoke_only':report['smoke_only'],'report':str(args.output/'run.json')}))
    elif args.command == 'evaluate':
        from .timing_experiments import evaluate_head
        from .timing_model import load_timing_head, MaiaFeatures
        from .model import load_model
        if args.crps_points < 2: parser.error('--crps-points must be at least 2')
        if args.training_positions and args.training_positions.name == 'test.jsonl':
            parser.error('Reserved test rows cannot form the training repetition index')
        cache = FeatureCache(args.cache); head=None; groups=[]; extractor=None
        sha = digest(args.backbone) if args.backbone else cache.manifest['checkpoint_sha256']
        if args.head:
            head,payload=load_timing_head(args.head,base_checkpoint_sha256=sha)
            groups=payload['extra']['feature_groups']
            from .timing_data import feature_names
            if feature_names(groups)!=payload['feature_names']:raise ValueError('Checkpoint feature ordering mismatch')
        if args.backbone: extractor=MaiaFeatures(load_model(args.backbone))
        report=evaluate_head(head,cache,groups,batch_size=args.batch_size,conditional=args.conditional,
                             feasible=not args.unconstrained,extractor=extractor, crps_points=args.crps_points,
                             training_rows=read_rows(args.training_positions) if args.training_positions else None)
        report['cache_manifest']=cache.manifest
        report['training_positions_sha256']=digest(args.training_positions) if args.training_positions else None
        report['head_checkpoint_sha256']=digest(args.head) if args.head else None
        report['test_evaluated']=False
        args.output.parent.mkdir(parents=True,exist_ok=True); write_json(args.output,report)
    elif args.command == 'calibrate':
        from .timing_calibration import fit_calibration
        fit_calibration(args.fit_cache,args.val_cache,args.output,mode=args.mode,epochs=args.epochs,seed=args.seed)
    elif args.command == 'playback':
        from .timing_predict import TimingPredictor
        from .timing_runtime import recorded_replay, virtual_game
        predictor=TimingPredictor(args.checkpoint,args.head,calibration=args.calibration,event=args.event,
                                  mode=args.mode,feasible=not args.unconstrained,reserve_seconds=args.reserve_seconds,seed=args.seed)
        if args.positions:
            report=recorded_replay(read_rows(args.positions),predictor,reserve_seconds=args.reserve_seconds,guard=not args.unconstrained)
        else:
            if args.games < 1: parser.error('--games must be positive')
            report={'mode':'virtual_games','seed':args.seed,'games':[
                virtual_game(predictor,args.base,args.increment,max_plies=args.max_plies,
                             reserve_seconds=args.reserve_seconds,guard=not args.unconstrained,game_id=f'virtual-{i}',
                             first_move_rule=args.first_move_rule,clock_charge_mode=args.clock_charge_mode)
                for i in range(args.games)]}
        args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,report)
    elif args.command == 'export':
        from .timing_export import export_timing_head
        print(export_timing_head(args.head,args.output))
    elif args.command == 'export-backbone':
        from .timing_export import export_maia_features
        print(export_maia_features(args.checkpoint,args.output))
    elif args.command == 'browser-fixtures':
        from .timing_fixtures import export_browser_fixtures
        print(export_browser_fixtures(args.checkpoint,args.head,args.positions,args.output))
    elif args.command == 'audit':
        report=json.loads((args.data/'manifest.json').read_text())
        report['label_details']={}
        for split in ('train','val'):
            rows=read_rows(args.data/f'{split}.jsonl')
            report['label_details'][split]={'rows':len(rows),'missing':sum(r.get('elapsed_seconds') is None for r in rows),
                'zero_recorded_deltas':sum(r.get('elapsed_seconds') == 0 for r in rows),
                'long_labels':[{'game_id':r['game_id'],'ply':r['ply'],'elapsed_seconds':r['elapsed_seconds'],
                                'platform':r.get('platform')} for r in rows if (r.get('elapsed_seconds') or 0)>=99.9]}
        args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,report)
    elif args.command in ('partition','public-check'):
        from .timing_data import chronological_partition,validate_public_rows
        rows=read_rows(args.rows); reference=json.loads(args.reserved_audit.read_text())
        reserved=set(reference.get('reserved_test_game_ids',reference.get('splits',{}).get('test',{}).get('game_ids',[])))
        if not reserved: parser.error('Reserved-game audit must list protected IDs')
        if args.command=='public-check':
            report=validate_public_rows(rows,target_accounts=args.target_accounts,reserved_game_ids=reserved,cutoff=args.cutoff)
            output=fresh_directory(args.output)
            shutil.copyfile(args.rows,output/'train.jsonl')
            write_json(output/'manifest.json',{'schema_version':'timing-data-v2','public_validation':{'passed':True,'rows':len(report),'source_sha256':digest(args.rows)},
                       'target_accounts':args.target_accounts,'cutoff':args.cutoff,'reserved_test_game_ids':sorted(reserved)})
        else:
            partitioned=chronological_partition(rows,args.cutoff,reserved_game_ids=reserved)
            train,val=partitioned['train'],partitioned['val']
            former=set(reference.get('splits',{}).get('val',{}).get('game_ids',[]))
            reused=former & {r['game_id'] for r in train}
            if reused and not args.retire_validation:
                parser.error('New training contains former validation games; explicitly --retire-validation first')
            from .timing_data import add_familiarity
            add_familiarity(train,val)
            output=fresh_directory(args.output)
            for split,data in [('train',train),('val',val)]:
                (output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
            inherited={}
            if (args.rows.parent/'manifest.json').exists():
                inherited=json.loads((args.rows.parent/'manifest.json').read_text())
            write_json(output/'manifest.json',{**inherited,'schema_version':'timing-data-v2','refit_cutoff':args.cutoff,
                       'reserved_test_game_ids':sorted(reserved),'source_sha256':digest(args.rows),'retired_validation_game_ids':sorted(reused)})
    elif args.command == 'augment':
        augment_cache(args)
    elif args.command == 'events':
        from .timing_events import fit_events
        fit_events(args.training,args.validation,args.output,epochs=args.epochs)
    elif args.command == 'compare':
        from .timing_metrics import compare_scorecards
        left=json.loads(args.left.read_text());right=json.loads(args.right.read_text())
        result=compare_scorecards(left,right)
        args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,result)


def augment_cache(args):
    cache=FeatureCache(args.cache)
    if bool(args.training_cache)==bool(args.engine):
        raise ValueError('Choose exactly one of retrieval --training-cache or --engine')
    output=fresh_directory(args.output)
    rows=[dict(r) for r in cache.rows]
    if args.training_cache:
        from .timing_data import retrieval_features
        training=FeatureCache(args.training_cache)
        if training.manifest['checkpoint_sha256'] != cache.manifest['checkpoint_sha256']:
            raise ValueError('Retrieval requires identical backbone feature spaces')
        enrich=retrieval_features(training.rows,torch.stack([training[i]['pooled'] for i in range(len(training))]).numpy(),
                                 rows,torch.stack([cache[i]['pooled'] for i in range(len(cache))]).numpy())
        for row,extra in zip(rows,enrich):row.update(extra)
    else:
        import chess, chess.engine
        if args.shallow_depth<1 or args.deep_depth<=args.shallow_depth:
            raise ValueError('Engine depths must be positive and ordered')
        with chess.engine.SimpleEngine.popen_uci(str(args.engine),timeout=30) as engine:
            engine.configure({'Threads':1})
            for row in rows:
                board=chess.Board(row['fen'])
                low=engine.analyse(board,chess.engine.Limit(depth=args.shallow_depth,time=.25))
                high=engine.analyse(board,chess.engine.Limit(depth=args.deep_depth,time=1.))
                row['engine_score_disagreement']=abs(high['score'].pov(board.turn).score(mate_score=10000)-low['score'].pov(board.turn).score(mate_score=10000))/100
                row['engine_move_disagreement']=float(low.get('pv',[None])[0]!=high.get('pv',[None])[0])
    for shard in cache.manifest['shards']:shutil.copyfile(cache.path/shard['file'],output/shard['file'])
    (output/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest=dict(cache.manifest,rows_sha256=digest(output/'rows.jsonl'))
    manifest['augmentation']={'kind':'retrieval' if args.training_cache else 'engine',
                              'source_cache':str(args.cache),'training_cache':str(args.training_cache),
                              'engine':str(args.engine),'depths':[args.shallow_depth,args.deep_depth]}
    write_json(output/'manifest.json',manifest)


if __name__=='__main__':
    main()

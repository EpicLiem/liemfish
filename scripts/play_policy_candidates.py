"""Bounded sampled-policy games with fixed synthetic clocks and Stockfish budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch

from chess_self.model import load_model, predict


MATERIAL_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                   chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def capture_gain(board, move):
    captured = 1 if board.is_en_passant(move) else MATERIAL_VALUES.get(board.piece_type_at(move.to_square), 0)
    return captured + (MATERIAL_VALUES[move.promotion] - 1 if move.promotion else 0)


def exchange_gain(board, square, depth=0):
    """Best optional legal same-square recapture sequence, capped at 12 plies."""
    if depth >= 12:
        return 0
    best = 0
    for move in list(board.generate_legal_captures()):
        if move.to_square != square:
            continue
        value = capture_gain(board, move)
        after = board.copy(stack=False)
        after.push(move)
        best = max(best, value - exchange_gain(after, square, depth + 1))
    return best


def annotate_material_loss(record):
    """Return a new annotated record; require the actual engine PV reply capture."""
    board = chess.Board(record['fen'])
    move = chess.Move.from_uci(record['uci'])
    if move not in board.legal_moves:
        raise ValueError('Illegal recorded move')
    pv = record.get('pv', [])
    if pv and pv[0] != record['uci']:
        raise ValueError('Chosen PV does not start with recorded move')
    gain = 0
    reply_capture = None
    board.push(move)
    if len(pv) >= 2:
        reply = chess.Move.from_uci(pv[1])
        if reply not in board.legal_moves:
            raise ValueError('Illegal engine PV reply')
        if board.is_capture(reply):
            reply_capture = reply.uci()
            gain = capture_gain(board, reply)
            board.push(reply)
            gain -= exchange_gain(board, reply.to_square)
    return {**record, 'pv_reply_available': len(pv) >= 2,
            'pv_reply_capture': reply_capture, 'pv_reply_capture_gain': gain,
            'direct_material_loss': record['loss_cp'] >= 200 and gain >= 3}


def sample_move(policy, generator):
    weights = torch.tensor([move['probability'] for move in policy], dtype=torch.float64)
    index = int(torch.multinomial(weights, 1, generator=generator))
    return chess.Move.from_uci(policy[index]['uci']), index + 1, policy[index]['probability']


def score(info, color):
    return info['score'].pov(color).score(mate_score=10000)


def play_game(model, engine, *, seed, color, elo_self=680, elo_oppo=680,
              max_plies=100, opponent_nodes=10000, analysis_nodes=30000):
    generator = torch.Generator().manual_seed(seed)
    board = chess.Board()
    history = [board.fen()]
    clocks = {chess.WHITE: 600., chess.BLACK: 600.}
    game = chess.pgn.Game()
    game.headers.update({'Event': 'Liemfish fixed-clock policy comparison',
                         'White': 'Liemfish' if color else 'Stockfish',
                         'Black': 'Stockfish' if color else 'Liemfish',
                         'TimeControl': '600+0', 'Seed': str(seed),
                         'SelfRatingInput': str(elo_self), 'OpponentRatingInput': str(elo_oppo),
                         'ClockSimulation': '2 seconds per ply for both players; no timing head'})
    node = game
    records = []
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        mover = board.turn
        if mover == color:
            row = {'fen': board.fen(), 'history_fens': history[-8:],
                   'elo_self': elo_self, 'elo_oppo': elo_oppo,
                   'initial_seconds': 600., 'increment_seconds': 0.,
                   'remaining_seconds': clocks[mover]}
            policy = predict(model, row)['moves']
            move, rank, probability = sample_move(policy, generator)
            # Each analysis starts with an empty engine hash to remove cross-candidate history.
            engine.configure({'Clear Hash': None})
            best = engine.analyse(board, chess.engine.Limit(nodes=analysis_nodes))
            engine.configure({'Clear Hash': None})
            chosen = engine.analyse(board, chess.engine.Limit(nodes=analysis_nodes), root_moves=[move])
            best_cp, chosen_cp = score(best, mover), score(chosen, mover)
            records.append({'ply': ply + 1, 'fen': board.fen(), 'row': row,
                            'uci': move.uci(), 'san': board.san(move), 'rank': rank,
                            'probability': probability, 'top3': policy[:3],
                            'best_cp': best_cp, 'chosen_cp': chosen_cp,
                            'loss_cp': max(0, best_cp - chosen_cp),
                            'engine_best': best['pv'][0].uci(),
                            'pv': [m.uci() for m in chosen.get('pv', [])]})
            records[-1] = annotate_material_loss(records[-1])
        else:
            engine.configure({'Clear Hash': None})
            move = engine.play(board, chess.engine.Limit(nodes=opponent_nodes)).move
        if move not in board.legal_moves:
            raise ValueError('Policy or opponent returned an illegal move')
        clocks[mover] -= 2.
        if clocks[mover] <= 0:
            game.headers['Result'] = '0-1' if mover else '1-0'
            game.headers['Termination'] = 'synthetic clock exhausted'
            break
        board.push(move)
        history.append(board.fen())
        node = node.add_variation(move)
        node.set_clock(clocks[mover])
    if game.headers['Result'] == '*':
        game.headers['Result'] = board.result(claim_draw=True)
        game.headers['Termination'] = 'game over' if board.is_game_over(claim_draw=True) else 'ply cap'
    game.headers['PlyCount'] = str(board.ply())
    return game, records


def summarize(records):
    count = len(records)
    return {'ai_moves': count,
            'direct_material_loss': sum(bool(r.get('direct_material_loss')) for r in records),
            'direct_material_loss_evaluated': sum('direct_material_loss' in r for r in records),
            **{f'loss_ge_{threshold}cp': sum(r['loss_cp'] >= threshold for r in records)
                                 for threshold in (200, 500)},
            'mean_loss_cp': sum(r['loss_cp'] for r in records) / count if count else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stockfish', default='stockfish')
    parser.add_argument('--games', type=int, default=12)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--max-plies', type=int, default=100)
    parser.add_argument('--elo-self', type=int, default=680)
    parser.add_argument('--elo-oppo', type=int, default=680)
    parser.add_argument('--opponent-nodes', type=int, default=10000)
    parser.add_argument('--analysis-nodes', type=int, default=30000)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output exists; choose a fresh directory')
    if not 1 <= args.games <= 100 or not 1 <= args.max_plies <= 300:
        parser.error('Games must be 1..100 and max plies 1..300')
    if args.opponent_nodes < 1 or args.analysis_nodes < 1:
        parser.error('Node budgets must be positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = load_model(args.checkpoint, 'cpu')
    args.output.mkdir(parents=True)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    all_records, games, summaries = [], [], []
    report = {'checkpoint': str(args.checkpoint),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'settings': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              'engine': engine.id, 'engine_options': {'Threads': 1, 'Hash': 64},
              'limitations': ['Synthetic 2-second clocks isolate move policy; timing head is not exercised.',
                              'Finite-budget Stockfish centipawn losses are diagnostic, not human blunder labels.',
                              'Direct material loss requires loss >=200cp and the PV reply capture net >=3 material points after same-square recaptures; not exhaustive or human-validated.',
                              'Same seeds reproduce sampling only with the same checkpoint and software; trajectories diverge across policies.'],
              'games': summaries}
    try:
        engine.configure({'Threads': 1, 'Hash': 64})
        for index in range(args.games):
            game, records = play_game(model, engine, seed=args.seed + index, color=index % 2 == 0,
                                      elo_self=args.elo_self, elo_oppo=args.elo_oppo,
                                      max_plies=args.max_plies, opponent_nodes=args.opponent_nodes,
                                      analysis_nodes=args.analysis_nodes)
            for record in records:
                record['game'] = index + 1
            all_records.extend(records)
            games.append(str(game))
            summary = {'game': index + 1, 'seed': args.seed + index,
                       'color': 'white' if index % 2 == 0 else 'black',
                       'result': game.headers['Result'], 'termination': game.headers['Termination'],
                       'plies': int(game.headers['PlyCount']), **summarize(records)}
            summaries.append(summary)
            report['all'] = summarize(all_records)
            (args.output / 'games.pgn').write_text('\n\n'.join(games) + '\n')
            (args.output / 'moves.json').write_text(json.dumps(all_records, indent=2) + '\n')
            (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(summary), flush=True)
    finally:
        engine.quit()
    return report


if __name__ == '__main__':
    main()

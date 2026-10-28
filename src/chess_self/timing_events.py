"""Explicit terminal-event classifier; censored turns never become duration targets."""
from __future__ import annotations

import torch
from torch import nn
from .timing_cache import read_rows, fresh_directory, write_json
from .timing_data import feature_names, feature_vector

EVENTS = {'completed': 0, 'timeout': 1}


def event_data(rows):
    selected = []
    rejected = 0
    for row in rows:
        if row.get('event_type') not in EVENTS or row.get('event_verified') is not True:
            rejected += 1
            continue
        if row['remaining_seconds'] <= 0 or not row.get('game_id'):
            raise ValueError('Event requires known positive pre-turn budget and game identity')
        if row['event_type'] == 'timeout' and row.get('elapsed_seconds') is not None:
            raise ValueError('A timeout is censored, not an exact elapsed-time target')
        selected.append(row)
    if not selected or len({r['event_type'] for r in selected}) < 2:
        raise ValueError('Need verified completed and timeout turns; ambiguous endings cannot supply event labels')
    groups = ['clock', 'phase', 'chess', 'history_summary']
    x = torch.tensor([feature_vector(r, groups) for r in selected], dtype=torch.float32)
    y = torch.tensor([EVENTS[r['event_type']] for r in selected], dtype=torch.float32)
    return selected, x, y, rejected


def fit_events(training_path, validation_path, output, *, epochs=100, learning_rate=.01, seed=42):
    train, x, y, excluded = event_data(read_rows(training_path))
    val, vx, vy, val_excluded = event_data(read_rows(validation_path))
    if {r['game_id'] for r in train} & {r['game_id'] for r in val}:
        raise ValueError('Event train/validation games overlap')
    if epochs < 1 or learning_rate <= 0: raise ValueError('Invalid event optimization settings')
    output = fresh_directory(output); torch.manual_seed(seed)
    model = nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        loss = nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        logits = model(vx).squeeze(1); probabilities = logits.sigmoid()
        report = {'architecture': 'timing-timeout-event-v1', 'seed': seed, 'epochs': epochs,
                  'validation_nll': float(nn.functional.binary_cross_entropy_with_logits(logits, vy)),
                  'validation_brier': float(((probabilities - vy) ** 2).mean()),
                  'predicted_timeout_rate': float(probabilities.mean()), 'observed_timeout_rate': float(vy.mean()),
                  'excluded_ambiguous_training_turns': excluded, 'excluded_ambiguous_validation_turns': val_excluded,
                  'test_evaluated': False, 'duration_model': 'separate completed-move law; no intended-time labels inferred',
                  'rows': [{'game_id': r['game_id'], 'probability': float(p), 'event': r['event_type']} for r,p in zip(val, probabilities)]}
    torch.save({'architecture': report['architecture'], 'state_dict': model.state_dict(),
                'features': feature_names(['clock', 'phase', 'chess', 'history_summary'])}, output / 'event.pt')
    write_json(output / 'run.json', report)
    return report

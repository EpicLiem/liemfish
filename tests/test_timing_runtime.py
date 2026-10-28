import chess
import pytest

from chess_self.timing_runtime import TurnScheduler, execution_budget, recorded_replay, virtual_game


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def test_scheduler_counts_inference_and_only_adds_increment_after_commit():
    clock = Clock()
    scheduler = TurnScheduler(clock)
    token = scheduler.begin(chess.STARTING_FEN, 10)
    scheduler.inference_started(token)
    clock.value = 2
    plan = scheduler.prediction_ready(token, 'e2e4', 5)
    assert plan['wait_seconds'] == 3
    clock.value = 5
    event = scheduler.commit(token, chess.STARTING_FEN, increment_seconds=2)
    assert event['charged_seconds'] == 5
    assert event['post_clock_seconds'] == 7
    assert event['inference_end'] == 2
    assert event['commit_time'] == 5


def test_scheduler_logs_latency_floor_and_recorded_charge_separately():
    clock = Clock()
    scheduler = TurnScheduler(clock)
    token = scheduler.begin(chess.STARTING_FEN, 10)
    scheduler.inference_started(token)
    clock.value = 2
    assert scheduler.prediction_ready(token, 'e2e4', .1)['wait_seconds'] == 0
    event = scheduler.commit(token, chess.STARTING_FEN, charged_seconds=.1)
    assert event['visible_delay_seconds'] == 2
    assert event['charged_seconds'] == .1
    assert event['target_unattainable']


def test_scheduler_cancels_takebacks_and_terminal_games():
    scheduler = TurnScheduler(Clock())
    token = scheduler.begin(chess.STARTING_FEN, 10)
    scheduler.takeback()
    with pytest.raises(ValueError, match='Stale'):
        scheduler.prediction_ready(token, 'e2e4', 1)
    scheduler.end_game()
    with pytest.raises(RuntimeError, match='ended'):
        scheduler.begin(chess.STARTING_FEN, 10)


def test_scheduler_rejects_changed_board_and_early_commit():
    clock = Clock()
    scheduler = TurnScheduler(clock)
    token = scheduler.begin(chess.STARTING_FEN, 10)
    scheduler.inference_started(token)
    scheduler.prediction_ready(token, 'e2e4', 1)
    with pytest.raises(ValueError, match='not elapsed'):
        scheduler.commit(token, chess.STARTING_FEN)
    clock.value = 1
    with pytest.raises(ValueError, match='Position changed'):
        scheduler.commit(token, chess.Board().copy().fen().replace(' w ', ' b '))
    assert scheduler.active is None


def test_exhausted_budget_and_inference_overrun_never_earn_increment():
    clock = Clock()
    scheduler = TurnScheduler(clock, reserve_seconds=.1)
    token = scheduler.begin(chess.STARTING_FEN, 1)
    scheduler.inference_started(token)
    clock.value = 1.1
    assert scheduler.prediction_ready(token, 'e2e4', .2)['status'] == 'budget_exhausted'
    event = scheduler.commit(token, chess.STARTING_FEN, increment_seconds=3)
    assert event['status'] == 'budget_exhausted'
    assert event['post_clock_seconds'] == 0
    assert execution_budget(0, 0) == 0
    assert execution_budget(.1, .2) == 0
    assert 0 < execution_budget(1, 0) < 1


def test_replay_removes_human_move_labels_and_termination():
    def predict(board, row):
        assert not {'elapsed_seconds', 'move_uci', 'termination', 'observation_interval_seconds'} & row.keys()
        return {'move_uci': 'e2e4', 'target_seconds': 2}
    rows = [dict(fen=chess.STARTING_FEN, history_fens=[chess.STARTING_FEN], remaining_seconds=10,
                 elapsed_seconds=3, move_uci='d2d4', termination='resigned', ply=1)]
    report = recorded_replay(rows, predict)
    assert report['events'][0]['observed_seconds'] == 3
    assert report['mode'] == 'recorded_state_replay'


def test_virtual_game_uses_generated_board_clocks_and_history():
    seen = []
    moves = iter(['e2e4', 'e7e5', 'g1f3'])
    def predict(board, row):
        seen.append(row)
        return {'move_uci': next(moves), 'target_seconds': 2, 'inference_seconds': .1}
    report = virtual_game(predict, 10, 1, max_plies=3)
    assert seen[1]['opponent_remaining_seconds'] == 9
    assert seen[1]['opponent_time_history'] == [2]
    assert seen[2]['remaining_seconds'] == 9
    assert seen[2]['own_time_history'] == [2]
    assert len(seen[2]['history_fens']) == 3
    assert seen[2]['fen'] == chess.Board('rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2').fen()
    assert report['final_clocks'] == {'white': 8, 'black': 9}
    assert report['summary']['completed_moves'] == 3


def test_virtual_unconstrained_timeout_does_not_apply_increment_or_push_move():
    report = virtual_game(lambda b, r: {'move_uci': 'e2e4', 'target_seconds': 2}, 1, 5, guard=False)
    assert report['termination'] == 'flagged'
    assert report['final_fen'] == chess.STARTING_FEN
    assert report['final_clocks']['white'] == 0
    assert report['summary']['flags'] == 1


def test_virtual_latency_and_first_move_rules_are_explicit():
    predictor = lambda b, r: {'move_uci': 'e2e4', 'target_seconds': .1, 'inference_seconds': 1}
    target = virtual_game(predictor, 10, max_plies=1)
    wall = virtual_game(predictor, 10, max_plies=1, clock_charge_mode='wall')
    free = virtual_game(predictor, 10, 1, max_plies=1, first_move_rule='free_first_turn')
    assert target['final_clocks']['white'] == 9.9
    assert wall['final_clocks']['white'] == 9
    assert free['final_clocks']['white'] == 11
    assert target['events'][0]['visible_delay_seconds'] == 1
    assert target['summary']['latency_floor_misses'] == 1


def test_zero_budget_is_not_renormalized_into_a_positive_duration():
    report = virtual_game(lambda b, r: {'move_uci': 'e2e4', 'target_seconds': 1}, .1, reserve_seconds=.1)
    assert report['termination'] == 'no_feasible_budget'
    assert report['summary']['completed_moves'] == 0


@pytest.mark.parametrize('value', [-1, float('inf'), float('nan')])
def test_invalid_clock_rejected(value):
    with pytest.raises(ValueError):
        execution_budget(value)


def test_exhausted_simulation_does_not_invoke_model_with_empty_support():
    def forbidden(board, row):
        raise AssertionError('Do not normalize an empty budget')
    assert virtual_game(forbidden, 0)['termination'] == 'flagged'
    assert virtual_game(forbidden, 1, reserve_seconds=1)['termination'] == 'no_feasible_budget'
    report = recorded_replay([{'fen': chess.STARTING_FEN, 'remaining_seconds': 0}], forbidden)
    assert report['summary']['flags'] == 1


def test_predictor_mutation_cannot_change_recorded_history():
    history = [chess.STARTING_FEN]
    def predict(board, row):
        row['history_fens'].clear()
        return {'move_uci': 'e2e4', 'target_seconds': 0}
    recorded_replay([{'fen': chess.STARTING_FEN, 'history_fens': history, 'remaining_seconds': 1}], predict)
    assert history == [chess.STARTING_FEN]


def test_illegal_moves_rejected_before_simulated_clock_updates():
    with pytest.raises(ValueError, match='illegal'):
        virtual_game(lambda b, r: {'move_uci': 'e2e5', 'target_seconds': 1}, 10)

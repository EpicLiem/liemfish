import json
from types import SimpleNamespace

import chess
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from chess_self import export
from chess_self.model import MOVES, PersonalModel, config, encode


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    cfg = vars(config()).copy()
    cfg.update(dim_vit=32, dim_emb=8, head_hid_dim=16, num_blocks=1,
               num_heads=4, gab_gen_size=8, gab_per_square_dim=4, gab_intermediate_dim=8)
    torch.manual_seed(31)
    model = PersonalModel(SimpleNamespace(**cfg)).eval()
    directory = tmp_path_factory.mktemp('onnx')
    checkpoint = directory / 'tiny.pt'
    torch.save(model.state_dict(), checkpoint)
    output = directory / 'tiny.onnx'
    with pytest.MonkeyPatch.context() as patch:
        def load(path, device):
            restored = PersonalModel(SimpleNamespace(**cfg))
            restored.load_state_dict(torch.load(path, weights_only=True))
            return restored.to(device).eval()
        patch.setattr(export, 'load_model', load)
        report = export.export_model(checkpoint, output)
    return model, output, report


def test_real_onnx_runtime_parity(exported):
    model, output, report = exported
    assert report['parity']['passed']
    assert len(report['parity']['cases']) == 9
    assert {case['batch_size'] for case in report['parity']['cases']} == {1, 2}
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    assert all(x.data_type == onnx.TensorProto.FLOAT for x in graph.graph.initializer)
    assert json.loads(output.with_suffix('.moves.json').read_text()) == MOVES
    assert json.loads(output.with_suffix('.json').read_text()) == report
    assert any(isinstance(module, torch.nn.RMSNorm) for module in model.modules())


def test_probability_and_expectation_contract(exported):
    _, output, report = exported
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    inputs = export._inputs(export.parity_rows()[-2:])
    moves, logits, probabilities, seconds = session.run(
        None, {name: tensor.numpy() for name, tensor in zip(export.INPUT_NAMES, inputs)})
    assert moves.shape == (2, 4352)
    assert logits.shape == probabilities.shape == (2, 30)
    assert seconds.shape == (2,)
    np.testing.assert_allclose(probabilities.sum(-1), 1, atol=1e-6)
    np.testing.assert_allclose(probabilities, torch.tensor(logits).softmax(-1).numpy(), atol=1e-6)
    np.testing.assert_allclose(seconds, probabilities @ np.array(report['time_bin_centers_seconds']), rtol=1e-6)


def test_history_clocks_and_elapsed_are_encoded_correctly():
    for row in export.parity_rows():
        tokens = encode(row)['tokens']
        assert tokens.shape == (64, 99)
        altered = dict(row, elapsed_seconds=42)
        torch.testing.assert_close(tokens, encode(altered)['tokens'])
        torch.testing.assert_close(tokens[:, 96:], torch.tensor(
            [row['initial_seconds'], row['increment_seconds'], row['remaining_seconds']]
        ).repeat(64, 1) / 100)
        if len(row['history_fens']) == 8:
            assert not torch.equal(tokens[:, :12], tokens[:, 84:96])
        if len(row['history_fens']) == 1:
            torch.testing.assert_close(tokens[:, :12], tokens[:, 84:96])
        board = chess.Board(row['fen'])
        if 'promotion' in row['name']:
            assert any(move.promotion for move in board.legal_moves)


def test_checkpoint_cannot_be_overwritten(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'preserve')
    with pytest.raises(ValueError, match='different files'):
        export.export_model(checkpoint, checkpoint)
    assert checkpoint.read_bytes() == b'preserve'

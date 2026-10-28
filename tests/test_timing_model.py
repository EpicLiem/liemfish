from dataclasses import replace

import chess
import pytest
import torch
from torch import nn

from chess_self.model import MOVE_INDEX, MOVES
from chess_self.timing_model import (MaiaFeatures, TimingHead, TimingHeadConfig,
                                    history_tensor, load_timing_head, save_timing_head,
                                    selected_move_inputs)
from chess_self.timing_export import sample_inputs


def test_absolute_residual_preserves_base_and_trains():
    head = TimingHead(TimingHeadConfig(side_dim=3, pooled_dim=8))
    inputs = sample_inputs(head.config)
    result = head(*inputs)
    assert torch.equal(result, inputs[1])
    result[:, 0].sum().backward()
    assert head.output.weight.grad.abs().sum() > 0


def test_new_support_does_not_relabel_old_logits():
    head = TimingHead(TimingHeadConfig(side_dim=2, pooled_dim=8, output_family='extended', output_bins=40))
    inputs = sample_inputs(head.config)
    output = head(*inputs)
    torch.testing.assert_close(output[:, :30].softmax(-1), inputs[1].softmax(-1))
    assert torch.equal(output[:, 30:], torch.full((2, 10), -12.))
    for family in ('fraction', 'beta'):
        head = TimingHead(TimingHeadConfig(side_dim=2, pooled_dim=8, output_family=family))
        inputs = sample_inputs(head.config)
        assert torch.equal(head(*inputs), head(inputs[0], inputs[1]+100, *inputs[2:]))
        assert head(*inputs).shape[-1] == (2 if family == 'beta' else 30)


def test_extended_prior_is_offset_invariant_with_fixed_initial_tail_mass():
    import math
    head = TimingHead(TimingHeadConfig(side_dim=2, pooled_dim=8, output_family='extended', output_bins=40)).double()
    inputs = [v.double() if v.is_floating_point() else v for v in sample_inputs(head.config)]
    expected = head(*inputs).softmax(-1)
    for offset in (-1000., 1000.):
        shifted = head(inputs[0], inputs[1] + offset, *inputs[2:]).softmax(-1)
        torch.testing.assert_close(shifted, expected, rtol=1e-12, atol=1e-14)
    expected_mass = 10 * math.exp(-12) / (1 + 10 * math.exp(-12))
    torch.testing.assert_close(expected[:, 30:].sum(-1), torch.full((2,), expected_mass, dtype=torch.float64))
    nn.init.normal_(head.output.weight)
    torch.testing.assert_close(head(*inputs), head(inputs[0], inputs[1]+1000., *inputs[2:]), rtol=1e-12, atol=1e-12)


def test_extended_checkpoint_requires_prior_semantics(tmp_path):
    path = tmp_path/'extended.pt'
    head = TimingHead(TimingHeadConfig(side_dim=2, pooled_dim=8, output_family='extended', output_bins=40))
    save_timing_head(path, head, base_checkpoint_sha256='abc', feature_names=['a', 'b'])
    loaded, meta = load_timing_head(path)
    inputs = sample_inputs(head.config)
    torch.testing.assert_close(head(*inputs), loaded(*inputs))
    del meta['extended_prior']
    torch.save(meta, path)
    with pytest.raises(ValueError, match='Legacy extended timing prior'):
        load_timing_head(path)


def test_refit_control_uses_pooled_features_without_side_information():
    import json
    from pathlib import Path
    config = json.loads((Path(__file__).parents[1]/'configs/timing/e03-refit-control.json').read_text())
    assert config['features'] == []
    assert config['interval_labels'] and config['sampling'] == 'natural'
    head = TimingHead(TimingHeadConfig(side_dim=0, **config['head']))
    assert head.hidden.in_features == 512 and head.hidden.out_features == 64
    inputs = sample_inputs(head.config)
    assert torch.equal(head(*inputs), inputs[1])
    head(*inputs)[:, 0].sum().backward()
    assert head.output.weight.grad.abs().sum() > 0


def test_move_conditioned_square_selection_and_history():
    cfg = TimingHeadConfig(side_dim=2, pooled_dim=8, move_mode='squares', history_mode='gru')
    head = TimingHead(cfg)
    nn.init.normal_(head.output.weight)
    inputs = list(sample_inputs(cfg))
    before = head(*inputs)
    inputs[4] = (inputs[4] + 1) % 64
    assert not torch.equal(before, head(*inputs))
    with pytest.raises(ValueError, match='attributes'):
        head(inputs[0], inputs[1], inputs[2])


def test_black_square_orientation_and_selected_move_surprisal():
    board = chess.Board()
    board.push_uci('e2e4')
    row = {'fen': board.fen(), 'move_uci': 'e7e5'}
    logits = torch.zeros(1, len(MOVES))
    indices, attrs = selected_move_inputs([row], logits)
    assert indices.tolist() == [[chess.E2, chess.E4]]
    assert attrs[0, -1] == pytest.approx(torch.log(torch.tensor(20.)).item())
    assert attrs[0, :7].sum() == 0
    with pytest.raises(ValueError, match='illegal'):
        selected_move_inputs([row], logits, ['e2e4'])


def test_promotion_attributes():
    row={'fen': '7k/P7/8/8/8/8/8/4K3 w - - 0 1', 'move_uci': 'a7a8q'}
    _, attrs = selected_move_inputs([row], torch.zeros(1, len(MOVES)))
    assert attrs[0,:4].tolist() == [1,0,0,0]
    assert attrs[0,5] == 1


def test_history_mask_padding_and_labels_ignored():
    rows=[{'own_time_history':[1.,None,3.], 'opponent_time_history':[2.], 'elapsed_seconds':999}]
    h=history_tensor(rows, length=4)
    assert h.shape == (1,4,4)
    assert h[0,:,2].tolist() == [0,1,0,1]
    assert h[0,:,3].tolist() == [0,0,0,1]
    rows[0]['elapsed_seconds']=-999
    assert torch.equal(h, history_tensor(rows, length=4))


def test_checkpoint_contract(tmp_path):
    path=tmp_path/'head.pt';head=TimingHead(TimingHeadConfig(side_dim=1, pooled_dim=8))
    save_timing_head(path,head,base_checkpoint_sha256='abc',feature_names=['clock'])
    loaded,meta=load_timing_head(path,base_checkpoint_sha256='abc',feature_names=['clock'])
    inputs=sample_inputs(head.config)
    assert torch.equal(head(*inputs),loaded(*inputs))
    with pytest.raises(ValueError,match='another backbone'):
        load_timing_head(path,base_checkpoint_sha256='other')
    with pytest.raises(ValueError,match='schema'):
        load_timing_head(path,feature_names=['label'])


class Trunk(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([nn.Linear(8,8) for _ in range(3)])
    def forward(self,x):
        for layer in self.layers:x=layer(x)
        return x


class ToyMaia(nn.Module):
    def __init__(self):
        super().__init__();self.transformer=Trunk();self.last_ln=nn.LayerNorm(8)
        self.move=nn.Linear(8,4);self.time=nn.Linear(8,30)
    def forward(self,tokens,own,opp):
        x=self.transformer(tokens);pooled=self.last_ln(x.mean(1))
        return self.move(x[:,0]),self.time(pooled),pooled[:,0]


def test_frozen_features_and_limited_backbone_gradient():
    model=ToyMaia();tokens=torch.randn(2,64,8);ratings=torch.zeros(2)
    expected=model(tokens,ratings,ratings)
    wrapper=MaiaFeatures(model)
    features=wrapper(tokens,ratings,ratings)
    assert torch.equal(features['move_logits'],expected[0])
    assert features['squares'].shape == (2,64,8)
    assert not features['pooled'].requires_grad
    wrapper.close()
    wrapper=MaiaFeatures(model,final_blocks=1)
    features=wrapper(tokens,ratings,ratings)
    features['base_logits'].sum().backward()
    assert model.transformer.layers[-1].weight.grad is not None
    assert model.transformer.layers[0].weight.grad is None
    assert model.time.weight.grad is None
    head=TimingHead(TimingHeadConfig(side_dim=1,pooled_dim=8))
    assert len(wrapper.parameter_groups(head,1e-3,1e-5))==2
    with pytest.raises(ValueError,match='lower'):
        wrapper.parameter_groups(head,1e-3,1e-3)
    wrapper.close()

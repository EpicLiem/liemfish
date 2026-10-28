import pytest
import torch

from chess_self.timing_model import TimingHead, TimingHeadConfig, save_timing_head
from chess_self.timing_export import export_timing_head


@pytest.mark.parametrize('family,move,history', [('absolute','none','none'), ('extended','attributes','none'), ('fraction','squares','none'), ('beta','squares','gru')])
def test_side_head_export_parity(tmp_path,family,move,history):
    pytest.importorskip('onnxruntime')
    head=TimingHead(TimingHeadConfig(side_dim=3,pooled_dim=8,output_family=family,
                                   output_bins=40 if family=='extended' else 30,
                                   move_mode=move,history_mode=history))
    # Test learned computation too, not just the identity residual initialization.
    torch.nn.init.normal_(head.output.weight,std=.05)
    checkpoint=tmp_path/'head.pt'
    save_timing_head(checkpoint,head,base_checkpoint_sha256='example',feature_names=['a','b','c'])
    report=export_timing_head(checkpoint,tmp_path/'head.onnx')
    assert report['parity']['passed']
    assert report['browser_execution_verified'] is False
    assert [x['batch'] for x in report['parity']['cases']]==[1,2,5]
    with pytest.raises(ValueError,match='fresh'):
        export_timing_head(checkpoint,tmp_path/'head.onnx')


def test_feature_backbone_export_preserves_native_representations(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import onnxruntime as ort
    from chess_self import model as source_model
    from chess_self import timing_export
    from chess_self.export import _inputs, parity_rows
    from chess_self.model import PersonalModel, config, MOVES
    from chess_self.timing_model import MaiaFeatures

    cfg=vars(config()).copy()
    cfg.update(dim_vit=32, dim_emb=8, head_hid_dim=16, num_blocks=1,
               num_heads=4, gab_gen_size=8, gab_per_square_dim=4, gab_intermediate_dim=8)
    torch.manual_seed(83)
    native=PersonalModel(SimpleNamespace(**cfg)).eval()
    checkpoint=tmp_path/'tiny.pt'
    torch.save(native.state_dict(),checkpoint)
    def load(path,device='cpu'):
        m=PersonalModel(SimpleNamespace(**cfg))
        m.load_state_dict(torch.load(path,weights_only=True),strict=True)
        return m.to(device).eval()
    monkeypatch.setattr(source_model,'load_model',load)
    output=tmp_path/'features.onnx'
    report=timing_export.export_maia_features(checkpoint,output)
    assert report['parity']['passed']
    assert len(report['parity']['cases'])==9
    assert report['outputs']['pooled']==['batch',32]
    assert report['browser_execution_verified'] is False
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(output),sess_options=options,providers=['CPUExecutionProvider'])
    inputs=_inputs(parity_rows()[-2:])
    actual=session.run(None,dict(zip(timing_export.BACKBONE_INPUT_NAMES,[x.numpy() for x in inputs])))
    wrapped=MaiaFeatures(native)
    with torch.no_grad():
        expected=wrapped(*inputs)
    for name,value in zip(timing_export.BACKBONE_OUTPUT_NAMES,actual):
        torch.testing.assert_close(torch.tensor(value),expected[name],atol=2e-4,rtol=2e-4)
    wrapped.close()
    with pytest.raises(ValueError,match='fresh'):
        timing_export.export_maia_features(checkpoint,output)
    with pytest.raises(ValueError,match='different'):
        timing_export.export_maia_features(checkpoint,checkpoint)

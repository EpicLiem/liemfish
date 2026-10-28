# Timing side-head contract, version 1

`timing_model.py` implements timing heads over the representations produced by `PersonalModel`.

## Representations and training boundaries

`MaiaFeatures(PersonalModel, final_blocks=0)` returns `pooled [B,512]`, `squares [B,64,512]`, `base_logits [B,30]` and `move_logits [B,4352]`. Hooks observe the original transformer output and the final pooled normalization. The unchanged model still computes its normal outputs. A feature extractor must not be shared across concurrent forward calls.

All original parameters are frozen by default and the backbone stays in evaluation mode, including when the side head trains. `final_blocks=1` or `2` exposes only those final transformer layers for adaptation. Use `parameter_groups(head, head_lr, backbone_lr)` with a lower positive backbone learning rate. Updating shared blocks can change policy predictions even though move projections stay frozen. In that case the caller must save the adapted backbone as well as the timing head, use its hash in the timing checkpoint, and rerun move evaluations. Cached features cannot be used to train shared blocks.

## Side-head inputs

The forward order is:

1. `pooled`: normalized pooled Maia representation, float32 `[B,512]`.
2. `base_logits`: original timing logits, float32 `[B,30]`.
3. `side_features`: causal external features, float32 `[B,side_dim]`, in checkpoint `feature_names` order. Their missingness masks and normalization belong to the versioned feature builder.
4. `squares`: transformer square representations, float32 `[B,64,512]`; required only for square conditioning.
5. `move_indices`: int64 `[B,2]`, source then destination in Maia's current side-to-move orientation.
6. `move_attributes`: float32 `[B,8]`: promotion queen/rook/bishop/knight, capture, gives check, castling, and negative log legal-policy probability.
7. `history`: float32 `[B,8,4]` by default. Channels are log1p prior own duration, log1p prior opponent duration, own observed mask, opponent observed mask. Oldest to newest; pad left with zero values and zero masks. Missing measurements remain masked gaps, not guessed times. The two player windows describe completed own/opponent turns, not simultaneous events.

`selected_move_inputs(rows, policy_logits, moves=...)` constructs selected-move inputs and checks legality. Omitting `moves` uses supervised human targets. That option is appropriate for conditional training/scoring, not an unconditional validation win or generation. During play supply the move actually selected. Equal-information timing comparisons must marginalize the conditional distribution over **all legal moves under the same policy**, not select the human move or take a top-k approximation silently. The helper preserves gradients through policy surprisal for shared-block experiments.

`history_tensor` reads only explicit prior-history fields; it never reads current elapsed time. Fixed-window and recurrent arms should receive the same length. History values used in virtual play must come from generated completed turns.

## Outputs and initialization

The architecture identifier is `maia3-timing-side-v1`.

- `absolute`, 30 logits: a residual correction to the unchanged base logits. The residual output layer is zero-initialized, so initial probabilities match exactly.
- `extended`, 40 logits: first 30 retain base logits plus a zero residual; ten new tail logits start at bias -12. The larger softmax denominator changes original probabilities. The distribution module defines matching first-30 boundaries and tail edges.
- `fraction`, 30 logits: newly initialized replacement outputs, with no semantic reuse of absolute-time logits.
- `beta`, 2 raw parameters: newly initialized outputs; the distribution decoder applies `softplus + 0.05` to obtain alpha/beta. No seconds logits are relabeled as Beta parameters.

`use_pooled=False` permits a compact tabular model. `move_mode='attributes'` adds only the eight move descriptors; `'squares'` adds descriptors plus both square vectors. `history_mode='gru'` adds a small recurrent encoder. Summary and fixed-window history arms use the ordinary side-feature input.

## Checkpoints and export

`save_timing_head(path, head, base_checkpoint_sha256=..., feature_names=..., extra=...)` stores config, architecture, ordered feature names, base hash, strict state dict and caller metadata. `load_timing_head(..., base_checkpoint_sha256=..., feature_names=...)` rejects mismatches. Pass these expected values when integrating with caches or deployment. New module names and initialization policy are recorded.

`export_timing_head(checkpoint, output)` exports the timing **side head only**, with raw logits/parameters as output. ONNX prunes inputs unused by the configured arm; the adjacent JSON reports the actual input names and shapes. It verifies CPU ONNX Runtime parity at batch sizes 1, 2 and 5 against the native head, including nonzero learned residual weights in tests. `export_maia_features(checkpoint, output)` supplies the matching feature-producing backbone path. It exports `tokens`, `self_elos`, and `oppo_elos` to `move_logits`, `base_logits`, `pooled`, and `squares`, and records architecture `maia3-timing-features-v1`. Its checkpoint SHA must match the side head. Move logits are unmasked; the consumer applies legal-move masking before sampling or calculating surprisal. Representation parity is checked against the original native model over nine cases covering white/black, padded/full histories, promotions, low clocks and batches one/two. RMSNorm decomposition is export-only and restored before native comparison. This establishes both ONNX components, not an end-to-end browser runtime.

The consumer must use the versioned output-family decoder, interval likelihood and feasible sampler. It must not add the next increment to the pre-move duration budget. Scheduler rules, missing-value semantics, learned feature order and the actual selected move are part of the runtime contract. ONNX CPU parity does not certify browser WASM/WebGPU execution or latency.

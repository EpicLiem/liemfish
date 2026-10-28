import { Chess } from 'chess.js';
import { moveIndex, type ModelPosition } from './model-encoding';

const known = (value: number | null): value is number => value !== null && Number.isFinite(value) && value >= 0;

export function timingFeatures(position: ModelPosition): Float32Array {
  const board = new Chess(position.fen);
  const base = position.initialSeconds;
  const increment = position.incrementSeconds;
  const own = position.remainingSeconds;
  const opponentKnown = known(position.opponentRemainingSeconds);
  const opponent = opponentKnown ? position.opponentRemainingSeconds! : 0;
  const pieces = board.board().flat().filter(piece => piece !== null);
  const material = pieces.reduce((sum, piece) => sum + ({ p: 0, n: 3, b: 3, r: 5, q: 9, k: 0 })[piece.type], 0);
  const legal = board.moves({ verbose: true });
  const features = [
    Math.log1p(base), Math.log1p(increment), Math.log1p(own), Math.log1p(opponent),
    1, Number(opponentKnown), own / Math.max(base, 1), opponent / Math.max(base, 1),
    opponentKnown ? (own - opponent) / Math.max(base, 1) : 0, increment / Math.max(own, 1),
    Number(position.fen.split(' ')[5]) / 100, pieces.length / 32, material / 62,
    Number(board.isCheck()), Math.log1p(legal.length), Number(legal.length === 1),
    Number(legal.some(move => move.isCapture() || move.isEnPassant())), Number(position.previousMoveWasCapture),
  ];
  for (const times of [position.ownTimeHistory, position.opponentTimeHistory]) {
    const history = times.slice(-8);
    while (history.length < 8) history.unshift(null);
    for (const duration of history) features.push(known(duration) ? Math.log1p(duration) : 0, Number(known(duration)));
  }
  if (features.some(value => !Number.isFinite(value))) throw new Error('Nonfinite timing feature.');
  return Float32Array.from(features);
}

export function selectedMoveInputs(fen: string, uci: string, logits: Float32Array) {
  const board = new Chess(fen);
  const black = board.turn() === 'b';
  const legal = board.moves({ verbose: true });
  const selected = legal.find(move => move.from + move.to + (move.promotion ?? '') === uci);
  if (!selected) throw new Error('Timing requested for an illegal move.');
  const scores = legal.map(move => logits[moveIndex(move.from + move.to + (move.promotion ?? ''), black)]);
  const max = Math.max(...scores);
  const logSum = max + Math.log(scores.reduce((sum, score) => sum + Math.exp(score - max), 0));
  const orient = (square: string) => BigInt(square.charCodeAt(0) - 97 + (black ? 8 - Number(square[1]) : Number(square[1]) - 1) * 8);
  const indices = BigInt64Array.of(orient(selected.from), orient(selected.to));
  const capture = selected.isCapture() || selected.isEnPassant();
  const castling = selected.isKingsideCastle() || selected.isQueensideCastle();
  board.move(selected);
  const attributes = Float32Array.from([
    ...Array.from('qrbn', piece => Number(selected.promotion === piece)),
    Number(capture), Number(board.isCheck()), Number(castling), logSum - logits[moveIndex(uci, black)],
  ]);
  return { indices, attributes };
}

export function nextDown(value: number): number {
  if (!(value > 0) || !Number.isFinite(value)) throw new Error('Clock budget must be finite and positive.');
  const buffer = new ArrayBuffer(8);
  const view = new DataView(buffer);
  view.setFloat64(0, value);
  view.setBigUint64(0, view.getBigUint64(0) - BigInt(1));
  return view.getFloat64(0);
}

export function sampleTiming(logits: Float32Array, remainingSeconds: number, random = Math.random): number {
  if (logits.length !== 30 || Array.from(logits).some(value => !Number.isFinite(value))) throw new Error('Invalid timing model output.');
  const budget = nextDown(remainingSeconds);
  if (budget <= 0) throw new Error('No feasible duration remains.');
  const edges = Array.from({ length: 31 }, (_, index) => index === 0 ? 0 : 10 ** (-1 + index / 10) - 0.1);
  // Reweight partial bins before normalization; do not clamp a sampled duration.
  const masses = Array.from(logits, (logit, index) => {
    const lower = edges[index];
    const upper = Math.min(edges[index + 1], budget);
    if (upper <= lower) return -Infinity;
    const fraction = Math.log1p((upper - lower) / (lower + 0.1)) / Math.log1p((edges[index + 1] - lower) / (lower + 0.1));
    return logit + Math.log(fraction);
  });
  const max = Math.max(...masses);
  if (!Number.isFinite(max)) throw new Error('No probability in feasible duration support.');
  const weights = masses.map(mass => Math.exp(mass - max));
  let threshold = random() * weights.reduce((sum, weight) => sum + weight, 0);
  let bin = weights.length - 1;
  for (let index = 0; index < weights.length; index++) {
    threshold -= weights[index];
    if (threshold < 0) { bin = index; break; }
  }
  const lower = edges[bin];
  const upper = Math.min(edges[bin + 1], budget);
  const duration = lower + (lower + 0.1) * Math.expm1(random() * Math.log1p((upper - lower) / (lower + 0.1)));
  return Math.max(lower, Math.min(nextDown(upper), duration));
}

import { Chess } from 'chess.js';

export interface ModelPosition {
  fen: string;
  historyFens: string[];
  remainingSeconds: number;
  initialSeconds: number;
  incrementSeconds: number;
  eloSelf: number;
  eloOpponent: number;
  opponentRemainingSeconds: number | null;
  ownTimeHistory: (number | null)[];
  opponentTimeHistory: (number | null)[];
  previousMoveWasCapture: boolean;
}

export function encodePosition(position: ModelPosition): Float32Array {
  const { historyFens, fen } = position;
  if (!historyFens.length || historyFens[historyFens.length - 1] !== fen) {
    throw new Error('History must contain actual positions and end with the current FEN.');
  }
  const clocks = [position.initialSeconds, position.incrementSeconds, position.remainingSeconds];
  if ([...clocks, position.eloSelf, position.eloOpponent].some(value => !Number.isFinite(value) || value < 0)) {
    throw new Error('Clock and rating inputs must be finite nonnegative numbers.');
  }
  const history = historyFens.slice(-8);
  while (history.length < 8) history.unshift(history[0]);
  const tokens = new Float32Array(64 * 99);
  history.forEach((historicalFen, index) => {
    const board = new Chess(historicalFen);
    const mirror = board.turn() === 'b';
    for (const rank of board.board()) for (const piece of rank) {
      if (!piece) continue;
      const file = piece.square.charCodeAt(0) - 97;
      const originalRank = Number(piece.square[1]) - 1;
      const square = (mirror ? 7 - originalRank : originalRank) * 8 + file;
      const white = mirror ? piece.color === 'b' : piece.color === 'w';
      const channel = 'pnbrqk'.indexOf(piece.type) + (white ? 0 : 6);
      tokens[square * 99 + index * 12 + channel] = 1;
    }
  });
  for (let square = 0; square < 64; square++) {
    clocks.forEach((seconds, index) => { tokens[square * 99 + 96 + index] = seconds / 100; });
  }
  return tokens;
}

export function moveIndex(uci: string, black: boolean): number {
  const square = (file: string, rank: string) => (file.charCodeAt(0) - 97) + (black ? 8 - Number(rank) : Number(rank) - 1) * 8;
  if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uci)) throw new Error('Invalid UCI move.');
  if (uci.length === 5) {
    return 4096 + 32 * (uci.charCodeAt(0) - 97) + 4 * (uci.charCodeAt(2) - 97) + 'qrbn'.indexOf(uci[4]);
  }
  return 64 * square(uci[0], uci[1]) + square(uci[2], uci[3]);
}

export function sampleLegalMove(fen: string, logits: Float32Array, random = Math.random): string {
  if (logits.length !== 4352) throw new Error('Unexpected model move output.');
  const board = new Chess(fen);
  const moves = board.moves({ verbose: true }).map(move => move.from + move.to + (move.promotion ?? ''));
  if (!moves.length) throw new Error('This position has no legal moves.');
  const scores = moves.map(move => logits[moveIndex(move, board.turn() === 'b')]);
  if (scores.some(score => !Number.isFinite(score))) throw new Error('Model returned invalid legal move scores.');
  const max = Math.max(...scores);
  const weights = scores.map(score => Math.exp(score - max));
  let threshold = random() * weights.reduce((sum, weight) => sum + weight, 0);
  for (let index = 0; index < moves.length; index++) {
    threshold -= weights[index];
    if (threshold < 0) return moves[index];
  }
  return moves[moves.length - 1];
}

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Chess } from 'chess.js';
import { encodePosition, moveIndex, sampleLegalMove } from './model-encoding';
const defaults = { opponentRemainingSeconds: 300, ownTimeHistory: [], opponentTimeHistory: [], previousMoveWasCapture: false, initialSeconds: 300, incrementSeconds: 2, remainingSeconds: 125, eloSelf: 1500, eloOpponent: 1600 };

test('all 4352 vocabulary entries match the exported model for both orientations', () => {
  const vocabulary = JSON.parse(readFileSync(new URL('../../../models/export/personalized-timing-features.moves.json', import.meta.url), 'utf8'));
  const moves: string[] = Array.isArray(vocabulary) ? vocabulary : vocabulary.moves;
  assert.equal(moves.length, 4352);
  moves.forEach((move, index) => {
    assert.equal(moveIndex(move, false), index);
    const mirrored = move[0] + (9 - Number(move[1])) + move[2] + (9 - Number(move[3])) + move.slice(4);
    assert.equal(moveIndex(mirrored, true), index);
  });
});

test('history pads earliest board and orients each actual position independently', () => {
  const board = new Chess();
  const start = board.fen();
  board.move('e4');
  const after = board.fen();
  const tokens = encodePosition({ ...defaults, fen: after, historyFens: [start, after] });
  // e2 white pawn in six padded + first actual frames; e4 becomes e5 black pawn in final frame.
  for (let frame = 0; frame < 7; frame++) assert.equal(tokens[12 * 99 + frame * 12], 1);
  assert.equal(tokens[36 * 99 + 84 + 6], 1);
  assert.equal(tokens[12 * 99 + 84], 1); // Black e7 pawn mirrors to white e2.
  for (let square = 0; square < 64; square++) {
    assert.equal(tokens[square * 99 + 96], 3);
    assert.ok(Math.abs(tokens[square * 99 + 97] - 0.02) < 1e-8);
    assert.equal(tokens[square * 99 + 98], 1.25);
  }
  assert.throws(() => encodePosition({ ...defaults, fen: after, historyFens: [start] }));
});

test('full histories retain the latest eight real positions', () => {
  const board = new Chess();
  const history = [board.fen()];
  for (const move of ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'Ba4', 'Nf6', 'O-O']) { board.move(move); history.push(board.fen()); }
  const all = encodePosition({ ...defaults, fen: board.fen(), historyFens: history });
  const eight = encodePosition({ ...defaults, fen: board.fen(), historyFens: history.slice(-8) });
  assert.deepEqual(all, eight);
  assert.notDeepEqual(all, encodePosition({ ...defaults, fen: board.fen(), historyFens: [board.fen()] }));
});

test('sampling masks illegal scores before stable softmax and preserves all promotions', () => {
  const start = new Chess().fen();
  const logits = new Float32Array(4352).fill(10000);
  for (const move of new Chess(start).moves({ verbose: true })) logits[moveIndex(move.from + move.to, false)] = -10000;
  assert.doesNotThrow(() => new Chess(start).move(sampleLegalMove(start, logits, () => 0.5)));
  for (const [fen, prefix] of [['7k/P7/8/8/8/8/8/7K w - - 0 1', 'a7a8'], ['7k/8/8/8/8/8/p7/7K b - - 0 1', 'a2a1']]) {
    for (const promotion of 'qrbn') {
      const scores = new Float32Array(4352).fill(-10000);
      scores[moveIndex(prefix + promotion, fen.includes(' b '))] = 10000;
      assert.equal(sampleLegalMove(fen, scores, () => 0.5), prefix + promotion);
    }
  }
  assert.throws(() => sampleLegalMove('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1', logits));
});

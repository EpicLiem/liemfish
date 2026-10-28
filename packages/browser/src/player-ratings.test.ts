import test from 'node:test';
import assert from 'node:assert/strict';
import { RECENT_PLAYER_RATINGS, ratingContextForControl } from './player-ratings';

test('rating context follows base plus forty increments and preserves category boundaries', () => {
  for (const [base, increment, rating] of [
    [60, 1, 230], [180, 0, 220], [300, 3, 220], [600, 0, 680], [900, 10, 680],
    [179, 0, 230], [599, 0, 220], [0, 4.5, 220], [0, 15, 680],
  ]) {
    assert.deepEqual(ratingContextForControl(base, increment), { eloSelf: rating, eloOpponent: rating });
  }
  assert.deepEqual(RECENT_PLAYER_RATINGS.rapid, { rating: 680, median: 675, games: 23 });
});

test('rating context rejects invalid controls and returns independent objects', () => {
  for (const [base, increment] of [[-1, 0], [0, -1], [NaN, 0], [0, NaN], [Infinity, 0], [0, Infinity], [1e308, 1e308]]) {
    assert.throws(() => ratingContextForControl(base, increment), RangeError);
  }
  const changed = ratingContextForControl(600, 0);
  changed.eloSelf = 220;
  assert.deepEqual(ratingContextForControl(600, 0), { eloSelf: 680, eloOpponent: 680 });
});

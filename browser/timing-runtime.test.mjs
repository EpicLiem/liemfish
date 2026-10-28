import test from 'node:test';
import assert from 'node:assert/strict';
import { TurnScheduler, executeTurn, executionBudget } from './timing-runtime.mjs';

test('inference time is part of the requested delay', async () => {
  let time = 0;
  const scheduler = new TurnScheduler({ now: () => time });
  const token = scheduler.begin('position', 10);
  const waits = [];
  const moves = [];
  const event = await executeTurn({
    scheduler, token, infer: async () => { time = 2; return { move_uci: 'e2e4', target_seconds: 5 }; },
    wait: async seconds => { waits.push(seconds); time += seconds; },
    currentPosition: () => 'position', validateMove: () => true,
    commitMove: move => { moves.push(move); }, incrementSeconds: 1,
  });
  assert.deepEqual(waits, [3]);
  assert.deepEqual(moves, ['e2e4']);
  assert.equal(event.visible_delay_seconds, 5);
  assert.equal(event.post_clock_seconds, 6);
});

test('takeback during inference prevents a stale move', async () => {
  const scheduler = new TurnScheduler({ now: () => 0 });
  const token = scheduler.begin('position', 10);
  let committed = false;
  await assert.rejects(executeTurn({
    scheduler, token,
    infer: async () => { scheduler.takeback(); return { move_uci: 'e2e4', target_seconds: 0 }; },
    wait: async () => {}, currentPosition: () => 'position', validateMove: () => true,
    commitMove: () => { committed = true; },
  }), /Stale turn/);
  assert.equal(committed, false);
});

test('game ending during wait cancels commit', async () => {
  let time = 0;
  const scheduler = new TurnScheduler({ now: () => time });
  const token = scheduler.begin('position', 10);
  await assert.rejects(executeTurn({
    scheduler, token, infer: async () => ({ move_uci: 'e2e4', target_seconds: 1 }),
    wait: async seconds => { time += seconds; scheduler.endGame(); },
    currentPosition: () => 'position', validateMove: () => true,
    commitMove: () => assert.fail('Must not commit'),
  }), /Stale turn/);
});

test('position mutation without cancellation is also rejected', async () => {
  const scheduler = new TurnScheduler({ now: () => 0 });
  const token = scheduler.begin('position', 10);
  await assert.rejects(executeTurn({
    scheduler, token, infer: async () => ({ move_uci: 'e2e4', target_seconds: 0 }),
    wait: async () => {}, currentPosition: () => 'different', validateMove: () => true,
    commitMove: () => assert.fail('Must not commit'),
  }), /Position changed/);
});

test('inference overrunning the clock cannot commit or earn increment', async () => {
  let time = 0;
  const scheduler = new TurnScheduler({ now: () => time });
  const token = scheduler.begin('position', 1);
  const plan = await executeTurn({
    scheduler, token,
    infer: async () => { time = 2; return { move_uci: 'e2e4', target_seconds: .1 }; },
    wait: async () => assert.fail('Must not wait'),
    currentPosition: () => 'position', validateMove: () => true,
    commitMove: () => assert.fail('Must not commit'), incrementSeconds: 10,
  });
  assert.equal(plan.status, 'budget_exhausted');
  assert.equal(scheduler.active, null);
});

test('timer delay past deadline cannot commit', async () => {
  let time = 0;
  const scheduler = new TurnScheduler({ now: () => time });
  const token = scheduler.begin('position', 1);
  const event = await executeTurn({
    scheduler, token, infer: async () => ({ move_uci: 'e2e4', target_seconds: .5 }),
    wait: async () => { time = 2; }, currentPosition: () => 'position', validateMove: () => true,
    commitMove: () => assert.fail('Must not commit'), incrementSeconds: 10,
  });
  assert.equal(event.status, 'budget_exhausted');
  assert.equal(event.post_clock_seconds, 0);
});

test('early timer checks remaining duration again', async () => {
  let time = 0, waits = 0;
  const scheduler = new TurnScheduler({ now: () => time });
  const token = scheduler.begin('position', 10);
  const event = await executeTurn({
    scheduler, token, infer: async () => ({ move_uci: 'e2e4', target_seconds: 1 }),
    wait: async seconds => { time += ++waits === 1 ? .4 : seconds; },
    currentPosition: () => 'position', validateMove: () => true, commitMove: () => {},
  });
  assert.equal(waits, 2);
  assert.equal(event.visible_delay_seconds, 1);
});

test('zero budget and invalid clocks are explicit', () => {
  assert.equal(executionBudget(0), 0);
  assert.equal(executionBudget(.1, .2), 0);
  assert.ok(executionBudget(1) < 1);
  for (const value of [-1, NaN, Infinity]) assert.throws(() => executionBudget(value));
});

test('failed host commit is not logged as a completed move', async () => {
  const scheduler = new TurnScheduler({ now: () => 0 });
  const token = scheduler.begin('position', 10);
  await assert.rejects(executeTurn({
    scheduler, token, infer: async () => ({ move_uci: 'e2e4', target_seconds: 0 }),
    wait: async () => {}, currentPosition: () => 'position', validateMove: () => true,
    commitMove: () => { throw new Error('host failed'); },
  }), /host failed/);
  assert.equal(scheduler.events.at(-1).status, 'commit_failed');
  assert.equal(scheduler.events.at(-1).post_clock_seconds, null);
});

test('exhausted clock never invokes inference with empty support', async () => {
  const scheduler = new TurnScheduler({ now: () => 0 });
  const token = scheduler.begin('position', 0);
  const result = await executeTurn({ scheduler, token, infer: async () => assert.fail('Must not infer') });
  assert.equal(result.status, 'budget_exhausted');
});

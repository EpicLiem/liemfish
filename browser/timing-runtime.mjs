/** Monotonic timestamps are seconds. The host owns chess legality and the model. */
export class TurnScheduler {
  constructor({ now = () => performance.now() / 1000, reserveSeconds = 0 } = {}) {
    this.now = now;
    this.reserveSeconds = nonnegative(reserveSeconds, 'reserveSeconds');
    this.generation = 0;
    this.active = null;
    this.events = [];
    this.ended = false;
  }

  begin(positionKey, remainingSeconds) {
    if (this.ended) throw new Error('Game has ended');
    this.cancel('superseded');
    this.active = {
      token: ++this.generation, position_key: positionKey, turn_start: this.now(),
      pre_clock_seconds: nonnegative(remainingSeconds, 'remainingSeconds'),
      inference_start: null, inference_end: null,
    };
    return this.generation;
  }

  turn(token) {
    if (!this.active || this.active.token !== token) throw new Error('Stale turn token');
    return this.active;
  }

  inferenceStarted(token) {
    const turn = this.turn(token);
    if (turn.inference_start !== null) throw new Error('Inference already started');
    turn.inference_start = this.now();
  }

  predictionReady(token, move, targetSeconds) {
    const turn = this.turn(token);
    if (turn.inference_start === null) throw new Error('Record inferenceStarted first');
    if (turn.inference_end !== null) throw new Error('Prediction already supplied');
    const now = this.now();
    const elapsed = now - turn.turn_start;
    if (elapsed < 0) throw new Error('Clock moved backwards');
    const target = nonnegative(targetSeconds, 'targetSeconds');
    const budget = executionBudget(turn.pre_clock_seconds, this.reserveSeconds);
    Object.assign(turn, {
      inference_end: now, move_uci: move, target_seconds: target,
      effective_target_seconds: Math.min(target, budget), execution_budget_seconds: budget,
      guard_intervened: target > budget,
    });
    const status = budget > 0 && elapsed <= budget ? 'ready' : 'budget_exhausted';
    return { token, status, wait_seconds: status === 'ready' ? Math.max(turn.effective_target_seconds - elapsed, 0) : 0,
      guard_intervened: turn.guard_intervened, execution_budget_seconds: budget };
  }

  commit(token, positionKey, { incrementSeconds = 0, chargedSeconds = null } = {}) {
    const turn = this.turn(token);
    if (turn.inference_end === null) throw new Error('No prediction ready');
    if (positionKey !== turn.position_key) {
      this.cancel('position_changed');
      throw new Error('Position changed before commit');
    }
    const now = this.now();
    const elapsed = now - turn.turn_start;
    if (elapsed < turn.effective_target_seconds) throw new Error('Requested delay has not elapsed');
    const charged = chargedSeconds === null ? elapsed : nonnegative(chargedSeconds, 'chargedSeconds');
    const increment = nonnegative(incrementSeconds, 'incrementSeconds');
    const status = turn.execution_budget_seconds > 0 && elapsed <= turn.execution_budget_seconds && charged < turn.pre_clock_seconds ? 'committed' : 'budget_exhausted';
    const event = { ...turn, status, commit_time: status === 'committed' ? now : null,
      observed_at: now, visible_delay_seconds: elapsed, charged_seconds: charged,
      latency_floor_seconds: turn.inference_end - turn.turn_start,
      target_unattainable: turn.inference_end - turn.turn_start > turn.target_seconds,
      post_clock_seconds: Math.max(turn.pre_clock_seconds - charged, 0) + (status === 'committed' ? increment : 0) };
    this.events.push(event);
    this.active = null;
    return event;
  }

  cancel(reason = 'cancelled') {
    if (this.active) this.events.push({ ...this.active, status: 'cancelled', reason, observed_at: this.now() });
    this.active = null;
  }

  takeback() { this.cancel('takeback'); }
  endGame() { this.cancel('game_ended'); this.ended = true; }
}

function nonnegative(value, name) {
  if (!Number.isFinite(value) || value < 0) throw new Error(`${name} must be finite and nonnegative`);
  return value;
}

export function executionBudget(remaining, reserve = 0) {
  nonnegative(remaining, 'remaining');
  nonnegative(reserve, 'reserve');
  if (remaining <= reserve) return 0;
  return Math.max(0, Math.min(remaining - reserve, remaining - Math.max(Number.MIN_VALUE, Math.abs(remaining) * Number.EPSILON)));
}

/** infer returns {move_uci,target_seconds}; wait(seconds) must be cancellable by the host if desired.
 * commitMove is synchronous and must atomically validate and apply the move. No UI/network action
 * is performed by this module. Clocks returned here are local estimates, not platform receipts.
 */
export async function executeTurn({ scheduler, token, infer, wait, currentPosition, validateMove, commitMove, incrementSeconds = 0 }) {
  const initial = scheduler.turn(token);
  if (executionBudget(initial.pre_clock_seconds, scheduler.reserveSeconds) === 0) {
    scheduler.cancel('budget_exhausted');
    return { token, status: 'budget_exhausted', wait_seconds: 0 };
  }
  scheduler.inferenceStarted(token);
  const prediction = await infer();
  scheduler.turn(token);
  if (!validateMove(prediction.move_uci)) throw new Error('Prediction contains an illegal move');
  const plan = scheduler.predictionReady(token, prediction.move_uci, prediction.target_seconds);
  if (plan.status !== 'ready') {
    scheduler.cancel('budget_exhausted');
    return plan;
  }
  // Timer callbacks can fire early; recheck against the monotonic clock before committing.
  while (true) {
    const turn = scheduler.turn(token);
    const remaining = turn.effective_target_seconds - (scheduler.now() - turn.turn_start);
    if (remaining <= 0) break;
    await wait(remaining);
  }
  const turn = scheduler.turn(token);
  if (!validateMove(prediction.move_uci)) {
    scheduler.cancel('illegal_at_commit');
    throw new Error('Move no longer legal');
  }
  const position = currentPosition();
  if (position !== turn.position_key) {
    scheduler.cancel('position_changed');
    throw new Error('Position changed before commit');
  }
  const event = scheduler.commit(token, position, { incrementSeconds });
  if (event.status === 'committed') {
    try {
      const result = commitMove(prediction.move_uci);
      if (result && typeof result.then === 'function') throw new Error('commitMove must be synchronous');
    } catch (error) {
      event.status = 'commit_failed';
      event.commit_time = null;
      event.post_clock_seconds = null;
      event.error = String(error);
      throw error;
    }
  }
  return event;
}

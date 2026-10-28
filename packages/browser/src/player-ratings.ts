/** Rounded human rating context from 2026 Chess.com development games, not measured bot Elo. */
export const RECENT_PLAYER_RATINGS = {
  bullet: { rating: 230, median: 232, games: 1 },
  blitz: { rating: 220, median: 224, games: 15 },
  rapid: { rating: 680, median: 675, games: 23 },
} as const;

/** Optional context for personal-play mode; callers can supply their own ratings instead. */
export function ratingContextForControl(initialSeconds: number, incrementSeconds: number) {
  const estimatedSeconds = initialSeconds + 40 * incrementSeconds;
  if (!Number.isFinite(initialSeconds) || !Number.isFinite(incrementSeconds)
      || initialSeconds < 0 || incrementSeconds < 0 || !Number.isFinite(estimatedSeconds)) {
    throw new RangeError('Time control must contain finite, nonnegative seconds.');
  }
  const category = estimatedSeconds < 180 ? 'bullet' : estimatedSeconds < 600 ? 'blitz' : 'rapid';
  const rating = RECENT_PLAYER_RATINGS[category].rating;
  return { eloSelf: rating, eloOpponent: rating };
}

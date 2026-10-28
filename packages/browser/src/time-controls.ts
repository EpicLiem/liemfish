// Unique base/increment pairs in both processed and timing-v2 training rows.
export const TRAINED_TIME_CONTROLS = [
  [60, 0], [60, 1], [120, 0], [120, 1], [180, 0], [180, 2],
  [300, 0], [300, 3], [600, 0], [900, 10], [1200, 0], [7200, 0],
].map(([initialSeconds, incrementSeconds]) => ({
  id: `${initialSeconds / 60}+${incrementSeconds}`,
  initialSeconds,
  incrementSeconds,
  label: `${initialSeconds / 60} min + ${incrementSeconds} sec/move`,
}));
export const DEFAULT_TIME_CONTROL = TRAINED_TIME_CONTROLS.find(control => control.id === "10+0")!;

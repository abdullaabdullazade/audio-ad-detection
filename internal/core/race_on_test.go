//go:build race

package core

// raceEnabled reports whether the binary was built with the race detector.
const raceEnabled = true

// raceFactor multiplies every per-test detection deadline under -race. The detector's
// inner loops are instrumented on every memory access there, which measured 5-8x slower
// on this workload; without the multiplier the tests fail with "context deadline
// exceeded", which reads like a detection bug rather than a harness one.
const raceFactor = 8

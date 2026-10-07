package features

import "math"

// TrimSilence removes leading and trailing near-silence from a mono signal and
// returns the trimmed samples plus the amount of leading time removed (seconds).
//
// This keeps reference clips and aired copies aligned on their audible content start
// regardless of edge silence / ID3 padding (§8 class 11), so detected timestamps
// point at the content rather than at padding.
func TrimSilence(samples []float32, sampleRate int) (trimmed []float32, leadSec float64) {
	if len(samples) == 0 {
		return samples, 0
	}
	var peak float64
	for _, s := range samples {
		if v := math.Abs(float64(s)); v > peak {
			peak = v
		}
	}
	if peak == 0 {
		return samples, 0
	}
	thresh := math.Max(0.01, 0.02*peak)

	win := sampleRate / 100 // 10 ms RMS window
	if win < 1 {
		win = 1
	}
	loud := func(center int) bool {
		a := center - win/2
		b := center + win/2
		if a < 0 {
			a = 0
		}
		if b > len(samples) {
			b = len(samples)
		}
		var s2 float64
		for i := a; i < b; i++ {
			v := float64(samples[i])
			s2 += v * v
		}
		rms := math.Sqrt(s2 / float64(b-a))
		return rms >= thresh
	}

	start := 0
	for start < len(samples) && !loud(start) {
		start += win
	}
	if start >= len(samples) {
		return samples, 0 // all quiet; leave as-is
	}
	end := len(samples) - 1
	for end > start && !loud(end) {
		end -= win
	}
	// Small margins so we don't clip transients.
	margin := sampleRate / 50 // 20 ms
	start -= margin
	end += margin
	if start < 0 {
		start = 0
	}
	if end > len(samples) {
		end = len(samples)
	}
	return samples[start:end], float64(start) / float64(sampleRate)
}

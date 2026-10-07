package match

import "math"

// Envelope cross-correlation: a GLOBAL alignment check that complements DTW.
//
// DTW is greedy and locally elastic, so on a clip whose content repeats (a music bed,
// a repeated jingle) it can lock onto a time-shifted copy of the same material and
// still report high coverage and score. A whole-length normalized cross-correlation of
// the two energy envelopes cannot: a shift misaligns the entire remaining structure, so
// its correlation drops sharply. This is the same principle as the GCC time-delay
// estimation used in the audio-fingerprinting/TDOA literature, applied here on the
// frame-energy envelope so it stays robust to broadcast EQ, gain and compression.

// Envelope reduces a log-mel frame sequence to a per-frame energy value, mean-removed
// and unit-variance normalized so correlations are comparable across material.
func Envelope(mel [][]float32) []float64 {
	if len(mel) == 0 {
		return nil
	}
	env := make([]float64, len(mel))
	for i, f := range mel {
		var s float64
		for _, v := range f {
			s += float64(v) * float64(v)
		}
		env[i] = math.Sqrt(s)
	}
	var mean float64
	for _, v := range env {
		mean += v
	}
	mean /= float64(len(env))
	var varr float64
	for i := range env {
		env[i] -= mean
		varr += env[i] * env[i]
	}
	if varr > 0 {
		inv := 1 / math.Sqrt(varr/float64(len(env)))
		for i := range env {
			env[i] *= inv
		}
	}
	return env
}

// RefineByEnvelope searches lags in [startFrame-radius, startFrame+radius] for the
// offset where the reference envelope best correlates with the recording envelope, with
// the reference's time axis scaled by `scale` (aired/reference). It returns the best
// offset (recording frame index where reference frame 0 sits) and the normalized
// correlation at that offset in [-1,1].
//
// The correlation value doubles as evidence: a genuine airing scores high (typically
// > 0.5), while a shifted fit on self-similar content scores much lower, which is what
// lets the caller discard the shifted one.
func RefineByEnvelope(refEnv, recEnv []float64, startFrame, radius int, scale float64) (int, float64) {
	if len(refEnv) < 4 || len(recEnv) < 4 || radius < 0 {
		return startFrame, 0
	}
	bestOff, bestCorr := startFrame, math.Inf(-1)
	for off := startFrame - radius; off <= startFrame+radius; off++ {
		var dot, nRef, nRec float64
		n := 0
		for i := 0; i < len(refEnv); i++ {
			j := off + int(math.Round(float64(i)*scale))
			if j < 0 || j >= len(recEnv) {
				continue
			}
			a, b := refEnv[i], recEnv[j]
			dot += a * b
			nRef += a * a
			nRec += b * b
			n++
		}
		// Require most of the reference to overlap the recording, else the correlation is
		// computed on a fragment and is not comparable.
		if n < len(refEnv)*3/4 || nRef <= 0 || nRec <= 0 {
			continue
		}
		corr := dot / math.Sqrt(nRef*nRec)
		if corr > bestCorr {
			bestCorr, bestOff = corr, off
		}
	}
	if math.IsInf(bestCorr, -1) {
		return startFrame, 0
	}
	return bestOff, bestCorr
}

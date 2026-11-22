package features

import (
	"math"
	"testing"
)

// A non-finite magnitude must not yield a NaN cosine: NaN passes `score < threshold`
// gates unnoticed and json.Encoder rejects it.
func TestCosineNonFiniteFeatureIsZero(t *testing.T) {
	flat := func() []float32 {
		m := make([]float32, FFTSize/2+1)
		for i := range m {
			m[i] = 0.1
		}
		return m
	}
	good := LogMelCentered(&Spectrogram{Frames: [][]float32{flat()}})[0]
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		m := flat()
		m[50] = bad
		for _, center := range []bool{false, true} {
			v := logMel(&Spectrogram{Frames: [][]float32{m}}, center)[0]
			if got := Cosine(v, good); got != 0 {
				t.Fatalf("bad=%v center=%v: Cosine = %v, want 0", bad, center, got)
			}
		}
	}
}

package core

import (
	"testing"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/fingerprint"
)

// TestQuadInvariance validates the core assumption of quad-based fingerprinting: a
// naive-resampled (pitch-shifted + time-scaled) copy of a clip shares many quad hashes
// with the original, whereas the landmark-pair hash does not. This is the cheap probe
// that decides whether the quad approach is worth full integration.
func TestQuadInvariance(t *testing.T) {
	requireSamples(t)
	ctx, cancel := ctxWithBudget(60 * time.Second)
	defer cancel()

	ad := decodeAd(t, ctx, "ad_1.mpeg")
	trimmed, _ := features.TrimSilence(ad.Samples, features.SampleRate)

	refKeys := keySet(fingerprint.QuadsFromPCM(trimmed, fingerprint.RefQuadParams))

	for _, factor := range []float64{1.0, 1.05, 0.95, 1.02} {
		q := features.Resample(trimmed, factor)
		qh := fingerprint.QuadsFromPCM(q, fingerprint.QueryQuadParams)
		matched := 0
		for _, h := range qh {
			if _, ok := refKeys[h.Key]; ok {
				matched++
			}
		}
		t.Logf("resample factor=%.2f: query quads=%d, matched ref keys=%d", factor, len(qh), matched)
		if factor == 1.0 && matched < 20 {
			t.Fatalf("even at factor 1.0 only %d quad matches — grouping/hash broken", matched)
		}
		if (factor == 1.05 || factor == 0.95) && matched < 10 {
			t.Errorf("resample %.2f: only %d quad matches — invariance weak", factor, matched)
		}
	}
}

func keySet(hs []fingerprint.QuadHash) map[uint32]int {
	m := make(map[uint32]int, len(hs))
	for _, h := range hs {
		m[h.Key]++
	}
	return m
}

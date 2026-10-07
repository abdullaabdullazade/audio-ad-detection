package core

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
)

// sampleAds are the three reference clips at the repo root.
var sampleAds = []string{"ad_1.mpeg", "ad_2.mpeg", "ad_3.mpeg"}

// requireSamples skips the test if the sample ads are not present.
func requireSamples(t *testing.T) {
	t.Helper()
	for _, n := range sampleAds {
		if _, err := os.Stat(filepath.Join("..", "..", n)); err != nil {
			t.Skipf("sample %s missing", n)
		}
	}
}

func decodeAd(t *testing.T, ctx context.Context, name string) audio.PCM {
	t.Helper()
	pcm, err := audio.Decode(ctx, filepath.Join("..", "..", name), audio.DecodeOptions{SampleRate: features.SampleRate})
	if err != nil {
		t.Fatalf("decode %s: %v", name, err)
	}
	return pcm
}

// decodeAdFiltered decodes a sample ad through an ffmpeg filter chain (for time-scale,
// pause-compression, etc.).
func decodeAdFiltered(t *testing.T, ctx context.Context, name, filter string) audio.PCM {
	t.Helper()
	pcm, err := audio.DecodeFiltered(ctx, filepath.Join("..", "..", name), filter, features.SampleRate, 120)
	if err != nil {
		t.Fatalf("decode filtered %s: %v", name, err)
	}
	return pcm
}

// detectorWithSamples builds a detector over the three sample ads.
func detectorWithSamples(t *testing.T, ctx context.Context) *Detector {
	t.Helper()
	var clips []RefClip
	for _, n := range sampleAds {
		clips = append(clips, RefClip{AdID: n, Samples: decodeAd(t, ctx, n).Samples})
	}
	det, err := NewDetector(Config{}, clips)
	if err != nil {
		t.Fatalf("build detector: %v", err)
	}
	return det
}

// air returns `seconds` of deterministic low-level background noise at 16 kHz.
func air(seconds float64, seed uint32) audio.PCM {
	n := int(seconds * features.SampleRate)
	s := make([]float32, n)
	st := seed*2654435761 + 1
	for i := range s {
		st = st*1664525 + 1013904223
		s[i] = float32((float64(st)/float64(math.MaxUint32)*2 - 1) * 0.02)
	}
	return audio.PCM{SampleRate: features.SampleRate, Samples: s}
}

// mixAt overlays src into dst starting at offsetSec (dst modified in place).
func mixAt(dst audio.PCM, src []float32, offsetSec float64) {
	at := int(offsetSec * float64(dst.SampleRate))
	for i, v := range src {
		j := at + i
		if j >= 0 && j < len(dst.Samples) {
			dst.Samples[j] += float32(float64(v) * 0.9)
		}
	}
}

// detect runs detection and keeps only matches at or above minConf.
func detect(t *testing.T, ctx context.Context, det *Detector, rec audio.PCM, minConf float64) []Match {
	t.Helper()
	matches, err := det.Detect(ctx, rec)
	if err != nil {
		t.Fatalf("detect: %v", err)
	}
	var out []Match
	for _, m := range matches {
		if m.Confidence >= minConf {
			out = append(out, m)
		}
	}
	return out
}

// countAd counts matches for a given ad id.
func countAd(ms []Match, adID string) int {
	n := 0
	for _, m := range ms {
		if m.AdID == adID {
			n++
		}
	}
	return n
}

// operatingConf is a conservative operating threshold for the robustness tests: true
// matches score ~0.99+, while clean-air false matches on this material stay below it.
const operatingConf = 0.99

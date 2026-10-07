package audio

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// sampleAd returns the path to a real sample reference clip at the repo root, or
// skips the test if it is not present.
func sampleAd(t *testing.T, name string) string {
	t.Helper()
	p := filepath.Join("..", "..", name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("sample %s not available: %v", name, err)
	}
	return p
}

func TestProbeSampleAd(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	info, err := Probe(ctx, sampleAd(t, "ad_1.mpeg"), Limits{})
	if err != nil {
		t.Fatalf("probe: %v", err)
	}
	if info.Codec == "" || info.SampleRate <= 0 || info.Channels <= 0 || info.DurationS <= 0 {
		t.Fatalf("implausible probe info: %+v", info)
	}
	t.Logf("probe ad_1: codec=%s sr=%d ch=%d dur=%.3fs size=%d",
		info.Codec, info.SampleRate, info.Channels, info.DurationS, info.SizeBytes)
}

func TestDecodeSampleAd(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	const targetSR = 16000
	pcm, err := Decode(ctx, sampleAd(t, "ad_1.mpeg"), DecodeOptions{SampleRate: targetSR})
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if pcm.SampleRate != targetSR {
		t.Fatalf("sample rate = %d, want %d", pcm.SampleRate, targetSR)
	}
	if len(pcm.Samples) == 0 {
		t.Fatal("no samples decoded")
	}
	// ad_1 is ~28.5s; at 16kHz that is ~456k samples. Allow a wide tolerance.
	if pcm.DurationS() < 20 || pcm.DurationS() > 35 {
		t.Fatalf("decoded duration = %.3fs, expected ~28.5s", pcm.DurationS())
	}
	// Sanity: audio is not silent and not clipping to NaN/Inf.
	var peak float64
	for _, s := range pcm.Samples {
		v := math.Abs(float64(s))
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatal("decoded sample is NaN/Inf")
		}
		if v > peak {
			peak = v
		}
	}
	if peak == 0 {
		t.Fatal("decoded audio is all-zero (silent)")
	}
	t.Logf("decode ad_1: samples=%d dur=%.3fs peak=%.3f", len(pcm.Samples), pcm.DurationS(), peak)
}

func TestDecodeRejectsMissingFile(t *testing.T) {
	ctx := context.Background()
	_, err := Decode(ctx, filepath.Join(t.TempDir(), "nope.mp3"), DecodeOptions{SampleRate: 16000})
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestDecodeRejectsZeroLength(t *testing.T) {
	ctx := context.Background()
	p := filepath.Join(t.TempDir(), "empty.mp3")
	if err := os.WriteFile(p, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := Decode(ctx, p, DecodeOptions{SampleRate: 16000})
	if err == nil {
		t.Fatal("expected error for zero-length file")
	}
}

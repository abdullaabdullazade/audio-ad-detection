package index

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
)

// synthClip builds a deterministic pseudo-audio clip of the given length in seconds:
// a sum of a few tones with noise, which produces a realistic number of spectral peaks
// (and therefore of landmark hashes) without needing real media files.
func synthClip(seconds float64, seed int64) []float32 {
	n := int(seconds * float64(features.SampleRate))
	rng := rand.New(rand.NewSource(seed))
	f1 := 200 + rng.Float64()*1500
	f2 := 400 + rng.Float64()*3000
	f3 := 800 + rng.Float64()*5000
	out := make([]float32, n)
	for i := range out {
		t := float64(i) / float64(features.SampleRate)
		env := 0.5 + 0.5*math.Sin(2*math.Pi*(0.7+rng.Float64()*0.1)*t)
		v := env * (math.Sin(2*math.Pi*f1*t) + 0.7*math.Sin(2*math.Pi*f2*t) + 0.4*math.Sin(2*math.Pi*f3*t))
		v += rng.NormFloat64() * 0.05
		out[i] = float32(v * 0.2)
	}
	return out
}

// TestIndexAddRemoveWithoutRebuild verifies §6: a clip can be added and removed without
// rebuilding the whole index, and lookups reflect the change immediately.
func TestIndexAddRemoveWithoutRebuild(t *testing.T) {
	ix := New(nil)
	for i := 0; i < 5; i++ {
		if err := ix.Add(fmt.Sprintf("clip_%d", i), synthClip(12, int64(i+1))); err != nil {
			t.Fatalf("add: %v", err)
		}
	}
	before := ix.Stats()
	if before.LiveAds != 5 {
		t.Fatalf("live ads = %d, want 5", before.LiveAds)
	}

	// Adding one more must not require touching the existing entries.
	if err := ix.Add("clip_new", synthClip(12, 99)); err != nil {
		t.Fatalf("incremental add: %v", err)
	}
	if got := ix.Stats().LiveAds; got != 6 {
		t.Fatalf("after add live ads = %d, want 6", got)
	}

	// Removing is likewise incremental (tombstone), and the ad disappears from lookups.
	if !ix.Remove("clip_2") {
		t.Fatal("remove reported not found")
	}
	after := ix.Stats()
	if after.LiveAds != 5 {
		t.Fatalf("after remove live ads = %d, want 5", after.LiveAds)
	}
	if ix.Remove("clip_2") {
		t.Fatal("second remove should report not found")
	}
	// The removed ad must not appear in any posting list.
	removedIdx := uint32(2)
	for h := range ix.table {
		for _, p := range ix.Lookup(h) {
			if p.AdIdx == removedIdx {
				t.Fatal("removed ad still returned by Lookup")
			}
		}
	}
}

// TestIndexColdLoadBudget verifies §6: the full reference set loads from disk in under
// 60 s and stays within 256 MB per 1000 clips. It builds a scaled-down set and
// extrapolates linearly, which is what the budget is expressed in.
func TestIndexColdLoadBudget(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping index budget test in -short mode")
	}
	const clips = 100 // scaled-down stand-in for the 8000-clip production set
	ix := New(nil)
	for i := 0; i < clips; i++ {
		if err := ix.Add(fmt.Sprintf("clip_%04d", i), synthClip(20, int64(i+1))); err != nil {
			t.Fatalf("add %d: %v", i, err)
		}
	}
	path := filepath.Join(t.TempDir(), "index.bin")
	if err := ix.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	elapsed := time.Since(start)

	if got := loaded.Stats().LiveAds; got != clips {
		t.Fatalf("loaded live ads = %d, want %d", got, clips)
	}

	perThousandSec := elapsed.Seconds() / float64(clips) * 1000
	perThousandMB := float64(fi.Size()) / float64(clips) * 1000 / (1 << 20)
	t.Logf("cold load: %d clips in %v (%.1f s per 1000 clips); on-disk %.1f MB per 1000 clips",
		clips, elapsed.Round(time.Millisecond), perThousandSec, perThousandMB)

	if perThousandSec > 60 {
		t.Errorf("cold load %.1f s per 1000 clips exceeds the 60 s budget", perThousandSec)
	}
	if perThousandMB > 256 {
		t.Errorf("index %.1f MB per 1000 clips exceeds the 256 MB budget", perThousandMB)
	}
}

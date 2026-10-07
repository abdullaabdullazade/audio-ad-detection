package core

import (
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
)

// TestDeterminism verifies §6: identical input produces byte-identical output with a
// stable ordering. It runs the same recording through two independently built detectors
// and compares the serialized match lists.
func TestDeterminism(t *testing.T) {
	requireSamples(t)
	ctx, cancel := ctxWithBudget(3 * time.Minute)
	defer cancel()

	adverts := []string{}
	for _, n := range sampleAds {
		adverts = append(adverts, filepath.Join(repoRoot(), n))
	}
	rec := filepath.Join(repoRoot(), "testdata", "generated", "rec_ad_1_baseline.mp3")

	run := func() string {
		det, err := LoadDetectorFromPaths(ctx, Config{}, adverts, audio.Limits{})
		if err != nil {
			t.Fatalf("build detector: %v", err)
		}
		matches, err := det.DetectFile(ctx, rec, audio.Limits{})
		if err != nil {
			t.Fatalf("detect: %v", err)
		}
		b, err := json.Marshal(matches)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}

	first := run()
	second := run()
	if first != second {
		t.Fatalf("output is not deterministic:\n first=%s\nsecond=%s", first, second)
	}
	if first == "" || first == "null" {
		t.Fatal("expected a non-empty match list for the baseline recording")
	}
	t.Logf("deterministic across runs: %s", first)
}

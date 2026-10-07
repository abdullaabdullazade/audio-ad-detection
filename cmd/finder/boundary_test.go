package main

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/core"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/obs"
)

// TestSegmentBoundary covers §8.9 at the BATCH level: the recorder writes 61-minute
// segments, so an airing that starts near the end of one file continues into the next and
// is truncated in both. Neither per-file pass can see the whole occurrence — only the
// boundary bridge can — so this exercises bridgeBoundaries directly rather than the
// detector.
//
// The two synthetic recordings are deliberately short (60 s and 60 s) but the split is the
// real one: the advert starts 45 s into the first file, so ~15 s of it lands in file A and
// the remainder in file B. bridgeSec (45 s) covers that straddle.
func TestSegmentBoundary(t *testing.T) {
	if _, err := os.Stat("../../testdata/adverts/ad_1.mpeg"); err != nil {
		t.Skip("reference advert not available")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	const sr = 16000
	ad, err := audio.DecodeFiltered(ctx, "../../testdata/adverts/ad_1.mpeg", "", sr, 120)
	if err != nil {
		t.Skipf("decode advert: %v", err)
	}
	if len(ad.Samples) < 20*sr {
		t.Skipf("advert too short for a boundary split: %d samples", len(ad.Samples))
	}

	// Air on both sides of the split. Different seeds so the two files are not identical.
	airA, err := audio.SynthAir(ctx, "brown", 11, 0.05, 60, sr)
	if err != nil {
		t.Skipf("synth air: %v", err)
	}
	airB, err := audio.SynthAir(ctx, "brown", 12, 0.05, 60, sr)
	if err != nil {
		t.Skipf("synth air: %v", err)
	}

	// Advert starts 45 s into file A and runs past its end into file B.
	const startSec = 45.0
	at := int(startSec * sr)
	inA := len(airA.Samples) - at // advert samples that fit in A
	if inA <= 0 || inA >= len(ad.Samples) {
		t.Fatalf("split does not straddle the boundary: inA=%d adLen=%d", inA, len(ad.Samples))
	}
	copy(airA.Samples[at:], ad.Samples[:inA])
	copy(airB.Samples[:len(ad.Samples)-inA], ad.Samples[inA:])

	dir := t.TempDir()
	// Names must carry the archive's timestamp_station shape: bridging only joins
	// consecutive recordings of the SAME station (see stationOf).
	fileA := filepath.Join(dir, "2026-06-11-00-00-00_teststation-60s.mp3")
	fileB := filepath.Join(dir, "2026-06-11-00-01-00_teststation-60s.mp3")
	if err := audio.EncodePCMToMP3(ctx, airA, fileA, 128); err != nil {
		t.Skipf("encode A: %v", err)
	}
	if err := audio.EncodePCMToMP3(ctx, airB, fileB, 128); err != nil {
		t.Skipf("encode B: %v", err)
	}

	det, err := core.NewDetector(core.DefaultConfig(), []core.RefClip{{AdID: "ad_1", Samples: ad.Samples}})
	if err != nil {
		t.Fatalf("build detector: %v", err)
	}

	records := []string{fileA, fileB}
	recs := []RecordOutput{{File: filepath.Base(fileA)}, {File: filepath.Base(fileB)}}
	byName := map[string]int{filepath.Base(fileA): 0, filepath.Base(fileB): 1}

	log := obs.NewLogger(slog.LevelWarn, false)
	added := bridgeBoundaries(ctx, det, records, byName, recs, log, 2, DefaultThreshold)
	if added == 0 {
		t.Fatalf("boundary occurrence not recovered: bridge added no matches")
	}

	// The occurrence is attributed to the EARLIER file, at its true start there.
	var got *core.Match
	for i := range recs[0].Matches {
		if recs[0].Matches[i].AdID == "ad_1" {
			got = &recs[0].Matches[i]
			break
		}
	}
	if got == nil {
		t.Fatalf("no ad_1 match attributed to the earlier recording; recs[0]=%+v", recs[0].Matches)
	}
	if d := got.TimeSec - startSec; d < -1.5 || d > 1.5 {
		t.Errorf("boundary start %.2f s, want %.2f s (±1.5 s per §2)", got.TimeSec, startSec)
	}
}

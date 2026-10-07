package core

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/eval"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
)

// repoRoot locates the repo root from the package directory.
func repoRoot() string { return filepath.Join("..", "..") }

func loadDetectorFromSamples(t *testing.T, ctx context.Context) *Detector {
	t.Helper()
	clips := []RefClip{}
	for _, name := range []string{"ad_1.mpeg", "ad_2.mpeg", "ad_3.mpeg"} {
		p := filepath.Join(repoRoot(), name)
		if _, err := os.Stat(p); err != nil {
			t.Skipf("sample %s missing", name)
		}
		pcm, err := audio.Decode(ctx, p, audio.DecodeOptions{SampleRate: features.SampleRate})
		if err != nil {
			t.Fatalf("decode %s: %v", name, err)
		}
		clips = append(clips, RefClip{AdID: adID(name), Samples: pcm.Samples})
	}
	det, err := NewDetector(Config{}, clips)
	if err != nil {
		t.Fatalf("build detector: %v", err)
	}
	return det
}

// TestEvalCorpus runs the detector over the generated adversarial corpus and reports
// recall/precision, per-class recall, timestamp error, and clean-air FPs. It is a
// measurement harness, not a strict pass/fail (thresholds are tuned in Day 3).
func TestEvalCorpus(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full-corpus eval in -short mode")
	}
	labelsPath := filepath.Join(repoRoot(), "testdata", "generated", "labels.json")
	if _, err := os.Stat(labelsPath); err != nil {
		t.Skip("no generated corpus; run: go run ./cmd/gencorpus -adverts ad_1.mpeg,ad_2.mpeg,ad_3.mpeg")
	}
	labels, err := eval.LoadLabels(labelsPath)
	if err != nil {
		t.Fatalf("load labels: %v", err)
	}

	ctx, cancel := ctxWithBudget(5 * time.Minute)
	defer cancel()
	det := loadDetectorFromSamples(t, ctx)

	recDir := filepath.Join(repoRoot(), "testdata", "generated")
	var preds []eval.Prediction
	for _, r := range labels.Recordings {
		pcm, err := audio.Decode(ctx, filepath.Join(recDir, r.File), audio.DecodeOptions{SampleRate: features.SampleRate})
		if err != nil {
			t.Fatalf("decode %s: %v", r.File, err)
		}
		matches, err := det.Detect(ctx, pcm)
		if err != nil {
			t.Fatalf("detect %s: %v", r.File, err)
		}
		for _, m := range matches {
			preds = append(preds, eval.Prediction{File: r.File, AdID: m.AdID, TimeSec: m.TimeSec, Confidence: m.Confidence})
		}
	}

	// Threshold sweep (mini PR curve) — shows whether true matches separate from FPs.
	t.Logf("\n=== THRESHOLD SWEEP ===")
	t.Logf("%-6s %5s %5s %5s %8s %9s %6s", "thr", "TP", "FP", "FN", "recall", "precision", "clnFP")
	for _, thr := range []float64{0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99} {
		r := eval.Score(labels, preds, thr)
		t.Logf("%-6.2f %5d %5d %5d %8.3f %9.3f %6d", thr, r.TP, r.FP, r.FN, r.Recall, r.Precision, r.CleanFP)
	}

	// Dump false positives at a high threshold to diagnose precision.
	fpRep := eval.Score(labels, preds, 0.98)
	t.Logf("\n=== FALSE POSITIVES at thr=0.98 (count=%d) ===", len(fpRep.FPList))
	sort.Slice(fpRep.FPList, func(i, j int) bool { return fpRep.FPList[i].File < fpRep.FPList[j].File })
	for _, fp := range fpRep.FPList {
		t.Logf("  FP: %-32s ad=%s t=%.2f conf=%.4f", fp.File, fp.AdID, fp.TimeSec, fp.Confidence)
	}

	rep := eval.Score(labels, preds, 0.0)
	t.Logf("\n=== CORPUS EVAL (threshold=0) ===")
	t.Logf("TP=%d FP=%d FN=%d  recall=%.3f precision=%.3f F1=%.3f  cleanFP=%d",
		rep.TP, rep.FP, rep.FN, rep.Recall, rep.Precision, rep.F1, rep.CleanFP)
	t.Logf("timestamp |Δ|: median=%.3fs p95=%.3fs max=%.3fs",
		rep.TimestampMedian, rep.TimestampP95, rep.TimestampMax)

	classes := make([]string, 0, len(rep.PerClass))
	for c := range rep.PerClass {
		classes = append(classes, c)
	}
	sort.Strings(classes)
	for _, c := range classes {
		cs := rep.PerClass[c]
		t.Logf("  %-20s occ=%d TP=%d FN=%d recall=%s", c, cs.Occurrences, cs.TP, cs.FN, fmtRecall(cs))
	}
}

func fmtRecall(cs eval.ClassStat) string {
	if cs.Occurrences == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.2f", cs.Recall())
}

package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/core"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/eval"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/obs"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/pipeline"
)

func runEval(ctx context.Context, args []string) error {
	fs := newFlagSet("eval")
	recordsDir := fs.String("records-dir", "", "directory of recordings (required)")
	advertsDir := fs.String("adverts-dir", "", "directory of reference adverts (required)")
	labelsPath := fs.String("labels", "", "ground-truth labels.json (required)")
	out := fs.String("out", "report.json", "output report path")
	targetPrec := fs.Float64("target-precision", 0.995, "operating-point precision target")
	workers := fs.Int("workers", intEnv("FINDER_WORKERS", 8), "concurrent workers")
	minAgree := fs.Float64("min-peak-agree", 0, "override peak-agreement gate (0 = default 0.35; negative disables)")
	enableResample := fs.Bool("enable-resample", boolEnv("FINDER_RESAMPLE", false), "enable naive-resample (pitch-shift) recovery path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *recordsDir == "" || *advertsDir == "" || *labelsPath == "" {
		return errors.New("--records-dir, --adverts-dir and --labels are required")
	}

	log := obs.NewLogger(slog.LevelInfo, false)
	labels, err := eval.LoadLabels(*labelsPath)
	if err != nil {
		return err
	}
	adverts, err := listAudio(*advertsDir)
	if err != nil {
		return err
	}
	sort.Strings(adverts)

	cfg := core.DefaultConfig()
	cfg.EnableResample = *enableResample
	if *minAgree != 0 {
		cfg.MinPeakAgree = *minAgree
	}
	det, err := core.LoadDetectorFromPaths(ctx, cfg, adverts, audio.Limits{})
	if err != nil {
		return fmt.Errorf("build index: %w", err)
	}

	start := time.Now()
	type pred struct {
		file    string
		matches []core.Match
	}
	results := pipeline.Map(ctx, *workers, labels.Recordings,
		func(ctx context.Context, r eval.Recording) (pred, error) {
			path := filepath.Join(*recordsDir, r.File)
			matches, derr := det.DetectFile(ctx, path, audio.Limits{})
			if derr != nil {
				log.Warn("eval recording failed", "file", r.File, "err", derr)
				return pred{file: r.File}, nil
			}
			return pred{file: r.File, matches: matches}, nil
		}, nil)

	var preds []eval.Prediction
	for _, r := range results {
		if r.Err != nil {
			continue
		}
		for _, m := range r.Value.matches {
			preds = append(preds, eval.Prediction{File: r.Value.file, AdID: m.AdID, TimeSec: m.TimeSec, Confidence: m.Confidence})
		}
	}

	report := eval.BuildReport(labels, preds, *targetPrec)
	if err := report.Save(*out); err != nil {
		return err
	}

	op := report.Operating
	for _, fp := range op.FPList {
		log.Warn("FP", "file", fp.File, "ad", fp.AdID, "t", round2(fp.TimeSec), "conf", fp.Confidence)
	}
	log.Info("eval complete",
		"chosen_threshold", round2(report.ChosenThreshold),
		"recall", round2(op.Recall), "precision", round2(op.Precision), "f1", round2(op.F1),
		"tp", op.TP, "fp", op.FP, "fn", op.FN, "clean_fp", op.CleanFP,
		"ts_median", round2(op.TimestampMedian), "ts_p95", round2(op.TimestampP95),
		"elapsed_sec", round1(time.Since(start).Seconds()), "out", *out)

	fmt.Fprintf(os.Stderr, "\n=== EVAL @ threshold %.3f ===\n", report.ChosenThreshold)
	fmt.Fprintf(os.Stderr, "recall=%.3f precision=%.3f F1=%.3f  (TP=%d FP=%d FN=%d cleanFP=%d)\n",
		op.Recall, op.Precision, op.F1, op.TP, op.FP, op.FN, op.CleanFP)
	fmt.Fprintf(os.Stderr, "timestamp |Δ|: median=%.3fs p95=%.3fs max=%.3fs\n", op.TimestampMedian, op.TimestampP95, op.TimestampMax)
	fmt.Fprintf(os.Stderr, "calibration: ECE=%.4f MCE=%.4f (%d bins)  [target ECE <= 0.05]\n",
		report.Calibration.ECE, report.Calibration.MCE, report.Calibration.Bins)
	classes := make([]string, 0, len(op.PerClass))
	for c := range op.PerClass {
		classes = append(classes, c)
	}
	sort.Strings(classes)
	for _, c := range classes {
		cs := op.PerClass[c]
		fmt.Fprintf(os.Stderr, "  %-20s occ=%d TP=%d FN=%d recall=%.2f\n", c, cs.Occurrences, cs.TP, cs.FN, cs.Recall())
	}
	return nil
}

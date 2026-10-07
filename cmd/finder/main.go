// Command finder detects reference advertisements inside audio recordings. It has
// three modes: single (one record x one advert), batch (directories, bounded
// concurrency, per-file fault isolation), and eval (metrics report against labels).
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/core"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	// Context cancelled on SIGINT/SIGTERM for graceful shutdown.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	var err error
	switch os.Args[1] {
	case "batch":
		err = runBatch(ctx, os.Args[2:])
	case "eval":
		err = runEval(ctx, os.Args[2:])
	case "-h", "--help", "help":
		usage()
		return
	default:
		// Single mode: flags only (no subcommand).
		err = runSingle(ctx, os.Args[1:])
	}
	if err != nil {
		if errors.Is(err, context.Canceled) {
			fmt.Fprintln(os.Stderr, "cancelled")
			os.Exit(130) // 128 + SIGINT
		}
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `finder — advertisement detection in audio

USAGE:
  finder --record R.mp3 --advert A.mp3 [--output json|text] [--min-confidence F] [--timeout D]
  finder batch --records-dir D --adverts-dir D [--workers N] [--output json|text] [--min-confidence F] [--metrics-addr :6060]
  finder eval  --records-dir D --adverts-dir D --labels labels.json --out report.json [--target-precision 0.995]

Flags may also be set via env (FINDER_OUTPUT, FINDER_MIN_CONFIDENCE, FINDER_WORKERS, FINDER_TIMEOUT, FINDER_METRICS_ADDR, FINDER_LOG_JSON).
`)
}

// envOr returns env[name] if set, else the provided default.
func envOr(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
}

// emitMatches writes matches as a JSON array or a text table.
func emitMatches(w *os.File, matches []core.Match, format string) error {
	switch format {
	case "text":
		if len(matches) == 0 {
			fmt.Fprintln(w, "(no matches)")
			return nil
		}
		for _, m := range matches {
			fmt.Fprintf(w, "%-16s t=%8.3fs dur=%7.3fs conf=%.3f score=%.3f scale=%.3f\n",
				m.AdID, m.TimeSec, m.DurationSec, m.Confidence, m.Score, m.ScaleFactor)
		}
		return nil
	default: // json
		if matches == nil {
			matches = []core.Match{}
		}
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(matches)
	}
}

func runSingle(ctx context.Context, args []string) error {
	fs := newFlagSet("single")
	record := fs.String("record", "", "path to the recording (required)")
	advert := fs.String("advert", "", "path to the reference advert (required)")
	output := fs.String("output", envOr("FINDER_OUTPUT", "json"), "output format: json|text")
	minConf := fs.Float64("min-confidence", floatEnv("FINDER_MIN_CONFIDENCE", 0.0), "drop matches below this calibrated confidence")
	timeout := fs.Duration("timeout", durEnv("FINDER_TIMEOUT", 3*time.Minute), "overall timeout")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *record == "" || *advert == "" {
		return errors.New("--record and --advert are required")
	}

	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()

	cfg := core.DefaultConfig()
	det, err := core.LoadDetectorFromPaths(ctx, cfg, []string{*advert}, audio.Limits{})
	if err != nil {
		return err
	}
	matches, err := det.DetectFile(ctx, *record, audio.Limits{})
	if err != nil {
		return err
	}
	matches = filterConfidence(matches, *minConf)
	return emitMatches(os.Stdout, matches, *output)
}

func filterConfidence(in []core.Match, min float64) []core.Match {
	if min <= 0 {
		return in
	}
	out := make([]core.Match, 0, len(in))
	for _, m := range in {
		if m.Confidence >= min {
			out = append(out, m)
		}
	}
	return out
}

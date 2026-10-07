// Command gencorpus builds the adversarial test corpus (§8) from reference ad clips:
// distorted on-air copies inserted into synthetic air at known offsets, plus a
// labels.json manifest. Deterministic for fixed flags.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/corpus"
)

func main() {
	var (
		advertsCSV = flag.String("adverts", "", "comma-separated reference ad file paths (required)")
		outDir     = flag.String("out", "testdata/generated", "output directory for recordings")
		labelsOut  = flag.String("labels", "", "labels.json path (default <out>/labels.json)")
		sampleRate = flag.Int("sample-rate", 16000, "working sample rate")
		seed       = flag.Int("seed", 1, "base RNG seed (determinism)")
		timeout    = flag.Duration("timeout", 10*time.Minute, "overall timeout")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	if *advertsCSV == "" {
		log.Error("missing required flag", "flag", "-adverts")
		os.Exit(2)
	}
	adverts := splitCSV(*advertsCSV)
	for _, a := range adverts {
		if _, err := os.Stat(a); err != nil {
			log.Error("advert not found", "path", a, "err", err)
			os.Exit(2)
		}
	}
	if *labelsOut == "" {
		*labelsOut = filepath.Join(*outDir, "labels.json")
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Error("mkdir out", "err", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	ctx, stop := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	start := time.Now()
	labels, err := corpus.Generate(ctx, corpus.Options{
		Adverts:    adverts,
		OutDir:     *outDir,
		SampleRate: *sampleRate,
		Seed:       *seed,
	})
	if err != nil {
		log.Error("generate corpus", "err", err)
		os.Exit(1)
	}
	if err := labels.Save(*labelsOut); err != nil {
		log.Error("save labels", "err", err)
		os.Exit(1)
	}

	var occ int
	for _, r := range labels.Recordings {
		occ += len(r.Occurrences)
	}
	log.Info("corpus generated",
		"recordings", len(labels.Recordings),
		"occurrences", occ,
		"out", *outDir,
		"labels", *labelsOut,
		"elapsed", time.Since(start).Round(time.Millisecond))
	fmt.Println("OK")
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := parts[:0]
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

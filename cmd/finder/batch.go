package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/core"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/obs"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/pipeline"
)

var audioExts = map[string]bool{".mp3": true, ".mp4": true, ".aac": true, ".m4a": true, ".wav": true, ".flac": true, ".mpeg": true, ".mpga": true}

// RecordOutput is the per-recording result in batch/eval JSON.
type RecordOutput struct {
	File       string       `json:"file"`
	Matches    []core.Match `json:"matches"`
	LatencySec float64      `json:"latency_sec,omitempty"`
	Error      string       `json:"error,omitempty"`
}

// DefaultThreshold is the shipping operating point. It sits in the middle of the gap
// between genuine airings and spurious matches as MEASURED on real broadcast audio:
//
//	genuine airings, real air   confidence 0.9695 … 1.0000
//	spurious matches, real air  confidence 0.0000 … 0.0024
//
// The corpus cannot set this value — every corpus true positive saturates at 1.0, so a
// sweep over it returns 1.0 and would discard the weakest real airing at 0.9695. Half-way
// between the two measured populations leaves ~0.5 of margin on each side, so the operating
// point does not depend on the exact numbers above holding on unseen data.
//
// An earlier confidence model (raw score through a saturated sigmoid) put BOTH populations
// inside [0.985, 0.995] and forced a threshold of 0.994 with 0.00008 of margin. See
// core.confidence for what changed and why.
const DefaultThreshold = 0.5

// BatchStats summarizes a batch run.
type BatchStats struct {
	Records         int     `json:"records"`
	DecodeFailures  int     `json:"decode_failures"`
	TotalMatches    int     `json:"total_matches"`
	ElapsedSec      float64 `json:"elapsed_sec"`
	AudioHours      float64 `json:"audio_hours"`
	ThroughputRPM   float64 `json:"throughput_records_per_min"`
	LatencyP50Sec   float64 `json:"latency_p50_sec"`
	LatencyP95Sec   float64 `json:"latency_p95_sec"`
	LatencyP99Sec   float64 `json:"latency_p99_sec"`
	PeakRSSMB       float64 `json:"peak_rss_mb"`
	IndexPostings   int     `json:"index_postings"`
	IndexBytesPer1k int64   `json:"index_bytes_per_1000_clips"`
}

// BatchOutput is the batch JSON document.
type BatchOutput struct {
	Recordings []RecordOutput `json:"recordings"`
	Stats      BatchStats     `json:"stats"`
}

func runBatch(ctx context.Context, args []string) error {
	fs := newFlagSet("batch")
	recordsDir := fs.String("records-dir", "", "directory of recordings (required)")
	advertsDir := fs.String("adverts-dir", "", "directory of reference adverts (required)")
	workers := fs.Int("workers", intEnv("FINDER_WORKERS", 8), "number of concurrent workers")
	output := fs.String("output", envOr("FINDER_OUTPUT", "json"), "output format: json|text")
	minConf := fs.Float64("min-confidence", floatEnv("FINDER_MIN_CONFIDENCE", DefaultThreshold),
		"drop matches below this confidence")
	metricsAddr := fs.String("metrics-addr", envOr("FINDER_METRICS_ADDR", ""), "if set, serve /healthz /metrics /debug/pprof here")
	logJSON := fs.Bool("log-json", boolEnv("FINDER_LOG_JSON", false), "log in JSON")
	timeout := fs.Duration("timeout", durEnv("FINDER_TIMEOUT", 0), "overall timeout (0 = none)")
	noBridge := fs.Bool("no-boundary-bridge", false, "disable segment-boundary bridging")
	enableResample := fs.Bool("enable-resample", boolEnv("FINDER_RESAMPLE", false), "enable naive-resample (pitch-shift) recovery path (~3x latency)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *recordsDir == "" || *advertsDir == "" {
		return errors.New("--records-dir and --adverts-dir are required")
	}
	if *timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}

	log := obs.NewLogger(slog.LevelInfo, *logJSON)
	metrics := obs.NewMetrics()
	if *metricsAddr != "" {
		obs.NewServer(*metricsAddr, metrics, log).Start(ctx)
		log.Info("observability server", "addr", *metricsAddr)
	}

	adverts, err := listAudio(*advertsDir)
	if err != nil {
		return err
	}
	records, err := listAudio(*recordsDir)
	if err != nil {
		return err
	}
	if len(adverts) == 0 || len(records) == 0 {
		return fmt.Errorf("need at least one advert and one record (adverts=%d records=%d)", len(adverts), len(records))
	}
	sort.Strings(records)

	start := time.Now()
	cfg := core.DefaultConfig()
	cfg.EnableResample = *enableResample
	det, err := core.LoadDetectorFromPaths(ctx, cfg, adverts, audio.Limits{})
	if err != nil {
		return fmt.Errorf("build index: %w", err)
	}
	ixStats := det.Index().Stats()
	log.Info("index built",
		"adverts", len(adverts), "records", len(records),
		"postings", ixStats.Postings, "distinct_hashes", ixStats.DistinctHash,
		"build_sec", time.Since(start).Seconds())

	var done int64
	// Per-file detection with bounded concurrency and per-file fault isolation.
	results := pipeline.Map(ctx, *workers, records,
		func(ctx context.Context, rec string) (RecordOutput, error) {
			t0 := time.Now()
			matches, derr := det.DetectFile(ctx, rec, audio.Limits{})
			metrics.ObserveLatency(time.Since(t0))
			name := filepath.Base(rec)
			if derr != nil {
				metrics.Inc("decode_failures", 1)
				log.Warn("recording failed", "file", name, "err", derr)
				return RecordOutput{File: name, Error: derr.Error()}, nil // isolate: batch survives
			}
			matches = filterConfidence(matches, *minConf)
			metrics.Inc("emitted_matches", int64(len(matches)))
			// Per-recording progress. Without it a 385-file run is a black box for over an
			// hour, and a single pathological recording is indistinguishable from a hang —
			// which is exactly how the Stage-2 candidate blow-up below stayed hidden.
			secs := time.Since(t0).Seconds()
			n := atomic.AddInt64(&done, 1)
			log.Info("recording done", "file", name, "n", n, "of", len(records),
				"sec", math.Round(secs*10)/10, "matches", len(matches))
			// Emit each match to the log as well as to the JSON. The JSON is written once,
			// at the very end; a crash or a kill in a later stage would otherwise discard
			// an hour of completed detection work.
			for _, m := range matches {
				log.Info("match", "file", name, "ad", m.AdID, "t", round2(m.TimeSec),
					"dur", round2(m.DurationSec), "conf", m.Confidence, "scale", m.ScaleFactor)
			}
			return RecordOutput{File: name, Matches: matches, LatencySec: secs}, nil
		},
		func(s pipeline.Stats) { metrics.SetGauge("in_flight", s.InFlight) },
	)

	recs := make([]RecordOutput, 0, len(records))
	byName := map[string]int{}
	var decodeFailures, totalMatches int
	for _, r := range results {
		if r.Err != nil {
			continue
		}
		if r.Value.Error != "" {
			decodeFailures++
		}
		byName[r.Value.File] = len(recs)
		recs = append(recs, r.Value)
		totalMatches += len(r.Value.Matches)
	}

	// Segment-boundary bridging (§8 class 9 / TestSegmentBoundary).
	if !*noBridge {
		bridged := bridgeBoundaries(ctx, det, records, byName, recs, log, *workers, *minConf)
		totalMatches += bridged
	}

	// Sum audio hours from probing durations of successful records.
	audioHours := estimateAudioHours(ctx, records)

	elapsed := time.Since(start)
	p50, p95, p99 := metrics.Percentiles()
	var bytesPer1k int64
	if ixStats.Ads > 0 {
		bytesPer1k = int64(float64(estimateIndexBytes(ixStats.Postings)) / float64(ixStats.Ads) * 1000)
	}
	stats := BatchStats{
		Records:         len(records),
		DecodeFailures:  decodeFailures,
		TotalMatches:    totalMatches,
		ElapsedSec:      elapsed.Seconds(),
		AudioHours:      audioHours,
		ThroughputRPM:   float64(len(records)) / elapsed.Minutes(),
		LatencyP50Sec:   p50,
		LatencyP95Sec:   p95,
		LatencyP99Sec:   p99,
		PeakRSSMB:       peakRSSMB(),
		IndexPostings:   ixStats.Postings,
		IndexBytesPer1k: bytesPer1k,
	}
	log.Info("batch complete",
		"records", stats.Records, "matches", stats.TotalMatches, "decode_failures", stats.DecodeFailures,
		"elapsed_sec", round1(stats.ElapsedSec), "throughput_rpm", round1(stats.ThroughputRPM),
		"p50", round2(p50), "p95", round2(p95), "p99", round2(p99), "peak_rss_mb", round1(stats.PeakRSSMB))

	if *output == "text" {
		for _, r := range recs {
			fmt.Printf("== %s ==\n", r.File)
			_ = emitMatches(os.Stdout, r.Matches, "text")
		}
		fmt.Printf("\nrecords=%d matches=%d decode_failures=%d elapsed=%.1fs throughput=%.1f rec/min p99=%.2fs peakRSS=%.0fMB\n",
			stats.Records, stats.TotalMatches, stats.DecodeFailures, stats.ElapsedSec, stats.ThroughputRPM, p99, stats.PeakRSSMB)
		return nil
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(BatchOutput{Recordings: recs, Stats: stats})
}

// bridgeBoundaries detects occurrences spanning a 61-minute segment boundary by
// running detection on a short PCM bridge (tail of file i + head of file i+1) for each
// adjacent pair, attributing new matches to the earlier file. Returns the count added.
// stationOf extracts the capture-source key from an archive filename of the form
// YYYY-MM-DD-HH-MM-SS_<station>-<length>.mp3. Recordings only continue into each other
// within one station, so only same-station neighbours may be bridged.
func stationOf(path string) string {
	base := filepath.Base(path)
	i := strings.IndexByte(base, '_')
	if i < 0 || i+1 >= len(base) {
		return ""
	}
	return strings.TrimSuffix(base[i+1:], filepath.Ext(base))
}

func bridgeBoundaries(ctx context.Context, det *core.Detector, records []string, byName map[string]int, recs []RecordOutput, log *slog.Logger, workers int, minConf float64) int {
	const bridgeSec = 45.0 // covers the longest expected advert straddling a boundary

	// Group by station and order by name (which begins with the timestamp), so each
	// bridge joins two consecutive recordings of the SAME stream. Bridging unrelated
	// streams — which plain sorted order does, because the filename starts with the
	// timestamp — both wastes work and manufactures false positives at the junction.
	byStation := map[string][]string{}
	for _, r := range records {
		s := stationOf(r)
		byStation[s] = append(byStation[s], r)
	}
	stations := make([]string, 0, len(byStation))
	for s := range byStation {
		stations = append(stations, s)
		sort.Strings(byStation[s])
	}
	sort.Strings(stations)

	var pairs [][2]string
	for _, s := range stations {
		files := byStation[s]
		for i := 0; i+1 < len(files); i++ {
			pairs = append(pairs, [2]string{files[i], files[i+1]})
		}
	}
	return bridgePairs(ctx, det, pairs, byName, recs, log, bridgeSec, workers, minConf)
}

// bridgePairs runs the boundary check over explicit consecutive pairs, concurrently.
//
// The bridge for one pair is independent of every other pair, so this runs at the same
// worker bound as the main pass. It used to be a plain sequential loop, which made the
// boundary stage the longest single phase of a 385-file archive run — the per-file pass
// finished in 39 minutes and this stage then ran on one core for another 20+. Detection
// is applied to the shared results AFTER the concurrent phase, walking pairs in their
// original order, so the emitted output stays byte-identical run to run (§6).
func bridgePairs(ctx context.Context, det *core.Detector, pairs [][2]string, byName map[string]int, recs []RecordOutput, log *slog.Logger, bridgeSec float64, workers int, minConf float64) int {
	if workers < 1 {
		workers = 1
	}
	type bridgeFind struct {
		nameI   string
		matches []core.Match // times already rebased onto the earlier recording
	}
	var doneN int64
	results := pipeline.Map(ctx, workers, pairs,
		func(ctx context.Context, pair [2]string) (bridgeFind, error) {
			out := bridgeFind{nameI: filepath.Base(pair[0])}
			defer func() {
				if n := atomic.AddInt64(&doneN, 1); n%50 == 0 || int(n) == len(pairs) {
					log.Info("boundary bridges done", "n", n, "of", len(pairs))
				}
			}()
			durI, err := probeDuration(ctx, pair[0])
			if err != nil || durI <= bridgeSec {
				return out, nil
			}
			tail, err := audio.DecodeSegment(ctx, pair[0], durI-bridgeSec, bridgeSec, decodeOpts())
			if err != nil {
				return out, nil
			}
			head, err := audio.DecodeSegment(ctx, pair[1], 0, bridgeSec, decodeOpts())
			if err != nil {
				return out, nil
			}
			bridge := audio.PCM{SampleRate: tail.SampleRate,
				Samples: append(append([]float32{}, tail.Samples...), head.Samples...)}
			matches, err := det.DetectPCM(ctx, bridge)
			if err != nil {
				return out, nil
			}
			boundary := durI - bridgeSec
			for _, m := range matches {
				// Keep only occurrences that actually straddle the boundary.
				if m.TimeSec+m.DurationSec <= bridgeSec || m.TimeSec >= bridgeSec {
					continue // fully within one side — already found by the per-file pass
				}
				// Bridge matches go through the SAME confidence gate as the per-file pass.
				// Skipping it published detections the operating threshold rejects — a bug
				// the previous, saturated confidence model hid, because everything it
				// produced scored above the threshold anyway.
				if m.Confidence < minConf {
					continue
				}
				m.TimeSec = boundary + m.TimeSec
				out.matches = append(out.matches, m)
			}
			return out, nil
		},
		nil,
	)

	added := 0
	for _, r := range results {
		if r.Err != nil {
			continue
		}
		idx, ok := byName[r.Value.nameI]
		if !ok {
			continue
		}
		for _, m := range r.Value.matches {
			if alreadyHave(recs[idx].Matches, m) {
				continue
			}
			recs[idx].Matches = append(recs[idx].Matches, m)
			added++
			log.Info("boundary occurrence", "file", r.Value.nameI, "ad", m.AdID, "t", round2(m.TimeSec))
		}
	}
	return added
}

func alreadyHave(ms []core.Match, m core.Match) bool {
	for _, x := range ms {
		if x.AdID == m.AdID && absf(x.TimeSec-m.TimeSec) <= 1.5 {
			return true
		}
	}
	return false
}

func decodeOpts() audio.DecodeOptions {
	return audio.DecodeOptions{SampleRate: 16000}
}

func listAudio(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if audioExts[strings.ToLower(filepath.Ext(e.Name()))] {
			out = append(out, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(out)
	return out, nil
}

func probeDuration(ctx context.Context, path string) (float64, error) {
	info, err := audio.Probe(ctx, path, audio.Limits{})
	if err != nil {
		return 0, err
	}
	return info.DurationS, nil
}

func estimateAudioHours(ctx context.Context, records []string) float64 {
	var total float64
	for _, r := range records {
		if d, err := probeDuration(ctx, r); err == nil {
			total += d
		}
	}
	return total / 3600
}

// estimateIndexBytes approximates the resident index size: each posting is ~9 bytes
// (adIdx u32 + scaleTag u8 + time f32) plus map overhead.
func estimateIndexBytes(postings int) int64 { return int64(postings) * 12 }

// peakRSSMB reads VmHWM (peak resident set size) from /proc/self/status.
func peakRSSMB() float64 {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "VmHWM:") {
			f := strings.Fields(line)
			if len(f) >= 2 {
				if kb, err := strconv.ParseFloat(f[1], 64); err == nil {
					return kb / 1024
				}
			}
		}
	}
	return 0
}

func absf(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
func round1(f float64) float64 { return float64(int(f*10+0.5)) / 10 }
func round2(f float64) float64 { return float64(int(f*100+0.5)) / 100 }

package core

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/index"
)

// DefaultConfig returns the detector configuration tuned on the dev set.
func DefaultConfig() Config {
	var c Config
	c.withDefaults()
	return c
}

// LoadDetectorFromPaths decodes each advert file and builds a detector. Advert IDs
// are the file basenames (without extension). Decoding uses lim for validation.
func LoadDetectorFromPaths(ctx context.Context, cfg Config, advertPaths []string, lim audio.Limits) (*Detector, error) {
	cfg.withDefaults()
	sorted := append([]string(nil), advertPaths...)
	sort.Strings(sorted) // deterministic index order
	det := &Detector{cfg: cfg, ix: index.New(nil), byIdx: map[uint32]int{}, quadPost: map[uint32][]quadPosting{}, baseByAd: map[string]int{}, refPeaks: map[string][]features.QPeak{}}
	for _, p := range sorted {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		pcm, err := audio.Decode(ctx, p, audio.DecodeOptions{SampleRate: features.SampleRate, Limits: lim})
		if err != nil {
			return nil, fmt.Errorf("advert %s: %w", filepath.Base(p), err)
		}
		if err := det.AddRef(RefClip{AdID: adID(p), Samples: pcm.Samples}); err != nil {
			return nil, fmt.Errorf("index advert %s: %w", filepath.Base(p), err)
		}
	}
	return det, nil
}

// DetectFile decodes a recording and runs detection. When the resample path is
// enabled, it also runs recording-side pitch compensation from the file to recover
// naive-resampled (pitch-shifted) airings, merging the results.
func (d *Detector) DetectFile(ctx context.Context, recordPath string, lim audio.Limits) ([]Match, error) {
	pcm, err := audio.Decode(ctx, recordPath, audio.DecodeOptions{SampleRate: d.cfg.SampleRate, Limits: lim})
	if err != nil {
		return nil, err
	}
	baseRaws, cands, err := d.detectRawsWithCands(ctx, pcm, d.cfg.MinPeakAgree)
	if err != nil {
		return nil, err
	}
	// The naive-resample class is handled at INDEX time: resampled copies of each
	// reference are indexed alongside the original (see AddRef), so a pitch-shifted
	// airing retrieves the variant that underwent the same transformation. That replaces
	// the older per-window pitch-compensation pass, which re-decoded audio through ffmpeg
	// for every candidate window and was far too slow for 61-minute recordings.
	_ = cands
	return d.matchesFromRaws(baseRaws), nil
}

// DetectPCM runs detection on an already-decoded recording (used for segment-boundary
// bridging in batch mode).
func (d *Detector) DetectPCM(ctx context.Context, pcm audio.PCM) ([]Match, error) {
	return d.Detect(ctx, pcm)
}

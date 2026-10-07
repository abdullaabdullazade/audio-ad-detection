// Package eval defines the ground-truth manifest (labels.json) schema and the
// scoring/metrics used by the `finder eval` subcommand. The schema is authored here
// and documented in the README; the adversarial corpus generator emits it and the
// eval harness consumes it.
package eval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
)

// Distortion class identifiers, one per §8 test condition, used for the per-class
// recall/precision breakdown required by §7.
const (
	ClassClean           = "clean"               // true negative: air with no target clips
	ClassBaseline        = "baseline"            // clean insertion, no distortion
	ClassTranscode       = "transcode"           // bitrate/codec/resample change
	ClassGainDynamics    = "gain_dynamics"       // gain, loudnorm, compression, EQ, clip
	ClassTempo           = "time_scale_tempo"    // pitch-preserving linear time-scale
	ClassResample        = "time_scale_resample" // naive-resample time-scale (pitch shifts)
	ClassPauseCompress   = "pause_compress"      // non-linear shortening of internal pauses
	ClassPartial         = "partial"             // first/last 50% or fade
	ClassOverlay         = "overlay"             // announcer/music bed over <=50%
	ClassJingleTrap      = "jingle_trap"         // two ads sharing a 3s bed
	ClassBackToBack      = "back_to_back"        // same clip aired twice in a row
	ClassSegmentBoundary = "segment_boundary"    // occurrence spanning two 61-min files
	ClassGap             = "gap"                 // 0.2-0.5s recorder gap inside an occurrence
	ClassEdgeSilence     = "edge_silence"        // ref with leading/trailing silence + ID3
)

// Occurrence is a single ground-truth airing of a reference clip in a recording.
type Occurrence struct {
	AdID        string  `json:"ad_id"`
	StartSec    float64 `json:"start_sec"`              // ground-truth start t_gt
	DurationSec float64 `json:"duration_sec"`           // aired duration
	Class       string  `json:"class"`                  // §8 distortion class
	ScaleFactor float64 `json:"scale_factor,omitempty"` // expected linear time-scale (1.0 = none)
}

// Recording is one media file plus its ground-truth occurrences.
type Recording struct {
	File        string       `json:"file"`        // path relative to the records dir
	Occurrences []Occurrence `json:"occurrences"` // empty for a true-negative (clean) recording
}

// Labels is the full ground-truth manifest.
type Labels struct {
	SampleRate  int         `json:"sample_rate,omitempty"`
	MatchTolSec float64     `json:"match_tol_sec,omitempty"` // TP tolerance; default 1.5s per §2
	Recordings  []Recording `json:"recordings"`
}

// LoadLabels reads and validates a labels.json manifest.
func LoadLabels(path string) (*Labels, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}
	var l Labels
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&l); err != nil {
		return nil, fmt.Errorf("parse labels %s: %w", path, err)
	}
	if len(l.Recordings) == 0 {
		return nil, fmt.Errorf("labels %s: no recordings", path)
	}
	for i, r := range l.Recordings {
		if r.File == "" {
			return nil, fmt.Errorf("labels %s: recording %d has empty file", path, i)
		}
		for j, o := range r.Occurrences {
			if o.AdID == "" || o.StartSec < 0 || o.DurationSec <= 0 {
				return nil, fmt.Errorf("labels %s: recording %q occurrence %d invalid", path, r.File, j)
			}
		}
	}
	return &l, nil
}

// Save writes the manifest as indented JSON.
func (l *Labels) Save(path string) error {
	b, err := json.MarshalIndent(l, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

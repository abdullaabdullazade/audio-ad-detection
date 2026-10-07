package core

import (
	"context"
	"math"
	"sort"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/match"
)

// pitchCompFactors are the recording-side pitch/time compensation factors tried for
// naive-resample robustness. Each applies ffmpeg asetrate=SR/f + high-quality
// (soxr) aresample, shifting the whole recording's pitch and time; the base pipeline
// then matches a resampled airing whose distortion the factor cancels.
var pitchCompFactors = []float64{0.95, 1.05}

// resampleScaleTol accepts only detections whose residual scale is within this of 1.0
// — i.e. the compensation factor nearly cancelled the airing's resampling. This both
// selects the correct factor and rejects the wrong-factor spurious detections.
const resampleScaleTol = 0.05

// resampleMinConf gates resample-path detections to high confidence, since a
// well-compensated true airing scores near the top; this keeps the path from
// perturbing the operating threshold with borderline matches.
const resampleMinConf = 0.97

// resampleWindowPadSec pads each candidate window before pitch compensation.
const resampleWindowPadSec = 6.0

// resampleMatches recovers naive-resampled (pitch-shifted) airings.
//
// Cost note: compensating the WHOLE recording at each pitch factor is correct but far
// too slow on 61-minute files (measured p99 ≈ 610 s vs a 180 s budget). Instead we use
// the cheap pitch-tolerant retrieval already available (the pitch-invariant landmark
// hashes) to locate candidate WINDOWS first, then pitch-compensate only those short
// windows (~30-40 s each) and re-run the proven pipeline there. That is the same
// "generate the fingerprint first, then adjust it" principle used in production
// broadcast-monitoring systems, and it makes the path budget-viable.
func (d *Detector) resampleMatches(ctx context.Context, rec audio.PCM, baseCands []candWindow) ([]Match, error) {
	if len(baseCands) == 0 {
		return nil, nil
	}
	sr := d.cfg.SampleRate
	var out []Match
	for _, cw := range baseCands {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		a := int((cw.startSec - resampleWindowPadSec) * float64(sr))
		b := int((cw.endSec + resampleWindowPadSec) * float64(sr))
		if a < 0 {
			a = 0
		}
		if b > len(rec.Samples) {
			b = len(rec.Samples)
		}
		if b-a < features.FFTSize*4 {
			continue
		}
		winStartSec := float64(a) / float64(sr)
		win := rec.Samples[a:b]

		for _, f := range pitchCompFactors {
			// Pitch-compensate the WINDOW with ffmpeg's high-quality soxr resampler. A
			// naive linear interpolation low-passes the spectrum enough to break landmark
			// retrieval; soxr preserves it. The window is short, so this is cheap.
			comp, err := audio.PitchShiftPCM(ctx, audio.PCM{SampleRate: sr, Samples: win}, 1.0/f)
			if err != nil || len(comp.Samples) < features.FFTSize*2 {
				continue
			}
			raws, err := d.detectRaws(ctx, comp, -1)
			if err != nil {
				return nil, err
			}
			for _, r := range raws {
				conf := confidence(r.score, r.peakAgree)
				if math.Abs(r.scale-1.0) > resampleScaleTol || conf < resampleMinConf {
					continue // not well-compensated / not confident enough
				}
				// Window time -> recording time: undo the f stretch, then add the offset.
				out = append(out, Match{
					AdID:        r.adID,
					TimeSec:     round3(winStartSec + r.startSec/f),
					DurationSec: round3((r.endSec - r.startSec) / f),
					Confidence:  conf,
					Score:       round3(r.score),
					ScaleFactor: round3(f),
				})
			}
		}
	}
	return out, nil
}

// candWindow is a coarse time window where some ad may be present (from the cheap
// pitch-tolerant retrieval), used to bound the resample path's work.
type candWindow struct {
	startSec float64
	endSec   float64
}

// resampleCandidateWindows derives coarse candidate windows from Stage-1 retrieval,
// merging overlapping hypotheses. It is cheap: retrieval already ran for the base pass.
func resampleCandidateWindows(cands []match.Candidate, d *Detector) []candWindow {
	var ws []candWindow
	for _, c := range cands {
		ri, ok := d.byIdx[c.AdIdx]
		if !ok {
			continue
		}
		dur := d.refs[ri].origDur
		ws = append(ws, candWindow{startSec: c.StartSec, endSec: c.StartSec + dur})
	}
	sort.Slice(ws, func(i, j int) bool { return ws[i].startSec < ws[j].startSec })
	var merged []candWindow
	for _, w := range ws {
		if n := len(merged); n > 0 && w.startSec <= merged[n-1].endSec {
			if w.endSec > merged[n-1].endSec {
				merged[n-1].endSec = w.endSec
			}
			continue
		}
		merged = append(merged, w)
	}
	return merged
}

// mergeMatches combines base and resample-path matches. A non-empty resample result
// means the recording is pitch-shifted (the base pipeline only found it after
// compensation), and on such a recording the base pipeline's own matches are
// high-confidence but MIS-TIMED partial/cross-ad false positives. So when the resample
// path finds anything, only base matches that coincide with a resample find (same ad,
// within the 1.5 s tolerance) are kept; the rest are dropped. Remaining duplicates
// collapse to the higher confidence. Output order is stable.
// resampleAgreeCut: an ad the base pass matched with peak agreement at least this high
// is present un-pitch-shifted, so the resample path must not override it.
const resampleAgreeCut = 0.6

func mergeMatches(base, extra []Match, baseAgree map[string]float64) []Match {
	claimed := map[string]bool{} // ads the resample path found (pitch-shifted airings)
	for _, e := range extra {
		claimed[e.AdID] = true
	}
	all := make([]Match, 0, len(base)+len(extra))
	for _, b := range base {
		// Drop a base match ONLY when it is likely a pitch-shift false positive: the base
		// matched this ad only weakly (low peak agreement) AND the resample path found the
		// same ad after compensation. A legitimate but weakly-aligned match (e.g. a
		// partial airing — not pitch-shifted, so the resample path does NOT claim it) is
		// kept.
		if baseAgree[b.AdID] < resampleAgreeCut && claimed[b.AdID] {
			continue
		}
		all = append(all, b)
	}
	for _, e := range extra {
		// Only trust the resample path for ads the base pass could NOT match cleanly.
		if baseAgree[e.AdID] < resampleAgreeCut {
			all = append(all, e)
		}
	}

	sort.SliceStable(all, func(i, j int) bool { return all[i].Confidence > all[j].Confidence })
	var kept []Match
	for _, m := range all {
		dup := false
		for _, k := range kept {
			if k.AdID == m.AdID && math.Abs(k.TimeSec-m.TimeSec) <= 1.5 {
				dup = true
				break
			}
		}
		if !dup {
			kept = append(kept, m)
		}
	}
	sort.Slice(kept, func(i, j int) bool {
		if kept[i].TimeSec != kept[j].TimeSec {
			return kept[i].TimeSec < kept[j].TimeSec
		}
		return kept[i].AdID < kept[j].AdID
	})
	return kept
}

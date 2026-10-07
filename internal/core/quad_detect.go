package core

import (
	"context"
	"sort"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/fingerprint"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/match"
)

// quadCandidate is a scale-invariant retrieval hypothesis with recovered time and
// frequency (pitch) scale factors.
type quadCandidate struct {
	adID  string
	start float64
	sTime float64 // aired/ref time scale
	sFreq float64 // aired/ref frequency (pitch) scale
	votes int
}

func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

// quadDetect runs the quad-based (pitch/time-scale-invariant) retrieval and a
// pitch-compensated Stage-2 verification, supplementing the landmark path so
// naive-resampled (pitch-shifted) airings are found. Returns pre-dedup detections.
func (d *Detector) quadDetect(ctx context.Context, rec audio.PCM, recSpec *features.Spectrogram) ([]rawDet, error) {
	qhs := fingerprint.Quads(features.QuadPeaks(recSpec), fingerprint.QueryQuadParams)
	if len(qhs) == 0 {
		return nil, nil
	}

	// Per-ad matches: keep the raw quad time positions and per-quad scale estimates.
	// Following the paper's verification, we first estimate the AD's scale robustly
	// (median s_time/s_freq over all its matches), THEN recompute each quad's offset
	// using that stable median — this removes the per-quad noise that start =
	// AtSec - atSec*s_time would otherwise amplify — and histogram-bin the offsets.
	type est struct{ qAt, rAt, sTime, sFreq float64 }
	byAd := map[string][]est{}
	for _, qh := range qhs {
		for _, p := range d.quadPost[qh.Key] {
			if p.abx <= 0 || p.aby <= 0 {
				continue
			}
			sTime := qh.ABx / p.abx
			sFreq := qh.ABy / p.aby
			if sTime < 0.90 || sTime > 1.11 || sFreq < 0.90 || sFreq > 1.11 {
				continue
			}
			byAd[p.adID] = append(byAd[p.adID], est{qh.AtSec, p.atSec, sTime, sFreq})
		}
	}

	const binSec = 0.15
	const minVotes = 4
	var pruned []quadCandidate
	for ad, es := range byAd {
		if len(es) < minVotes {
			continue
		}
		sts := make([]float64, len(es))
		sfs := make([]float64, len(es))
		for i, e := range es {
			sts[i], sfs[i] = e.sTime, e.sFreq
		}
		sTimeMed, sFreqMed := median(sts), median(sfs)

		// Histogram of scale-corrected offsets (with neighbour merging for the peak).
		hist := map[int32]int{}
		offByBin := map[int32][]float64{}
		for _, e := range es {
			off := e.qAt - e.rAt*sTimeMed
			if off < -1 {
				continue
			}
			b := int32(off / binSec)
			hist[b]++
			offByBin[b] = append(offByBin[b], off)
		}
		// Emit every bin whose (self + neighbours) count clears the threshold (supports
		// multiple occurrences), suppressing adjacent duplicates.
		var bins []int32
		for b := range hist {
			bins = append(bins, b)
		}
		sort.Slice(bins, func(i, j int) bool { return bins[i] < bins[j] })
		lastEmit := int32(-1 << 30)
		for _, b := range bins {
			cnt := hist[b] + hist[b-1] + hist[b+1]
			if cnt < minVotes || b-lastEmit < 3 {
				continue
			}
			var offs []float64
			offs = append(offs, offByBin[b-1]...)
			offs = append(offs, offByBin[b]...)
			offs = append(offs, offByBin[b+1]...)
			pruned = append(pruned, quadCandidate{
				adID:  ad,
				start: median(offs),
				sTime: sTimeMed,
				sFreq: sFreqMed,
				votes: cnt,
			})
			lastEmit = b
		}
	}

	sr := d.cfg.SampleRate
	slack := int(d.cfg.WindowSlackSec * float64(sr))
	band := features.SecToFrame(d.cfg.WindowSlackSec)
	var aligner match.Aligner
	var raws []rawDet

	for _, c := range pruned {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ri, ok := d.baseByAd[c.adID]
		if !ok {
			continue
		}
		ref := d.refs[ri]
		if ref.frames < 2 {
			continue
		}
		startSample := int(c.start * float64(sr))
		airedDur := ref.origDur * c.sTime
		ws := startSample - slack
		if ws < 0 {
			ws = 0
		}
		we := startSample + int(airedDur*float64(sr)) + 2*slack
		if we > len(rec.Samples) {
			we = len(rec.Samples)
		}
		if we-ws < features.FFTSize {
			continue
		}
		win := rec.Samples[ws:we]
		// Undo the pitch shift: resample by 1/sFreq restores the reference pitch (and
		// stretches time by sFreq). Remaining time scale vs ref = sTime*sFreq.
		comp := features.Resample(win, 1.0/c.sFreq)
		if len(comp) < features.FFTSize {
			continue
		}
		compMel := features.LogMel(features.STFT(comp))
		slope := c.sTime * c.sFreq
		wsSec := float64(ws) / float64(sr)
		// Candidate start within the compensated window (in comp-time = orig-window-time*sFreq).
		startInComp := (c.start - wsSec) * c.sFreq
		res := aligner.Align(ref.mel, compMel, features.SecToFrame(startInComp), slope, band)
		if !res.OK || res.Coverage < d.cfg.AcceptCoverage || res.Score < d.cfg.AcceptScore {
			continue
		}
		// Map compensated-window time back to recording time (undo the sFreq stretch).
		recStart := wsSec + res.StartSec/c.sFreq
		recEnd := wsSec + res.EndSec/c.sFreq
		reported := c.sTime
		if reported < 0.90 || reported > 1.11 {
			continue
		}
		raws = append(raws, rawDet{
			adID:        c.adID,
			refDur:      ref.origDur,
			startSec:    recStart,
			endSec:      recEnd,
			impliedClip: recStart - (res.RefStartSec/c.sFreq)*reported,
			score:       res.Score,
			coverage:    res.Coverage,
			scale:       reported,
		})
	}
	return raws, nil
}

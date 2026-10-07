package core

import (
	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
)

// peakGrid indexes recording spectral peaks by frame for fast neighbour lookup.
type peakGrid struct {
	byFrame map[int][]int // frame -> bins
}

func buildPeakGrid(peaks []features.QPeak) peakGrid {
	g := peakGrid{byFrame: make(map[int][]int, len(peaks))}
	for _, p := range peaks {
		g.byFrame[p.Frame] = append(g.byFrame[p.Frame], p.Bin)
	}
	return g
}

const (
	peakAgreeDT = 3 // frame tolerance
	peakAgreeDF = 2 // bin tolerance
)

// hasPeakNear reports whether the recording has a peak near (frame, bin).
func (g peakGrid) hasPeakNear(frame, bin int) bool {
	for df := -peakAgreeDT; df <= peakAgreeDT; df++ {
		bins := g.byFrame[frame+df]
		for _, b := range bins {
			if b >= bin-peakAgreeDF && b <= bin+peakAgreeDF {
				return true
			}
		}
	}
	return false
}

// peakAgreement is the fraction of the MATCHED reference span's spectral peaks that
// have a recording peak at the SAME (linear) frequency bin once projected onto the
// recording. It is the key pitch-shift discriminator: unlike the log-mel score,
// log-band votes and coverage (all pitch-tolerant), it uses absolute frequency bins, so
// a naive-resampled (pitch-shifted) false positive — whose recording peaks sit at
// shifted bins — scores low, while a genuine same-pitch airing scores high. Restricting
// to the matched span [refStartFrame, refEndFrame] keeps partial airings high (their
// aired half matches) rather than penalizing them for the absent half.
//
// Reference peaks are placed using refToRec, the EXACT per-frame mapping from the DTW
// warping path. Projecting linearly with the observed scale instead would accumulate
// drift (a 5% scale error is ~0.7 s after 14 s — far beyond the few-frame tolerance),
// which is what made legitimate partial airings look like non-matches.
// refPeaks are in FULL-reference frame coordinates; refToRec is indexed by TEMPLATE
// frames, so tmplOffset (the template's start within the reference, in frames) shifts
// between them. refStartFrame/refEndFrame bound the considered range in full-reference
// coordinates.
// binScale is the frequency scaling the aired copy underwent (naive resample): a
// reference peak at bin b appears at bin b*binScale in the recording.
// pitchSearchScales are the frequency scalings searched by peakAgreementBest. They
// cover the ±5% naive-resample range plus unity.
var pitchSearchScales = []float64{1.0, 0.95, 1.05, 0.975, 1.025}

// peakAgreementBest is peakAgreement with the frequency scaling ESTIMATED from the data
// instead of assumed. This implements Panako's verification principle: a genuine airing
// — pitch-shifted or not — has ONE consistent frequency offset at which (almost) all
// reference peaks line up, whereas a spurious match against unrelated broadcast audio
// lines up at no scaling at all. Searching the scale therefore keeps the gate's
// precision on real air while no longer punishing legitimately pitch-shifted airings.
// It returns the best agreement and the scaling that achieved it.
func peakAgreementBest(refPeaks []features.QPeak, grid peakGrid, refToRec []int, refStartFrame, refEndFrame, tmplOffset int) (float64, float64) {
	best, bestScale := 0.0, 1.0
	for _, s := range pitchSearchScales {
		if a := peakAgreement(refPeaks, grid, refToRec, refStartFrame, refEndFrame, tmplOffset, s); a > best {
			best, bestScale = a, s
		}
	}
	return best, bestScale
}

func peakAgreement(refPeaks []features.QPeak, grid peakGrid, refToRec []int, refStartFrame, refEndFrame, tmplOffset int, binScale float64) float64 {
	matched, considered := 0, 0
	for _, rp := range refPeaks {
		if rp.Frame < refStartFrame || rp.Frame > refEndFrame {
			continue
		}
		ti := rp.Frame - tmplOffset
		if ti < 0 || ti >= len(refToRec) {
			continue
		}
		recFrame := refToRec[ti]
		if recFrame < 0 {
			continue // reference frame not on the path
		}
		considered++
		bin := rp.Bin
		if binScale != 1 {
			bin = int(float64(rp.Bin)*binScale + 0.5)
		}
		if grid.hasPeakNear(recFrame, bin) {
			matched++
		}
	}
	if considered == 0 {
		return 0
	}
	return float64(matched) / float64(considered)
}

package core

import (
	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
)

// Rigid head snapping.
//
// Subsequence DTW is locally elastic, so on a clip whose content repeats it can settle
// on a time-SHIFTED copy of the same material and still report high coverage and score:
// the warping absorbs the mismatch. A RIGID (non-elastic) correlation of the reference's
// opening seconds cannot absorb it — at a shifted position the opening lines up against
// the wrong material and scores lower. Searching that rigid score around the DTW start
// therefore pulls the reported timestamp back onto the true occurrence start. Measured on
// the dev corpus: naive-resampled airings that DTW placed 2-5 s late snap back to within
// ~1 s of ground truth, while already-correct starts move only a few tens of ms.
const (
	headSnapSec       = 4.0 // how much of the reference's opening to correlate rigidly
	headSnapRadiusSec = 8.0 // how far around the DTW start to search
	headSnapStepSec   = 0.032
)

// rigidHeadScore is the mean cosine of the reference's first headSec against the
// recording starting at startSec, advanced at the given time scale WITHOUT warping.
func rigidHeadScore(refMel, recMel [][]float32, startSec, scale, headSec float64) float64 {
	n := features.SecToFrame(headSec)
	if n > len(refMel) {
		n = len(refMel)
	}
	if n < 2 {
		return 0
	}
	s0 := features.SecToFrame(startSec)
	var sum float64
	cnt := 0
	for i := 0; i < n; i++ {
		j := s0 + int(float64(i)*scale+0.5)
		if j < 0 || j >= len(recMel) {
			continue
		}
		sum += features.Cosine(refMel[i], recMel[j])
		cnt++
	}
	if cnt == 0 {
		return 0
	}
	return sum / float64(cnt)
}

// snapScales are the time scales tried alongside the alignment's own estimate. DTW's
// scale estimate is unreliable on naive-resampled airings (measured 0.89 where the truth
// is 0.95), and a wrong scale drags the rigid correlation off over several seconds, so
// the search covers the ±5% grid explicitly.
var snapScales = []float64{1.0, 1.0 / 1.05, 1.0 / 0.95, 1.0 / 1.025, 1.0 / 0.975}

// snapStart returns the start (seconds) near startSec that maximizes the rigid head
// correlation, together with that score, searching over both the offset and a small set
// of plausible time scales. Only full-clip alignments should be snapped: for a partial
// airing the reference's opening may legitimately be absent.
func snapStart(refMel, recMel [][]float32, startSec, scale float64) (float64, float64) {
	scales := append([]float64{scale}, snapScales...)
	best, bestT := -1.0, startSec
	for _, sc := range scales {
		for off := -headSnapRadiusSec; off <= headSnapRadiusSec; off += headSnapStepSec {
			st := startSec + off
			if st < 0 {
				continue
			}
			if s := rigidHeadScore(refMel, recMel, st, sc, headSnapSec); s > best {
				best, bestT = s, st
			}
		}
	}
	return bestT, best
}

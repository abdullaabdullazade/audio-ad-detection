package match

import (
	"math"
	"sort"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
)

// AlignResult is the outcome of aligning a reference against a recording window.
type AlignResult struct {
	OK          bool
	StartSec    float64 // recording time where the matched audio begins
	EndSec      float64 // recording time where it ends
	Score       float64 // robust similarity in [0,1] (trimmed-mean cosine along the run)
	Coverage    float64 // fraction of the reference covered by the matched run
	Scale       float64 // observed linear time-scale (aired/ref)
	RefStartSec float64 // offset into the reference where the run begins (0 = clip start)
	RefEndSec   float64 // offset into the reference where the run ends
	// FullStartSec/FullEndSec are the recording times where reference frame 0 / the last
	// reference frame align on the full DTW path — the clip start/end for a full airing,
	// which (unlike the Kadane sub-run) does not drift on self-similar content.
	FullStartSec float64
	FullEndSec   float64
	// RefToRec maps each reference frame index to the recording frame it aligned to on
	// the DTW path (-1 where unmapped). Verification should use this exact, non-linear
	// mapping rather than projecting with Scale: a small scale error accumulates into
	// seconds of drift across a clip, which would break frame-accurate peak matching.
	RefToRec []int
}

// Aligner runs banded subsequence DTW (full reference consumed, free start on the
// recording axis) with reusable buffers. The band absorbs residual scale error and
// non-linear drift (pause compression, recorder gaps). Not safe for concurrent use.
type Aligner struct {
	prev []float64
	cur  []float64
	dir  []uint8 // Rf*W: 0=diag 1=up(ref adv) 2=left(rec adv) 3=none
}

const (
	coverFloor   = 0.55 // per-frame cosine counted as covered
	trimFraction = 0.30
	infCost      = 1e18
)

// Align aligns the full reference against the recording near startFrame along slope
// scaleExpected within ±band recording frames.
func (a *Aligner) Align(ref, rec [][]float32, startFrame int, scaleExpected float64, band int) AlignResult {
	Rf := len(ref)
	if Rf < 2 || len(rec) < 2 || band < 1 {
		return AlignResult{}
	}
	W := 2*band + 1
	if cap(a.prev) < W {
		a.prev = make([]float64, W)
		a.cur = make([]float64, W)
	}
	prev := a.prev[:W]
	cur := a.cur[:W]
	need := Rf * W
	if cap(a.dir) < need {
		a.dir = make([]uint8, need)
	}
	dir := a.dir[:need]

	center := func(i int) int { return startFrame + int(math.Round(float64(i)*scaleExpected)) }
	local := func(i, j int) float64 {
		if j < 0 || j >= len(rec) {
			return infCost
		}
		return 1 - features.Cosine(ref[i], rec[j])
	}

	c0 := center(0)
	anyFinite := false
	for k := 0; k < W; k++ {
		lc := local(0, c0-band+k)
		prev[k] = lc
		dir[k] = 3
		if lc < infCost {
			anyFinite = true
		}
	}
	if !anyFinite {
		return AlignResult{}
	}

	for i := 1; i < Rf; i++ {
		ci := center(i)
		shift := ci - center(i-1)
		base := i * W
		for k := 0; k < W; k++ {
			lc := local(i, ci-band+k)
			if lc >= infCost {
				cur[k] = infCost
				dir[base+k] = 3
				continue
			}
			best := infCost
			var d uint8 = 3
			if kp := k - 1 + shift; kp >= 0 && kp < W && prev[kp] < best {
				best, d = prev[kp], 0
			}
			if kp := k + shift; kp >= 0 && kp < W && prev[kp] < best {
				best, d = prev[kp], 1
			}
			if k > 0 && cur[k-1] < best {
				best, d = cur[k-1], 2
			}
			if best >= infCost {
				cur[k] = infCost
				dir[base+k] = 3
			} else {
				cur[k] = lc + best
				dir[base+k] = d
			}
		}
		copy(prev, cur)
	}

	endK := -1
	bestEnd := infCost
	for k := 0; k < W; k++ {
		if prev[k] < bestEnd {
			bestEnd, endK = prev[k], k
		}
	}
	if endK < 0 || bestEnd >= infCost {
		return AlignResult{}
	}

	// Backtrack, collecting the (refFrame, recFrame, cosine) path (end -> start).
	type step struct {
		i, j int
		c    float64
	}
	path := make([]step, 0, Rf)
	i, k := Rf-1, endK
	for {
		j := center(i) - band + k
		path = append(path, step{i, j, features.Cosine(ref[i], rec[j])})
		if i == 0 {
			break
		}
		d := dir[i*W+k]
		shift := center(i) - center(i-1)
		switch d {
		case 0:
			i, k = i-1, k-1+shift
		case 1:
			i, k = i-1, k+shift
		case 2:
			k--
		default:
			i = 0
		}
		if k < 0 {
			k = 0
		}
		if k >= W {
			k = W - 1
		}
	}
	for l, r := 0, len(path)-1; l < r; l, r = l+1, r-1 {
		path[l], path[r] = path[r], path[l]
	}

	// Extract the best contiguous matched sub-run (Kadane over cosine>=coverFloor).
	// For a full airing this is the whole path (start = clip start); for a partial
	// airing it is the present half, giving the correct span and start; interior
	// dips from an overlay are tolerated.
	lo, hi := kadaneRun(path, func(s step) bool { return s.c >= coverFloor })
	if lo < 0 {
		return AlignResult{}
	}
	refStart, refEnd := path[lo].i, path[hi].i
	recStart, recEnd := path[lo].j, path[hi].j
	if recStart < 0 {
		recStart = 0
	}

	coverage := float64(refEnd-refStart+1) / float64(Rf)
	if coverage > 1 {
		coverage = 1
	}
	runCos := make([]float64, 0, hi-lo+1)
	for l := lo; l <= hi; l++ {
		runCos = append(runCos, path[l].c)
	}
	score := trimmedMean(runCos, trimFraction)

	scale := scaleExpected
	if refEnd > refStart {
		scale = float64(recEnd-recStart) / float64(refEnd-refStart)
	}

	// Exact reference-frame -> recording-frame mapping from the warping path.
	refToRec := make([]int, Rf)
	for i := range refToRec {
		refToRec[i] = -1
	}
	for _, s := range path {
		if s.i >= 0 && s.i < Rf {
			refToRec[s.i] = s.j
		}
	}

	return AlignResult{
		OK:           true,
		RefToRec:     refToRec,
		StartSec:     features.FrameToSec(recStart),
		EndSec:       features.FrameToSec(recEnd),
		Score:        score,
		Coverage:     coverage,
		Scale:        scale,
		RefStartSec:  features.FrameToSec(refStart),
		RefEndSec:    features.FrameToSec(refEnd),
		FullStartSec: features.FrameToSec(path[0].j),
		FullEndSec:   features.FrameToSec(path[len(path)-1].j),
	}
}

// kadaneRun returns the [lo,hi] index range of the maximum-sum subarray where each
// element contributes +1 if good(elem) else -1. Returns (-1,-1) if no good element.
func kadaneRun[T any](xs []T, good func(T) bool) (int, int) {
	bestSum, bestLo, bestHi := 0, -1, -1
	curSum, curLo := 0, 0
	for i, x := range xs {
		v := -1
		if good(x) {
			v = 1
		}
		if curSum <= 0 {
			curSum, curLo = v, i
		} else {
			curSum += v
		}
		if curSum > bestSum {
			bestSum, bestLo, bestHi = curSum, curLo, i
		}
	}
	return bestLo, bestHi
}

// trimmedMean drops the lowest `frac` fraction of values and averages the rest.
func trimmedMean(v []float64, frac float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := make([]float64, len(v))
	copy(s, v)
	sort.Float64s(s)
	drop := int(float64(len(s)) * frac)
	s = s[drop:]
	if len(s) == 0 {
		return 0
	}
	var sum float64
	for _, x := range s {
		sum += x
	}
	return sum / float64(len(s))
}

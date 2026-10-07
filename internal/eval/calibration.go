package eval

import (
	"math"
	"sort"
)

// Calibration measures how well the emitted confidences behave as probabilities.
//
// A confidence is calibrated when, among detections reported with confidence ≈ p, a
// fraction ≈ p are true. Expected Calibration Error (ECE) quantifies the gap: bin the
// detections by confidence, and average |accuracy − mean confidence| over bins, weighted
// by bin population. The task requires ECE ≤ 0.05 over 10 bins at the operating point.
type Calibration struct {
	Bins  int          `json:"bins"`
	ECE   float64      `json:"ece"`
	MCE   float64      `json:"mce"` // worst single-bin gap
	Table []Calibrated `json:"table"`
}

// Calibrated is one confidence bin.
type Calibrated struct {
	Lo       float64 `json:"lo"`
	Hi       float64 `json:"hi"`
	Count    int     `json:"count"`
	MeanConf float64 `json:"mean_confidence"`
	Accuracy float64 `json:"accuracy"` // fraction of the bin that were true positives
}

// ComputeECE bins the scored predictions by confidence and returns the calibration
// report. Predictions are labelled true/false by the same greedy 1:1 matching the
// metrics use, so "accuracy" here means "was this detection a TP".
func ComputeECE(labels *Labels, preds []Prediction, threshold float64, bins int) Calibration {
	if bins <= 0 {
		bins = 10
	}
	isTP := labelPredictions(labels, preds, threshold)
	cal := Calibration{Bins: bins}
	type acc struct {
		n, tp int
		sum   float64
	}
	buckets := make([]acc, bins)
	total := 0
	for i, p := range preds {
		if p.Confidence < threshold {
			continue
		}
		b := int(p.Confidence * float64(bins))
		if b >= bins {
			b = bins - 1
		}
		if b < 0 {
			b = 0
		}
		buckets[b].n++
		buckets[b].sum += p.Confidence
		if isTP[i] {
			buckets[b].tp++
		}
		total++
	}
	if total == 0 {
		return cal
	}
	for i, b := range buckets {
		lo := float64(i) / float64(bins)
		hi := float64(i+1) / float64(bins)
		if b.n == 0 {
			cal.Table = append(cal.Table, Calibrated{Lo: lo, Hi: hi})
			continue
		}
		meanConf := b.sum / float64(b.n)
		acc := float64(b.tp) / float64(b.n)
		gap := math.Abs(acc - meanConf)
		cal.ECE += float64(b.n) / float64(total) * gap
		if gap > cal.MCE {
			cal.MCE = gap
		}
		cal.Table = append(cal.Table, Calibrated{Lo: lo, Hi: hi, Count: b.n, MeanConf: meanConf, Accuracy: acc})
	}
	return cal
}

// labelPredictions returns, per input prediction index, whether it is a true positive
// under the same greedy matching rule used for the metrics (§2).
func labelPredictions(labels *Labels, preds []Prediction, threshold float64) []bool {
	tol := labels.MatchTolSec
	if tol == 0 {
		tol = 1.5
	}
	type occ struct {
		Occurrence
		claimed bool
	}
	byFile := map[string][]*occ{}
	for _, r := range labels.Recordings {
		for i := range r.Occurrences {
			byFile[r.File] = append(byFile[r.File], &occ{Occurrence: r.Occurrences[i]})
		}
	}
	order := make([]int, 0, len(preds))
	for i, p := range preds {
		if p.Confidence >= threshold {
			order = append(order, i)
		}
	}
	sort.SliceStable(order, func(a, b int) bool { return preds[order[a]].Confidence > preds[order[b]].Confidence })

	out := make([]bool, len(preds))
	for _, idx := range order {
		p := preds[idx]
		occs := byFile[p.File]
		best, bestErr := -1, tol+1
		for i, o := range occs {
			if o.claimed || o.AdID != p.AdID {
				continue
			}
			if e := math.Abs(p.TimeSec - o.StartSec); e <= tol && e < bestErr {
				best, bestErr = i, e
			}
		}
		if best >= 0 {
			occs[best].claimed = true
			out[idx] = true
		}
	}
	return out
}

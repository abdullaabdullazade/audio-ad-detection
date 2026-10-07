package eval

import (
	"math"
	"sort"
)

// Prediction is one emitted detection to be scored against the labels.
type Prediction struct {
	File       string
	AdID       string
	TimeSec    float64
	Confidence float64
}

// ClassStat holds per-distortion-class counts.
type ClassStat struct {
	Occurrences int
	TP          int
	FN          int
}

// Recall returns TP/(TP+FN) for the class.
func (c ClassStat) Recall() float64 {
	if c.Occurrences == 0 {
		return math.NaN()
	}
	return float64(c.TP) / float64(c.Occurrences)
}

// Report is the scored result at one operating threshold.
type Report struct {
	Threshold       float64              `json:"threshold"`
	TP              int                  `json:"tp"`
	FP              int                  `json:"fp"`
	FN              int                  `json:"fn"`
	Recall          float64              `json:"recall"`
	Precision       float64              `json:"precision"`
	F1              float64              `json:"f1"`
	CleanFP         int                  `json:"clean_fp"` // FPs on true-negative recordings
	TimestampMedian float64              `json:"timestamp_median"`
	TimestampP95    float64              `json:"timestamp_p95"`
	TimestampMax    float64              `json:"timestamp_max"`
	PerClass        map[string]ClassStat `json:"per_class"`
	FPList          []Prediction         `json:"-"` // false positives (for diagnostics)
}

// Score matches predictions to ground-truth occurrences (§2): greedy by descending
// confidence, 1:1, same ad, same recording, within tolerance. Predictions below
// `threshold` are ignored.
func Score(labels *Labels, preds []Prediction, threshold float64) Report {
	tol := labels.MatchTolSec
	if tol == 0 {
		tol = 1.5
	}

	// Index occurrences and clean-recording flags per file.
	type occ struct {
		Occurrence
		claimed bool
	}
	byFile := map[string][]*occ{}
	cleanFile := map[string]bool{}
	rep := Report{Threshold: threshold, PerClass: map[string]ClassStat{}}
	for _, r := range labels.Recordings {
		if len(r.Occurrences) == 0 {
			cleanFile[r.File] = true
		}
		for i := range r.Occurrences {
			o := &occ{Occurrence: r.Occurrences[i]}
			byFile[r.File] = append(byFile[r.File], o)
			cs := rep.PerClass[o.Class]
			cs.Occurrences++
			rep.PerClass[o.Class] = cs
		}
	}

	// Greedy match, highest confidence first.
	kept := make([]Prediction, 0, len(preds))
	for _, p := range preds {
		if p.Confidence >= threshold {
			kept = append(kept, p)
		}
	}
	sort.SliceStable(kept, func(i, j int) bool { return kept[i].Confidence > kept[j].Confidence })

	var tsErrors []float64
	for _, p := range kept {
		occs := byFile[p.File]
		best := -1
		bestErr := tol + 1
		for i, o := range occs {
			if o.claimed || o.AdID != p.AdID {
				continue
			}
			e := math.Abs(p.TimeSec - o.StartSec)
			if e <= tol && e < bestErr {
				best, bestErr = i, e
			}
		}
		if best >= 0 {
			occs[best].claimed = true
			rep.TP++
			tsErrors = append(tsErrors, bestErr)
			cs := rep.PerClass[occs[best].Class]
			cs.TP++
			rep.PerClass[occs[best].Class] = cs
		} else {
			rep.FP++
			rep.FPList = append(rep.FPList, p)
			if cleanFile[p.File] {
				rep.CleanFP++
			}
		}
	}

	// Unclaimed occurrences are false negatives.
	for _, occs := range byFile {
		for _, o := range occs {
			if !o.claimed {
				rep.FN++
				cs := rep.PerClass[o.Class]
				cs.FN++
				rep.PerClass[o.Class] = cs
			}
		}
	}

	rep.Recall = ratio(rep.TP, rep.TP+rep.FN)
	rep.Precision = ratio(rep.TP, rep.TP+rep.FP)
	if rep.Recall+rep.Precision > 0 {
		rep.F1 = 2 * rep.Recall * rep.Precision / (rep.Recall + rep.Precision)
	}
	rep.TimestampMedian, rep.TimestampP95, rep.TimestampMax = percentiles(tsErrors)
	return rep
}

func ratio(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}

func percentiles(v []float64) (median, p95, max float64) {
	if len(v) == 0 {
		return 0, 0, 0
	}
	s := make([]float64, len(v))
	copy(s, v)
	sort.Float64s(s)
	median = s[len(s)/2]
	p95 = s[int(math.Ceil(0.95*float64(len(s))))-1]
	max = s[len(s)-1]
	return
}

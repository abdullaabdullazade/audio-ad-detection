package eval

import (
	"encoding/json"
	"os"
	"sort"
)

// PRPoint is one point on the precision-recall curve.
type PRPoint struct {
	Threshold float64 `json:"threshold"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
}

// FullReport is the complete eval output written by `finder eval`.
type FullReport struct {
	ChosenThreshold float64     `json:"chosen_threshold"`
	TargetPrecision float64     `json:"target_precision"`
	Operating       Report      `json:"operating_point"`
	PRCurve         []PRPoint   `json:"pr_curve"`
	NumRecordings   int         `json:"num_recordings"`
	NumOccurrences  int         `json:"num_occurrences"`
	NumPredictions  int         `json:"num_predictions"`
	Calibration     Calibration `json:"calibration"`
}

// BuildReport computes the PR curve over candidate thresholds, chooses the operating
// point as the threshold giving maximum recall subject to precision >= targetPrecision
// (falling back to best F1 if the target is unreachable), and returns the full report.
func BuildReport(labels *Labels, preds []Prediction, targetPrecision float64) FullReport {
	thresholds := candidateThresholds(preds)
	var curve []PRPoint
	for _, t := range thresholds {
		r := Score(labels, preds, t)
		curve = append(curve, PRPoint{Threshold: t, Precision: r.Precision, Recall: r.Recall, F1: r.F1})
	}

	// Choose operating point.
	chosen := 0.0
	bestRecall := -1.0
	found := false
	for _, p := range curve {
		// Ties break toward the HIGHER threshold. When the detector's own gates already
		// remove every false positive, precision is 1.000 across the whole sweep and a
		// "first best recall wins" rule would return threshold 0 — numerically optimal but
		// useless in production, since it admits anything the gates ever emit. The highest
		// threshold reaching the same recall is the same operating point with margin.
		if p.Precision >= targetPrecision && (p.Recall > bestRecall ||
			(p.Recall == bestRecall && found && p.Threshold > chosen)) {
			bestRecall, chosen, found = p.Recall, p.Threshold, true
		}
	}
	if !found {
		bestF1 := -1.0
		for _, p := range curve {
			if p.F1 > bestF1 {
				bestF1, chosen = p.F1, p.Threshold
			}
		}
	}

	op := Score(labels, preds, chosen)
	var occ int
	for _, r := range labels.Recordings {
		occ += len(r.Occurrences)
	}
	return FullReport{
		Calibration:     ComputeECE(labels, preds, chosen, 10),
		ChosenThreshold: chosen,
		TargetPrecision: targetPrecision,
		Operating:       op,
		PRCurve:         curve,
		NumRecordings:   len(labels.Recordings),
		NumOccurrences:  occ,
		NumPredictions:  len(preds),
	}
}

// candidateThresholds returns a sorted, de-duplicated set of thresholds to sweep:
// every distinct prediction confidence plus 0 and 1.
func candidateThresholds(preds []Prediction) []float64 {
	set := map[float64]struct{}{0: {}, 1.0: {}}
	for _, p := range preds {
		set[p.Confidence] = struct{}{}
	}
	out := make([]float64, 0, len(set))
	for t := range set {
		out = append(out, t)
	}
	sort.Float64s(out)
	return out
}

// Save writes the full report as indented JSON.
func (fr *FullReport) Save(path string) error {
	b, err := json.MarshalIndent(fr, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

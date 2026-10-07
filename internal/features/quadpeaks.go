package features

import (
	"math"
	"sort"
)

// QPeak is a spectral peak in (time-frame, linear-frequency-bin) space, used for
// quad-based, time/frequency-scale-invariant fingerprinting.
type QPeak struct {
	Frame int
	Bin   int
	Mag   float32
}

const (
	quadPeakDT      = 3  // max-filter half-width in frames
	quadPeakDF      = 3  // max-filter half-width in bins
	quadFMinBin     = 6  // ignore bins below this (~94 Hz at 16k/1024)
	quadPeaksPerSec = 34 // density cap
)

// QuadPeaks extracts strong local maxima from a magnitude spectrogram in
// (frame, bin) coordinates: a point is a peak if it is the maximum in a
// (±quadPeakDT, ±quadPeakDF) neighbourhood and exceeds an adaptive floor. Working in
// LINEAR frequency bins (not log bands) is what makes the later quad hash invariant
// to frequency scaling (pitch shift): a pitch shift multiplies bin indices, which the
// A→B span normalisation divides out.
func QuadPeaks(spec *Spectrogram) []QPeak {
	F := len(spec.Frames)
	if F == 0 {
		return nil
	}
	// Global adaptive floor: mean + 1*std of log-magnitude.
	var sum, sum2 float64
	var cnt int
	for _, fr := range spec.Frames {
		for b := quadFMinBin; b < len(fr); b++ {
			v := math.Log1p(float64(fr[b]))
			sum += v
			sum2 += v * v
			cnt++
		}
	}
	if cnt == 0 {
		return nil
	}
	mean := sum / float64(cnt)
	varr := sum2/float64(cnt) - mean*mean
	if varr < 0 {
		varr = 0
	}
	floor := mean + math.Sqrt(varr)

	var peaks []QPeak
	for f := 0; f < F; f++ {
		row := spec.Frames[f]
		for b := quadFMinBin; b < len(row); b++ {
			v := math.Log1p(float64(row[b]))
			if v < floor {
				continue
			}
			if isLocalMax2D(spec.Frames, f, b, F, len(row), float32(v)) {
				peaks = append(peaks, QPeak{Frame: f, Bin: b, Mag: float32(v)})
			}
		}
	}
	return capQuadDensity(peaks, F)
}

// QuadPeaksRange extracts peaks only within the frame range [a,b) of the spectrogram,
// using a floor computed from that range. This is used to compute peak agreement over a
// candidate window without paying the O(frames·bins·neighbourhood) cost of running peak
// picking over an entire 61-minute recording — critical for the window budget.
func QuadPeaksRange(spec *Spectrogram, a, b int) []QPeak {
	F := len(spec.Frames)
	if a < 0 {
		a = 0
	}
	if b > F {
		b = F
	}
	if b-a < 1 {
		return nil
	}
	var sum, sum2 float64
	var cnt int
	for f := a; f < b; f++ {
		fr := spec.Frames[f]
		for bin := quadFMinBin; bin < len(fr); bin++ {
			v := math.Log1p(float64(fr[bin]))
			sum += v
			sum2 += v * v
			cnt++
		}
	}
	if cnt == 0 {
		return nil
	}
	mean := sum / float64(cnt)
	varr := sum2/float64(cnt) - mean*mean
	if varr < 0 {
		varr = 0
	}
	floor := mean + math.Sqrt(varr)

	var peaks []QPeak
	for f := a; f < b; f++ {
		row := spec.Frames[f]
		for bin := quadFMinBin; bin < len(row); bin++ {
			v := math.Log1p(float64(row[bin]))
			if v < floor {
				continue
			}
			if isLocalMax2D(spec.Frames, f, bin, F, len(row), float32(v)) {
				peaks = append(peaks, QPeak{Frame: f, Bin: bin, Mag: float32(v)})
			}
		}
	}
	return peaks
}

func isLocalMax2D(frames [][]float32, f, b, F, nb int, v float32) bool {
	for df := -quadPeakDT; df <= quadPeakDT; df++ {
		g := f + df
		if g < 0 || g >= F {
			continue
		}
		row := frames[g]
		for db := -quadPeakDF; db <= quadPeakDF; db++ {
			c := b + db
			if c < 0 || c >= nb || (df == 0 && db == 0) {
				continue
			}
			if float32(math.Log1p(float64(row[c]))) > v {
				return false
			}
		}
	}
	return true
}

func capQuadDensity(peaks []QPeak, F int) []QPeak {
	if len(peaks) == 0 {
		return peaks
	}
	seconds := FrameToSec(F) + 1
	budget := int(quadPeaksPerSec * seconds)
	if len(peaks) > budget {
		byMag := make([]QPeak, len(peaks))
		copy(byMag, peaks)
		sort.Slice(byMag, func(i, j int) bool { return byMag[i].Mag > byMag[j].Mag })
		peaks = byMag[:budget]
	}
	sort.Slice(peaks, func(i, j int) bool {
		if peaks[i].Frame != peaks[j].Frame {
			return peaks[i].Frame < peaks[j].Frame
		}
		return peaks[i].Bin < peaks[j].Bin
	})
	return peaks
}

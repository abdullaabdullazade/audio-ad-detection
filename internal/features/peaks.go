package features

import (
	"math"
	"sort"
)

// NumBands is the number of log-spaced frequency bands used to anchor fingerprint
// peaks. Log spacing makes a peak's band robust to the <=5% frequency shift caused by
// naive-resample time-scaling: a 5% shift is a fraction of one band, so most peaks
// keep their band.
const NumBands = 24

const (
	peakFMin   = 200.0 // ignore very low frequencies (hum, DC)
	peakTimeR  = 3     // local-max time radius in frames
	peakPerSec = 28    // density cap: strongest peaks kept per second
)

// binToBand maps each FFT bin to a log-frequency band (or -1 if below peakFMin).
var binToBand = func() []int {
	binHz := float64(SampleRate) / float64(FFTSize)
	fMax := float64(SampleRate) / 2
	logMin, logMax := math.Log(peakFMin), math.Log(fMax)
	m := make([]int, NumBins)
	for b := 0; b < NumBins; b++ {
		f := float64(b) * binHz
		if f < peakFMin {
			m[b] = -1
			continue
		}
		band := int(float64(NumBands) * (math.Log(f) - logMin) / (logMax - logMin))
		if band >= NumBands {
			band = NumBands - 1
		}
		m[b] = band
	}
	return m
}()

// Peak is a spectral peak anchored to a time frame and a log-frequency band.
type Peak struct {
	Frame int
	Band  int
	Mag   float32
}

// PeakPick extracts robust spectral peaks from a magnitude spectrogram. For each
// frame it reduces the spectrum to per-band energies (max within band), then keeps
// (frame,band) points that are local maxima in time and stand above the band's
// background, subject to a per-second density cap.
func PeakPick(spec *Spectrogram) []Peak {
	F := len(spec.Frames)
	if F == 0 {
		return nil
	}
	// Per-band energy in dB-like log domain.
	energy := make([][]float32, F)
	for f := 0; f < F; f++ {
		row := make([]float32, NumBands)
		mag := spec.Frames[f]
		for b, band := range binToBand {
			if band < 0 {
				continue
			}
			if mag[b] > row[band] {
				row[band] = mag[b]
			}
		}
		for b := range row {
			row[b] = float32(math.Log1p(float64(row[b])))
		}
		energy[f] = row
	}

	// Per-band background threshold: mean + 0.5*std over time.
	thresh := make([]float32, NumBands)
	for b := 0; b < NumBands; b++ {
		var sum, sum2 float64
		for f := 0; f < F; f++ {
			v := float64(energy[f][b])
			sum += v
			sum2 += v * v
		}
		mean := sum / float64(F)
		varr := sum2/float64(F) - mean*mean
		if varr < 0 {
			varr = 0
		}
		thresh[b] = float32(mean + 0.5*math.Sqrt(varr))
	}

	var peaks []Peak
	for f := 0; f < F; f++ {
		for b := 0; b < NumBands; b++ {
			v := energy[f][b]
			if v < thresh[b] {
				continue
			}
			if !isLocalTimeMax(energy, f, b, F) {
				continue
			}
			peaks = append(peaks, Peak{Frame: f, Band: b, Mag: v})
		}
	}
	return capDensity(peaks, F)
}

func isLocalTimeMax(energy [][]float32, f, b, F int) bool {
	v := energy[f][b]
	for d := -peakTimeR; d <= peakTimeR; d++ {
		if d == 0 {
			continue
		}
		g := f + d
		if g < 0 || g >= F {
			continue
		}
		if energy[g][b] > v {
			return false
		}
	}
	return true
}

// capDensity keeps only the strongest peaks up to peakPerSec per second, preserving
// time order in the result.
func capDensity(peaks []Peak, F int) []Peak {
	if len(peaks) == 0 {
		return peaks
	}
	seconds := FrameToSec(F) + 1
	budget := int(peakPerSec * seconds)
	if len(peaks) <= budget {
		sort.Slice(peaks, func(i, j int) bool { return peaks[i].Frame < peaks[j].Frame })
		return peaks
	}
	// Keep the strongest `budget` peaks, then restore time order.
	byMag := make([]Peak, len(peaks))
	copy(byMag, peaks)
	sort.Slice(byMag, func(i, j int) bool { return byMag[i].Mag > byMag[j].Mag })
	kept := byMag[:budget]
	sort.Slice(kept, func(i, j int) bool {
		if kept[i].Frame != kept[j].Frame {
			return kept[i].Frame < kept[j].Frame
		}
		return kept[i].Band < kept[j].Band
	})
	return kept
}

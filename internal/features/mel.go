package features

import (
	"math"
	"sync"
)

// NumMel is the number of mel bands in the Stage-2 alignment feature. 32 bands is
// rich enough to discriminate two ads that share a short common bed (jingle trap)
// while staying cheap for DTW.
const NumMel = 32

// melFilterbank precomputes triangular mel filters over the linear FFT bins.
var melFilterbank = buildMelFilterbank(NumMel, SampleRate, FFTSize)

type melFilters struct {
	weights [][]float32 // len NumMel, each len NumBins
}

func hzToMel(f float64) float64 { return 2595 * math.Log10(1+f/700) }
func melToHz(m float64) float64 { return 700 * (math.Pow(10, m/2595) - 1) }

var (
	warpedMu    sync.Mutex
	warpedCache = map[int]*melFilters{}
)

func buildMelFilterbank(nMel, sr, fftSize int) *melFilters {
	return buildMelFilterbankScaled(nMel, sr, fftSize, 1.0)
}

// buildMelFilterbankScaled builds the mel filterbank with all centre frequencies
// multiplied by freqScale (1.0 = standard).
func buildMelFilterbankScaled(nMel, sr, fftSize int, freqScale float64) *melFilters {
	nBins := fftSize/2 + 1
	fMin, fMax := 0.0, float64(sr)/2
	mMin, mMax := hzToMel(fMin), hzToMel(fMax)

	// nMel+2 mel points -> nMel triangular filters.
	points := make([]float64, nMel+2)
	for i := range points {
		mel := mMin + (mMax-mMin)*float64(i)/float64(nMel+1)
		points[i] = melToHz(mel) * freqScale
	}
	binHz := float64(sr) / float64(fftSize)

	mf := &melFilters{weights: make([][]float32, nMel)}
	for m := 0; m < nMel; m++ {
		lo, ctr, hi := points[m], points[m+1], points[m+2]
		w := make([]float32, nBins)
		for b := 0; b < nBins; b++ {
			f := float64(b) * binHz
			var v float64
			switch {
			case f >= lo && f <= ctr && ctr > lo:
				v = (f - lo) / (ctr - lo)
			case f > ctr && f <= hi && hi > ctr:
				v = (hi - f) / (hi - ctr)
			}
			w[b] = float32(v)
		}
		mf.weights[m] = w
	}
	return mf
}

// LogMel converts a magnitude spectrogram to per-frame mean-centered, L2-normalized
// log-mel vectors. Mean-centering removes the overall level, leaving spectral SHAPE,
// so cosine similarity is gain/level invariant. `center` toggles the mean subtraction.
func LogMel(spec *Spectrogram) [][]float32 { return logMel(spec, false) }

// LogMelCentered is LogMel with per-frame mean subtraction.
func LogMelCentered(spec *Spectrogram) [][]float32 { return logMel(spec, true) }

// LogMelWarped is LogMel computed with a mel filterbank whose centre frequencies are
// multiplied by freqScale. Naive-resample (playback-rate) distortion scales the WHOLE
// spectrum uniformly — aired_freq = ref_freq × r — so building the reference's log-mel
// with filters at ref_freq × r models that distortion exactly (unlike vocal pitch shift,
// which preserves formants). This lets a resampled airing be verified against a cheap
// reference-side template instead of re-decoding the recording at several pitch factors.
func LogMelWarped(spec *Spectrogram, freqScale float64) [][]float32 {
	fb := warpedFilterbank(freqScale)
	out := make([][]float32, len(spec.Frames))
	for i, mag := range spec.Frames {
		vec := make([]float32, NumMel)
		for m := 0; m < NumMel; m++ {
			w := fb.weights[m]
			var e float64
			for b, wb := range w {
				if wb != 0 {
					p := float64(mag[b])
					e += float64(wb) * p * p
				}
			}
			vec[m] = float32(math.Log1p(e))
		}
		l2Normalize(vec)
		out[i] = vec
	}
	return out
}

// warpedFilterbank builds (and caches) a mel filterbank with centres scaled by s.
func warpedFilterbank(s float64) *melFilters {
	warpedMu.Lock()
	defer warpedMu.Unlock()
	key := int(math.Round(s * 1000))
	if fb, ok := warpedCache[key]; ok {
		return fb
	}
	fb := buildMelFilterbankScaled(NumMel, SampleRate, FFTSize, s)
	warpedCache[key] = fb
	return fb
}

func logMel(spec *Spectrogram, center bool) [][]float32 {
	out := make([][]float32, len(spec.Frames))
	for i, mag := range spec.Frames {
		vec := make([]float32, NumMel)
		var mean float64
		for m := 0; m < NumMel; m++ {
			w := melFilterbank.weights[m]
			var e float64
			for b, wb := range w {
				if wb != 0 {
					p := float64(mag[b])
					e += float64(wb) * p * p // power
				}
			}
			v := math.Log1p(e)
			vec[m] = float32(v)
			mean += v
		}
		if center {
			mean /= float64(NumMel)
			for m := range vec {
				vec[m] -= float32(mean)
			}
		}
		l2Normalize(vec)
		out[i] = vec
	}
	return out
}

func l2Normalize(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s <= 0 {
		return
	}
	inv := float32(1 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
}

// Cosine returns the dot product of two L2-normalized vectors (== cosine similarity).
func Cosine(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

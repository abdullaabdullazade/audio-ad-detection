// Package features turns mono PCM into the time-frequency representations the
// detector uses: a magnitude spectrogram, a log-mel feature (for Stage-2 alignment),
// and spectral peaks in log-frequency bands (for Stage-1 fingerprinting).
//
// Fixed analysis parameters (task-tuned): 16 kHz, 1024-pt FFT, 256-sample hop
// (16 ms frames, ~62.5 fps). Frame resolution (16 ms) is well under the 0.3 s median
// timestamp target.
package features

import (
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
)

const (
	SampleRate = 16000
	FFTSize    = 1024
	HopSize    = 256
	NumBins    = FFTSize/2 + 1
	HopSec     = float64(HopSize) / float64(SampleRate)
)

// Spectrogram is a sequence of magnitude spectra.
type Spectrogram struct {
	Frames [][]float32 // F frames, each NumBins magnitudes
	HopSec float64
}

// NumFrames returns the number of frames.
func (s *Spectrogram) NumFrames() int { return len(s.Frames) }

// hannWindow returns a cached Hann window of length FFTSize.
var hannWindow = func() []float64 {
	w := make([]float64, FFTSize)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(FFTSize-1))
	}
	return w
}()

// STFT computes the magnitude spectrogram of samples (assumed at SampleRate). If the
// signal is shorter than one frame it returns an empty spectrogram.
func STFT(samples []float32) *Spectrogram {
	if len(samples) < FFTSize {
		return &Spectrogram{HopSec: HopSec}
	}
	nFrames := 1 + (len(samples)-FFTSize)/HopSize
	fft := fourier.NewFFT(FFTSize)
	frame := make([]float64, FFTSize)
	frames := make([][]float32, nFrames)

	for f := 0; f < nFrames; f++ {
		start := f * HopSize
		for i := 0; i < FFTSize; i++ {
			frame[i] = float64(samples[start+i]) * hannWindow[i]
		}
		coeff := fft.Coefficients(nil, frame) // NumBins complex values
		mag := make([]float32, NumBins)
		for b := 0; b < NumBins; b++ {
			mag[b] = float32(math.Hypot(real(coeff[b]), imag(coeff[b])))
		}
		frames[f] = mag
	}
	return &Spectrogram{Frames: frames, HopSec: HopSec}
}

// FrameToSec converts a frame index to seconds (frame center).
func FrameToSec(frame int) float64 { return float64(frame) * HopSec }

// SecToFrame converts seconds to the nearest frame index.
func SecToFrame(sec float64) int { return int(math.Round(sec / HopSec)) }

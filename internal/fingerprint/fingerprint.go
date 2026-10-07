// Package fingerprint turns spectral peaks into landmark hashes (Shazam-style
// constellation pairing). Each hash packs the two peaks' log-frequency bands and the
// quantized time gap between them; the landmark also carries the anchor time so the
// retrieval stage can vote on the implied clip start.
//
// A scale parameter stretches/compresses the peak time axis before pairing, which is
// how the index builds multiple time-scaled variants of a reference cheaply (in
// feature space, no re-decoding) to stay robust to ±5% linear time-scale.
package fingerprint

import (
	"sort"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
)

// Hash packs (band1[5b], band2[5b], dtBucket[8b]).
type Hash uint32

// Landmark is a hash together with the anchor peak's time (seconds).
type Landmark struct {
	Hash    Hash
	TimeSec float64
}

const (
	dtMinSec  = 0.05
	dtMaxSec  = 1.5
	dtQuant   = 0.03 // seconds per dt bucket
	fanout    = 6    // target peaks paired per anchor
	maxBucket = 255
)

// invariantBit tags a pitch-invariant hash so it shares the table with specific
// hashes without colliding.
const invariantBit = uint32(1) << 22

// pack builds the specific (high-precision) hash from the two absolute bands and the
// time gap.
func pack(band1, band2, dtBucket int) Hash {
	if dtBucket > maxBucket {
		dtBucket = maxBucket
	}
	return Hash(uint32(band1&0x1f) | uint32(band2&0x1f)<<5 | uint32(dtBucket&0xff)<<10)
}

// packInvariant builds a pitch-robust hash: a COARSE absolute band (stable under the
// <=5% frequency shift of naive-resample) plus the band DIFFERENCE (invariant to any
// uniform pitch shift) and the time gap. Emitted alongside the specific hash so
// resampled airings still retrieve, without weakening precision on undistorted air.
func packInvariant(band1, band2, dtBucket int) Hash {
	if dtBucket > maxBucket {
		dtBucket = maxBucket
	}
	coarse := band1 / 4 // 0..5
	dband := band2 - band1 + 23
	if dband < 0 {
		dband = 0
	}
	if dband > 63 {
		dband = 63
	}
	return Hash(invariantBit | uint32(coarse&0x7) | uint32(dband&0x3f)<<3 | uint32(dtBucket&0xff)<<9)
}

type timedPeak struct {
	t    float64
	band int
}

// Landmarks generates landmark hashes from peaks, with the time axis scaled by
// `scale` (1.0 = no change). Peaks are assumed roughly time-ordered.
func Landmarks(peaks []features.Peak, scale float64) []Landmark {
	if len(peaks) == 0 {
		return nil
	}
	tp := make([]timedPeak, len(peaks))
	for i, p := range peaks {
		tp[i] = timedPeak{t: features.FrameToSec(p.Frame) * scale, band: p.Band}
	}
	sort.Slice(tp, func(i, j int) bool { return tp[i].t < tp[j].t })

	var out []Landmark
	for i := 0; i < len(tp); i++ {
		paired := 0
		for j := i + 1; j < len(tp) && paired < fanout; j++ {
			dt := tp[j].t - tp[i].t
			if dt < dtMinSec {
				continue
			}
			if dt > dtMaxSec {
				break
			}
			bucket := int(dt / dtQuant)
			out = append(out,
				Landmark{Hash: pack(tp[i].band, tp[j].band, bucket), TimeSec: tp[i].t},
				Landmark{Hash: packInvariant(tp[i].band, tp[j].band, bucket), TimeSec: tp[i].t},
			)
			paired++
		}
	}
	return out
}

// FromPCM is a convenience: STFT -> peaks -> landmarks at the given scale.
func FromPCM(samples []float32, scale float64) []Landmark {
	spec := features.STFT(samples)
	peaks := features.PeakPick(spec)
	return Landmarks(peaks, scale)
}

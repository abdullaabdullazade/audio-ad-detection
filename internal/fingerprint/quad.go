package fingerprint

import (
	"sort"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
)

// Quad-based fingerprinting (Sonnleitner & Widmer, DAFx-2014): a quad is four spectral
// peaks A,B,C,D with A the earliest and B the latest-in-time, C,D inside the rectangle
// [A,B]. Normalizing C,D into the unit square defined by A→B yields a hash invariant to
// BOTH time and frequency scaling; the A→B spans of a matched query/reference pair
// recover the time- and frequency-scale factors.

// QuadHash is one quad's scale-invariant descriptor plus the data needed for scale
// recovery and start estimation.
type QuadHash struct {
	Key   uint32  // quantized (cx,cy,dx,dy)
	AtSec float64 // A's time (seconds)
	ABx   float64 // Bx-Ax in frames (time span)
	ABy   float64 // By-Ay in bins (frequency span)
}

// QuadParams controls quad grouping. Reference uses small (n,r,q); query uses large
// ones so that a scaled query still yields quads matching the reference (per the paper).
type QuadParams struct {
	N     int     // peaks considered per root region
	RSec  float64 // region width (seconds) to the right of the root
	Q     int     // max quads per root
	GridG int     // quantization levels per hash dimension
}

// RefQuadParams / QueryQuadParams are the paper's reference/query settings, scaled to
// our 16 kHz / 8 ms-hop front-end.
var (
	RefQuadParams   = QuadParams{N: 5, RSec: 2.0, Q: 2, GridG: 32}
	QueryQuadParams = QuadParams{N: 8, RSec: 7.9, Q: 500, GridG: 32}
)

func quantize(v float64, g int) uint32 {
	i := int(v * float64(g))
	if i < 0 {
		i = 0
	}
	if i >= g {
		i = g - 1
	}
	return uint32(i)
}

func quadKey(cx, cy, dx, dy float64, g int) uint32 {
	return quantize(cx, g) | quantize(cy, g)<<5 | quantize(dx, g)<<10 | quantize(dy, g)<<15
}

// Quads groups peaks into quads and returns their scale-invariant hashes.
func Quads(peaks []features.QPeak, p QuadParams) []QuadHash {
	if len(peaks) < 4 {
		return nil
	}
	sorted := make([]features.QPeak, len(peaks))
	copy(sorted, peaks)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].Frame != sorted[j].Frame {
			return sorted[i].Frame < sorted[j].Frame
		}
		return sorted[i].Bin < sorted[j].Bin
	})
	rFrames := int(p.RSec / features.HopSec)

	var out []QuadHash
	for ai := 0; ai < len(sorted); ai++ {
		A := sorted[ai]
		// Region: peaks strictly after A in time, within rFrames.
		var region []features.QPeak
		for j := ai + 1; j < len(sorted); j++ {
			if sorted[j].Frame <= A.Frame {
				continue
			}
			if sorted[j].Frame-A.Frame > rFrames {
				break
			}
			region = append(region, sorted[j])
			if len(region) >= p.N {
				break
			}
		}
		if len(region) < 3 {
			continue
		}
		made := 0
		for i := 0; i < len(region) && made < p.Q; i++ {
			for j := i + 1; j < len(region) && made < p.Q; j++ {
				for k := j + 1; k < len(region) && made < p.Q; k++ {
					if h, ok := makeQuad(A, region[i], region[j], region[k], p.GridG); ok {
						out = append(out, h)
						made++
					}
				}
			}
		}
	}
	return out
}

// makeQuad forms a valid quad from A and three region peaks (B = latest in time; the
// other two become C,D ordered by time) and returns its hash.
func makeQuad(A, p1, p2, p3 features.QPeak, g int) (QuadHash, bool) {
	trip := [3]features.QPeak{p1, p2, p3}
	// B = most distant in time from A.
	bi := 0
	for i := 1; i < 3; i++ {
		if trip[i].Frame > trip[bi].Frame {
			bi = i
		}
	}
	B := trip[bi]
	var cd []features.QPeak
	for i := 0; i < 3; i++ {
		if i != bi {
			cd = append(cd, trip[i])
		}
	}
	if cd[0].Frame > cd[1].Frame {
		cd[0], cd[1] = cd[1], cd[0]
	}
	C, D := cd[0], cd[1]

	spanX := float64(B.Frame - A.Frame)
	spanY := float64(B.Bin - A.Bin)
	if spanX <= 0 || spanY <= 0 {
		return QuadHash{}, false
	}
	// Validity: C,D inside rectangle [A,B].
	if !(A.Frame < C.Frame && C.Frame <= B.Frame && A.Frame < D.Frame && D.Frame <= B.Frame) {
		return QuadHash{}, false
	}
	if !(A.Bin < C.Bin && C.Bin <= B.Bin && A.Bin < D.Bin && D.Bin <= B.Bin) {
		return QuadHash{}, false
	}
	cx := float64(C.Frame-A.Frame) / spanX
	cy := float64(C.Bin-A.Bin) / spanY
	dx := float64(D.Frame-A.Frame) / spanX
	dy := float64(D.Bin-A.Bin) / spanY
	return QuadHash{
		Key:   quadKey(cx, cy, dx, dy, g),
		AtSec: features.FrameToSec(A.Frame),
		ABx:   spanX,
		ABy:   spanY,
	}, true
}

// QuadsFromPCM: STFT -> quad peaks -> quads.
func QuadsFromPCM(samples []float32, p QuadParams) []QuadHash {
	spec := features.STFT(samples)
	peaks := features.QuadPeaks(spec)
	return Quads(peaks, p)
}

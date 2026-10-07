package features

// Resample changes the playback rate of a mono signal by `factor` via linear
// interpolation — exactly the "naive resample" distortion: factor > 1 makes it shorter
// AND higher-pitched, factor < 1 longer and lower. The output length is len/factor.
//
// It is used to build pitch-shifted reference templates so the detector can match
// naive-resampled airings (which shift both time and every spectral peak) in both the
// retrieval and verification stages.
func Resample(in []float32, factor float64) []float32 {
	if factor <= 0 || len(in) == 0 {
		return in
	}
	outLen := int(float64(len(in)) / factor)
	if outLen < 1 {
		return in
	}
	out := make([]float32, outLen)
	for i := range out {
		pos := float64(i) * factor
		j := int(pos)
		if j+1 < len(in) {
			frac := float32(pos - float64(j))
			out[i] = in[j]*(1-frac) + in[j+1]*frac
		} else if j < len(in) {
			out[i] = in[j]
		}
	}
	return out
}

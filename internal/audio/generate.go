package audio

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"time"
)

// These helpers exist to build the adversarial test corpus (§8). They apply ffmpeg
// filter chains to reference clips, synthesize background "air", and encode PCM back
// to mp3 — all with the same subprocess discipline as Decode (arg slices, context,
// bounded output). They are not on the detection hot path.

// runFFmpegToPCM runs ffmpeg with the given pre-input args, a single input, and an
// optional audio-filter chain, capturing mono f32le output as PCM at sr. maxSeconds
// bounds the captured audio.
func runFFmpegToPCM(ctx context.Context, ffmpegPath string, preInput []string, input, filter string, sr int, maxSeconds float64) (PCM, error) {
	if ffmpegPath == "" {
		ffmpegPath = "ffmpeg"
	}
	args := []string{"-nostdin", "-v", "error"}
	args = append(args, preInput...)
	args = append(args, "-i", input)
	if filter != "" {
		args = append(args, "-af", filter)
	}
	args = append(args, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", fmt.Sprintf("%d", sr), "-f", "f32le", "-acodec", "pcm_f32le", "-")

	cmd := exec.CommandContext(ctx, ffmpegPath, args...)
	cmd.WaitDelay = 5 * time.Second
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return PCM{}, err
	}
	var stderr capBuf
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return PCM{}, err
	}
	maxBytes := int64((maxSeconds+2)*float64(sr))*4 + 4
	raw, readErr := io.ReadAll(io.LimitReader(stdout, maxBytes))
	waitErr := cmd.Wait()
	if ctx.Err() != nil {
		return PCM{}, ctx.Err()
	}
	if readErr != nil {
		return PCM{}, readErr
	}
	if waitErr != nil {
		var ee *exec.ExitError
		if errors.As(waitErr, &ee) {
			return PCM{}, fmt.Errorf("ffmpeg: %v: %s", waitErr, stderr.String())
		}
		return PCM{}, waitErr
	}
	return PCM{SampleRate: sr, Samples: bytesToFloat32(raw)}, nil
}

// DecodeFiltered decodes path applying an ffmpeg audio-filter chain (may be empty),
// producing mono PCM at sr. Used to synthesize distorted ad variants.
func DecodeFiltered(ctx context.Context, path, filter string, sr int, maxSeconds float64) (PCM, error) {
	return runFFmpegToPCM(ctx, "", nil, path, filter, sr, maxSeconds)
}

// SynthAir generates `seconds` of background air (lavfi noise source) as PCM. color
// is an ffmpeg anoisesrc color ("brown", "pink", "white"); seed makes it deterministic.
func SynthAir(ctx context.Context, color string, seed int, amplitude, seconds float64, sr int) (PCM, error) {
	src := fmt.Sprintf("anoisesrc=color=%s:seed=%d:amplitude=%.4f:duration=%.3f:sample_rate=%d", color, seed, amplitude, seconds, sr)
	return runFFmpegToPCM(ctx, "", []string{"-f", "lavfi"}, src, "", sr, seconds)
}

// EncodePCMToMP3 encodes mono PCM to an mp3 file at the given bitrate (kbps),
// piping the samples to ffmpeg's stdin as f32le (no temp file).
func EncodePCMToMP3(ctx context.Context, pcm PCM, outPath string, bitrateK int) error {
	args := []string{
		"-nostdin", "-v", "error", "-y",
		"-f", "f32le", "-ar", fmt.Sprintf("%d", pcm.SampleRate), "-ac", "1", "-i", "-",
		"-c:a", "libmp3lame", "-b:a", fmt.Sprintf("%dk", bitrateK),
		"--", outPath,
	}
	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	cmd.WaitDelay = 5 * time.Second
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	var stderr capBuf
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	writeErr := writeFloat32LE(stdin, pcm.Samples)
	stdin.Close()
	waitErr := cmd.Wait()
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if writeErr != nil {
		return writeErr
	}
	if waitErr != nil {
		return fmt.Errorf("mp3 encode: %v: %s", waitErr, stderr.String())
	}
	return nil
}

// AACRoundTrip encodes pcm to AAC at the given bitrate and decodes it straight back, so
// the result carries a real codec change (§8.1: "codec change MP3<->AAC"), not just a
// bitrate change. AAC's MDCT introduces different pre-echo and high-frequency behaviour
// than MP3, which is exactly what the class is meant to probe. The intermediate file lives
// in dir and is removed before returning.
func AACRoundTrip(ctx context.Context, pcm PCM, dir string, bitrateK int) (PCM, error) {
	tmp, err := os.CreateTemp(dir, "aac-*.m4a")
	if err != nil {
		return PCM{}, err
	}
	path := tmp.Name()
	tmp.Close()
	defer os.Remove(path)

	args := []string{
		"-nostdin", "-v", "error", "-y",
		"-f", "f32le", "-ar", fmt.Sprintf("%d", pcm.SampleRate), "-ac", "1", "-i", "-",
		"-c:a", "aac", "-b:a", fmt.Sprintf("%dk", bitrateK),
		"--", path,
	}
	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	cmd.WaitDelay = 5 * time.Second
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return PCM{}, err
	}
	var stderr capBuf
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return PCM{}, err
	}
	writeErr := writeFloat32LE(stdin, pcm.Samples)
	stdin.Close()
	waitErr := cmd.Wait()
	if ctx.Err() != nil {
		return PCM{}, ctx.Err()
	}
	if writeErr != nil {
		return PCM{}, writeErr
	}
	if waitErr != nil {
		return PCM{}, fmt.Errorf("aac encode: %v: %s", waitErr, stderr.String())
	}
	return DecodeFiltered(ctx, path, "", pcm.SampleRate, 300)
}

// PitchShiftPCM changes an in-memory buffer's playback rate by `factor` using ffmpeg's
// high-quality soxr resampler (asetrate + aresample), piping PCM in and out — no temp
// files, no re-decode of the source. factor > 1 raises pitch and shortens; < 1 lowers
// and lengthens. Used to pitch-compensate short candidate windows cheaply, which is what
// makes the naive-resample recovery path affordable on 61-minute recordings.
func PitchShiftPCM(ctx context.Context, pcm PCM, factor float64) (PCM, error) {
	if factor <= 0 || len(pcm.Samples) == 0 {
		return pcm, nil
	}
	sr := pcm.SampleRate
	newRate := int(float64(sr) / factor)
	args := []string{
		"-nostdin", "-v", "error",
		"-f", "f32le", "-ar", fmt.Sprintf("%d", sr), "-ac", "1", "-i", "-",
		"-af", fmt.Sprintf("asetrate=%d,aresample=%d:resampler=soxr", newRate, sr),
		"-f", "f32le", "-acodec", "pcm_f32le", "-",
	}
	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	cmd.WaitDelay = 5 * time.Second
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return PCM{}, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return PCM{}, err
	}
	var stderr capBuf
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return PCM{}, err
	}
	go func() {
		_ = writeFloat32LE(stdin, pcm.Samples)
		stdin.Close()
	}()
	maxBytes := int64(float64(len(pcm.Samples))*4/factor) + int64(sr)*8
	raw, readErr := io.ReadAll(io.LimitReader(stdout, maxBytes))
	waitErr := cmd.Wait()
	if ctx.Err() != nil {
		return PCM{}, ctx.Err()
	}
	if readErr != nil {
		return PCM{}, readErr
	}
	if waitErr != nil {
		return PCM{}, fmt.Errorf("pitch shift: %v: %s", waitErr, stderr.String())
	}
	return PCM{SampleRate: sr, Samples: bytesToFloat32(raw)}, nil
}

// WriteZeroFile writes a zero-length file (for the corrupt/zero-length test class).
func WriteZeroFile(path string) error { return os.WriteFile(path, nil, 0o600) }

// WriteGarbageFile writes non-media bytes (for the corrupt-input test class).
func WriteGarbageFile(path string, n int) error {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte((i*131 + 7) & 0xff)
	}
	return os.WriteFile(path, b, 0o600)
}

func bytesToFloat32(raw []byte) []float32 {
	n := len(raw) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func writeFloat32LE(w io.Writer, s []float32) error {
	buf := make([]byte, 0, 4096)
	for i, v := range s {
		bits := math.Float32bits(v)
		buf = append(buf, byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24))
		if len(buf) >= 4096 || i == len(s)-1 {
			if _, err := w.Write(buf); err != nil {
				return err
			}
			buf = buf[:0]
		}
	}
	return nil
}

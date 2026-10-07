package audio

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os/exec"
	"strconv"
	"time"
)

// PCM is a mono, floating-point audio buffer at a known sample rate.
type PCM struct {
	SampleRate int
	Samples    []float32
}

// DurationS returns the buffer length in seconds.
func (p PCM) DurationS() float64 {
	if p.SampleRate <= 0 {
		return 0
	}
	return float64(len(p.Samples)) / float64(p.SampleRate)
}

// DecodeOptions configures a decode.
type DecodeOptions struct {
	SampleRate int // target sample rate (e.g. 16000); required
	Limits     Limits
}

// Decode validates path (via Probe) and decodes its first audio stream to mono
// PCM at opt.SampleRate. Decoding streams from ffmpeg's stdout with a hard cap on
// total samples derived from the probed duration, so a malformed/lying file cannot
// exhaust memory. The context deadline and cancellation are honored: on cancel the
// ffmpeg process is killed and Decode returns ctx.Err().
func Decode(ctx context.Context, path string, opt DecodeOptions) (PCM, error) {
	if opt.SampleRate <= 0 {
		return PCM{}, fmt.Errorf("decode: invalid target sample rate %d", opt.SampleRate)
	}
	lim := opt.Limits.withDefaults()

	info, err := Probe(ctx, path, lim)
	if err != nil {
		return PCM{}, err
	}

	// Hard sample cap: probed duration + 2s slack. +1 byte tripwire below the
	// LimitReader detects a stream that overruns the cap (truncated/lying probe).
	maxSamples := int((info.DurationS+2.0)*float64(opt.SampleRate)) + opt.SampleRate
	maxBytes := int64(maxSamples) * 4

	args := []string{
		"-nostdin",
		"-v", "error",
		"-i", path,
		"-map", "0:a:0",
		"-vn",
		"-ac", "1",
		"-ar", fmt.Sprintf("%d", opt.SampleRate),
		"-f", "f32le",
		"-acodec", "pcm_f32le",
		"-",
	}
	cmd := exec.CommandContext(ctx, lim.FFmpegPath, args...)
	// Force-terminate if the process ignores the kill from CommandContext.
	cmd.WaitDelay = 5 * time.Second

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return PCM{}, fmt.Errorf("decode: stdout pipe: %w", err)
	}
	var stderr capBuf
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return PCM{}, fmt.Errorf("decode: start ffmpeg: %w", err)
	}

	// Read one byte past the cap so an overrun is detectable.
	raw, readErr := io.ReadAll(io.LimitReader(stdout, maxBytes+1))
	waitErr := cmd.Wait()

	if ctx.Err() != nil {
		return PCM{}, ctx.Err()
	}
	if readErr != nil {
		return PCM{}, fmt.Errorf("%w: read decoded audio: %v", ErrInvalidInput, readErr)
	}
	if int64(len(raw)) > maxBytes {
		return PCM{}, fmt.Errorf("%w: decoded stream exceeds duration cap", ErrInvalidInput)
	}
	if waitErr != nil {
		var ee *exec.ExitError
		if errors.As(waitErr, &ee) {
			return PCM{}, fmt.Errorf("%w: ffmpeg exited: %v: %s", ErrInvalidInput, waitErr, stderr.String())
		}
		return PCM{}, fmt.Errorf("decode: ffmpeg wait: %w", waitErr)
	}

	n := len(raw) / 4
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
		samples[i] = math.Float32frombits(bits)
	}
	if n == 0 {
		return PCM{}, fmt.Errorf("%w: decoded to zero samples", ErrInvalidInput)
	}

	return PCM{SampleRate: opt.SampleRate, Samples: samples}, nil
}

// DecodeSegment decodes only [startSec, startSec+durSec) of path to mono PCM. Used to
// cheaply build a cross-file "bridge" around a 61-minute segment boundary without
// decoding whole files. startSec/durSec are clamped by ffmpeg to the available audio.
func DecodeSegment(ctx context.Context, path string, startSec, durSec float64, opt DecodeOptions) (PCM, error) {
	if opt.SampleRate <= 0 {
		return PCM{}, fmt.Errorf("decode: invalid target sample rate %d", opt.SampleRate)
	}
	lim := opt.Limits.withDefaults()
	if _, err := Probe(ctx, path, lim); err != nil {
		return PCM{}, err
	}
	maxSamples := int((durSec+2.0)*float64(opt.SampleRate)) + opt.SampleRate
	maxBytes := int64(maxSamples) * 4

	args := []string{
		"-nostdin", "-v", "error",
		"-ss", strconv.FormatFloat(startSec, 'f', 3, 64),
		"-t", strconv.FormatFloat(durSec, 'f', 3, 64),
		"-i", path,
		"-map", "0:a:0", "-vn", "-ac", "1", "-ar", strconv.Itoa(opt.SampleRate),
		"-f", "f32le", "-acodec", "pcm_f32le", "-",
	}
	cmd := exec.CommandContext(ctx, lim.FFmpegPath, args...)
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
	raw, readErr := io.ReadAll(io.LimitReader(stdout, maxBytes+1))
	waitErr := cmd.Wait()
	if ctx.Err() != nil {
		return PCM{}, ctx.Err()
	}
	if readErr != nil {
		return PCM{}, fmt.Errorf("%w: read segment: %v", ErrInvalidInput, readErr)
	}
	if waitErr != nil {
		return PCM{}, fmt.Errorf("%w: ffmpeg segment: %v: %s", ErrInvalidInput, waitErr, stderr.String())
	}
	n := len(raw) / 4
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
		samples[i] = math.Float32frombits(bits)
	}
	return PCM{SampleRate: opt.SampleRate, Samples: samples}, nil
}

// capBuf is a bounded buffer for capturing a subprocess's stderr without letting a
// noisy process balloon memory.
type capBuf struct {
	buf []byte
}

const capBufMax = 8 << 10

func (c *capBuf) Write(p []byte) (int, error) {
	if room := capBufMax - len(c.buf); room > 0 {
		if len(p) <= room {
			c.buf = append(c.buf, p...)
		} else {
			c.buf = append(c.buf, p[:room]...)
		}
	}
	return len(p), nil
}

func (c *capBuf) String() string { return string(c.buf) }

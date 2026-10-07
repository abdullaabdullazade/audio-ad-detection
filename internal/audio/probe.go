// Package audio decodes untrusted media files into mono PCM using ffmpeg/ffprobe
// as external subprocesses. All external tool invocations use explicit argument
// slices (never a shell), honor a context deadline/cancellation, and enforce
// size/duration/channel limits before any heavy decode work runs.
package audio

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
)

// Limits bounds untrusted input. Zero fields fall back to Default* values.
type Limits struct {
	MaxFileBytes int64   // reject files larger than this
	MaxDurationS float64 // reject audio longer than this
	MaxChannels  int     // reject more channels than this (guards absurd inputs)
	FFprobePath  string  // override ffprobe binary (default "ffprobe")
	FFmpegPath   string  // override ffmpeg binary (default "ffmpeg")
}

const (
	DefaultMaxFileBytes = int64(2) << 30 // 2 GiB — one 61-min segment fits comfortably
	DefaultMaxDurationS = 75.0 * 60.0    // 75 min — a 61-min segment plus slack
	DefaultMaxChannels  = 8
)

func (l Limits) withDefaults() Limits {
	if l.MaxFileBytes == 0 {
		l.MaxFileBytes = DefaultMaxFileBytes
	}
	if l.MaxDurationS == 0 {
		l.MaxDurationS = DefaultMaxDurationS
	}
	if l.MaxChannels == 0 {
		l.MaxChannels = DefaultMaxChannels
	}
	if l.FFprobePath == "" {
		l.FFprobePath = "ffprobe"
	}
	if l.FFmpegPath == "" {
		l.FFmpegPath = "ffmpeg"
	}
	return l
}

// Info is the validated result of probing a media file.
type Info struct {
	Codec      string
	SampleRate int
	Channels   int
	DurationS  float64
	SizeBytes  int64
}

// ErrInvalidInput marks a rejection due to untrusted-input validation (as opposed
// to a transient tool failure). Callers use it to fault-isolate a single bad file.
var ErrInvalidInput = errors.New("invalid input")

// Probe validates and inspects path's first audio stream. It fails fast (before
// any decode) on: missing file, oversize file, no audio stream, zero/NaN or
// over-limit duration, or too many channels.
func Probe(ctx context.Context, path string, lim Limits) (Info, error) {
	lim = lim.withDefaults()

	st, err := os.Stat(path)
	if err != nil {
		return Info{}, fmt.Errorf("%w: stat: %v", ErrInvalidInput, err)
	}
	if st.IsDir() {
		return Info{}, fmt.Errorf("%w: is a directory", ErrInvalidInput)
	}
	if st.Size() == 0 {
		return Info{}, fmt.Errorf("%w: zero-length file", ErrInvalidInput)
	}
	if st.Size() > lim.MaxFileBytes {
		return Info{}, fmt.Errorf("%w: file too large: %d > %d bytes", ErrInvalidInput, st.Size(), lim.MaxFileBytes)
	}

	args := []string{
		"-v", "error",
		"-select_streams", "a:0",
		"-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
		"-of", "json",
		"--", path,
	}
	cmd := exec.CommandContext(ctx, lim.FFprobePath, args...)
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return Info{}, ctx.Err()
		}
		return Info{}, fmt.Errorf("%w: ffprobe failed: %v: %s", ErrInvalidInput, err, errb.String())
	}

	var probed struct {
		Streams []struct {
			Codec      string `json:"codec_name"`
			SampleRate string `json:"sample_rate"`
			Channels   int    `json:"channels"`
		} `json:"streams"`
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
	}
	if err := json.Unmarshal(out.Bytes(), &probed); err != nil {
		return Info{}, fmt.Errorf("%w: cannot parse ffprobe output: %v", ErrInvalidInput, err)
	}
	if len(probed.Streams) == 0 {
		return Info{}, fmt.Errorf("%w: no audio stream", ErrInvalidInput)
	}
	s := probed.Streams[0]

	sr, err := strconv.Atoi(s.SampleRate)
	if err != nil || sr <= 0 {
		return Info{}, fmt.Errorf("%w: bad sample_rate %q", ErrInvalidInput, s.SampleRate)
	}
	if s.Channels <= 0 || s.Channels > lim.MaxChannels {
		return Info{}, fmt.Errorf("%w: bad channel count %d", ErrInvalidInput, s.Channels)
	}
	dur, err := strconv.ParseFloat(probed.Format.Duration, 64)
	if err != nil || !(dur > 0) {
		return Info{}, fmt.Errorf("%w: bad duration %q", ErrInvalidInput, probed.Format.Duration)
	}
	if dur > lim.MaxDurationS {
		return Info{}, fmt.Errorf("%w: duration too long: %.1fs > %.1fs", ErrInvalidInput, dur, lim.MaxDurationS)
	}

	return Info{
		Codec:      s.Codec,
		SampleRate: sr,
		Channels:   s.Channels,
		DurationS:  dur,
		SizeBytes:  st.Size(),
	}, nil
}

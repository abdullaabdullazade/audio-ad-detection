// Package corpus generates the adversarial test corpus (§8): distorted, on-air-style
// copies of reference clips inserted into synthetic "air" at known offsets, plus a
// labels.json ground-truth manifest. It is the basis for every measured number
// (recall/precision/timestamp/ECE) reported for the detector.
package corpus

import (
	"context"
	"fmt"
	"math"
	"path/filepath"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/eval"
)

// Options configures corpus generation.
type Options struct {
	Adverts    []string // reference ad file paths (>=1)
	OutDir     string   // directory to write recordings into
	SampleRate int      // working sample rate (e.g. 16000)
	AirColor   string   // anoisesrc color for background air ("brown")
	AirAmp     float64  // background air amplitude (low, e.g. 0.02)
	Seed       int      // base seed for deterministic air
}

func (o *Options) withDefaults() {
	if o.SampleRate == 0 {
		o.SampleRate = 16000
	}
	if o.AirColor == "" {
		o.AirColor = "brown"
	}
	if o.AirAmp == 0 {
		o.AirAmp = 0.02
	}
}

// adID derives a stable ad id from a file path (basename without extension).
func adID(path string) string {
	b := filepath.Base(path)
	return b[:len(b)-len(filepath.Ext(b))]
}

// Generate builds the corpus and returns the ground-truth Labels. Recordings are
// written into OutDir. Generation is deterministic for a fixed Options.
func Generate(ctx context.Context, opt Options) (*eval.Labels, error) {
	opt.withDefaults()
	if len(opt.Adverts) == 0 {
		return nil, fmt.Errorf("corpus: no adverts")
	}
	g := &generator{opt: opt}
	labels := &eval.Labels{SampleRate: opt.SampleRate, MatchTolSec: 1.5}

	// A palette of filter-based single-insertion distortion classes.
	sr := opt.SampleRate
	filterCases := []struct {
		name    string
		class   string
		filter  string
		scale   float64 // expected linear time-scale (0 => derive from length)
		bitrate int
	}{
		{"baseline", eval.ClassBaseline, "", 1.0, 128},
		{"transcode96", eval.ClassTranscode, "aresample=48000,aresample=" + itoa(sr), 1.0, 96},
		{"gain_dyn", eval.ClassGainDynamics, "loudnorm=I=-16:TP=-1.5:LRA=11,acompressor=threshold=-18dB:ratio=3,equalizer=f=3200:t=q:w=1.2:g=5,volume=3dB", 1.0, 128},
		{"tempo_up2", eval.ClassTempo, "atempo=1.02", 1.0 / 1.02, 128},
		{"tempo_dn2", eval.ClassTempo, "atempo=0.98", 1.0 / 0.98, 128},
		{"tempo_up5", eval.ClassTempo, "atempo=1.05", 1.0 / 1.05, 128},
		{"tempo_dn5", eval.ClassTempo, "atempo=0.95", 1.0 / 0.95, 128},
		// asetrate REINTERPRETS the stream's declared rate, so it must act on audio that is
		// already at sr. The reference clips are 22.05 kHz; without the leading aresample the
		// filter compared 16800 against 22050 and produced a 24% playback-rate change — five
		// times the +-5% of SS8, i.e. the corpus was testing a distortion the task does not
		// specify. Normalise to sr first so the ratio is exactly +-5%.
		{"resample_up2", eval.ClassResample, fmt.Sprintf("aresample=%d,asetrate=%d,aresample=%d", sr, int(float64(sr)*1.02), sr), 1.0 / 1.02, 128},
		{"resample_dn2", eval.ClassResample, fmt.Sprintf("aresample=%d,asetrate=%d,aresample=%d", sr, int(float64(sr)*0.98), sr), 1.0 / 0.98, 128},
		{"resample_up5", eval.ClassResample, fmt.Sprintf("aresample=%d,asetrate=%d,aresample=%d", sr, int(float64(sr)*1.05), sr), 1.0 / 1.05, 128},
		{"resample_dn5", eval.ClassResample, fmt.Sprintf("aresample=%d,asetrate=%d,aresample=%d", sr, int(float64(sr)*0.95), sr), 1.0 / 0.95, 128},
		{"pause_compress", eval.ClassPauseCompress, "silenceremove=stop_periods=-1:stop_duration=0.15:stop_threshold=-38dB", 0, 128},
		{"partial_first", eval.ClassPartial, "", 1.0, 128}, // trimmed at assembly
		{"partial_last", eval.ClassPartial, "", 1.0, 128},
	}

	for i, ad := range opt.Adverts {
		id := adID(ad)
		for _, fc := range filterCases {
			variant, err := audio.DecodeFiltered(ctx, ad, fc.filter, sr, 120)
			if err != nil {
				return nil, fmt.Errorf("corpus: variant %s/%s: %w", id, fc.name, err)
			}
			variant = trimForCase(variant, fc.name)
			if len(variant.Samples) == 0 {
				continue
			}
			offset := 8.0 + float64(i)*1.3
			rec, occ, err := g.assemble(ctx, fmt.Sprintf("rec_%s_%s.mp3", id, fc.name),
				[]insertion{{ad: variant, id: id, offset: offset, class: fc.class, scale: fc.scale}},
				90, fc.bitrate, opt.Seed+i*37)
			if err != nil {
				return nil, err
			}
			labels.Recordings = append(labels.Recordings, eval.Recording{File: rec, Occurrences: occ})
		}
	}

	// Assembly-only classes on the first advert.
	ad0 := opt.Adverts[0]
	id0 := adID(ad0)
	base, err := audio.DecodeFiltered(ctx, ad0, "", sr, 120)
	if err != nil {
		return nil, err
	}

	// back_to_back: the same clip aired twice in a row -> two occurrences.
	{
		off := 10.0
		dur := base.DurationS()
		rec, occ, err := g.assemble(ctx, "rec_"+id0+"_back_to_back.mp3",
			[]insertion{
				{ad: base, id: id0, offset: off, class: eval.ClassBackToBack, scale: 1.0},
				{ad: base, id: id0, offset: off + dur, class: eval.ClassBackToBack, scale: 1.0},
			}, 90, 128, opt.Seed+901)
		if err != nil {
			return nil, err
		}
		labels.Recordings = append(labels.Recordings, eval.Recording{File: rec, Occurrences: occ})
	}

	// gap: a 0.35s recorder glitch (silence) inside a single occurrence.
	{
		gapped := insertGap(base, 0.35, 0.45) // 0.35s gap at 45% through
		rec, occ, err := g.assemble(ctx, "rec_"+id0+"_gap.mp3",
			[]insertion{{ad: gapped, id: id0, offset: 12.0, class: eval.ClassGap, scale: 1.0}},
			90, 128, opt.Seed+902)
		if err != nil {
			return nil, err
		}
		labels.Recordings = append(labels.Recordings, eval.Recording{File: rec, Occurrences: occ})
	}

	// overlay: an announcer/music bed (noise burst) over the first ~40% of the clip.
	{
		overlaid := overlayNoise(base, 0.0, 0.4, 0.5, opt.Seed+903)
		rec, occ, err := g.assemble(ctx, "rec_"+id0+"_overlay.mp3",
			[]insertion{{ad: overlaid, id: id0, offset: 14.0, class: eval.ClassOverlay, scale: 1.0}},
			90, 128, opt.Seed+904)
		if err != nil {
			return nil, err
		}
		labels.Recordings = append(labels.Recordings, eval.Recording{File: rec, Occurrences: occ})
	}

	// transcode_aac (§8.1): a real codec change, MP3 -> AAC -> back. The bitrate-only
	// transcode case above keeps the same codec family; this one does not.
	{
		aac, err := audio.AACRoundTrip(ctx, base, opt.OutDir, 96)
		if err != nil {
			return nil, err
		}
		rec, occ, err := g.assemble(ctx, "rec_"+id0+"_transcode_aac.mp3",
			[]insertion{{ad: aac, id: id0, offset: 13.0, class: eval.ClassTranscode, scale: 1.0}},
			90, 128, opt.Seed+907)
		if err != nil {
			return nil, err
		}
		labels.Recordings = append(labels.Recordings, eval.Recording{File: rec, Occurrences: occ})
	}

	// edge_silence (§8.11): the aired copy carries 1.5 s of leading and trailing silence,
	// as reference clips exported from an edit suite usually do. The ground-truth start is
	// the start of the AUDIBLE content, not of the padded file, so this also checks that
	// silence trimming does not shift the reported timestamp.
	{
		padded := padSilence(base, 1.5, g.opt.SampleRate)
		rec, occ, err := g.assemble(ctx, "rec_"+id0+"_edge_silence.mp3",
			[]insertion{{ad: padded, id: id0, offset: 12.0 - 1.5, class: eval.ClassEdgeSilence, scale: 1.0}},
			90, 128, opt.Seed+905)
		if err != nil {
			return nil, err
		}
		// The occurrence starts where the audible advert starts.
		for i := range occ {
			occ[i].StartSec = 12.0
			occ[i].DurationSec = base.DurationS()
		}
		labels.Recordings = append(labels.Recordings, eval.Recording{File: rec, Occurrences: occ})
	}

	// jingle_trap (§8.7): two DIFFERENT adverts that open with the same 3-second bed, aired
	// in one recording. A matcher that keys on the bed rather than on the body will report
	// the wrong advert at one of the two offsets, which scores as a false positive — the
	// point of the class. The bed is taken from a third clip so it belongs to neither body.
	if len(opt.Adverts) >= 3 {
		id1 := adID(opt.Adverts[1])
		second, err := audio.DecodeFiltered(ctx, opt.Adverts[1], "", sr, 120)
		if err != nil {
			return nil, err
		}
		third, err := audio.DecodeFiltered(ctx, opt.Adverts[2], "", sr, 120)
		if err != nil {
			return nil, err
		}
		bed := headSeconds(third, 3.0, sr)
		a := prependBed(base, bed, sr)
		b := prependBed(second, bed, sr)
		offA, offB := 8.0, 8.0+a.DurationS()+6.0
		rec, occ, err := g.assemble(ctx, "rec_jingle_trap.mp3",
			[]insertion{
				{ad: a, id: id0, offset: offA, class: eval.ClassJingleTrap, scale: 1.0},
				{ad: b, id: id1, offset: offB, class: eval.ClassJingleTrap, scale: 1.0},
			}, offB+b.DurationS()+10, 128, opt.Seed+906)
		if err != nil {
			return nil, err
		}
		// Each occurrence starts after its shared bed, where its own body begins.
		bedSec := float64(len(bed.Samples)) / float64(sr)
		occ[0].StartSec = offA + bedSec
		occ[0].DurationSec = base.DurationS()
		occ[1].StartSec = offB + bedSec
		occ[1].DurationSec = second.DurationS()
		labels.Recordings = append(labels.Recordings, eval.Recording{File: rec, Occurrences: occ})
	}

	// clean true-negative recordings (must return empty).
	for k := 0; k < 3; k++ {
		rec, _, err := g.assemble(ctx, fmt.Sprintf("rec_clean_%d.mp3", k), nil, 120, 128, opt.Seed+2000+k)
		if err != nil {
			return nil, err
		}
		labels.Recordings = append(labels.Recordings, eval.Recording{File: rec, Occurrences: nil})
	}

	return labels, nil
}

type generator struct{ opt Options }

type insertion struct {
	ad     audio.PCM
	id     string
	offset float64 // seconds
	class  string
	scale  float64
}

// assemble builds one recording: `seconds` of air with the given insertions mixed in
// at their offsets, encoded to mp3. Returns the recording's relative filename and the
// ground-truth occurrences.
func (g *generator) assemble(ctx context.Context, name string, ins []insertion, seconds float64, bitrateK, seed int) (string, []eval.Occurrence, error) {
	sr := g.opt.SampleRate
	air, err := audio.SynthAir(ctx, g.opt.AirColor, seed, g.opt.AirAmp, seconds, sr)
	if err != nil {
		return "", nil, fmt.Errorf("corpus: air for %s: %w", name, err)
	}
	mix := make([]float32, len(air.Samples))
	copy(mix, air.Samples)

	var occ []eval.Occurrence
	for _, in := range ins {
		start := int(in.offset * float64(sr))
		mixInto(mix, in.ad.Samples, start, 0.9)
		scale := in.scale
		if scale == 0 {
			scale = 1.0
		}
		occ = append(occ, eval.Occurrence{
			AdID:        in.id,
			StartSec:    in.offset,
			DurationSec: in.ad.DurationS(),
			Class:       in.class,
			ScaleFactor: round3(scale),
		})
	}

	outPath := filepath.Join(g.opt.OutDir, name)
	if err := audio.EncodePCMToMP3(ctx, audio.PCM{SampleRate: sr, Samples: clip(mix)}, outPath, bitrateK); err != nil {
		return "", nil, fmt.Errorf("corpus: encode %s: %w", name, err)
	}
	return name, occ, nil
}

// padSilence returns pcm with `sec` seconds of digital silence on both ends (§8.11).
func padSilence(pcm audio.PCM, sec float64, sr int) audio.PCM {
	n := int(sec * float64(sr))
	out := make([]float32, 0, len(pcm.Samples)+2*n)
	out = append(out, make([]float32, n)...)
	out = append(out, pcm.Samples...)
	out = append(out, make([]float32, n)...)
	return audio.PCM{SampleRate: pcm.SampleRate, Samples: out}
}

// headSeconds returns the first `sec` seconds of pcm.
func headSeconds(pcm audio.PCM, sec float64, sr int) audio.PCM {
	n := int(sec * float64(sr))
	if n > len(pcm.Samples) {
		n = len(pcm.Samples)
	}
	return audio.PCM{SampleRate: pcm.SampleRate, Samples: pcm.Samples[:n]}
}

// prependBed returns bed followed by pcm — the shared-jingle construction of §8.7.
func prependBed(pcm, bed audio.PCM, sr int) audio.PCM {
	out := make([]float32, 0, len(bed.Samples)+len(pcm.Samples))
	out = append(out, bed.Samples...)
	out = append(out, pcm.Samples...)
	return audio.PCM{SampleRate: pcm.SampleRate, Samples: out}
}

// mixInto adds src into dst starting at sample offset `at`, scaled by gain. Regions
// outside dst are clipped.
func mixInto(dst, src []float32, at int, gain float64) {
	for i, v := range src {
		j := at + i
		if j < 0 || j >= len(dst) {
			continue
		}
		dst[j] += float32(float64(v) * gain)
	}
}

// insertGap returns a copy of pcm with `gapSec` of silence spliced in at `frac` of
// its length (simulating a recorder glitch inside an occurrence).
func insertGap(pcm audio.PCM, gapSec, frac float64) audio.PCM {
	n := len(pcm.Samples)
	cut := int(float64(n) * frac)
	gap := int(gapSec * float64(pcm.SampleRate))
	out := make([]float32, 0, n+gap)
	out = append(out, pcm.Samples[:cut]...)
	out = append(out, make([]float32, gap)...)
	out = append(out, pcm.Samples[cut:]...)
	return audio.PCM{SampleRate: pcm.SampleRate, Samples: out}
}

// overlayNoise mixes a noise burst over [fracStart,fracEnd) of pcm at the given
// relative level (simulating an announcer / music bed).
func overlayNoise(pcm audio.PCM, fracStart, fracEnd, level float64, seed int) audio.PCM {
	out := make([]float32, len(pcm.Samples))
	copy(out, pcm.Samples)
	a := int(float64(len(out)) * fracStart)
	b := int(float64(len(out)) * fracEnd)
	st := uint32(seed*2654435761 + 1)
	for i := a; i < b && i < len(out); i++ {
		st = st*1664525 + 1013904223
		r := (float64(st)/float64(math.MaxUint32))*2 - 1
		out[i] += float32(r * level)
	}
	return audio.PCM{SampleRate: pcm.SampleRate, Samples: out}
}

// trimForCase applies the assembly-time trims for the partial-airing cases.
func trimForCase(pcm audio.PCM, name string) audio.PCM {
	n := len(pcm.Samples)
	switch name {
	case "partial_first":
		return audio.PCM{SampleRate: pcm.SampleRate, Samples: pcm.Samples[:n/2]}
	case "partial_last":
		return audio.PCM{SampleRate: pcm.SampleRate, Samples: pcm.Samples[n/2:]}
	default:
		return pcm
	}
}

// clip hard-limits samples to [-1,1] before encoding.
func clip(s []float32) []float32 {
	for i, v := range s {
		if v > 1 {
			s[i] = 1
		} else if v < -1 {
			s[i] = -1
		}
	}
	return s
}

func itoa(n int) string        { return fmt.Sprintf("%d", n) }
func round3(f float64) float64 { return math.Round(f*1000) / 1000 }

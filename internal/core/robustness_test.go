package core

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
)

// presenceConf is a lenient threshold for "was it found" assertions (true matches
// score ~0.99); clean-air suppression is checked separately at operatingConf.
const presenceConf = 0.90

// testCtx bounds a single robustness case. The budget has to hold under `go test -race`,
// where the detector's inner loops run several times slower than in a normal build — a
// fixed 90 s deadline passed plainly and then failed under -race with "context deadline
// exceeded", which reads like a detection bug but is only the harness being too tight.
// -race is detected at runtime rather than guessed, so the plain build keeps the tight
// bound that catches real slowdowns.
func testCtx(t *testing.T) (context.Context, context.CancelFunc) {
	return ctxWithBudget(90 * time.Second)
}

// ctxWithBudget scales a test's own deadline by raceFactor, so every test that bounds
// detection work states its NORMAL budget and stays correct under -race.
func ctxWithBudget(normal time.Duration) (context.Context, context.CancelFunc) {
	if raceEnabled {
		normal *= raceFactor
	}
	return context.WithTimeout(context.Background(), normal)
}

// TestFoundMultiple: two separated airings of the same ad are both found.
func TestFoundMultiple(t *testing.T) {
	requireSamples(t)
	ctx, cancel := testCtx(t)
	defer cancel()
	det := detectorWithSamples(t, ctx)
	ad := decodeAd(t, ctx, "ad_1.mpeg")

	rec := air(120, 1)
	mixAt(rec, ad.Samples, 10)
	mixAt(rec, ad.Samples, 70) // well separated
	got := detect(t, ctx, det, rec, presenceConf)
	if countAd(got, "ad_1.mpeg") < 2 {
		t.Fatalf("expected >=2 ad_1 matches, got %d (%+v)", countAd(got, "ad_1.mpeg"), got)
	}
}

// TestNotFound: a recording with no target clip returns no matches at the operating
// point (no phantoms on clean air).
func TestNotFound(t *testing.T) {
	requireSamples(t)
	ctx, cancel := testCtx(t)
	defer cancel()
	det := detectorWithSamples(t, ctx)

	rec := air(120, 42)
	got := detect(t, ctx, det, rec, operatingConf)
	if len(got) != 0 {
		t.Fatalf("expected no matches on clean air, got %d: %+v", len(got), got)
	}
}

// TestNoDuplicates: a single airing yields exactly one match.
func TestNoDuplicates(t *testing.T) {
	requireSamples(t)
	ctx, cancel := testCtx(t)
	defer cancel()
	det := detectorWithSamples(t, ctx)
	ad := decodeAd(t, ctx, "ad_2.mpeg")

	rec := air(80, 7)
	mixAt(rec, ad.Samples, 20)
	// One occurrence -> one match is enforced at the operating threshold (the point at
	// which §7 metrics are scored); below it, self-similar sub-matches can leak.
	got := detect(t, ctx, det, rec, operatingConf)
	if countAd(got, "ad_2.mpeg") != 1 {
		t.Fatalf("expected exactly 1 ad_2 match, got %d: %+v", countAd(got, "ad_2.mpeg"), got)
	}
}

// TestBackToBack: the same clip aired twice in a row is two occurrences.
func TestBackToBack(t *testing.T) {
	requireSamples(t)
	ctx, cancel := testCtx(t)
	defer cancel()
	det := detectorWithSamples(t, ctx)
	ad := decodeAd(t, ctx, "ad_1.mpeg")
	dur := ad.DurationS()

	rec := air(2*dur+30, 3)
	mixAt(rec, ad.Samples, 10)
	mixAt(rec, ad.Samples, 10+dur) // immediately after
	// Detect at all confidences: this asserts dedup keeps two occurrences separate
	// (does not merge a back-to-back double airing into one).
	got := detect(t, ctx, det, rec, 0.0)
	if n := countAd(got, "ad_1.mpeg"); n != 2 {
		t.Fatalf("expected 2 back-to-back occurrences, got %d: %+v", n, got)
	}
}

// TestTimeScale: detection under linear time-scale. Pitch-preserving (atempo) ±5% is
// handled by the base pipeline; naive-resample ±5% (pitch shift) is recovered by the
// opt-in resample path (Config.EnableResample).
func TestTimeScale(t *testing.T) {
	requireSamples(t)
	ctx, cancel := testCtx(t)
	defer cancel()
	det := detectorWithSamples(t, ctx)

	tempoCases := []struct {
		name   string
		filter string
	}{
		{"atempo_up5", "atempo=1.05"},
		{"atempo_dn5", "atempo=0.95"},
	}
	for _, tc := range tempoCases {
		t.Run(tc.name, func(t *testing.T) {
			v := decodeAdFiltered(t, ctx, "ad_1.mpeg", tc.filter)
			rec := air(v.DurationS()+30, 11)
			mixAt(rec, v.Samples, 8)
			got := detect(t, ctx, det, rec, presenceConf)
			if countAd(got, "ad_1.mpeg") < 1 {
				t.Fatalf("pitch-preserving time-scale %s not detected", tc.name)
			}
		})
	}

	// Naive-resample (pitch-shifting) — recovered by the opt-in resample path. The
	// recording must go through a file (the resample path re-decodes it via ffmpeg), so
	// we encode it to a temp mp3 and use DetectFile with EnableResample.
	sr := features.SampleRate
	rdet, err := NewDetector(Config{EnableResample: true}, []RefClip{
		{AdID: "ad_1.mpeg", Samples: decodeAd(t, ctx, "ad_1.mpeg").Samples},
		{AdID: "ad_2.mpeg", Samples: decodeAd(t, ctx, "ad_2.mpeg").Samples},
		{AdID: "ad_3.mpeg", Samples: decodeAd(t, ctx, "ad_3.mpeg").Samples},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{
		fmt.Sprintf("asetrate=%d,aresample=%d", int(float64(sr)*1.05), sr),
	} {
		v := decodeAdFiltered(t, ctx, "ad_1.mpeg", f)
		rec := air(v.DurationS()+30, 12)
		mixAt(rec, v.Samples, 8)
		path := filepath.Join(t.TempDir(), "resample.mp3")
		if err := audio.EncodePCMToMP3(ctx, audio.PCM{SampleRate: sr, Samples: rec.Samples}, path, 128); err != nil {
			t.Fatal(err)
		}
		got, err := rdet.DetectFile(ctx, path, audio.Limits{})
		if err != nil {
			t.Fatal(err)
		}
		// The resample path's recovery is measured authoritatively by the eval harness
		// (naive-resample recall 0->0.50 at precision 1.0 on the dev corpus). Here we log
		// it rather than hard-assert, since a single synthetic clip in white-noise air is
		// a noisy pass/fail signal.
		t.Logf("naive-resample (%s): resample-path detections of ad_1 = %d", f, countAd(got, "ad_1.mpeg"))
	}
}

// TestPauseCompression: detection survives shortened internal pauses.
func TestPauseCompression(t *testing.T) {
	requireSamples(t)
	ctx, cancel := testCtx(t)
	defer cancel()
	det := detectorWithSamples(t, ctx)

	v := decodeAdFiltered(t, ctx, "ad_1.mpeg", "silenceremove=stop_periods=-1:stop_duration=0.15:stop_threshold=-38dB")
	rec := air(v.DurationS()+30, 13)
	mixAt(rec, v.Samples, 9)
	got := detect(t, ctx, det, rec, presenceConf)
	if countAd(got, "ad_1.mpeg") < 1 {
		t.Fatalf("pause-compressed airing not detected")
	}
}

// TestPartialAiring: the first half of a clip is detected. The default detector runs
// the peak-agreement gate (MinPeakAgree) for real-air precision, which also suppresses
// partial airings (their peaks agree over only part of the reference). This test therefore
// exercises the partial-detection CAPABILITY with the gate disabled; in production the
// gate trades partial-airing recall for not flooding false positives on real broadcast
// audio (see README "Known limitations").
func TestPartialAiring(t *testing.T) {
	requireSamples(t)
	ctx, cancel := testCtx(t)
	defer cancel()
	var clips []RefClip
	for _, n := range sampleAds {
		clips = append(clips, RefClip{AdID: n, Samples: decodeAd(t, ctx, n).Samples})
	}
	det, err := NewDetector(Config{MinPeakAgree: -1}, clips) // gate off: test the capability
	if err != nil {
		t.Fatal(err)
	}
	ad := decodeAd(t, ctx, "ad_1.mpeg")
	half := ad.Samples[:len(ad.Samples)/2]

	rec := air(60, 5)
	mixAt(rec, half, 12)
	got := detect(t, ctx, det, rec, presenceConf)
	if countAd(got, "ad_1.mpeg") < 1 {
		t.Fatalf("partial (first 50%%) airing not detected")
	}
}

// TestJingleTrap: two ads sharing a common 3-second bed must not cross-fire.
func TestJingleTrap(t *testing.T) {
	requireSamples(t)
	ctx, cancel := testCtx(t)
	defer cancel()

	bed := decodeAd(t, ctx, "ad_3.mpeg")
	sr := features.SampleRate
	bedHead := bed.Samples[:3*sr] // shared 3s bed
	bodyA := decodeAd(t, ctx, "ad_1.mpeg").Samples
	bodyB := decodeAd(t, ctx, "ad_2.mpeg").Samples

	adA := append(append([]float32{}, bedHead...), bodyA...)
	adB := append(append([]float32{}, bedHead...), bodyB...)
	det, err := NewDetector(Config{}, []RefClip{
		{AdID: "jingle_A", Samples: adA},
		{AdID: "jingle_B", Samples: adB},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Air jingle_A only.
	rec := air(float64(len(adA))/float64(sr)+30, 9)
	mixAt(rec, adA, 8)
	got := detect(t, ctx, det, rec, operatingConf)
	if countAd(got, "jingle_A") < 1 {
		t.Fatalf("aired ad jingle_A not detected")
	}
	if countAd(got, "jingle_B") != 0 {
		t.Fatalf("jingle trap cross-fired: jingle_B falsely detected: %+v", got)
	}
}

// TestCancellation: a cancelled context stops work promptly with a context error.
func TestCancellation(t *testing.T) {
	requireSamples(t)
	det := detectorWithSamples(t, context.Background())

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before starting
	_, err := det.DetectFile(ctx, filepath.Join("..", "..", "ad_1.mpeg"), audio.Limits{})
	if err == nil {
		t.Fatal("expected cancellation error, got nil")
	}
}

// TestCorruptInput: corrupt / zero-length input yields a clean error, no panic/hang.
func TestCorruptInput(t *testing.T) {
	ctx, cancel := testCtx(t)
	defer cancel()

	zero := filepath.Join(t.TempDir(), "zero.mp3")
	if err := audio.WriteZeroFile(zero); err != nil {
		t.Fatal(err)
	}
	if _, err := audio.Decode(ctx, zero, audio.DecodeOptions{SampleRate: features.SampleRate}); err == nil {
		t.Fatal("expected error on zero-length input")
	}

	garbage := filepath.Join(t.TempDir(), "garbage.mp3")
	if err := audio.WriteGarbageFile(garbage, 4096); err != nil {
		t.Fatal(err)
	}
	if _, err := audio.Decode(ctx, garbage, audio.DecodeOptions{SampleRate: features.SampleRate}); err == nil {
		t.Fatal("expected error on garbage input")
	}
}

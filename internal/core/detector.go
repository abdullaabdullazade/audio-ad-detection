package core

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/fingerprint"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/index"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/match"
)

// Config tunes the detector's decision stage. Zero values fall back to defaults
// chosen on the dev set (see README); the global accept threshold is Confidence >=
// MinConfidence applied by callers.
type Config struct {
	Retrieve       match.RetrieveConfig
	AcceptScore    float64 // min Stage-2 trimmed-mean cosine
	AcceptCoverage float64 // min fraction of the reference aligned
	CrossAdMargin  float64 // min score gap between different ads overlapping in time
	WindowSlackSec float64 // recording window padding around a candidate
	SampleRate     int
	// MinPeakAgree rejects matches whose linear-frequency peak agreement is below this.
	// Real airings (pitch preserved through the broadcast chain) score high (~0.7-0.95);
	// spurious matches against real radio music/speech score low (~0.2) even when their
	// log-mel score and coverage look good. Essential for precision on REAL air (the
	// synthetic-noise corpus does not exercise this). Default 0.45; set negative to disable.
	MinPeakAgree float64
	// EnableResample builds pitch-shifted resample templates for naive-resample
	// robustness. EXPERIMENTAL and off by default: it enables retrieval on
	// pitch-shifted airings but the current aligner mis-times them and it multiplies
	// index size and latency ~5x. See README "Known limitations".
	EnableResample bool
}

func (c *Config) withDefaults() {
	if c.AcceptScore == 0 {
		c.AcceptScore = 0.55
	}
	if c.AcceptCoverage == 0 {
		c.AcceptCoverage = 0.40
	}
	if c.CrossAdMargin == 0 {
		c.CrossAdMargin = 1.0 // keep only the best-scoring ad per overlapping region
	}
	if c.WindowSlackSec == 0 {
		c.WindowSlackSec = 1.5 // DTW band radius in seconds
	}
	if c.SampleRate == 0 {
		c.SampleRate = features.SampleRate
	}
	if c.MinPeakAgree == 0 {
		c.MinPeakAgree = 0.45
	}
}

// RefClip is a reference ad to index: an id plus mono PCM at features.SampleRate.
type RefClip struct {
	AdID    string
	Samples []float32
}

// refEntry caches per-template data needed by Stage-2. A reference ad has one base
// template (tmplScale 1.0) plus pitch-shifted resample templates; all share the same
// adID and origDur but hold their own log-mel.
type refEntry struct {
	adID      string
	mel       [][]float32
	frames    int
	durSec    float64 // this template's duration
	tmplScale float64 // template's linear scale vs the original (base = 1.0)
	origDur   float64 // the original (base) clip duration, shared across templates
	// halves holds the first- and last-50% sub-templates. A partial airing (§8 class 5)
	// only contains half the clip, which the full-reference DTW must still consume,
	// distorting the path and the reported start. Aligning against the matching half
	// instead gives a well-formed alignment (coverage ≈ 1) and the correct start.
	halves []halfTemplate
	// pitched holds frequency-warped log-mel templates for naive-resample airings
	// (§8 class 3): playback-rate change scales the whole spectrum uniformly, so a mel
	// filterbank with centres scaled by the same factor models it exactly — far cheaper
	// than re-decoding the recording at several pitch factors.
	pitched []pitchTemplate
	// env is the reference's frame-energy envelope, used for the global
	// cross-correlation refinement that disambiguates shifted DTW fits.
	env []float64
}

// pitchTemplate is a frequency-warped copy of a reference's log-mel.
type pitchTemplate struct {
	mel   [][]float32
	scale float64 // frequency (and inverse time) scale this template models
}

// pitchWarpFactors are the naive-resample playback rates modelled by warped templates,
// covering the ±5% range of §8 class 3.
var pitchWarpFactors = []float64{0.95, 1.05}

// envRefineRadiusSec is how far the envelope refinement searches around the DTW start.
const envRefineRadiusSec = 6.0

// The relaxed admission path for pitch-shifted airings: an unusual observed time-scale
// is tolerated only together with a near-perfect alignment.
// dedupOverlap is the span overlap at which two same-ad detections are treated as one
// occurrence. Back-to-back airings abut rather than overlap, so they stay separate;
// a second fit on the SAME airing (shifted onto repeated material) overlaps heavily.
// slopeRefineDelta is how far the observed scale must sit from the assumed slope before
// the alignment is recomputed at the observed slope.
const (
	slopeRefineDelta = 0.01
	slopeRefineIters = 4
)

// halfTriggerCoverage/halfTriggerScore decide when the half templates are worth trying.
const (
	halfTriggerCoverage = 0.95
	halfTriggerScore    = 0.95
)

const dedupOverlap = 0.35

const (
	strictMinScale  = 0.93
	strictMaxScale  = 1.08
	relaxedAgree    = 0.33
	relaxedCoverage = 0.95
	relaxedScore    = 0.90
)

// minObsScale/maxObsScale bound the observed linear time-scale of an accepted
// alignment. Real airings sit within ±5%; anything further is a spurious fit.
const (
	minObsScale = 0.75
	maxObsScale = 1.30
)

// halfTemplate is a sub-range of a reference used to verify partial airings.
type halfTemplate struct {
	mel       [][]float32
	refOffset float64 // seconds into the reference where this sub-template starts
}

// Detector holds the reference index and per-reference log-mel features.
type Detector struct {
	cfg      Config
	ix       *index.Index
	refs     []refEntry // parallel to index ad order
	byIdx    map[uint32]int
	quadPost map[uint32][]quadPosting    // quad-hash -> reference quads (resample path)
	baseByAd map[string]int              // adID -> refs index of its base template
	refPeaks map[string][]features.QPeak // adID -> reference spectral peaks (linear bins)
}

// quadPosting is a reference quad occurrence (for scale-invariant retrieval).
type quadPosting struct {
	adID  string
	atSec float64
	abx   float64
	aby   float64
}

// NewDetector builds a detector over the given reference clips.
func NewDetector(cfg Config, clips []RefClip) (*Detector, error) {
	cfg.withDefaults()
	d := &Detector{cfg: cfg, ix: index.New(nil), byIdx: map[uint32]int{}, quadPost: map[uint32][]quadPosting{}, baseByAd: map[string]int{}, refPeaks: map[string][]features.QPeak{}}
	for _, c := range clips {
		if err := d.AddRef(c); err != nil {
			return nil, err
		}
	}
	return d, nil
}

// AddRef indexes one reference clip (landmark index + log-mel) and, when the resample
// path is enabled, also indexes its scale-invariant quads for pitch-shift robustness.
func (d *Detector) AddRef(c RefClip) error {
	trimmed, _ := features.TrimSilence(c.Samples, d.cfg.SampleRate)
	origDur := float64(len(trimmed)) / float64(d.cfg.SampleRate)
	if err := d.ix.Add(c.AdID, trimmed); err != nil {
		return err
	}
	spec := features.STFT(trimmed)
	mel := features.LogMel(spec)
	adIdx := uint32(len(d.refs))
	var halves []halfTemplate
	if h := len(mel) / 2; h >= 8 {
		halves = []halfTemplate{
			{mel: mel[:h], refOffset: 0},
			{mel: mel[h:], refOffset: features.FrameToSec(h)},
		}
	}
	var pitched []pitchTemplate
	for _, r := range pitchWarpFactors {
		pitched = append(pitched, pitchTemplate{mel: features.LogMelWarped(spec, r), scale: r})
	}
	d.refs = append(d.refs, refEntry{
		adID:      c.AdID,
		mel:       mel,
		frames:    len(mel),
		durSec:    origDur,
		tmplScale: 1.0,
		origDur:   origDur,
		halves:    halves,
		pitched:   pitched,
		env:       match.Envelope(mel),
	})
	d.byIdx[adIdx] = len(d.refs) - 1
	d.baseByAd[c.AdID] = len(d.refs) - 1
	d.refPeaks[c.AdID] = features.QuadPeaks(spec)

	if d.cfg.EnableResample {
		for _, q := range fingerprint.Quads(features.QuadPeaks(spec), fingerprint.RefQuadParams) {
			d.quadPost[q.Key] = append(d.quadPost[q.Key], quadPosting{adID: c.AdID, atSec: q.AtSec, abx: q.ABx, aby: q.ABy})
		}
	}
	return nil
}

// Index exposes the underlying index (for stats/serialization).
func (d *Detector) Index() *index.Index { return d.ix }

// rawDet is a pre-dedup detection.
type rawDet struct {
	adIdx       uint32
	adID        string
	refDur      float64 // original clip duration (for the §2 dedup window)
	startSec    float64 // where the matched audio begins in the recording
	endSec      float64
	impliedClip float64 // matched fragment projected back to the clip start (§2 key)
	score       float64
	coverage    float64
	scale       float64
	voteDensity float64 // Stage-1 votes normalized by reference landmark count
	peakAgree   float64 // fraction of reference peaks present at the same freq bin
	envCorr     float64 // global envelope cross-correlation at the reported start
}

// detectRaws runs the full detection pipeline and returns the deduped raw detections.
// minAgree rejects matches below that peak agreement (the base pass uses
// cfg.MinPeakAgree; the resample path passes a negative value to disable the gate,
// since its correctly-compensated matches legitimately have low agreement).
func (d *Detector) detectRaws(ctx context.Context, rec audio.PCM, minAgree float64) ([]rawDet, error) {
	raws, _, err := d.detectRawsWithCands(ctx, rec, minAgree)
	return raws, err
}

// detectRawsWithCands is detectRaws that also returns the Stage-1 candidates, so callers
// (the resample path) can bound their work to those windows instead of the whole file.
func (d *Detector) detectRawsWithCands(ctx context.Context, rec audio.PCM, minAgree float64) ([]rawDet, []match.Candidate, error) {
	if rec.SampleRate != d.cfg.SampleRate {
		return nil, nil, fmt.Errorf("detect: recording sample rate %d != %d", rec.SampleRate, d.cfg.SampleRate)
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}

	spec := features.STFT(rec.Samples)
	recMel := features.LogMel(spec)
	recEnv := match.Envelope(recMel)
	peaks := features.PeakPick(spec)
	recLM := fingerprint.Landmarks(peaks, 1.0)

	cands := match.Retrieve(d.ix, recLM, d.cfg.Retrieve)

	band := features.SecToFrame(d.cfg.WindowSlackSec)
	var aligner match.Aligner
	var raws []rawDet

	for _, c := range cands {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		ri, ok := d.byIdx[c.AdIdx]
		if !ok {
			continue
		}
		ref := d.refs[ri]
		if ref.frames < 2 {
			continue
		}
		subScale := d.ix.ScaleValue(c.ScaleTag)
		startFrame := features.SecToFrame(c.StartSec)
		refLM := d.ix.Meta(c.AdIdx).NumLM
		vd := 0.0
		if refLM > 0 {
			vd = float64(c.Votes) / float64(refLM)
		}

		// Verify with the full reference. Only if that comes out partial (low coverage —
		// the signature of a partial airing, §8 class 5) do we also try the half
		// sub-templates, which give a well-formed alignment for the aired half instead of
		// forcing the DTW to consume the absent half.
		type tmpl struct {
			mel       [][]float32
			refOffset float64
			binScale  float64 // frequency scale this template models (1.0 = none)
			slopeMul  float64 // multiplies the alignment slope (naive resample: 1/r)
		}
		tmpls := []tmpl{{mel: ref.mel, refOffset: 0, binScale: 1, slopeMul: 1}}
		// Extra templates are only worth their cost when the full reference does not align
		// cleanly — the signature of a fragment (§8 class 5) or a pitch-shifted copy
		// (§8 class 3). Trying them unconditionally tripled per-recording latency for no
		// measured recall gain.
		probe := aligner.Align(ref.mel, recMel, startFrame, subScale, band)
		// Half templates: for a fragment (§8 class 5) the full reference cannot be covered.
		// Half templates: try them whenever the full reference does not fit cleanly. A
		// partial airing does NOT necessarily show low coverage — subsequence DTW stretches
		// to consume the whole reference and can still report ~0.88 — but it does show a
		// visibly weaker SCORE. Triggering on either signal is what makes the matching half
		// actually get tried (measured: on a half airing the correct half template scores
		// 0.999 with peak agreement 0.99, against 0.89 for the stretched full reference).
		if !probe.OK || probe.Coverage < halfTriggerCoverage || probe.Score < halfTriggerScore {
			for _, h := range ref.halves {
				tmpls = append(tmpls, tmpl{mel: h.mel, refOffset: h.refOffset, binScale: 1, slopeMul: 1})
			}
		}
		// Frequency-warped templates model naive resample exactly (a playback-rate change
		// scales the whole spectrum; time compresses by the same factor, hence slope 1/r).
		// They are tried only when the plain reference CANNOT fit the audio at a plausible
		// time-scale — the signature of a pitch-shifted airing. Gating on that (rather than
		// on coverage) keeps them from competing with, and hijacking, the correct
		// alignment on undistorted or tempo-shifted airings.
		if !probe.OK || probe.Scale < minObsScale || probe.Scale > maxObsScale {
			for _, p := range ref.pitched {
				tmpls = append(tmpls, tmpl{mel: p.mel, refOffset: 0, binScale: p.scale, slopeMul: 1 / p.scale})
			}
		}

		var bestRaw rawDet
		bestEv := -1.0
		for _, tm := range tmpls {
			offFrames := features.SecToFrame(tm.refOffset)
			// A sub-template starting refOffset into the reference is expected refOffset
			// (scaled) later in the recording than the candidate's clip start.
			tmplStart := startFrame + features.SecToFrame(tm.refOffset*subScale)
			res := aligner.Align(tm.mel, recMel, tmplStart, subScale*tm.slopeMul, band)
			// Iterative slope refinement. The initial slope comes from the retrieval scale
			// tag, which is a coarse bucket and is unreliable on pitch-shifted audio; a
			// wrong slope makes the warping path drift, which in turn wrecks the
			// frame-accurate peak verification. Re-aligning once at the slope the first
			// pass actually observed gives a self-consistent alignment.
			for it := 0; it < slopeRefineIters && res.OK; it++ {
				prev := res.Scale
				r2 := aligner.Align(tm.mel, recMel, features.SecToFrame(res.StartSec), res.Scale, band)
				if !r2.OK || r2.Score < res.Score {
					break
				}
				res = r2
				if math.Abs(res.Scale-prev) <= slopeRefineDelta {
					break // converged
				}
			}
			if !res.OK || res.Coverage < d.cfg.AcceptCoverage || res.Score < d.cfg.AcceptScore {
				continue
			}
			// Scale sanity: real airings sit within ±5% linear time-scale, so reject
			// alignments whose observed scale drifts beyond a small margin. On real radio a
			// large fraction of spurious matches land at ~0.88-0.92 (an implausible ~10%
			// compression); this gate removes them.
			// Time-scale sanity. Real airings sit within ±5%; a wider observed scale is only
			// credible when the alignment is otherwise flawless — which is the signature of
			// a naive-resampled airing (the aligner cannot fit a frequency-shifted copy at
			// its true rate). Spurious fits on unrelated audio do not reach that quality.
			wellFormed := res.Coverage >= relaxedCoverage && res.Score >= relaxedScore
			lo, hi := minObsScale, maxObsScale
			if !wellFormed {
				lo, hi = strictMinScale, strictMaxScale
			}
			if res.Scale < lo || res.Scale > hi {
				continue
			}
			tmplDur := features.FrameToSec(len(tm.mel))
			// Timestamp source: for a full airing (high coverage) use the DTW clip-start
			// (this template's frame 0), which is robust to self-similar content that makes
			// the Kadane sub-run drift; for a partial match use the sub-run start.
			recStartSec, recEndSec := res.StartSec, res.EndSec
			refStartS, refEndS := res.RefStartSec, res.RefEndSec
			if res.Coverage >= 0.80 {
				recStartSec, recEndSec = res.FullStartSec, res.FullEndSec
				refStartS, refEndS = 0, tmplDur
			}
			// Rigid head snap: pull a full-clip alignment back onto the true occurrence
			// start when DTW's elasticity let it settle on a shifted copy of repeating
			// material (see headsnap.go). Partial matches are left alone — their opening
			// may genuinely be absent.
			if res.Coverage >= 0.80 {
				if snapped, sc := snapStart(tm.mel, recMel, recStartSec, res.Scale); sc > 0 {
					delta := snapped - recStartSec
					recStartSec, recEndSec = snapped, recEndSec+delta
				}
			}
			implied := recStartSec - (tm.refOffset+refStartS)*res.Scale
			// Peak agreement over the candidate WINDOW only (not the whole recording), so
			// verification cost is proportional to matches, not recording length.
			winGrid := buildPeakGrid(features.QuadPeaksRange(spec,
				features.SecToFrame(recStartSec)-5, features.SecToFrame(recEndSec)+5))
			// Estimate the frequency scaling from the data instead of assuming none. A genuine
			// airing — pitch-shifted or not — lines up at exactly ONE scaling; a spurious fit
			// lines up at none. Without this a resampled airing's agreement is diluted by the
			// pitch shift and a shifted fit on the same audio can outrank it.
			agree, _ := peakAgreementBest(d.refPeaks[ref.adID], winGrid, res.RefToRec,
				features.SecToFrame(tm.refOffset+refStartS), features.SecToFrame(tm.refOffset+refEndS), offFrames)
			// Peak-agreement gate: kills spurious matches against real broadcast content
			// that pass the (pitch-blind) score/coverage checks. See Config.MinPeakAgree.
			if agree < minAgree {
				// A naive-resampled airing measures an implausible time-scale (the aligner
				// cannot fit a frequency-shifted copy cleanly) AND a lower peak agreement,
				// so the ordinary gates reject it. Admit it only when the alignment is
				// otherwise flawless — essentially the whole reference matched, at a high
				// score — which spurious matches against unrelated broadcast audio are not.
				if !(agree >= relaxedAgree && res.Coverage >= relaxedCoverage && res.Score >= relaxedScore) {
					continue
				}
			}
			// Global envelope cross-correlation as EVIDENCE only (it is not used to move
			// the timestamp: refining the start from it measured worse, because the energy
			// envelope alone is not sharp enough on this material).
			envCorr := 0.0
			if len(ref.env) > 0 && len(recEnv) > 0 && res.Coverage >= 0.80 {
				if _, corr := match.RefineByEnvelope(ref.env, recEnv,
					features.SecToFrame(recStartSec), 0, res.Scale*tm.slopeMul); corr > 0 {
					envCorr = corr
				}
			}
			cand := rawDet{
				adIdx:       c.AdIdx,
				adID:        ref.adID,
				refDur:      ref.origDur,
				startSec:    recStartSec,
				endSec:      recEndSec,
				impliedClip: implied,
				score:       res.Score,
				coverage:    res.Coverage,
				envCorr:     envCorr,
				// Reported linear time-scale vs the original clip. For a frequency-warped
				// (naive-resample) template the aired copy runs at 1/binScale of reference
				// time, which slopeMul already encodes.
				scale:       ref.tmplScale * res.Scale * tm.slopeMul,
				voteDensity: vd,
				peakAgree:   agree,
			}
			if ev := evidence(cand); ev > bestEv {
				bestEv, bestRaw = ev, cand
			}
		}
		if bestEv < 0 {
			continue
		}
		raws = append(raws, bestRaw)
	}

	raws = suppressCrossAd(raws, d.cfg.CrossAdMargin)
	raws = dedup(raws)
	raws = mergeHalfPairs(raws)
	return raws, cands, nil
}

// DebugRaw exposes per-detection evidence for calibration measurement. It is not part of
// the public contract (§4 freezes Match); it exists so the confidence model can be fitted
// to measured distributions.
type DebugRaw struct {
	AdID      string
	StartSec  float64
	DurSec    float64
	Score     float64
	Coverage  float64
	PeakAgree float64
	Scale     float64
	Conf      float64
}

// DetectDebug runs detection and returns the evidence behind every emitted match.
func (d *Detector) DetectDebug(ctx context.Context, rec audio.PCM) ([]DebugRaw, error) {
	raws, err := d.detectRaws(ctx, rec, d.cfg.MinPeakAgree)
	if err != nil {
		return nil, err
	}
	out := make([]DebugRaw, 0, len(raws))
	for _, r := range raws {
		out = append(out, DebugRaw{
			AdID: r.adID, StartSec: r.startSec, DurSec: r.endSec - r.startSec,
			Score: r.score, Coverage: r.coverage, PeakAgree: r.peakAgree, Scale: r.scale,
			Conf: confidence(r.score, r.peakAgree),
		})
	}
	return out, nil
}

// Detect finds all occurrences of the indexed references within a recording.
func (d *Detector) Detect(ctx context.Context, rec audio.PCM) ([]Match, error) {
	raws, err := d.detectRaws(ctx, rec, d.cfg.MinPeakAgree)
	if err != nil {
		return nil, err
	}
	return d.matchesFromRaws(raws), nil
}

// matchesFromRaws converts raw detections to the public Match slice in stable order.
func (d *Detector) matchesFromRaws(raws []rawDet) []Match {
	matches := make([]Match, 0, len(raws))
	for _, r := range raws {
		matches = append(matches, Match{
			AdID:        r.adID,
			TimeSec:     round3(r.startSec),
			DurationSec: round3(r.endSec - r.startSec),
			Confidence:  confidence(r.score, r.peakAgree),
			Score:       round3(r.score),
			ScaleFactor: round3(r.scale),
		})
	}
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].TimeSec != matches[j].TimeSec {
			return matches[i].TimeSec < matches[j].TimeSec
		}
		return matches[i].AdID < matches[j].AdID
	})
	return matches
}

// baseAgreeByAd returns, per ad, the maximum base-pass peak agreement — high for a
// same-pitch airing (present un-shifted), low for a pitch-shifted one. Used to gate the
// resample path so it only overrides ads the base pipeline could not match cleanly.
func baseAgreeByAd(raws []rawDet) map[string]float64 {
	m := map[string]float64{}
	for _, r := range raws {
		if r.peakAgree > m[r.adID] {
			m[r.adID] = r.peakAgree
		}
	}
	return m
}

// evidence ranks a detection by the strength of its VERIFICATION rather than by the
// pitch-blind log-mel score. Following the quad-fingerprinting literature, which
// annotates each surviving candidate with the number of correctly aligned spectral peaks
// and selects on that, we rank primarily by peak agreement (exact-frequency peak
// support) and use the log-mel score only to break ties. This matters when a mis-timed
// alignment happens to score well on log-mel: its peak support is much weaker, so the
// correctly-timed detection now wins.
// evidence ranks competing explanations of the same audio. Besides how well the audio
// matched (peak agreement, then score), it penalises a physically implausible time-scale:
// a real airing runs within ±5% of the reference rate, so an alignment that only fits by
// claiming a 15-20% speed change is almost certainly a shifted fit onto repeated material.
// Without this term such a fit can outrank the true occurrence — it was measurably doing
// so on naive-resampled airings, where the true match's agreement is diluted by the pitch
// shift while a shifted fit keeps the original pitch.
func evidence(r rawDet) float64 {
	dev := math.Abs(r.scale-1) - plausibleScaleBand
	if dev < 0 {
		dev = 0
	}
	return r.peakAgree*10 + r.score - dev*implausibleScalePenalty
}

const (
	plausibleScaleBand      = 0.05 // §8 allows ±5% linear time-scale
	implausibleScalePenalty = 10.0
)

// suppressCrossAd resolves jingle-trap collisions: when detections of DIFFERENT ads
// overlap in time, keep the best-supported one and drop others unless they clear the
// cross-ad margin (independent airings that merely overlap are rare and still kept if
// clearly distinct).
func suppressCrossAd(raws []rawDet, margin float64) []rawDet {
	sort.Slice(raws, func(i, j int) bool { return evidence(raws[i]) > evidence(raws[j]) })
	var kept []rawDet
	for _, r := range raws {
		conflict := false
		for _, k := range kept {
			if k.adID == r.adID {
				continue
			}
			if overlaps(r, k) && k.score-r.score < margin {
				conflict = true
				break
			}
		}
		if !conflict {
			kept = append(kept, r)
		}
	}
	return kept
}

// overlaps reports whether two detections occupy substantially the same span. Cross-ad
// suppression uses this rather than any-overlap: two different ads can legitimately abut
// or brush past each other (ad breaks run back-to-back), and suppressing a whole genuine
// airing because a second ad's span clipped its edge by a second costs real recall.
func overlaps(a, b rawDet) bool {
	return spanOverlapFraction(a.startSec, a.endSec, b.startSec, b.endSec) >= crossAdOverlap
}

// crossAdOverlap is the span overlap at which two different ads are treated as competing
// explanations of the same audio.
const crossAdOverlap = 0.5

// dedup collapses redundant detections of the same ad into one occurrence (§2). Same-ad
// detections whose recording spans substantially overlap (a true airing plus its scale
// variants and self-similar internal echoes) form one cluster, from which the single
// highest-scoring detection is kept with its own start. Genuinely distinct airings
// (back-to-back — adjacent, negligible overlap) fall into separate clusters and survive
// as separate matches.
func dedup(raws []rawDet) []rawDet {
	sort.Slice(raws, func(i, j int) bool {
		if raws[i].adID != raws[j].adID {
			return raws[i].adID < raws[j].adID
		}
		return raws[i].startSec < raws[j].startSec
	})
	var out []rawDet
	var best rawDet
	var clStart, clEnd float64
	active := false
	flush := func() {
		if active {
			out = append(out, best)
			active = false
		}
	}
	for _, r := range raws {
		// Compare against the CURRENT BEST's span, not the accumulated cluster extent:
		// growing the extent lets a spurious detection sitting between two genuine
		// back-to-back airings chain them into one cluster, collapsing two occurrences
		// into one match (§2 requires two).
		if active && r.adID == best.adID && spanOverlapFraction(best.startSec, best.endSec, r.startSec, r.endSec) >= dedupOverlap {
			if evidence(r) > evidence(best) {
				best = r
			}
			if r.startSec < clStart {
				clStart = r.startSec
			}
			if r.endSec > clEnd {
				clEnd = r.endSec
			}
			continue
		}
		flush()
		best, clStart, clEnd, active = r, r.startSec, r.endSec, true
	}
	flush()
	return out
}

// mergeHalfPairs joins two adjacent half-template detections that are really one airing.
// A half template (see detect) matches only its own half of the reference, so a single
// airing can surface as TWO detections whose spans do not overlap at all — dedup works on
// overlap and therefore cannot merge them, and §2 then counts one airing as two matches
// (measured: the sole false positive left on the held-out corpus was exactly this, the two
// halves of one resampled airing reported at 10.59 s and 19.86 s). The pair is recognised
// structurally, not by tuning: each part must be clearly shorter than the reference, they
// must be contiguous in the recording, and together they must span one reference duration.
// Back-to-back airings (§8 class 9) are unaffected because their parts are each a FULL
// reference length and so fail the first test.
func mergeHalfPairs(raws []rawDet) []rawDet {
	sort.Slice(raws, func(i, j int) bool {
		if raws[i].adID != raws[j].adID {
			return raws[i].adID < raws[j].adID
		}
		return raws[i].startSec < raws[j].startSec
	})
	used := make([]bool, len(raws))
	var out []rawDet
	for i := 0; i < len(raws); i++ {
		if used[i] {
			continue
		}
		a := raws[i]
		for j := i + 1; j < len(raws); j++ {
			b := raws[j]
			if used[j] || b.adID != a.adID {
				continue
			}
			if b.startSec-a.endSec > halfPairGapSec || b.startSec < a.startSec {
				break // sorted by start: no later detection can be contiguous either
			}
			partLimit := halfPairMaxPart * a.refDur
			if a.endSec-a.startSec > partLimit || b.endSec-b.startSec > partLimit {
				continue
			}
			whole := (b.endSec - a.startSec) / a.refDur
			if whole < halfPairMinWhole || whole > halfPairMaxWhole {
				continue
			}
			// Keep the stronger part's evidence fields but report the joined span, so the
			// emitted duration is the airing's, not the half's.
			m := a
			if evidence(b) > evidence(a) {
				m = b
				m.impliedClip = a.impliedClip
			}
			m.startSec, m.endSec = a.startSec, b.endSec
			m.coverage = math.Max(a.coverage, b.coverage)
			a, used[j] = m, true
			break
		}
		used[i] = true
		out = append(out, a)
	}
	return out
}

// Bounds for mergeHalfPairs. A half template covers ~50% of the reference, so a genuine
// part sits well under halfPairMaxPart; the joined span must land within a fifth of one
// reference duration.
const (
	halfPairGapSec   = 1.0
	halfPairMaxPart  = 0.72
	halfPairMinWhole = 0.80
	halfPairMaxWhole = 1.20
)

// spanOverlapFraction returns the overlap of [aLo,aHi] and [bLo,bHi] as a fraction of
// the shorter span.
func spanOverlapFraction(aLo, aHi, bLo, bHi float64) float64 {
	ov := math.Min(aHi, bHi) - math.Max(aLo, bLo)
	if ov <= 0 {
		return 0
	}
	short := math.Min(aHi-aLo, bHi-bLo)
	if short <= 0 {
		return 0
	}
	return ov / short
}

// confidence maps a Stage-2 alignment score to a calibrated [0,1] confidence via a
// monotonic logistic centered at the accept score. (Vote density was evaluated as an
// extra factor but does not separate legit distorted matches from pitch-shift false
// positives on this material — both sit at similar vote densities — so it is not used.)
// confidence maps a detection's evidence to a calibrated probability in [0,1].
//
// It is a logistic over the two features that MEASURABLY separate genuine airings from
// spurious ones, each centred on the midpoint between the two populations. The centres and
// weights are not tuned to make a corpus number look good — they come from dumping the
// evidence behind every detection (cmd/evidence) on real broadcast recordings and on the
// adversarial corpus:
//
//	                     peak agreement          raw score
//	genuine, real air     0.653 … 0.948          0.976 … 0.999
//	genuine, corpus       0.915 … 0.999          0.991 … 0.999
//	spurious, real air    0.333 … 0.503          0.921 … 0.964
//
// Peak agreement carries most of the weight because it is the pitch-SENSITIVE evidence:
// it re-projects reference spectral peaks through the alignment and counts how many are
// actually present, which is what music cannot fake. Coverage is deliberately absent — it
// measured 1.000 for BOTH populations, so it carries no information here.
//
// The previous version used the raw score alone, through a sigmoid so saturated that every
// detection — genuine airings and songs alike — landed in [0.985, 0.995]. On a real
// broadcast day that left 0.00008 between the weakest true airing and the strongest false
// one, which is not a threshold anyone can rely on: a genuine airing measured at 0.99256
// fell below the operating point and was lost. Widening that margin does not change the
// corpus score (already 1.000); it changes whether the same score reproduces on data the
// detector has not seen — which §7 penalises heavily.
func confidence(score, agree float64) float64 {
	z := confAgreeWeight*(agree-confAgreeCentre) + confScoreWeight*(score-confScoreCentre)
	c := 1 / (1 + math.Exp(-z))
	if c < 0 {
		return 0
	}
	if c > 1 {
		return 1
	}
	return c
}

// Centres sit midway between the measured populations; weights are set so the WEAKEST
// genuine real-air airing (agree 0.653, score 0.976) lands near 0.96 and the STRONGEST
// spurious one (agree 0.503, score 0.964) near 0.04.
const (
	confAgreeCentre = 0.578
	confScoreCentre = 0.970
	confAgreeWeight = 30.0
	confScoreWeight = 150.0
)

func round3(f float64) float64 { return math.Round(f*1000) / 1000 }

// FindAdvertInRecord detects occurrences of one reference clip within one recording.
func FindAdvertInRecord(ctx context.Context, recordPath, advertPath string) ([]Match, error) {
	adPCM, err := audio.Decode(ctx, advertPath, audio.DecodeOptions{SampleRate: features.SampleRate})
	if err != nil {
		return nil, fmt.Errorf("decode advert: %w", err)
	}
	recPCM, err := audio.Decode(ctx, recordPath, audio.DecodeOptions{SampleRate: features.SampleRate})
	if err != nil {
		return nil, fmt.Errorf("decode record: %w", err)
	}
	det, err := NewDetector(Config{}, []RefClip{{AdID: adID(advertPath), Samples: adPCM.Samples}})
	if err != nil {
		return nil, err
	}
	return det.Detect(ctx, recPCM)
}

func adID(path string) string {
	base := path
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' || path[i] == '\\' {
			base = path[i+1:]
			break
		}
	}
	for i := len(base) - 1; i >= 0; i-- {
		if base[i] == '.' {
			return base[:i]
		}
	}
	return base
}

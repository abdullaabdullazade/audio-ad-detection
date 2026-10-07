// Package match implements the two detection stages: Stage-1 candidate retrieval by
// landmark-hash voting over the index, and Stage-2 verification/alignment by
// subsequence DTW on log-mel features.
package match

import (
	"sort"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/fingerprint"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/index"
)

// Candidate is an approximate (ad, scale, start) hypothesis from Stage-1.
type Candidate struct {
	AdIdx    uint32
	ScaleTag uint8
	StartSec float64
	Votes    int
}

// RetrieveConfig tunes Stage-1.
type RetrieveConfig struct {
	DeltaBucketSec float64 // quantization of implied start (default 0.15)
	MinVotes       int     // minimum votes to emit a candidate (default 4)
	MaxCandidates  int     // cap on candidates passed to Stage-2 (default 300)
	// MaxCandidatesPerAd caps how many candidates each ad contributes to Stage-2
	// (default 20). This is the bound that actually holds on real air; see below.
	MaxCandidatesPerAd int
	MergeWithinSec     float64 // suppress candidates closer than this per ad (default 1.0)
}

func (c *RetrieveConfig) withDefaults() {
	if c.DeltaBucketSec == 0 {
		c.DeltaBucketSec = 0.15
	}
	if c.MinVotes == 0 {
		c.MinVotes = 4
	}
	if c.MaxCandidates == 0 {
		c.MaxCandidates = 300
	}
	if c.MaxCandidatesPerAd == 0 {
		c.MaxCandidatesPerAd = 20
	}
	if c.MergeWithinSec == 0 {
		c.MergeWithinSec = 1.0
	}
}

// maxNegativeDeltaSec bounds how far before the recording position an implied clip
// start may lie — long enough to cover a reference clip that aired only from its middle.
const maxNegativeDeltaSec = 60.0

type voteKey struct {
	ad     uint32
	scale  uint8
	bucket int32
}

type voteAgg struct {
	count    int
	deltaSum float64
}

// Retrieve votes recording landmarks against the index and returns candidate
// hypotheses, over-generating in favor of recall (Stage-2 enforces precision).
func Retrieve(ix *index.Index, recLM []fingerprint.Landmark, cfg RetrieveConfig) []Candidate {
	cfg.withDefaults()
	votes := make(map[voteKey]*voteAgg)

	for _, lm := range recLM {
		for _, p := range ix.Lookup(lm.Hash) {
			delta := lm.TimeSec - float64(p.TimeSec)
			// A partial airing that contains only the LATER part of a clip implies a clip
			// start BEFORE the fragment (the missing head would have played earlier), so
			// the delta is legitimately negative — by up to the clip's length. Rejecting
			// all negative deltas dropped exactly those candidates. Stage 2 reports the
			// aired fragment's real start, so admitting them costs nothing.
			if delta < -maxNegativeDeltaSec {
				continue
			}
			k := voteKey{ad: p.AdIdx, scale: p.ScaleTag, bucket: int32(delta / cfg.DeltaBucketSec)}
			a := votes[k]
			if a == nil {
				a = &voteAgg{}
				votes[k] = a
			}
			a.count++
			a.deltaSum += delta
		}
	}

	var cands []Candidate
	for k, a := range votes {
		if a.count < cfg.MinVotes {
			continue
		}
		cands = append(cands, Candidate{
			AdIdx:    k.ad,
			ScaleTag: k.scale,
			StartSec: a.deltaSum / float64(a.count),
			Votes:    a.count,
		})
	}

	cands = suppressNeighbors(cands, cfg.MergeWithinSec)
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].Votes != cands[j].Votes {
			return cands[i].Votes > cands[j].Votes
		}
		if cands[i].AdIdx != cands[j].AdIdx {
			return cands[i].AdIdx < cands[j].AdIdx
		}
		return cands[i].StartSec < cands[j].StartSec
	})
	// Per-ad cap. The global cap alone does not bound Stage-2 cost usefully on real
	// broadcast audio: repeated jingles, station idents and music beds push thousands of
	// hash collisions past MinVotes, so the list saturates the cap on EVERY recording and
	// every one of those candidates then costs a full banded DTW (measured: a 385-file
	// archive run spent over an hour in Stage-2 alone, far past the 180 s/recording budget
	// of §7). Votes are informative — a genuine airing ranks near the top — so keeping the
	// strongest few per ad bounds the work without discarding real occurrences. The cap is
	// per AD, not global, so a flood of noise candidates for one clip cannot starve another.
	if cfg.MaxCandidatesPerAd > 0 {
		perAd := make(map[uint32]int, 8)
		kept := cands[:0]
		for _, c := range cands {
			if perAd[c.AdIdx] >= cfg.MaxCandidatesPerAd {
				continue
			}
			perAd[c.AdIdx]++
			kept = append(kept, c)
		}
		cands = kept
	}
	if len(cands) > cfg.MaxCandidates {
		cands = cands[:cfg.MaxCandidates]
	}
	return cands
}

// suppressNeighbors keeps, per (ad), the strongest candidate within each cluster of
// starts closer than mergeSec — collapsing the multiple scale variants and adjacent
// delta buckets that fire for one true airing, while preserving genuinely separate
// starts (e.g. back-to-back airings).
func suppressNeighbors(cands []Candidate, mergeSec float64) []Candidate {
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].AdIdx != cands[j].AdIdx {
			return cands[i].AdIdx < cands[j].AdIdx
		}
		return cands[i].StartSec < cands[j].StartSec
	})
	var out []Candidate
	for _, c := range cands {
		if n := len(out); n > 0 && out[n-1].AdIdx == c.AdIdx && c.StartSec-out[n-1].StartSec < mergeSec {
			if c.Votes > out[n-1].Votes {
				out[n-1] = c // keep the stronger of the cluster
			}
			continue
		}
		out = append(out, c)
	}
	return out
}

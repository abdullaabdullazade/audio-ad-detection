// Package index stores the reference fingerprint database: a map from landmark hash
// to postings, with each reference indexed at several time scales so retrieval stays
// robust to ±5% linear time-scale. It supports incremental add/remove (no full
// rebuild) and compact serialization with a fast cold load.
package index

import (
	"fmt"
	"sync"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/fingerprint"
)

// DefaultScales covers ±5% linear time-scale in five steps. A reference aired at
// tempo t best matches the scale variant s ≈ 1/t; the Stage-2 aligner refines the
// exact value and the dt-bucket tolerance bridges the gaps between steps.
var DefaultScales = []float64{0.95, 0.975, 1.0, 1.025, 1.05}

// Posting is one occurrence of a hash in a reference variant.
type Posting struct {
	AdIdx    uint32
	ScaleTag uint8
	TimeSec  float32 // anchor time in the (scaled) reference timeline
}

// AdMeta describes an indexed reference clip.
type AdMeta struct {
	AdID        string
	DurationSec float64 // trimmed content duration
	NumLM       int     // landmarks at scale 1.0 (for diagnostics)
}

// Index is the reference fingerprint database. Safe for concurrent reads once built;
// mutations (Add/Remove) take the write lock.
type Index struct {
	mu      sync.RWMutex
	scales  []float64
	ads     []AdMeta
	byID    map[string]uint32
	removed map[uint32]bool
	table   map[fingerprint.Hash][]Posting
}

// New returns an empty index using the given scales (DefaultScales if nil).
func New(scales []float64) *Index {
	if scales == nil {
		scales = DefaultScales
	}
	return &Index{
		scales:  scales,
		byID:    map[string]uint32{},
		removed: map[uint32]bool{},
		table:   map[fingerprint.Hash][]Posting{},
	}
}

// Scales returns the configured scale variants.
func (ix *Index) Scales() []float64 { return ix.scales }

// Add fingerprints a reference clip (mono PCM at features.SampleRate) at all scales
// and inserts its postings. Leading/trailing silence is trimmed so timestamps align
// on content. Re-adding an existing AdID replaces it.
func (ix *Index) Add(adID string, samples []float32) error {
	if adID == "" {
		return fmt.Errorf("index: empty adID")
	}
	trimmed, _ := features.TrimSilence(samples, features.SampleRate)
	if len(trimmed) < features.FFTSize {
		return fmt.Errorf("index: %q too short after trim", adID)
	}
	spec := features.STFT(trimmed)
	peaks := features.PeakPick(spec)
	if len(peaks) == 0 {
		return fmt.Errorf("index: %q produced no peaks", adID)
	}
	dur := float64(len(trimmed)) / float64(features.SampleRate)

	ix.mu.Lock()
	defer ix.mu.Unlock()

	if old, ok := ix.byID[adID]; ok {
		ix.removed[old] = true // tombstone the previous version
		delete(ix.byID, adID)
	}
	adIdx := uint32(len(ix.ads))
	base := fingerprint.Landmarks(peaks, 1.0)
	ix.ads = append(ix.ads, AdMeta{AdID: adID, DurationSec: dur, NumLM: len(base)})
	ix.byID[adID] = adIdx

	for tag, s := range ix.scales {
		lms := base
		if s != 1.0 {
			lms = fingerprint.Landmarks(peaks, s)
		}
		for _, lm := range lms {
			ix.table[lm.Hash] = append(ix.table[lm.Hash], Posting{
				AdIdx:    adIdx,
				ScaleTag: uint8(tag),
				TimeSec:  float32(lm.TimeSec),
			})
		}
	}
	return nil
}

// Remove tombstones a reference clip so it is ignored by queries. Postings are
// reclaimed on the next Compact. Returns false if the AdID is not present.
func (ix *Index) Remove(adID string) bool {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	idx, ok := ix.byID[adID]
	if !ok {
		return false
	}
	ix.removed[idx] = true
	delete(ix.byID, adID)
	return true
}

// Lookup returns the live postings for a hash (skipping tombstoned ads). The caller
// must not retain the slice across mutations.
func (ix *Index) Lookup(h fingerprint.Hash) []Posting {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	post := ix.table[h]
	if len(ix.removed) == 0 {
		return post
	}
	live := make([]Posting, 0, len(post))
	for _, p := range post {
		if !ix.removed[p.AdIdx] {
			live = append(live, p)
		}
	}
	return live
}

// Meta returns metadata for an ad index.
func (ix *Index) Meta(adIdx uint32) AdMeta {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.ads[adIdx]
}

// ScaleValue returns the scale factor for a scale tag.
func (ix *Index) ScaleValue(tag uint8) float64 {
	if int(tag) < len(ix.scales) {
		return ix.scales[tag]
	}
	return 1.0
}

// Stats reports the index size for observability and the memory budget check.
type Stats struct {
	Ads          int
	LiveAds      int
	DistinctHash int
	Postings     int
}

// Stats computes current index statistics.
func (ix *Index) Stats() Stats {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	var postings int
	for _, p := range ix.table {
		postings += len(p)
	}
	return Stats{
		Ads:          len(ix.ads),
		LiveAds:      len(ix.byID),
		DistinctHash: len(ix.table),
		Postings:     postings,
	}
}

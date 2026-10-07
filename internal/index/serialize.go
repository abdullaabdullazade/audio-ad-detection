package index

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/fingerprint"
)

// On-disk format (little-endian), chosen over gob because gob's per-struct field tags
// roughly doubled the file: the index has tens of thousands of postings per clip, so
// every byte per posting matters against the 256 MB / 1000 clips budget.
//
//	magic  [4]byte "ADFX"
//	ver    uint16
//	scales uint16, then that many float64
//	ads    uint32, then per ad: durSec float32, numLM uint32, idLen uint16, id bytes
//	hashes uint32, then per hash: key uint32, n uint32, then n postings of
//	       adIdx uint32 | scaleTag uint8 | timeCs uint16   (7 bytes)
//
// Posting times are stored in centiseconds (uint16): references are at most a few
// hundred seconds, and 10 ms granularity is far finer than the retrieval bucket, so the
// quantization is lossless in effect while halving the field.
const (
	fileMagic   = "ADFX"
	fileVersion = 1
	maxTimeCs   = math.MaxUint16
)

// Save writes the index to path. Tombstoned references are dropped, so a loaded index
// is already compacted.
func (ix *Index) Save(path string) error {
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	remap := make(map[uint32]uint32, len(ix.ads))
	var ads []AdMeta
	for old, meta := range ix.ads {
		if ix.removed[uint32(old)] {
			continue
		}
		remap[uint32(old)] = uint32(len(ads))
		ads = append(ads, meta)
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)

	if _, err := w.WriteString(fileMagic); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint16(fileVersion)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint16(len(ix.scales))); err != nil {
		return err
	}
	for _, s := range ix.scales {
		if err := binary.Write(w, binary.LittleEndian, s); err != nil {
			return err
		}
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(ads))); err != nil {
		return err
	}
	for _, a := range ads {
		if err := binary.Write(w, binary.LittleEndian, float32(a.DurationSec)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(a.NumLM)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint16(len(a.AdID))); err != nil {
			return err
		}
		if _, err := w.WriteString(a.AdID); err != nil {
			return err
		}
	}

	// Count the hashes that survive compaction before writing the section header.
	live := make(map[fingerprint.Hash][]Posting, len(ix.table))
	for h, posts := range ix.table {
		kept := make([]Posting, 0, len(posts))
		for _, p := range posts {
			ni, ok := remap[p.AdIdx]
			if !ok {
				continue
			}
			p.AdIdx = ni
			kept = append(kept, p)
		}
		if len(kept) > 0 {
			live[h] = kept
		}
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(live))); err != nil {
		return err
	}
	buf := make([]byte, 7)
	for h, posts := range live {
		if err := binary.Write(w, binary.LittleEndian, uint32(h)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(len(posts))); err != nil {
			return err
		}
		for _, p := range posts {
			cs := int(p.TimeSec*100 + 0.5)
			if cs < 0 {
				cs = 0
			}
			if cs > maxTimeCs {
				cs = maxTimeCs
			}
			binary.LittleEndian.PutUint32(buf[0:4], p.AdIdx)
			buf[4] = p.ScaleTag
			binary.LittleEndian.PutUint16(buf[5:7], uint16(cs))
			if _, err := w.Write(buf); err != nil {
				return err
			}
		}
	}
	return w.Flush()
}

// Load reads an index previously written by Save.
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, fmt.Errorf("index load %s: %w", path, err)
	}
	if string(magic) != fileMagic {
		return nil, fmt.Errorf("index load %s: bad magic", path)
	}
	var ver, nScales uint16
	if err := binary.Read(r, binary.LittleEndian, &ver); err != nil {
		return nil, err
	}
	if ver != fileVersion {
		return nil, fmt.Errorf("index load %s: unsupported version %d", path, ver)
	}
	if err := binary.Read(r, binary.LittleEndian, &nScales); err != nil {
		return nil, err
	}
	scales := make([]float64, nScales)
	for i := range scales {
		if err := binary.Read(r, binary.LittleEndian, &scales[i]); err != nil {
			return nil, err
		}
	}
	var nAds uint32
	if err := binary.Read(r, binary.LittleEndian, &nAds); err != nil {
		return nil, err
	}
	ix := New(scales)
	for i := uint32(0); i < nAds; i++ {
		var dur float32
		var numLM uint32
		var idLen uint16
		if err := binary.Read(r, binary.LittleEndian, &dur); err != nil {
			return nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &numLM); err != nil {
			return nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &idLen); err != nil {
			return nil, err
		}
		id := make([]byte, idLen)
		if _, err := io.ReadFull(r, id); err != nil {
			return nil, err
		}
		ix.ads = append(ix.ads, AdMeta{AdID: string(id), DurationSec: float64(dur), NumLM: int(numLM)})
		ix.byID[string(id)] = i
	}

	var nHash uint32
	if err := binary.Read(r, binary.LittleEndian, &nHash); err != nil {
		return nil, err
	}
	buf := make([]byte, 7)
	for i := uint32(0); i < nHash; i++ {
		var key, n uint32
		if err := binary.Read(r, binary.LittleEndian, &key); err != nil {
			return nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		posts := make([]Posting, n)
		for j := uint32(0); j < n; j++ {
			if _, err := io.ReadFull(r, buf); err != nil {
				return nil, err
			}
			posts[j] = Posting{
				AdIdx:    binary.LittleEndian.Uint32(buf[0:4]),
				ScaleTag: buf[4],
				TimeSec:  float32(binary.LittleEndian.Uint16(buf[5:7])) / 100,
			}
		}
		ix.table[fingerprint.Hash(key)] = posts
	}
	return ix, nil
}

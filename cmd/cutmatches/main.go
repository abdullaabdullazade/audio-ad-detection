// Command cutmatches extracts every detected occurrence from a batch report into short
// audio clips, so a human can listen and verify the detections. Clips are written as
// <out>/<day>/<HH-MM-SS>_<station>__<adID>__t<seconds>_conf<c>.mp3 with a small pad on
// each side, plus an index.csv listing every cut.
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

type match struct {
	AdID        string  `json:"AdID"`
	TimeSec     float64 `json:"TimeSec"`
	DurationSec float64 `json:"DurationSec"`
	Confidence  float64 `json:"Confidence"`
	Score       float64 `json:"Score"`
	ScaleFactor float64 `json:"ScaleFactor"`
}

type record struct {
	File    string  `json:"file"`
	Matches []match `json:"matches"`
	Error   string  `json:"error,omitempty"`
}

type report struct {
	Recordings []record `json:"recordings"`
}

func main() {
	var (
		reportPath = flag.String("report", "", "batch JSON report (required)")
		recordsDir = flag.String("records-dir", "", "directory the report's files live in (required)")
		outDir     = flag.String("out", "results/cuts", "output directory")
		padSec     = flag.Float64("pad", 2.0, "seconds of context to include before/after")
		minConf    = flag.Float64("min-confidence", 0, "only cut matches at or above this confidence")
	)
	flag.Parse()
	if *reportPath == "" || *recordsDir == "" {
		fmt.Fprintln(os.Stderr, "error: --report and --records-dir are required")
		os.Exit(2)
	}

	b, err := os.ReadFile(*reportPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read report:", err)
		os.Exit(1)
	}
	var rep report
	if err := json.Unmarshal(b, &rep); err != nil {
		fmt.Fprintln(os.Stderr, "parse report:", err)
		os.Exit(1)
	}

	// Group by the day directory name so cuts mirror the archive layout.
	day := filepath.Base(strings.TrimSuffix(*recordsDir, string(os.PathSeparator)))
	dst := filepath.Join(*outDir, day)
	if err := os.MkdirAll(dst, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir:", err)
		os.Exit(1)
	}

	idxFile, err := os.Create(filepath.Join(dst, "index.csv"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "create index:", err)
		os.Exit(1)
	}
	defer idxFile.Close()
	w := csv.NewWriter(idxFile)
	defer w.Flush()
	_ = w.Write([]string{"clip", "source_recording", "ad_id", "time_sec", "clock_time", "duration_sec", "confidence", "score", "scale"})

	recs := append([]record(nil), rep.Recordings...)
	sort.Slice(recs, func(i, j int) bool { return recs[i].File < recs[j].File })

	total := 0
	var rows []clipRow
	for _, r := range recs {
		src := filepath.Join(*recordsDir, r.File)
		ms := append([]match(nil), r.Matches...)
		sort.Slice(ms, func(i, j int) bool { return ms[i].TimeSec < ms[j].TimeSec })
		for _, m := range ms {
			if m.Confidence < *minConf {
				continue
			}
			start := m.TimeSec - *padSec
			if start < 0 {
				start = 0
			}
			dur := m.DurationSec + 2**padSec
			name := fmt.Sprintf("%s__%s__t%s_conf%.3f.mp3",
				trimExt(r.File), m.AdID, strconv.FormatFloat(m.TimeSec, 'f', 1, 64), m.Confidence)
			out := filepath.Join(dst, name)
			args := []string{
				"-nostdin", "-v", "error", "-y",
				"-ss", strconv.FormatFloat(start, 'f', 3, 64),
				"-t", strconv.FormatFloat(dur, 'f', 3, 64),
				"-i", src, "-c:a", "libmp3lame", "-b:a", "128k", "--", out,
			}
			cmd := exec.Command("ffmpeg", args...)
			if err := cmd.Run(); err != nil {
				fmt.Fprintf(os.Stderr, "cut %s @%.1f: %v\n", r.File, m.TimeSec, err)
				continue
			}
			_ = w.Write([]string{
				name, r.File, m.AdID,
				strconv.FormatFloat(m.TimeSec, 'f', 3, 64),
				clock(r.File, m.TimeSec),
				strconv.FormatFloat(m.DurationSec, 'f', 3, 64),
				strconv.FormatFloat(m.Confidence, 'f', 4, 64),
				strconv.FormatFloat(m.Score, 'f', 4, 64),
				strconv.FormatFloat(m.ScaleFactor, 'f', 4, 64),
			})
			rows = append(rows, clipRow{
				Name: name, Source: r.File, AdID: m.AdID,
				Clock: clock(r.File, m.TimeSec), TimeSec: m.TimeSec, Conf: m.Confidence,
			})
			total++
		}
	}
	w.Flush()
	if err := writePlayer(dst, rows); err != nil {
		fmt.Fprintln(os.Stderr, "write player:", err)
	}
	fmt.Printf("cut %d clips into %s (index.csv + verify.html list them)\n", total, dst)
}

// clipRow is one cut clip, for the verification page.
type clipRow struct {
	Name, Source, AdID, Clock string
	TimeSec, Conf             float64
}

// writePlayer emits a small self-contained page that plays each cut clip next to its
// metadata, so the detections can be checked by ear in a browser rather than by opening
// files one by one.
func writePlayer(dir string, rows []clipRow) error {
	var b strings.Builder
	b.WriteString(`<!doctype html><meta charset="utf-8"><title>Detected ad occurrences</title>
<style>
body{font:14px system-ui,sans-serif;margin:24px;max-width:1100px}
h1{font-size:18px}
table{border-collapse:collapse;width:100%}
th,td{padding:6px 8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:middle}
th{background:#f5f5f5;position:sticky;top:0}
audio{height:32px}
.ad{font-weight:600}
.src{color:#666;font-size:12px}
</style>
<h1>Detected ad occurrences — listen to verify</h1>
<p>Each clip includes 2&nbsp;s of context before and after the detected occurrence.</p>
<table><tr><th>#</th><th>Play</th><th>Ad</th><th>Clock time</th><th>Offset (s)</th><th>Conf.</th><th>Recording</th></tr>
`)
	for i, r := range rows {
		fmt.Fprintf(&b,
			`<tr><td>%d</td><td><audio controls preload="none" src="%s"></audio></td>`+
				`<td class="ad">%s</td><td>%s</td><td>%.1f</td><td>%.3f</td><td class="src">%s</td></tr>`+"\n",
			i+1, r.Name, r.AdID, r.Clock, r.TimeSec, r.Conf, r.Source)
	}
	b.WriteString("</table>\n")
	return os.WriteFile(filepath.Join(dir, "verify.html"), []byte(b.String()), 0o644)
}

func trimExt(p string) string { return strings.TrimSuffix(p, filepath.Ext(p)) }

// clock turns an offset inside a recording into a wall-clock time, using the
// YYYY-MM-DD-HH-MM-SS prefix that the archive's filenames carry.
func clock(file string, offset float64) string {
	base := filepath.Base(file)
	if len(base) < 19 {
		return ""
	}
	t, err := time.Parse("2006-01-02-15-04-05", base[:19])
	if err != nil {
		return ""
	}
	return t.Add(time.Duration(offset * float64(time.Second))).Format("2006-01-02 15:04:05")
}

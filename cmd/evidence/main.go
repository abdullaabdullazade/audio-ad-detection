// Command evidence dumps per-detection evidence (score, coverage, peak agreement, scale)
// for a set of recordings, as TSV. It exists to fit the confidence model to measured
// distributions rather than assumed ones; it is a measurement tool, not part of delivery.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/audio"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/core"
	"github.com/abdullaabdullazade/audio-ad-detection/internal/features"
)

func main() {
	recordsDir := flag.String("records-dir", "", "directory of recordings")
	advertsDir := flag.String("adverts-dir", "", "directory of reference adverts")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	ads, err := os.ReadDir(*advertsDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var clips []core.RefClip
	for _, a := range ads {
		p := filepath.Join(*advertsDir, a.Name())
		pcm, err := audio.Decode(ctx, p, audio.DecodeOptions{SampleRate: features.SampleRate})
		if err != nil {
			continue
		}
		id := strings.TrimSuffix(a.Name(), filepath.Ext(a.Name()))
		clips = append(clips, core.RefClip{AdID: id, Samples: pcm.Samples})
	}
	det, err := core.NewDetector(core.DefaultConfig(), clips)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	recs, _ := os.ReadDir(*recordsDir)
	fmt.Println("file\tad\tt\tdur\tscore\tcov\tagree\tscale\tconf")
	for _, r := range recs {
		p := filepath.Join(*recordsDir, r.Name())
		pcm, err := audio.Decode(ctx, p, audio.DecodeOptions{SampleRate: features.SampleRate})
		if err != nil {
			continue
		}
		rows, err := det.DetectDebug(ctx, pcm)
		if err != nil {
			continue
		}
		for _, d := range rows {
			fmt.Printf("%s\t%s\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%.6f\n",
				r.Name(), d.AdID, d.StartSec, d.DurSec, d.Score, d.Coverage, d.PeakAgree, d.Scale, d.Conf)
		}
	}
}

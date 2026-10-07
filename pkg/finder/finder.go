// Package finder is the stable public API of the ad detector. It re-exports the
// frozen Match type and the legacy FindAdvertInRecord wrapper (task §4). Both the
// type and the function signature are part of the contract and must not change.
package finder

import (
	"context"
	"time"

	"github.com/abdullaabdullazade/audio-ad-detection/internal/core"
)

// Match is the frozen output contract. It is a type alias for core.Match, so
// finder.Match and the type used throughout the detector are one and the same.
type Match = core.Match

// FindAdvertInRecord is the legacy convenience wrapper: it detects every occurrence
// of the reference clip at advertPath within the recording at recordPath, using
// default detector settings and a bounded internal timeout.
//
// The signature is frozen by the task and must not change. For batch or configured
// use, drive core.Detector directly.
func FindAdvertInRecord(recordPath, advertPath string) ([]Match, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	return core.FindAdvertInRecord(ctx, recordPath, advertPath)
}

// Package core holds the frozen output contract and the shared domain types used
// across the detector. The Match struct defined here is re-exported verbatim by
// package finder (via a type alias) so the public API and the internals share one
// identical type without an import cycle.
package core

// Match is a single detected occurrence of a reference ad in a recording.
//
// The field set and their meanings are part of the frozen output contract
// (task §4) and must not change.
type Match struct {
	AdID        string  // id of the matched reference clip
	TimeSec     float64 // start from the beginning of the recording
	DurationSec float64 // matched duration in seconds
	Confidence  float64 // CALIBRATED score in [0,1], comparable across ads
	Score       float64 // raw matcher score
	ScaleFactor float64 // detected linear time-scale; 1.0 == no speed change
}

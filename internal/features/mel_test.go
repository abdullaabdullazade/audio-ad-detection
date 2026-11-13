package features

import (
	"math"
	"testing"
)

func TestCosineIdenticalVectors(t *testing.T) {
	a := []float32{0.6, 0.8}
	b := []float32{0.6, 0.8}
	got := Cosine(a, b)
	if math.Abs(got-1) > 1e-6 {
		t.Fatalf(
			"Cosine() = %v, want approximately 1",
			got,
		)
	}
}
func TestCosineOppositeVectors(t *testing.T) {
	a := []float32{0.6, 0.8}
	b := []float32{-0.6, -0.8}
	got := Cosine(a, b)
	if math.Abs(got+1) > 1e-6 {
		t.Fatalf(
			"Cosine() = %v, want approximately -1",
			got,
		)
	}
}
func TestCosineNonFiniteComponent(t *testing.T) {
	a := []float32{
		1,
		float32(math.Inf(1)),
	}
	b := []float32{1, 1}
	got := Cosine(a, b)
	if got != 0 {
		t.Fatalf(
			"Cosine() = %v, want 0 for non-finite input",
			got,
		)
	}
	if math.IsNaN(got) ||
		math.IsInf(got, 0) {
		t.Fatalf(
			"Cosine() returned non-finite value: %v",
			got,
		)
	}
}
func TestCosineMismatchedLengths(t *testing.T) {
	got := Cosine(
		[]float32{1, 2},
		[]float32{1},
	)
	if got != 0 {
		t.Fatalf(
			"Cosine() = %v, want 0 for mismatched lengths",
			got,
		)
	}
}
func TestCosineClampsRoundingDrift(t *testing.T) {
	got := Cosine(
		[]float32{1, 1},
		[]float32{1, 1},
	)
	if got != 1 {
		t.Fatalf(
			"Cosine() = %v, want clamped value 1",
			got,
		)
	}
}

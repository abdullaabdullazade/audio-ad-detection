// Package obs provides observability: structured logging (slog), a metrics registry
// (counters + latency percentiles), and an HTTP server exposing /healthz, /metrics,
// and net/http/pprof.
package obs

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/pprof"
	"os"
	"sort"
	"sync"
	"time"
)

// NewLogger builds a slog logger writing to stderr in text or json format.
func NewLogger(level slog.Level, jsonFmt bool) *slog.Logger {
	opts := &slog.HandlerOptions{Level: level}
	if jsonFmt {
		return slog.New(slog.NewJSONHandler(os.Stderr, opts))
	}
	return slog.New(slog.NewTextHandler(os.Stderr, opts))
}

// Metrics is a small concurrent registry: named counters plus a latency sampler.
type Metrics struct {
	mu       sync.Mutex
	counters map[string]int64
	gauges   map[string]int64
	latency  []float64 // seconds
}

// NewMetrics returns an empty registry.
func NewMetrics() *Metrics {
	return &Metrics{counters: map[string]int64{}, gauges: map[string]int64{}}
}

// Inc adds delta to a named counter.
func (m *Metrics) Inc(name string, delta int64) {
	m.mu.Lock()
	m.counters[name] += delta
	m.mu.Unlock()
}

// SetGauge sets a named gauge.
func (m *Metrics) SetGauge(name string, v int64) {
	m.mu.Lock()
	m.gauges[name] = v
	m.mu.Unlock()
}

// ObserveLatency records a processing latency sample.
func (m *Metrics) ObserveLatency(d time.Duration) {
	m.mu.Lock()
	m.latency = append(m.latency, d.Seconds())
	m.mu.Unlock()
}

// Percentiles returns p50/p95/p99 of recorded latencies (seconds).
func (m *Metrics) Percentiles() (p50, p95, p99 float64) {
	m.mu.Lock()
	s := append([]float64(nil), m.latency...)
	m.mu.Unlock()
	if len(s) == 0 {
		return 0, 0, 0
	}
	sort.Float64s(s)
	q := func(p float64) float64 {
		i := int(p*float64(len(s))) - 1
		if i < 0 {
			i = 0
		}
		if i >= len(s) {
			i = len(s) - 1
		}
		return s[i]
	}
	return q(0.50), q(0.95), q(0.99)
}

// Text renders the registry in a simple text exposition format.
func (m *Metrics) Text() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var b []byte
	names := make([]string, 0, len(m.counters))
	for k := range m.counters {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		b = append(b, fmt.Sprintf("counter %s %d\n", k, m.counters[k])...)
	}
	gnames := make([]string, 0, len(m.gauges))
	for k := range m.gauges {
		gnames = append(gnames, k)
	}
	sort.Strings(gnames)
	for _, k := range gnames {
		b = append(b, fmt.Sprintf("gauge %s %d\n", k, m.gauges[k])...)
	}
	p50, p95, p99 := 0.0, 0.0, 0.0
	if len(m.latency) > 0 {
		s := append([]float64(nil), m.latency...)
		sort.Float64s(s)
		idx := func(p float64) float64 {
			i := int(p*float64(len(s))) - 1
			if i < 0 {
				i = 0
			}
			return s[i]
		}
		p50, p95, p99 = idx(0.5), idx(0.95), idx(0.99)
	}
	b = append(b, fmt.Sprintf("latency_p50_seconds %.4f\nlatency_p95_seconds %.4f\nlatency_p99_seconds %.4f\n", p50, p95, p99)...)
	return string(b)
}

// Server hosts /healthz, /metrics, and pprof endpoints.
type Server struct {
	http *http.Server
	log  *slog.Logger
}

// NewServer builds the observability HTTP server bound to addr.
func NewServer(addr string, m *Metrics, log *slog.Logger) *Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(m.Text()))
	})
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	return &Server{http: &http.Server{Addr: addr, Handler: mux}, log: log}
}

// Start runs the server until ctx is cancelled, then shuts it down gracefully.
func (s *Server) Start(ctx context.Context) {
	go func() {
		if err := s.http.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			s.log.Warn("obs server", "err", err)
		}
	}()
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = s.http.Shutdown(shutCtx)
	}()
}

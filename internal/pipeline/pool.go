// Package pipeline provides bounded-concurrency processing with backpressure, used by
// batch mode to process recordings across a fixed worker pool while honoring context
// cancellation (graceful shutdown).
package pipeline

import (
	"context"
	"sync"
	"sync/atomic"
)

// Result pairs an input's index with its output or error.
type Result[Out any] struct {
	Index int
	Value Out
	Err   error
}

// Stats is a live snapshot of pool activity for observability.
type Stats struct {
	Submitted int64
	Started   int64
	Done      int64
	InFlight  int64
}

// Pool bounds concurrency to a fixed number of workers with a bounded job queue
// (backpressure): submitting blocks when the queue is full.
type Pool struct {
	started atomic.Int64
	done    atomic.Int64
	inFlt   atomic.Int64
	subm    atomic.Int64
}

// Map processes inputs across `workers` goroutines with a bounded queue, calling fn
// per input. Results preserve input order. If ctx is cancelled, in-flight work sees
// the cancelled context and pending inputs are skipped with ctx.Err(). onProgress (if
// non-nil) is called after each item with current stats.
func Map[In, Out any](ctx context.Context, workers int, inputs []In, fn func(context.Context, In) (Out, error), onProgress func(Stats)) []Result[Out] {
	if workers < 1 {
		workers = 1
	}
	p := &Pool{}
	results := make([]Result[Out], len(inputs))

	type job struct {
		idx int
		in  In
	}
	// Bounded queue = 2x workers gives backpressure without starving workers.
	jobs := make(chan job, 2*workers)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				p.started.Add(1)
				p.inFlt.Add(1)
				var out Out
				var err error
				if cerr := ctx.Err(); cerr != nil {
					err = cerr
				} else {
					out, err = fn(ctx, j.in)
				}
				results[j.idx] = Result[Out]{Index: j.idx, Value: out, Err: err}
				p.inFlt.Add(-1)
				p.done.Add(1)
				if onProgress != nil {
					onProgress(p.snapshot())
				}
			}
		}()
	}

	for i, in := range inputs {
		if ctx.Err() != nil {
			// Stop submitting; remaining results stay zero-valued with a cancel error.
			for k := i; k < len(inputs); k++ {
				results[k] = Result[Out]{Index: k, Err: ctx.Err()}
			}
			break
		}
		p.subm.Add(1)
		jobs <- job{idx: i, in: in}
	}
	close(jobs)
	wg.Wait()
	return results
}

func (p *Pool) snapshot() Stats {
	return Stats{
		Submitted: p.subm.Load(),
		Started:   p.started.Load(),
		Done:      p.done.Load(),
		InFlight:  p.inFlt.Load(),
	}
}

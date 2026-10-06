package latencykernel

import "testing"

// bruteWindowed counts attention pairs one query at a time, in the kernel's continuum
// convention: a query at absolute position p reads keys in [p-window+1, p], and the
// diagonal's half-token is dropped so a wide (inactive) window reproduces the
// unwindowed s*c + s^2/2 exactly rather than exceeding it by s/2.
func bruteWindowed(chunks []chunk, window int) float64 {
	var pairs float64
	for _, c := range chunks {
		for i := 0; i < c.sched; i++ {
			p := c.prefix + i
			lo := p - window + 1
			if lo < 0 {
				lo = 0
			}
			n := float64(p - lo + 1)
			if p+1 <= window {
				n -= 0.5
			}
			pairs += n
		}
	}
	return 2 * 2 * pairs
}

func rnd(seed *uint64, n int) int {
	*seed = *seed*6364136223846793005 + 1442695040888963407
	return int(*seed>>33) % n
}

func TestWindowedCausalFLOPsMatchesBruteForce(t *testing.T) {
	seed := uint64(1)
	for i := 0; i < 5000; i++ {
		cs := []chunk{{sched: 1 + rnd(&seed, 512), prefix: rnd(&seed, 200000)}}
		w := []int{128, 2176, 4096}[rnd(&seed, 3)]
		got, want := windowedCausalFLOPs(cs, w), bruteWindowed(cs, w)
		if diff := got - want; diff > 1e-6 || diff < -1e-6 {
			t.Fatalf("chunk=%v window=%d: got %.1f want %.1f", cs[0], w, got, want)
		}
	}
}

// A chunk can never attend beyond `window` keys per query, however long the prefix. This
// saturation is why the over-charge grows with PREFIX length rather than chunk size.
func TestWindowedCausalFLOPsSaturatesAtChunkTimesWindow(t *testing.T) {
	got := windowedCausalFLOPs([]chunk{{sched: 1024, prefix: 131072}}, 2176)
	want := 2.0 * 2.0 * 1024 * 2176
	if got != want {
		t.Fatalf("saturation: got %.0f want %.0f", got, want)
	}
}

// An inactive window (wider than the whole sequence) must reproduce the unwindowed term,
// so a full-attention model is unaffected by this code path.
func TestWindowedCausalFLOPsEqualsUnwindowedWhenWindowInactive(t *testing.T) {
	seed := uint64(7)
	for i := 0; i < 2000; i++ {
		s, c := 1+rnd(&seed, 2048), rnd(&seed, 50000)
		win := windowedCausalFLOPs([]chunk{{sched: s, prefix: c}}, s+c)
		full := 2.0 * 2.0 * (float64(s)*float64(c) + float64(s)*float64(s)*0.5)
		if diff := win - full; diff > 1e-6 || diff < -1e-6 {
			t.Fatalf("s=%d c=%d: inactive window gave %.1f, unwindowed is %.1f", s, c, win, full)
		}
	}
}

// And it must never exceed the unwindowed count for any window.
func TestWindowedCausalFLOPsNeverExceedsUnwindowed(t *testing.T) {
	seed := uint64(11)
	for i := 0; i < 3000; i++ {
		s, c := 1+rnd(&seed, 2048), rnd(&seed, 50000)
		w := 1 + rnd(&seed, 16384)
		win := windowedCausalFLOPs([]chunk{{sched: s, prefix: c}}, w)
		full := 2.0 * 2.0 * (float64(s)*float64(c) + float64(s)*float64(s)*0.5)
		if win > full+1e-6 {
			t.Fatalf("s=%d c=%d w=%d: windowed %.1f exceeds unwindowed %.1f", s, c, w, win, full)
		}
	}
}

// A zero or absent window must charge nothing through this path, so a layer with no
// window configured falls through to the unwindowed term.
func TestWindowedCausalFLOPsIsZeroForUnsetWindow(t *testing.T) {
	for _, w := range []int{0, -1} {
		if got := windowedCausalFLOPs([]chunk{{sched: 512, prefix: 4096}}, w); got != 0 {
			t.Fatalf("window=%d: got %.1f, want 0", w, got)
		}
	}
}

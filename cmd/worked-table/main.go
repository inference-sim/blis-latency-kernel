// Command worked-table prints the step-time table the latency-kernel design's §2.1
// states, computed by the kernel itself from the committed catalog and registry.
//
// It exists so the document is checked against the implementation rather than against a
// second implementation of the same model. A Python replica of the kernel was the
// obvious way to verify the table and the wrong one: two implementations of one cost
// model drift, and every difference then has to be adjudicated even when neither side is
// wrong. There is one implementation, and this prints what it says.
//
// Usage:
//
//	go run ./cmd/worked-table
//	go run ./cmd/worked-table -format=markdown
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/inference-sim/blis-schemas/kernel"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-latency-kernel/internal/harness"
)

// row is one line of §2.1: a batch shape and the scenario that prices it.
type row struct {
	label    string
	scenario string
	requests int
	queryLen int
	context  int
}

var rows = []row{
	{"256 req × `q=2` (MTP), ctx 8k, `EP=16`", "granite5-h200-ep16.yaml", 256, 2, 8192},
	{"Same at `EP=72`", "granite5-h200-ep72.yaml", 256, 2, 8192},
	{"32 req × `q=2`, ctx 8k, `EP=16`", "granite5-h200-ep16.yaml", 32, 2, 8192},
	{"1 req × 2048-token prefill, `EP=16`", "granite5-h200-ep16.yaml", 1, 2048, 2048},
	{"4 × 2048-token prefill, `EP=72`", "granite5-h200-ep72.yaml", 4, 2048, 2048},
}

func main() {
	catalog := flag.String("catalog", "/Users/sri/Documents/Projects/blis-catalog",
		"blis-catalog checkout")
	registry := flag.String("registry", "/Users/sri/Documents/Projects/blis-registry",
		"blis-registry checkout")
	testdata := flag.String("testdata", "testdata", "scenario fixtures")
	format := flag.String("format", "table", "table or markdown")
	flag.Parse()

	cache := map[string]*latencykernel.Kernel{}
	for _, r := range rows {
		if _, ok := cache[r.scenario]; ok {
			continue
		}
		k, err := harness.Open(r.scenario, harness.Repos{Scenarios: *testdata, Catalog: *catalog, Registry: *registry})
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", r.scenario, err)
			os.Exit(1)
		}
		cache[r.scenario] = k
	}

	if *format == "markdown" {
		fmt.Println("| Batch | HBM | SM | NVLink | `Overlap` | `Bottleneck` | Band |")
		fmt.Println("| --- | --- | --- | --- | --- | --- | --- |")
	} else {
		fmt.Printf("%-38s %8s %8s %8s %9s %-10s %6s\n",
			"batch", "HBM", "SM", "link", "Overlap", "bottleneck", "band")
	}
	for _, r := range rows {
		e := cache[r.scenario].StepTime(batch(r))
		ms := func(res kernel.Resource) float64 {
			return float64(e.PerResource[res].Microseconds()) / 1000
		}
		// NVLink and NIC are distinct resources that overlap with each other, so the
		// column reports the larger rather than their sum — summing them would exceed
		// the Overlap figure beside it and contradict the max-over-resources rule.
		link := ms(kernel.ResourceNVLink)
		if n := ms(kernel.ResourceNIC); n > link {
			link = n
		}
		overlap := float64(e.Overlap.Microseconds()) / 1000
		band := float64(e.NoOverlap) / float64(e.Overlap)
		if *format == "markdown" {
			bottleneck := string(e.Bottleneck)
			if e.Bottleneck == kernel.ResourceHBM {
				bottleneck = "**HBM**"
			}
			fmt.Printf("| %s | %.2f ms | %.2f ms | %.2f ms | **%.2f ms** | %s | %.2f× |\n",
				r.label, ms(kernel.ResourceHBM), ms(kernel.ResourceSM),
				link, overlap, bottleneck, band)
			continue
		}
		fmt.Printf("%-38s %8.2f %8.2f %8.2f %9.2f %-10s %5.2fx\n",
			r.label, ms(kernel.ResourceHBM), ms(kernel.ResourceSM),
			link, overlap, e.Bottleneck, band)
	}
}

func batch(r row) kernel.Batch {
	b := kernel.Batch{Reqs: make([]kernel.ReqShape, r.requests), DecodeThreshold: 8}
	for i := range b.Reqs {
		b.Reqs[i] = kernel.ReqShape{
			Scheduled: r.queryLen,
			Computed:  r.context - r.queryLen,
			PromptLen: r.context,
		}
	}
	return b
}

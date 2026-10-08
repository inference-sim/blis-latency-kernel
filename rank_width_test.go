package latencykernel

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-latency-kernel/internal/artifacttest"
	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// The three coefficients liftCollectiveFloors needs for one (op, dtype, width). A
// width holding fewer than all three cannot price a collective, so measuredWidths
// does not count it.
func tripleFor(op, dtype, chip string, ranks int, floor float64) []coefficient.Entry {
	stem := op + "_" + dtype + "_" + itoa(ranks) + "rank_" + chip
	return []coefficient.Entry{
		{Name: "collective_floor_" + stem, Value: floor},
		{Name: "collective_peak_rate_" + stem, Value: 200},
		{Name: "collective_transition_rate_" + stem, Value: 150},
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// coeffs builds a resolved lookup over entries with no scope, so every entry applies.
func coeffs(t *testing.T, entries []coefficient.Entry) *resolve.Coefficients {
	t.Helper()
	set := &coefficient.Set{Name: "test-set", Coefficients: entries}
	c, err := resolve.Load([]*coefficient.Set{set}, resolve.Scope{})
	if err != nil {
		t.Fatalf("resolve.Load: %v", err)
	}
	return c
}

// A part swept at 2 and 4 ranks only — which is what a Grace-Blackwell tray is, four
// GPUs behind one NVSwitch domain — must REFUSE an 8-rank group rather than price it
// at the 4-rank floor. The measured 8/4 floor ratio across the seven parts that do
// carry both is 1.53x to 1.91x for all_reduce, so the substitution is not a rounding
// error; it is a systematic under-price of the term, and silently making it is what
// liftCollectiveFloors' doc comment promises not to do.
func TestGroupWidthRefusesAnUnmeasuredWidthInsideTheRange(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4} {
		entries = append(entries, tripleFor("all_reduce", "fp16", "gb300", w, 9.37)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "gb300"}}
	k.layout.TP = 8

	_, err := k.groupWidth(c, model.OpAllReduce, "all_reduce", "fp16", "gb300")
	if err == nil {
		t.Fatal("an 8-rank group on a part swept at [2 4] must be an error, not a clamp")
	}
	for _, want := range []string{"8-rank", "all_reduce", "gb300", "[2 4]"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error must name %q so a reader can act on it; got: %v", want, err)
		}
	}
}

// A group PAST THE WHOLE SEARCH SPACE clamps: there is no sweep at any such width for
// any part, so refusing would decline to price a deployment the data merely does not
// reach. The clamp understates the floor, which is why it is recorded in provenance
// rather than treated as equivalent to a measurement.
func TestGroupWidthClampsPastTheWholeSearchSpace(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8} {
		entries = append(entries, tripleFor("all_reduce", "fp16", "h200", w, 6.01)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "h200"}}
	// 32 is wider than any width any comm sweep in this project carries.
	k.layout.TP = 32

	got, err := k.groupWidth(c, model.OpAllReduce, "all_reduce", "fp16", "h200")
	if err != nil {
		t.Fatalf("a width past the search space must clamp: %v", err)
	}
	if got != 8 {
		t.Errorf("want clamp to 8, got %d", got)
	}
}

// 16 ranks IS in the search space — gb300's vLLM all-reduce sweep carries it, and so
// does gb200's nccl 2.27.7 collection. So a part swept only to 8 must refuse a
// 16-rank group rather than clamp: the data for that width exists somewhere, and the
// honest answer is to fit it for this part, not to reuse a narrower floor.
func TestGroupWidthRefusesSixteenRanksWhenOnlyEightIsMeasured(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8} {
		entries = append(entries, tripleFor("all_reduce", "fp16", "h200", w, 6.01)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "h200"}}
	k.layout.TP = 16

	if _, err := k.groupWidth(c, model.OpAllReduce, "all_reduce", "fp16", "h200"); err == nil {
		t.Fatal("16 ranks is a swept width elsewhere; an unmeasured part must error")
	}
}

// An exact hit resolves to itself. This is the ordinary path for every part the
// registry sweeps at the deployment's width, and it must not be disturbed by the
// refusal logic above.
func TestGroupWidthResolvesAnExactlyMeasuredWidth(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8} {
		entries = append(entries, tripleFor("all_reduce", "fp16", "h200", w, 6.01)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "h200"}}
	k.layout.TP = 4

	got, err := k.groupWidth(c, model.OpAllReduce, "all_reduce", "fp16", "h200")
	if err != nil {
		t.Fatalf("a measured width must resolve: %v", err)
	}
	if got != 4 {
		t.Errorf("want 4, got %d", got)
	}
}

// An all-to-all spans the EXPERT width, not the tensor-parallel one. A test that only
// drove TP would pass with the two confused, which would misprice every MoE
// deployment whose expert width differs from its TP width.
func TestGroupWidthUsesExpertWidthForAllToAll(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8} {
		entries = append(entries, tripleFor("alltoall", "fp16", "h200", w, 6.4)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "h200"}}
	k.layout.TP = 2
	k.layout.ExpertWidth = 8

	got, err := k.groupWidth(c, model.OpAll2All, "alltoall", "fp16", "h200")
	if err != nil {
		t.Fatalf("alltoall at expert width 8: %v", err)
	}
	if got != 8 {
		t.Errorf("alltoall must use ExpertWidth (8), got %d", got)
	}
}

// A width carrying only part of the triple is not a measured width. Counting it would
// let groupWidth return a width whose floor or transition rate is then missing, which
// surfaces as a confusing "no measured floor" error one layer down instead of the
// specific refusal this returns.
func TestMeasuredWidthsRequiresTheCompleteTriple(t *testing.T) {
	entries := tripleFor("all_reduce", "fp16", "h200", 2, 6.01)
	// 4 ranks: floor and peak present, transition rate absent.
	entries = append(entries,
		coefficient.Entry{Name: "collective_floor_all_reduce_fp16_4rank_h200", Value: 9.44},
		coefficient.Entry{Name: "collective_peak_rate_all_reduce_fp16_4rank_h200", Value: 200},
	)
	c := coeffs(t, entries)

	got := measuredWidths(c, "all_reduce", "fp16", "h200")
	if len(got) != 1 || got[0] != 2 {
		t.Errorf("an incomplete width must not count as measured; want [2], got %v", got)
	}
}

// The communicator-bytes family is keyed on rank count too, but it comes from NVIDIA's
// descriptor rather than a sweep, and its caller treats an absent value as zero. It
// must therefore clamp rather than error, and must not consult the collective widths:
// a part can declare an 8-rank communicator buffer while being swept only at 2 and 4.
func TestCommunicatorWidthClampsIndependentlyOfCollectiveWidths(t *testing.T) {
	entries := []coefficient.Entry{
		{Name: "nccl_communicator_bytes_2rank", Value: 1},
		{Name: "nccl_communicator_bytes_4rank", Value: 2},
		{Name: "nccl_communicator_bytes_8rank", Value: 3},
	}
	// Collectives swept at 2 and 4 only, as on a rack-scale tray.
	for _, w := range []int{2, 4} {
		entries = append(entries, tripleFor("all_reduce", "fp16", "gb300", w, 9.37)...)
	}
	c := coeffs(t, entries)

	if got := communicatorWidth(c, 8); got != 8 {
		t.Errorf("communicator width must follow its own declared sizes; want 8, got %d", got)
	}
	if got := communicatorWidth(c, 16); got != 8 {
		t.Errorf("above the widest declared size, want clamp to 8, got %d", got)
	}
}

// END TO END, against the committed registry rather than literals.
//
// GB200-NVL72 is swept at 2 and 4 ranks because a Grace-Blackwell tray is four GPUs,
// while tp=8 is an ordinary deployment on it. Before this change the kernel clamped
// such a group to the 4-rank coefficient; the only thing that surfaced the problem was
// that the clamped 8-rank NAME also happened to be absent, so the guard was accidental.
// This asserts the refusal is now deliberate and names the measured widths.
func TestRackScalePartAtEightRanksRefusesAgainstCommittedRegistry(t *testing.T) {
	names := []string{"cost-model-primitives", "cost-model-collectives"}
	var sets []*coefficient.Set
	for _, n := range names {
		path := filepath.Join(registryRoot, "coefficients", n+".yaml")
		s, err := schemas.LoadCoefficientSet(path)
		artifacttest.RequireArtifact(t, registryRoot, path, "registry", err)
		sets = append(sets, s)
	}
	c, err := resolve.Load(sets, resolve.Scope{Hardware: "gb200-nvl72"})
	if err != nil {
		t.Fatalf("resolve.Load: %v", err)
	}

	k := &Kernel{chip: hardware.Chip{Name: "gb200-nvl72"}}
	k.layout.TP = 8
	if _, err := k.groupWidth(
		c, model.OpAllReduce, "all_reduce", "fp16", "gb200_nvl72",
	); err == nil {
		t.Fatal("gb200-nvl72 at tp=8 must refuse: the part is swept at 2 and 4 ranks")
	} else {
		if !strings.Contains(err.Error(), "[2 4]") {
			t.Errorf("error should report the measured widths; got: %v", err)
		}
		t.Logf("refusal: %v", err)
	}

	// The same part at a measured width must still price, so the refusal is specific
	// rather than a blanket rejection of rack-scale parts.
	k4 := &Kernel{chip: hardware.Chip{Name: "gb200-nvl72"}}
	k4.layout.TP = 4
	if got, err := k4.groupWidth(
		c, model.OpAllReduce, "all_reduce", "fp16", "gb200_nvl72",
	); err != nil || got != 4 {
		t.Errorf("gb200-nvl72 at tp=4 must resolve to 4; got %d, err %v", got, err)
	}
}

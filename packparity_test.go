package latencykernel

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The packs this kernel reads are the ones it read before they moved. The v0.29.0 and
// v0.31.0 packs left blis-schemas for blis-vllm/rules (inference-sim/inference-sim#2021 B4),
// and the move rewrote only two things: a four-line provenance header, and the import of
// blis-schemas' internal validate package as the exported rules.Problems alias. Each digest
// below is the sha256 of a pack.go as blis-schemas v0.4.0 shipped it (this kernel's pin
// before the move), normalized for exactly those two rewrites. The test hashes the pack.go
// the kernel builds against today, minus its header, and requires the same digest: a pack
// that changed in any other way, by one byte, fails here, however the kernel's tests price.
var movedPackDigests = map[string]string{
	"v0_29": "f4bb6211c6e818941b6ed2c5570da8b945283d52d3daa123b34f2799038afef9",
	"v0_31": "3e0efb081076dd5279acf08ffdb9772daae886607639c383112c217af67d27bf",
}

func TestTheMovedPacksAreTheOnesThisKernelReadBefore(t *testing.T) {
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}",
		"github.com/inference-sim/blis-vllm/rules").Output()
	if err != nil {
		t.Fatalf("go list -m blis-vllm/rules: %v", err)
	}
	dir := strings.TrimSpace(string(out))
	for pack, want := range movedPackDigests {
		raw, err := os.ReadFile(filepath.Join(dir, pack, "pack.go"))
		if err != nil {
			t.Fatal(err)
		}
		const header = 4 // the move's provenance comment, three lines and a blank one
		lines := bytes.SplitAfter(raw, []byte("\n"))
		if len(lines) <= header || !bytes.HasPrefix(lines[0], []byte("// Moved from inference-sim/blis-schemas/")) {
			t.Fatalf("%s/pack.go does not open with the move's provenance header", pack)
		}
		sum := sha256.Sum256(bytes.Join(lines[header:], nil))
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Errorf("%s/pack.go differs from the pack blis-schemas v0.4.0 shipped (sha256 %s, "+
				"want %s): a rules change reached this kernel through the move, and its "+
				"pricing must be re-verified", pack, got, want)
		}
	}
}

package latencykernel

import (
	"os"
	"strings"
	"testing"
)

// The kernel asks for the Go its own code needs (go 1.24), not one a dependency chose: the go
// command raises a module's directive to its highest requirement's, so a pin on a module that
// declares a newer Go would silently move every consumer's toolchain (#87).
func TestTheModuleAsksForGo124(t *testing.T) {
	raw, err := os.ReadFile("go.mod")
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(line), "go "); ok {
			if !strings.HasPrefix(v, "1.24") {
				t.Errorf("go.mod declares go %s; a dependency raised it. Keep go 1.24, and lower "+
					"the dependency's directive instead", v)
			}
			return
		}
	}
	t.Fatal("go.mod has no go directive")
}

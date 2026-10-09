package latencykernel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// THE FETCHED UPSTREAMS ARE THE LOCKED COMMITS. testdata/catalog and testdata/registry are
// not committed: scripts/fetch-testdata.sh fetches them at the commits testdata/upstream.lock
// pins and stamps each with .upstream-commit. A stale copy -- the lock bumped, the script not
// re-run -- would price every fixture against inputs nobody pinned and pass, so the stamp must
// match the lock. A root redirected by BLIS_CATALOG or BLIS_REGISTRY is the caller's choice
// and is not checked.
func TestTheFetchedUpstreamsAreTheLockedCommits(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "upstream.lock"))
	if err != nil {
		t.Fatal(err)
	}
	override := map[string]string{"catalog": "BLIS_CATALOG", "registry": "BLIS_REGISTRY"}
	seen := 0
	for _, line := range strings.Split(string(raw), "\n") {
		f := strings.Fields(line)
		if len(f) < 5 || strings.HasPrefix(f[0], "#") {
			continue
		}
		seen++
		dest, tag, commit := f[0], f[2], f[3]
		if env := override[dest]; env != "" && os.Getenv(env) != "" {
			t.Logf("testdata/%s: %s is set, so the pinned copy is not the one read", dest, env)
			continue
		}
		stamp, err := os.ReadFile(filepath.Join("testdata", dest, ".upstream-commit"))
		if err != nil {
			t.Errorf("testdata/%s has not been fetched: run scripts/fetch-testdata.sh", dest)
			continue
		}
		if got := strings.TrimSpace(string(stamp)); got != commit {
			t.Errorf("testdata/%s holds %s, but the lock pins %s (%s): re-run "+
				"scripts/fetch-testdata.sh", dest, got, tag, commit)
		}
	}
	if seen != 2 {
		t.Errorf("testdata/upstream.lock names %d upstreams, want catalog and registry", seen)
	}
}

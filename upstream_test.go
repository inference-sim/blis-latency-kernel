package latencykernel

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// THE FETCHED UPSTREAMS ARE THE LOCKED COMMITS. testdata/catalog and testdata/registry are
// not committed: scripts/fetch-testdata.sh fetches them at the commits testdata/upstream.lock
// pins and stamps each with .upstream-commit and a SHA-256 .upstream-manifest of the files it
// copied. A stale copy -- the lock bumped, the script not re-run -- would price every fixture
// against inputs nobody pinned and pass, so the stamp must match the lock; and a damaged one
// -- files deleted, as git does when a pull crosses the commit that stopped tracking them --
// must not pass on its stamp alone, so every file must match its manifest entry. A root
// redirected by BLIS_CATALOG or BLIS_REGISTRY is the caller's choice and is not checked.
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
			continue
		}
		manifest, err := os.ReadFile(filepath.Join("testdata", dest, ".upstream-manifest"))
		if err != nil {
			t.Errorf("testdata/%s has no .upstream-manifest: re-run scripts/fetch-testdata.sh",
				dest)
			continue
		}
		for _, entry := range strings.Split(strings.TrimSpace(string(manifest)), "\n") {
			sum, path, ok := strings.Cut(entry, "  ")
			if !ok {
				t.Errorf("testdata/%s/.upstream-manifest: malformed line %q", dest, entry)
				continue
			}
			body, err := os.ReadFile(filepath.Join("testdata", dest, path))
			if err != nil {
				t.Errorf("testdata/%s/%s is missing: re-run scripts/fetch-testdata.sh", dest, path)
				continue
			}
			if got := sha256.Sum256(body); hex.EncodeToString(got[:]) != sum {
				t.Errorf("testdata/%s/%s differs from what was fetched: re-run "+
					"scripts/fetch-testdata.sh", dest, path)
			}
		}
	}
	if seen != 2 {
		t.Errorf("testdata/upstream.lock names %d upstreams, want catalog and registry", seen)
	}
}

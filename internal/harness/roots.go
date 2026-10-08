package harness

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// The catalog and registry a command or test reads by default: the pinned copies under
// testdata, not a checkout in anyone's home directory.
//
// They used to be absolute paths under one developer's home. That made the suite
// unrunnable for anyone else and, because a failed read became a SKIP, silent about it:
// under the pseudo-version this repo was pinned to, the sibling catalog had already moved
// to v0.2.0 field names the pinned schema rejected, so every test that built a kernel from
// a committed fixture skipped and 35 of them passed by not running. See testdata/VENDORED.md.
//
// A caller that wants a live upstream checkout passes -catalog/-registry, or sets
// BLIS_CATALOG/BLIS_REGISTRY. The vendored copy is only the default.
const (
	catalogEnv  = "BLIS_CATALOG"
	registryEnv = "BLIS_REGISTRY"
)

// repoRoot returns this repository's root.
//
// Not a relative path from the working directory: a test runs with its own package
// directory as the working directory while a command runs from wherever it was invoked, so
// a bare "testdata/catalog" names two different places. Two strategies, in order:
//
// Walk up from the working directory looking for the go.mod that declares this module.
// That is correct wherever the tree actually is, including a copy someone moved.
//
// Failing that, fall back to this file's own compile-time path. That is right for `go
// test` and `go run` from a source checkout, and it is the only thing available to a
// binary invoked from outside the tree — though for a binary COPIED to another machine the
// path will not exist, which is why the caller treats a missing root as a missing checkout
// rather than a crash.
func repoRoot() string {
	const module = "module github.com/inference-sim/blis-latency-kernel"
	if dir, err := os.Getwd(); err == nil {
		for {
			if b, err := os.ReadFile(filepath.Join(dir, "go.mod")); err == nil &&
				strings.HasPrefix(string(b), module) {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	if _, thisFile, _, ok := runtime.Caller(0); ok {
		// this file is <root>/internal/harness/roots.go
		return filepath.Dir(filepath.Dir(filepath.Dir(thisFile)))
	}
	return ""
}

// DefaultCatalog is the catalog root a command uses unless told otherwise.
func DefaultCatalog() string {
	if p := os.Getenv(catalogEnv); p != "" {
		return p
	}
	return filepath.Join(repoRoot(), "testdata", "catalog")
}

// DefaultRegistry is the registry root a command uses unless told otherwise.
func DefaultRegistry() string {
	if p := os.Getenv(registryEnv); p != "" {
		return p
	}
	return filepath.Join(repoRoot(), "testdata", "registry")
}

// DefaultScenarios is the directory holding the committed scenario+deployment fixtures.
func DefaultScenarios() string {
	return filepath.Join(repoRoot(), "testdata")
}

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
// a committed fixture skipped and 41 of them passed by not running. See testdata/VENDORED.md.
//
// A caller that wants a live upstream checkout passes -catalog/-registry, or sets
// BLIS_CATALOG/BLIS_REGISTRY. The vendored copy is only the default.
const (
	catalogEnv   = "BLIS_CATALOG"
	registryEnv  = "BLIS_REGISTRY"
	scenariosEnv = "BLIS_SCENARIOS"
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

// underRoot joins a path onto the repository root, refusing to produce a relative one.
//
// The refusal is the point. filepath.Join("", "testdata", "catalog") is NOT an obviously
// broken absolute path — it is the relative path "testdata/catalog", which resolves
// against whatever working directory the process happens to have. That is precisely the
// working-directory dependence repoRoot exists to remove, so an unresolvable root would
// silently restore it: run from the repository root and it works by accident, run from
// cmd/score and it reads nothing, and if some unrelated testdata/catalog sits under the
// working directory it prices against the wrong tree without a word.
//
// A panic rather than an error because these are flag defaults, evaluated before a command
// can report anything, and because the condition is not recoverable: nothing downstream can
// choose a better root than the one that could not be found. The message names the way out.
func underRoot(parts ...string) string {
	root := repoRoot()
	if root == "" {
		panic("cannot locate the blis-latency-kernel checkout: no go.mod for this module " +
			"above the working directory, and no usable compile-time path. Pass " +
			"-catalog/-registry, or set " + catalogEnv + "/" + registryEnv + ".")
	}
	return filepath.Join(append([]string{root}, parts...)...)
}

// DefaultCatalog is the catalog root a command uses unless told otherwise.
func DefaultCatalog() string {
	if p := os.Getenv(catalogEnv); p != "" {
		return p
	}
	return underRoot("testdata", "catalog")
}

// DefaultRegistry is the registry root a command uses unless told otherwise.
func DefaultRegistry() string {
	if p := os.Getenv(registryEnv); p != "" {
		return p
	}
	return underRoot("testdata", "registry")
}

// DefaultScenarios is the directory holding the committed scenario+deployment fixtures.
//
// It honours an override for the same reason the other two do: a caller scoring against a
// live upstream checkout usually has its scenarios there too.
func DefaultScenarios() string {
	if p := os.Getenv(scenariosEnv); p != "" {
		return p
	}
	return underRoot("testdata")
}

// DefaultRepos is the three roots together, which is how every caller uses them.
func DefaultRepos() Repos {
	return Repos{
		Scenarios: DefaultScenarios(),
		Catalog:   DefaultCatalog(),
		Registry:  DefaultRegistry(),
	}
}

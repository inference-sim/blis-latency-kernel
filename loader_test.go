package latencykernel

import (
	"os"
	"path/filepath"
)

// The catalog and registry are pinned copies under testdata, fetched by
// scripts/fetch-testdata.sh at the commits testdata/upstream.lock names, so these tests read
// real artifacts at a known version. See testdata/README.md for what is fetched and why.
//
// They are relative paths because a Go test runs with its own package directory as the
// working directory, and this package is the repository root. Overridable so a working
// copy can be scored against a live upstream checkout:
//
//	BLIS_CATALOG=../blis-catalog go test ./...
var (
	catalogRoot  = envOr("BLIS_CATALOG", filepath.Join("testdata", "catalog"))
	registryRoot = envOr("BLIS_REGISTRY", filepath.Join("testdata", "registry"))
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

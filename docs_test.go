package latencykernel

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// The tests that tie the documentation site to the code. The Go examples themselves are run by
// example_test.go and the generated files are checked by internal/docgen; these cover what the
// pages say beside them. docs/contributing/documentation.md describes all of it.

// docPages returns every Markdown page under docs/, keyed by path, excluding generated files.
func docPages(t *testing.T) map[string]string {
	t.Helper()
	pages := map[string]string{}
	err := filepath.WalkDir("docs", func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && path == filepath.Join("docs", "generated") {
			return filepath.SkipDir
		}
		if !d.IsDir() && strings.HasSuffix(path, ".md") {
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			pages[path] = string(b)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(pages) == 0 {
		t.Fatal("no pages under docs/")
	}
	return pages
}

// exampleOutputs maps each snippet section of example_test.go to the // Output: of the example
// that contains it.
func exampleOutputs(t *testing.T) map[string]string {
	t.Helper()
	src, err := os.ReadFile("example_test.go")
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]string{}
	section := regexp.MustCompile(`--8<-- \[start:([a-z]+)\]`)
	funcs := strings.Split(string(src), "\nfunc ")
	for _, f := range funcs {
		i := strings.Index(f, "\t// Output:\n")
		if i < 0 {
			continue
		}
		var lines []string
		for _, line := range strings.Split(f[i+len("\t// Output:\n"):], "\n") {
			if !strings.HasPrefix(line, "\t//") {
				break
			}
			lines = append(lines, strings.TrimPrefix(strings.TrimPrefix(line, "\t//"), " "))
		}
		for _, m := range section.FindAllStringSubmatch(f, -1) {
			out[m[1]] = strings.Join(lines, "\n")
		}
	}
	return out
}

// TestDocOutputsMatchTheirExamples checks that the output a page shows beside a Go snippet is
// the output of the example the snippet comes from. Go checks the example against its own
// // Output: comment; this checks the copy on the page against that, so the page cannot show a
// number the code no longer computes.
func TestDocOutputsMatchTheirExamples(t *testing.T) {
	outputs := exampleOutputs(t)
	snippet := regexp.MustCompile("(?s)--8<-- \"example_test.go:([a-z]+)\"\\n```\\n\\n```text\\n(.*?)\\n```")
	seen := 0
	for path, page := range docPages(t) {
		for _, m := range snippet.FindAllStringSubmatch(page, -1) {
			seen++
			want, ok := outputs[m[1]]
			if !ok {
				t.Errorf("%s shows output for snippet %q, which is in no example with an "+
					"// Output: comment", path, m[1])
				continue
			}
			if m[2] != want {
				t.Errorf("%s shows this output for snippet %q:\n%s\nbut the example prints:\n%s",
					path, m[1], m[2], want)
			}
		}
	}
	if seen == 0 {
		t.Error("no page shows an example's output; the pattern this test matches is stale")
	}
}

// tableCodes returns the code spans in the first column of the table that follows marker on a
// page.
func tableCodes(t *testing.T, path, marker string) []string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	page := string(b)
	i := strings.Index(page, marker)
	if i < 0 {
		t.Fatalf("%s has no %s", path, marker)
	}
	code := regexp.MustCompile("`([^`]+)`")
	var out []string
	started := false
	for _, line := range strings.Split(page[i+len(marker):], "\n") {
		switch {
		case strings.HasPrefix(line, "|"):
			started = true
			first := strings.SplitN(strings.Trim(line, "|"), "|", 2)[0]
			for _, m := range code.FindAllStringSubmatch(first, -1) {
				out = append(out, m[1])
			}
		case started:
			return out
		}
	}
	return out
}

// sourceNames returns the first string argument of every call matching call in the package's
// non-test Go files. A literal followed by + names a prefix completed by a kind, written
// <kind> on the page.
func sourceNames(t *testing.T, call string) []string {
	t.Helper()
	re := regexp.MustCompile(call + `"([a-z_0-9]+)"(\+)?`)
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	set := map[string]bool{}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range re.FindAllStringSubmatch(string(b), -1) {
			name := m[1]
			if m[2] == "+" {
				name += "<kind>"
			}
			set[name] = true
		}
	}
	var out []string
	for n := range set {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

func sameSet(t *testing.T, what string, page, code []string) {
	t.Helper()
	inPage, inCode := map[string]bool{}, map[string]bool{}
	for _, n := range page {
		inPage[n] = true
	}
	for _, n := range code {
		inCode[n] = true
	}
	for n := range inCode {
		if !inPage[n] {
			t.Errorf("the code can record %s %q, and the page does not list it", what, n)
		}
	}
	for n := range inPage {
		if !inCode[n] {
			t.Errorf("the page lists %s %q, which the code no longer records", what, n)
		}
	}
	if len(code) == 0 {
		t.Errorf("found no %s in the code; the pattern this test matches is stale", what)
	}
}

// TestDocAssumptionsMatchTheCode checks the table of assumptions against every name the code
// passes to assume, so the page can neither miss one nor list one that no longer exists.
func TestDocAssumptionsMatchTheCode(t *testing.T) {
	page := tableCodes(t, filepath.Join("docs", "reference", "provenance.md"), "<!-- assumptions -->")
	sameSet(t, "the assumption", page, sourceNames(t, `k\.assume\(`))
}

// TestDocOptionalCoefficientsMatchTheCode does the same for the optional coefficients, the
// names the code reads through optional.
func TestDocOptionalCoefficientsMatchTheCode(t *testing.T) {
	page := tableCodes(t, filepath.Join("docs", "reference", "provenance.md"),
		"### Optional coefficients that were missing")
	sameSet(t, "the optional coefficient", page, sourceNames(t, `k\.optional\(c, `))
}

// TestDocSourceLinksExist checks that every link from the pages into this repository's source
// names a path that exists, so a renamed file cannot leave a page pointing nowhere.
func TestDocSourceLinksExist(t *testing.T) {
	link := regexp.MustCompile(`github\.com/inference-sim/blis-latency-kernel/(?:blob|tree)/main/([^)\s#"]+)`)
	pages := docPages(t)
	if b, err := os.ReadFile(filepath.Join("docs", "generated", "divergences.md")); err == nil {
		pages["docs/generated/divergences.md"] = string(b)
	}
	seen := 0
	for path, page := range pages {
		for _, m := range link.FindAllStringSubmatch(page, -1) {
			seen++
			if _, err := os.Stat(m[1]); err != nil {
				t.Errorf("%s links to %s, which does not exist", path, m[1])
			}
		}
	}
	if seen == 0 {
		t.Error("no page links into the source; the pattern this test matches is stale")
	}
}

//go:build linux && nvlink

package collector_test

import (
	"testing"
)

// NVLink tests require a live DCGM hostengine and GPU hardware.
// They are intentionally omitted from automated unit tests.
// Integration tests should be run on a node with DCGM installed:
//
//	CGO_ENABLED=1 go test -tags nvlink -run TestNVLink ./internal/collector/
func TestNVLinkCollector_SkipWithoutHardware(t *testing.T) {
	t.Skip("NVLink collector requires DCGM hostengine; run on a GPU node with -tags nvlink")
}

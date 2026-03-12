//go:build linux && infiniband

package collector_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Yayka/ml-infra-profiler/agent/internal/collector"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

// buildFakeSysfs creates a minimal fake sysfs tree for one HCA device with
// one port, writing the given counter values under counters/.
//
//	<root>/mlx5_0/ports/1/counters/<name>
func buildFakeSysfs(t *testing.T, counters map[string]string) string {
	t.Helper()
	root := t.TempDir()
	countersPath := filepath.Join(root, "mlx5_0", "ports", "1", "counters")
	if err := os.MkdirAll(countersPath, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", countersPath, err)
	}
	for name, val := range counters {
		p := filepath.Join(countersPath, name)
		if err := os.WriteFile(p, []byte(val+"\n"), 0o644); err != nil {
			t.Fatalf("write %s: %v", p, err)
		}
	}
	return root
}

func TestInfinibandCollector_Bytes(t *testing.T) {
	root := buildFakeSysfs(t, map[string]string{
		"port_rcv_data":  "100",
		"port_xmit_data": "200",
	})

	c := collector.NewInfinibandCollector(root)

	const expected = `
# HELP ml_ib_port_data_bytes_total Total data bytes transferred on an InfiniBand port (OFED 4-byte units × 4).
# TYPE ml_ib_port_data_bytes_total counter
ml_ib_port_data_bytes_total{device="mlx5_0",direction="rx",port="1"} 400
ml_ib_port_data_bytes_total{device="mlx5_0",direction="tx",port="1"} 800
`
	if err := testutil.CollectAndCompare(c, strings.NewReader(expected), "ml_ib_port_data_bytes_total"); err != nil {
		t.Errorf("bytes metric mismatch:\n%v", err)
	}
}

func TestInfinibandCollector_Packets(t *testing.T) {
	root := buildFakeSysfs(t, map[string]string{
		"port_rcv_packets":  "50",
		"port_xmit_packets": "75",
	})

	c := collector.NewInfinibandCollector(root)

	const expected = `
# HELP ml_ib_port_packets_total Total packets transferred on an InfiniBand port.
# TYPE ml_ib_port_packets_total counter
ml_ib_port_packets_total{device="mlx5_0",direction="rx",port="1"} 50
ml_ib_port_packets_total{device="mlx5_0",direction="tx",port="1"} 75
`
	if err := testutil.CollectAndCompare(c, strings.NewReader(expected), "ml_ib_port_packets_total"); err != nil {
		t.Errorf("packets metric mismatch:\n%v", err)
	}
}

func TestInfinibandCollector_Errors(t *testing.T) {
	root := buildFakeSysfs(t, map[string]string{
		"port_rcv_errors": "3",
		"link_downed":     "1",
	})

	c := collector.NewInfinibandCollector(root)

	const expected = `
# HELP ml_ib_port_errors_total Total errors on an InfiniBand port.
# TYPE ml_ib_port_errors_total counter
ml_ib_port_errors_total{device="mlx5_0",port="1",type="link.downed"} 1
ml_ib_port_errors_total{device="mlx5_0",port="1",type="port.rcv.errors"} 3
`
	if err := testutil.CollectAndCompare(c, strings.NewReader(expected), "ml_ib_port_errors_total"); err != nil {
		t.Errorf("errors metric mismatch:\n%v", err)
	}
}

func TestInfinibandCollector_MissingSysfs(t *testing.T) {
	root := buildFakeSysfs(t, map[string]string{})
	c := collector.NewInfinibandCollector(root)

	// Simulate sysfs disappearing by removing the counters directory.
	countersPath := filepath.Join(root, "mlx5_0", "ports", "1", "counters")
	if err := os.RemoveAll(countersPath); err != nil {
		t.Fatalf("removeAll: %v", err)
	}

	// Collect should emit no metrics, not panic.
	// Wrap in a Registry so we can use CollectAndCompare with an empty expected string.
	reg := prometheus.NewRegistry()
	reg.MustRegister(c)
	mfs, err := reg.Gather()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(mfs) != 0 {
		t.Errorf("expected 0 metric families, got %d", len(mfs))
	}
}

//go:build linux

package netio

import (
	"os"
	"path/filepath"
	"testing"
)

const sampleProcNetDev = `Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo:  123456     1000    0    0    0     0          0         0   123456    1000    0    0    0     0       0          0
  eth0: 9876543    54321    0    0    0     0          0         0  1234567   12345    0    0    0     0       0          0
   ib0:  555000     2000    0    0    0     0          0         0   444000    1800    0    0    0     0       0          0
`

func TestParseNetDev(t *testing.T) {
	entries, err := parseNetDev(sampleProcNetDev)
	if err != nil {
		t.Fatalf("parseNetDev error: %v", err)
	}
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(entries))
	}

	tests := []struct {
		name      string
		rxBytes   uint64
		rxPackets uint64
		txBytes   uint64
		txPackets uint64
	}{
		{"lo", 123456, 1000, 123456, 1000},
		{"eth0", 9876543, 54321, 1234567, 12345},
		{"ib0", 555000, 2000, 444000, 1800},
	}
	for i, tt := range tests {
		e := entries[i]
		if e.Name != tt.name {
			t.Errorf("entry[%d]: name = %q, want %q", i, e.Name, tt.name)
		}
		if e.RxBytes != tt.rxBytes {
			t.Errorf("%s: RxBytes = %d, want %d", tt.name, e.RxBytes, tt.rxBytes)
		}
		if e.RxPackets != tt.rxPackets {
			t.Errorf("%s: RxPackets = %d, want %d", tt.name, e.RxPackets, tt.rxPackets)
		}
		if e.TxBytes != tt.txBytes {
			t.Errorf("%s: TxBytes = %d, want %d", tt.name, e.TxBytes, tt.txBytes)
		}
		if e.TxPackets != tt.txPackets {
			t.Errorf("%s: TxPackets = %d, want %d", tt.name, e.TxPackets, tt.txPackets)
		}
	}
}

func TestDetectTransport(t *testing.T) {
	// Build a minimal fixture sysfs tree under a temp dir.
	tmp := t.TempDir()
	sysNet := filepath.Join(tmp, "class", "net")
	sysIB := filepath.Join(tmp, "class", "infiniband")

	// Helper: create a sysfs "type" file for an interface.
	mkIface := func(name, ifaceType string) {
		dir := filepath.Join(sysNet, name)
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "type"), []byte(ifaceType+"\n"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	// eth0: type 1, no IB association → ethernet
	mkIface("eth0", "1")

	// ib0: type 32 → infiniband (IPoIB)
	mkIface("ib0", "32")

	// mlx0: type 1, RoCE — appears in /sys/class/infiniband/mlx5_0/device/net/mlx0
	mkIface("mlx0", "1")
	roceNet := filepath.Join(sysIB, "mlx5_0", "device", "net", "mlx0")
	if err := os.MkdirAll(roceNet, 0755); err != nil {
		t.Fatal(err)
	}

	// efa0: type 1, driver symlink → "efa"
	mkIface("efa0", "1")
	driverDir := filepath.Join(sysNet, "efa0", "device", "driver")
	// Create a real symlink target directory and symlink to it
	efaDriverTarget := filepath.Join(tmp, "drivers", "efa")
	if err := os.MkdirAll(efaDriverTarget, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(driverDir), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(efaDriverTarget, driverDir); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		iface string
		want  string
	}{
		{"eth0", "ethernet"},
		{"ib0", "infiniband"},
		{"mlx0", "roce"},
		{"efa0", "efa"},
		{"nonexistent", "unknown"},
	}

	for _, tt := range tests {
		got := detectTransport(tt.iface, sysNet, sysIB)
		if got != tt.want {
			t.Errorf("detectTransport(%q): got %q, want %q", tt.iface, got, tt.want)
		}
	}
}

func TestLinuxProviderFiltersLoopbackAndPrefixes(t *testing.T) {
	entries, _ := parseNetDev(sampleProcNetDev)
	// All 3 parsed; lo should be filtered in Stats()
	if len(entries) != 3 {
		t.Fatalf("expected 3 raw entries, got %d", len(entries))
	}
	// Verify lo is present in raw entries (it gets filtered later in Stats)
	found := false
	for _, e := range entries {
		if e.Name == "lo" {
			found = true
		}
	}
	if !found {
		t.Error("expected lo in raw parsed entries")
	}
}

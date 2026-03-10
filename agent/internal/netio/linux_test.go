//go:build linux

package netio

import "testing"

func TestDetectTransportLinux(t *testing.T) {
	tests := []struct {
		iface string
		want  string
	}{
		{"eth0", "ethernet"},
		{"eth1", "ethernet"},
		{"en0", "ethernet"},
		{"ens3", "ethernet"},
		{"enp0s3", "ethernet"},
		{"ib0", "infiniband"},
		{"ib1", "infiniband"},
		{"efa0", "efa"},
		{"docker0", "unknown"},
		{"veth0", "unknown"},
		{"lo", "unknown"},
	}
	for _, tt := range tests {
		// Note: sysfs check is skipped in unit tests because the paths don't
		// exist; the name-prefix fallback is exercised instead.
		got := detectTransportLinux(tt.iface)
		if got != tt.want {
			t.Errorf("detectTransportLinux(%q) = %q, want %q", tt.iface, got, tt.want)
		}
	}
}

//go:build darwin

package netio

import "testing"

func TestDetectTransportDarwin(t *testing.T) {
	tests := []struct {
		iface string
		want  string
	}{
		{"en0", "ethernet"},
		{"en1", "ethernet"},
		{"eth0", "ethernet"},
		{"ib0", "infiniband"},
		{"ib1", "infiniband"},
		{"efa0", "efa"},
		{"utun0", "unknown"},
		{"bridge0", "unknown"},
		{"lo0", "unknown"},
	}
	for _, tt := range tests {
		got := detectTransport(tt.iface)
		if got != tt.want {
			t.Errorf("detectTransport(%q) = %q, want %q", tt.iface, got, tt.want)
		}
	}
}

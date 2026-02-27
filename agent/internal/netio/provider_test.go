package netio

import "testing"

func TestHasPrefix(t *testing.T) {
	tests := []struct {
		s        string
		prefixes []string
		want     bool
	}{
		{"eth0", []string{"eth"}, true},
		{"eth0", []string{"ib", "eth"}, true},
		{"ib0", []string{"eth"}, false},
		{"ib0", []string{}, false},
		{"en0", []string{"en", "ib"}, true},
		{"", []string{"eth"}, false},
	}
	for _, tt := range tests {
		got := hasPrefix(tt.s, tt.prefixes)
		if got != tt.want {
			t.Errorf("hasPrefix(%q, %v) = %v, want %v", tt.s, tt.prefixes, got, tt.want)
		}
	}
}

//go:build darwin

package netio

import "testing"

const sampleNetstatIB = `Name  Mtu   Network     Address            Ipkts Ierrs     Ibytes    Opkts Oerrs     Obytes  Coll
lo0   16384 <Link#1>                        1000     0      56789     1000     0      56789     0
lo0   16384 127          127.0.0.1           500     0      12345      500     0      12345     0
en0   1500  <Link#2>    aa:bb:cc:dd:ee:ff   5000     0     500000     4000     0     400000     0
en0   1500  192.168.1    192.168.1.10        800     0      80000      700     0      70000     0
en1   1500  <Link#3>    bb:cc:dd:ee:ff:00   1000     0     100000      900     0      90000     0
`

func TestParseNetstatIB(t *testing.T) {
	stats, err := parseNetstatIB(sampleNetstatIB)
	if err != nil {
		t.Fatalf("parseNetstatIB error: %v", err)
	}
	// Should have lo0, en0, en1 (one entry per interface, Link# rows only)
	if len(stats) != 3 {
		t.Fatalf("expected 3 entries, got %d: %+v", len(stats), stats)
	}

	tests := []struct {
		name      string
		rxBytes   uint64
		txBytes   uint64
		rxPackets uint64
		txPackets uint64
		transport string
	}{
		{"lo0", 56789, 56789, 1000, 1000, "unknown"},
		{"en0", 500000, 400000, 5000, 4000, "ethernet"},
		{"en1", 100000, 90000, 1000, 900, "ethernet"},
	}
	for i, tt := range tests {
		s := stats[i]
		if s.Name != tt.name {
			t.Errorf("[%d] name = %q, want %q", i, s.Name, tt.name)
		}
		if s.RxBytes != tt.rxBytes {
			t.Errorf("%s RxBytes = %d, want %d", tt.name, s.RxBytes, tt.rxBytes)
		}
		if s.TxBytes != tt.txBytes {
			t.Errorf("%s TxBytes = %d, want %d", tt.name, s.TxBytes, tt.txBytes)
		}
		if s.RxPackets != tt.rxPackets {
			t.Errorf("%s RxPackets = %d, want %d", tt.name, s.RxPackets, tt.rxPackets)
		}
		if s.TxPackets != tt.txPackets {
			t.Errorf("%s TxPackets = %d, want %d", tt.name, s.TxPackets, tt.txPackets)
		}
		if s.Transport != tt.transport {
			t.Errorf("%s Transport = %q, want %q", tt.name, s.Transport, tt.transport)
		}
	}
}

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

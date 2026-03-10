//go:build linux

package netio

import (
	"fmt"
	"os"

	gopsnet "github.com/shirou/gopsutil/v3/net"
)

// linuxProvider uses gopsutil to get per-interface byte/packet counters.
// Transport type is inferred from sysfs first, then interface name patterns.
type linuxProvider struct{}

// NewProvider returns the Linux implementation of NetworkStatsProvider.
// Called by main.go; resolved at compile time via the linux build tag.
func NewProvider() NetworkStatsProvider {
	return &linuxProvider{}
}

func (p *linuxProvider) Stats(include []string) ([]InterfaceStat, error) {
	counters, err := gopsnet.IOCounters(true) // true = per-interface
	if err != nil {
		return nil, fmt.Errorf("get interface counters: %w", err)
	}

	var stats []InterfaceStat
	for _, c := range counters {
		if c.Name == "lo" || c.Name == "lo0" {
			continue
		}
		if len(include) > 0 && !hasPrefix(c.Name, include) {
			continue
		}
		stats = append(stats, InterfaceStat{
			Name:      c.Name,
			Transport: detectTransportLinux(c.Name),
			RxBytes:   c.BytesRecv,
			TxBytes:   c.BytesSent,
			RxPackets: c.PacketsRecv,
			TxPackets: c.PacketsSent,
		})
	}
	return stats, nil
}

// detectTransportLinux infers the transport type from sysfs first, then
// falls back to interface name prefix patterns.
//
//   - sysfs /sys/class/net/<iface>/device/infiniband present → "infiniband"
//   - "ib*"              → "infiniband"
//   - "efa*"             → "efa"
//   - "eth*","en*"       → "ethernet"
//   - anything else      → "unknown"
func detectTransportLinux(iface string) string {
	// sysfs check: presence of device/infiniband indicates an IB-capable NIC.
	if _, err := os.Stat("/sys/class/net/" + iface + "/device/infiniband"); err == nil {
		return "infiniband"
	}
	switch {
	case hasPrefix(iface, []string{"ib"}):
		return "infiniband"
	case hasPrefix(iface, []string{"efa"}):
		return "efa"
	case hasPrefix(iface, []string{"eth", "en"}):
		return "ethernet"
	default:
		return "unknown"
	}
}

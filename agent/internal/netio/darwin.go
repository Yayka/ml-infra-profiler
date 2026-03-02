//go:build darwin

package netio

import (
	"fmt"

	gopsnet "github.com/shirou/gopsutil/v3/net"
)

// darwinProvider uses gopsutil to get per-interface byte/packet counters.
// Transport type is inferred from interface name patterns since macOS does
// not expose sysfs.
type darwinProvider struct{}

// NewProvider returns the macOS implementation of NetworkStatsProvider.
// Called by main.go; resolved at compile time via the darwin build tag.
func NewProvider() NetworkStatsProvider {
	return &darwinProvider{}
}

func (p *darwinProvider) Stats(include []string) ([]InterfaceStat, error) {
	counters, err := gopsnet.IOCounters(true) // true = per-interface
	if err != nil {
		return nil, fmt.Errorf("get interface counters: %w", err)
	}

	var stats []InterfaceStat
	for _, c := range counters {
		if c.Name == "lo0" || c.Name == "lo" {
			continue
		}
		if len(include) > 0 && !hasPrefix(c.Name, include) {
			continue
		}
		stats = append(stats, InterfaceStat{
			Name:      c.Name,
			Transport: detectTransport(c.Name),
			RxBytes:   c.BytesRecv,
			TxBytes:   c.BytesSent,
			RxPackets: c.PacketsRecv,
			TxPackets: c.PacketsSent,
		})
	}
	return stats, nil
}

// detectTransport infers the transport type from the interface name on macOS.
// macOS does not expose sysfs, so name patterns are the only signal available.
//
//   - "ib*"        → "infiniband"
//   - "efa*"       → "efa"
//   - "en*","eth*" → "ethernet"
//   - anything else → "unknown"
func detectTransport(iface string) string {
	switch {
	case hasPrefix(iface, []string{"ib"}):
		return "infiniband"
	case hasPrefix(iface, []string{"efa"}):
		return "efa"
	case hasPrefix(iface, []string{"en", "eth"}):
		return "ethernet"
	default:
		return "unknown"
	}
}

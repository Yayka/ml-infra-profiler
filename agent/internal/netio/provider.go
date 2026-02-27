// Package netio provides platform-specific network interface statistics.
// The NetworkStatsProvider interface is implemented separately for Linux
// (linux.go, reads /proc/net/dev + sysfs) and macOS (darwin.go, runs netstat).
// The correct implementation is selected at compile time via build tags.
package netio

// InterfaceStat holds one snapshot of a network interface's counters.
type InterfaceStat struct {
	Name      string // e.g. "eth0", "ib0", "efa0"
	Transport string // "ethernet", "infiniband", "roce", "efa", "unknown"
	RxBytes   uint64
	TxBytes   uint64
	RxPackets uint64
	TxPackets uint64
}

// NetworkStatsProvider reads current per-interface counters from the OS.
// Implementations are platform-specific and selected at compile time.
type NetworkStatsProvider interface {
	// Stats returns counters for all non-loopback interfaces.
	// If includeInterfaces is non-empty, only interfaces whose name starts
	// with one of the given prefixes are returned.
	Stats(includeInterfaces []string) ([]InterfaceStat, error)
}

// hasPrefix reports whether s starts with any prefix in the list.
// Used by both platform implementations.
func hasPrefix(s string, prefixes []string) bool {
	for _, p := range prefixes {
		if len(s) >= len(p) && s[:len(p)] == p {
			return true
		}
	}
	return false
}

//go:build darwin

package netio

import (
	"bufio"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// darwinProvider runs `netstat -ib -n` to get per-interface byte counters.
// Transport type is inferred from interface name patterns since macOS does
// not expose sysfs.
type darwinProvider struct {
	netstatPath string // default: "netstat"
}

// NewProvider returns the macOS implementation of NetworkStatsProvider.
// Called by main.go; resolved at compile time via the darwin build tag.
func NewProvider() NetworkStatsProvider {
	return &darwinProvider{netstatPath: "netstat"}
}

func (p *darwinProvider) Stats(include []string) ([]InterfaceStat, error) {
	out, err := exec.Command(p.netstatPath, "-ib", "-n").Output()
	if err != nil {
		return nil, fmt.Errorf("run netstat: %w", err)
	}

	all, err := parseNetstatIB(string(out))
	if err != nil {
		return nil, err
	}

	var stats []InterfaceStat
	for _, s := range all {
		if s.Name == "lo0" || s.Name == "lo" {
			continue
		}
		if len(include) > 0 && !hasPrefix(s.Name, include) {
			continue
		}
		stats = append(stats, s)
	}
	return stats, nil
}

// parseNetstatIB parses the output of `netstat -ib -n`.
// Only rows containing "<Link#N>" are used — these carry hardware-level
// byte counters rather than per-IP-address rows.
//
// The Address column is optional: loopback and some interfaces omit it,
// shifting all subsequent columns left by one. We detect this by checking
// whether field[3] is numeric (no address) or not (address present).
//
// With address (11 fields):
//
//	name mtu <Link#N> addr  ipkts ierrs ibytes opkts oerrs obytes coll
//	  0    1     2      3     4     5      6     7     8      9    10
//
// Without address (10 fields):
//
//	name mtu <Link#N> ipkts ierrs ibytes opkts oerrs obytes coll
//	  0    1     2      3     4      5     6     7      8     9
//
// Package-level so tests can call it with fixture strings.
func parseNetstatIB(output string) ([]InterfaceStat, error) {
	var stats []InterfaceStat
	seen := make(map[string]bool)

	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := scanner.Text()
		// Only process Link-level rows
		if !strings.Contains(line, "<Link#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 10 {
			continue
		}
		name := fields[0]
		if seen[name] {
			continue
		}
		seen[name] = true

		// Determine column offset based on presence of Address field.
		// If fields[3] parses as a number, there is no address column.
		ipktsIdx := 4
		ibytesIdx := 6
		opktsIdx := 7
		obytesIdx := 9
		if _, err := strconv.ParseUint(fields[3], 10, 64); err == nil {
			// No address column — shift indices left by 1
			ipktsIdx = 3
			ibytesIdx = 5
			opktsIdx = 6
			obytesIdx = 8
		}
		if len(fields) <= obytesIdx {
			continue
		}

		rxPackets, err := strconv.ParseUint(fields[ipktsIdx], 10, 64)
		if err != nil {
			continue
		}
		rxBytes, err := strconv.ParseUint(fields[ibytesIdx], 10, 64)
		if err != nil {
			continue
		}
		txPackets, err := strconv.ParseUint(fields[opktsIdx], 10, 64)
		if err != nil {
			continue
		}
		txBytes, err := strconv.ParseUint(fields[obytesIdx], 10, 64)
		if err != nil {
			continue
		}

		stats = append(stats, InterfaceStat{
			Name:      name,
			Transport: detectTransport(name),
			RxBytes:   rxBytes,
			TxBytes:   txBytes,
			RxPackets: rxPackets,
			TxPackets: txPackets,
		})
	}
	return stats, scanner.Err()
}

// detectTransport infers the transport type from the interface name on macOS.
// macOS does not expose sysfs, so name patterns are the only signal available.
//
//   - "ib*"       → "infiniband"
//   - "efa*"      → "efa"
//   - "en*","eth*"→ "ethernet"
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

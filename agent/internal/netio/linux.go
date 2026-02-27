//go:build linux

package netio

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// linuxProvider reads /proc/net/dev for interface counters and
// /sys/class/net/ + /sys/class/infiniband/ for transport type detection.
type linuxProvider struct {
	procNetDev  string // default: "/proc/net/dev"
	sysClassNet string // default: "/sys/class/net"
	sysClassIB  string // default: "/sys/class/infiniband"
}

// NewProvider returns the Linux implementation of NetworkStatsProvider.
// Called by main.go; resolved at compile time via the linux build tag.
func NewProvider() NetworkStatsProvider {
	return &linuxProvider{
		procNetDev:  "/proc/net/dev",
		sysClassNet: "/sys/class/net",
		sysClassIB:  "/sys/class/infiniband",
	}
}

func (p *linuxProvider) Stats(include []string) ([]InterfaceStat, error) {
	data, err := os.ReadFile(p.procNetDev)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", p.procNetDev, err)
	}

	entries, err := parseNetDev(string(data))
	if err != nil {
		return nil, err
	}

	var stats []InterfaceStat
	for _, e := range entries {
		if e.Name == "lo" {
			continue
		}
		if len(include) > 0 && !hasPrefix(e.Name, include) {
			continue
		}
		stats = append(stats, InterfaceStat{
			Name:      e.Name,
			Transport: detectTransport(e.Name, p.sysClassNet, p.sysClassIB),
			RxBytes:   e.RxBytes,
			TxBytes:   e.TxBytes,
			RxPackets: e.RxPackets,
			TxPackets: e.TxPackets,
		})
	}
	return stats, nil
}

// procNetDevEntry holds the raw columns parsed from /proc/net/dev.
type procNetDevEntry struct {
	Name      string
	RxBytes   uint64
	RxPackets uint64
	TxBytes   uint64
	TxPackets uint64
}

// parseNetDev parses /proc/net/dev content.
// Format (after two header lines):
//
//	  iface: rx_bytes rx_packets rx_errs rx_drop rx_fifo rx_frame rx_compressed rx_multicast \
//	         tx_bytes tx_packets tx_errs tx_drop ...
//
// Package-level (not a method) so tests can call it directly with fixture strings.
func parseNetDev(content string) ([]procNetDevEntry, error) {
	var entries []procNetDevEntry
	scanner := bufio.NewScanner(strings.NewReader(content))
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		if lineNum <= 2 {
			continue // skip two header lines
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		colonIdx := strings.Index(line, ":")
		if colonIdx < 0 {
			continue
		}
		name := strings.TrimSpace(line[:colonIdx])
		fields := strings.Fields(line[colonIdx+1:])
		if len(fields) < 10 {
			continue
		}
		rxBytes, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil {
			continue
		}
		rxPackets, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			continue
		}
		txBytes, err := strconv.ParseUint(fields[8], 10, 64)
		if err != nil {
			continue
		}
		txPackets, err := strconv.ParseUint(fields[9], 10, 64)
		if err != nil {
			continue
		}
		entries = append(entries, procNetDevEntry{
			Name:      name,
			RxBytes:   rxBytes,
			RxPackets: rxPackets,
			TxBytes:   txBytes,
			TxPackets: txPackets,
		})
	}
	return entries, scanner.Err()
}

// detectTransport resolves the network transport type for iface using sysfs.
//
// Detection order:
//  1. /sys/class/net/<iface>/type == "32" → "infiniband" (IPoIB interface)
//  2. type "1" (Ethernet) + any /sys/class/infiniband/*/device/net/<iface> exists → "roce"
//  3. type "1" + /sys/class/net/<iface>/device/driver symlink base == "efa" → "efa"
//  4. type "1" → "ethernet"
//  5. read error → "unknown"
func detectTransport(iface, sysClassNet, sysClassIB string) string {
	typeFile := filepath.Join(sysClassNet, iface, "type")
	typeData, err := os.ReadFile(typeFile)
	if err != nil {
		return "unknown"
	}
	ifaceType := strings.TrimSpace(string(typeData))

	switch ifaceType {
	case "32":
		return "infiniband"
	case "1":
		// Check RoCE: does any IB device have a net/<iface> entry?
		if isRoCE(iface, sysClassIB) {
			return "roce"
		}
		// Check EFA driver
		driverLink := filepath.Join(sysClassNet, iface, "device", "driver")
		if target, err := os.Readlink(driverLink); err == nil {
			if filepath.Base(target) == "efa" {
				return "efa"
			}
		}
		return "ethernet"
	default:
		return "unknown"
	}
}

// isRoCE checks whether any InfiniBand device in sysClassIB advertises iface
// under its device/net/ directory, which indicates a RoCE-capable NIC.
func isRoCE(iface, sysClassIB string) bool {
	ibDevs, err := os.ReadDir(sysClassIB)
	if err != nil {
		return false
	}
	for _, dev := range ibDevs {
		netPath := filepath.Join(sysClassIB, dev.Name(), "device", "net", iface)
		if _, err := os.Stat(netPath); err == nil {
			return true
		}
	}
	return false
}

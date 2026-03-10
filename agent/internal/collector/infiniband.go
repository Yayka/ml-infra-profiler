//go:build linux && infiniband

package collector

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
)

// InfinibandCollector scrapes per-port counters from the InfiniBand sysfs tree.
// It reads /sys/class/infiniband/<hca>/ports/<port>/counters/.
//
// All metrics are Prometheus Counters (cumulative, monotonically increasing).
// OFED encodes port_*_data values in 4-byte units; this collector multiplies by 4.
type InfinibandCollector struct {
	sysfsPath  string
	bytesDesc  *prometheus.Desc
	pktsDesc   *prometheus.Desc
	errorsDesc *prometheus.Desc
}

// NewInfinibandCollector creates an InfinibandCollector rooted at sysfsPath
// (normally /sys/class/infiniband). If the path does not exist at startup, the
// agent exits loudly — a missing sysfs path means the IB driver is not loaded.
func NewInfinibandCollector(sysfsPath string) *InfinibandCollector {
	if _, err := os.Stat(sysfsPath); err != nil {
		log.Fatalf("infiniband: sysfs path %q not found; is the IB driver loaded? (%v)", sysfsPath, err)
	}
	return &InfinibandCollector{
		sysfsPath: sysfsPath,
		bytesDesc: prometheus.NewDesc(
			"ml_ib_port_data_bytes_total",
			"Total data bytes transferred on an InfiniBand port (OFED 4-byte units × 4).",
			[]string{"device", "port", "direction"}, nil,
		),
		pktsDesc: prometheus.NewDesc(
			"ml_ib_port_packets_total",
			"Total packets transferred on an InfiniBand port.",
			[]string{"device", "port", "direction"}, nil,
		),
		errorsDesc: prometheus.NewDesc(
			"ml_ib_port_errors_total",
			"Total errors on an InfiniBand port.",
			[]string{"device", "port", "type"}, nil,
		),
	}
}

func (c *InfinibandCollector) Name() string { return "infiniband" }

func (c *InfinibandCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.bytesDesc
	ch <- c.pktsDesc
	ch <- c.errorsDesc
}

func (c *InfinibandCollector) Collect(ch chan<- prometheus.Metric) {
	// Walk HCA devices: /sys/class/infiniband/<device>/
	devices, err := os.ReadDir(c.sysfsPath)
	if err != nil {
		// sysfs disappeared mid-run — emit nothing for this scrape.
		return
	}

	for _, dev := range devices {
		if !dev.IsDir() {
			continue
		}
		device := dev.Name()
		portsPath := filepath.Join(c.sysfsPath, device, "ports")

		ports, err := os.ReadDir(portsPath)
		if err != nil {
			continue
		}

		for _, p := range ports {
			if !p.IsDir() {
				continue
			}
			port := p.Name()
			countersPath := filepath.Join(portsPath, port, "counters")

			c.collectCounters(ch, device, port, countersPath)
		}
	}
}

// collectCounters reads all counter files in countersPath and emits metrics.
func (c *InfinibandCollector) collectCounters(ch chan<- prometheus.Metric, device, port, countersPath string) {
	entries, err := os.ReadDir(countersPath)
	if err != nil {
		return
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		val, err := readUint64File(filepath.Join(countersPath, name))
		if err != nil {
			continue
		}

		switch name {
		case "port_rcv_data":
			ch <- prometheus.MustNewConstMetric(c.bytesDesc, prometheus.CounterValue,
				float64(val*4), device, port, "rx")
		case "port_xmit_data":
			ch <- prometheus.MustNewConstMetric(c.bytesDesc, prometheus.CounterValue,
				float64(val*4), device, port, "tx")
		case "port_rcv_packets":
			ch <- prometheus.MustNewConstMetric(c.pktsDesc, prometheus.CounterValue,
				float64(val), device, port, "rx")
		case "port_xmit_packets":
			ch <- prometheus.MustNewConstMetric(c.pktsDesc, prometheus.CounterValue,
				float64(val), device, port, "tx")
		case "port_rcv_errors", "port_xmit_discards", "symbol_error",
			"link_error_recovery", "link_downed":
			errType := strings.ReplaceAll(name, "_", ".")
			ch <- prometheus.MustNewConstMetric(c.errorsDesc, prometheus.CounterValue,
				float64(val), device, port, errType)
		}
	}
}

// readUint64File reads a sysfs counter file and parses its trimmed content as uint64.
func readUint64File(path string) (uint64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, fmt.Errorf("read %s: %w", path, err)
	}
	v, err := strconv.ParseUint(strings.TrimSpace(string(raw)), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parse %s: %w", path, err)
	}
	return v, nil
}

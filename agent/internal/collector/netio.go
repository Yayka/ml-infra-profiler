package collector

import (
	"github.com/Yayka/ml-infra-profiler/agent/internal/netio"
	"github.com/prometheus/client_golang/prometheus"
)

// NetIOCollector collects per-interface network byte and packet counters and
// exposes them as Prometheus counters labeled by interface name, transport
// technology (ethernet/infiniband/roce/efa), and direction (rx/tx).
type NetIOCollector struct {
	provider    netio.NetworkStatsProvider
	include     []string
	bytesDesc   *prometheus.Desc
	packetsDesc *prometheus.Desc
}

// NewNetIOCollector creates a NetIOCollector using the given provider.
// includeInterfaces is forwarded to provider.Stats(); empty means all non-loopback.
func NewNetIOCollector(provider netio.NetworkStatsProvider, includeInterfaces []string) *NetIOCollector {
	labels := []string{"interface", "transport", "direction"}
	return &NetIOCollector{
		provider: provider,
		include:  includeInterfaces,
		bytesDesc: prometheus.NewDesc(
			"ml_net_interface_bytes_total",
			"Total bytes transferred on a network interface, labeled by transport technology and direction.",
			labels, nil,
		),
		packetsDesc: prometheus.NewDesc(
			"ml_net_interface_packets_total",
			"Total packets transferred on a network interface, labeled by transport technology and direction.",
			labels, nil,
		),
	}
}

func (c *NetIOCollector) Name() string { return "netio" }

func (c *NetIOCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.bytesDesc
	ch <- c.packetsDesc
}

func (c *NetIOCollector) Collect(ch chan<- prometheus.Metric) {
	stats, err := c.provider.Stats(c.include)
	if err != nil {
		// Emit nothing on error; the Prometheus scrape will show staleness.
		return
	}
	for _, s := range stats {
		ch <- prometheus.MustNewConstMetric(
			c.bytesDesc, prometheus.CounterValue,
			float64(s.RxBytes), s.Name, s.Transport, "rx",
		)
		ch <- prometheus.MustNewConstMetric(
			c.bytesDesc, prometheus.CounterValue,
			float64(s.TxBytes), s.Name, s.Transport, "tx",
		)
		ch <- prometheus.MustNewConstMetric(
			c.packetsDesc, prometheus.CounterValue,
			float64(s.RxPackets), s.Name, s.Transport, "rx",
		)
		ch <- prometheus.MustNewConstMetric(
			c.packetsDesc, prometheus.CounterValue,
			float64(s.TxPackets), s.Name, s.Transport, "tx",
		)
	}
}

package collector_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/Yayka/ml-infra-profiler/agent/internal/collector"
	"github.com/Yayka/ml-infra-profiler/agent/internal/netio"
	"github.com/Yayka/ml-infra-profiler/agent/internal/netio/netiotest"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestNetIOCollector(t *testing.T) {
	fake := &netiotest.FakeProvider{
		Results: []netio.InterfaceStat{
			{
				Name:      "eth0",
				Transport: "ethernet",
				RxBytes:   1000,
				TxBytes:   2000,
				RxPackets: 10,
				TxPackets: 20,
			},
			{
				Name:      "ib0",
				Transport: "infiniband",
				RxBytes:   3000,
				TxBytes:   4000,
				RxPackets: 30,
				TxPackets: 40,
			},
		},
	}

	c := collector.NewNetIOCollector(fake, nil)

	// Validate byte counter values.
	const expectedBytes = `
# HELP ml_net_interface_bytes_total Total bytes transferred on a network interface, labeled by transport technology and direction.
# TYPE ml_net_interface_bytes_total counter
ml_net_interface_bytes_total{direction="rx",interface="eth0",transport="ethernet"} 1000
ml_net_interface_bytes_total{direction="rx",interface="ib0",transport="infiniband"} 3000
ml_net_interface_bytes_total{direction="tx",interface="eth0",transport="ethernet"} 2000
ml_net_interface_bytes_total{direction="tx",interface="ib0",transport="infiniband"} 4000
`
	if err := testutil.CollectAndCompare(c, strings.NewReader(expectedBytes), "ml_net_interface_bytes_total"); err != nil {
		t.Errorf("bytes metric mismatch:\n%v", err)
	}

	// Validate packet counter values.
	const expectedPackets = `
# HELP ml_net_interface_packets_total Total packets transferred on a network interface, labeled by transport technology and direction.
# TYPE ml_net_interface_packets_total counter
ml_net_interface_packets_total{direction="rx",interface="eth0",transport="ethernet"} 10
ml_net_interface_packets_total{direction="rx",interface="ib0",transport="infiniband"} 30
ml_net_interface_packets_total{direction="tx",interface="eth0",transport="ethernet"} 20
ml_net_interface_packets_total{direction="tx",interface="ib0",transport="infiniband"} 40
`
	if err := testutil.CollectAndCompare(c, strings.NewReader(expectedPackets), "ml_net_interface_packets_total"); err != nil {
		t.Errorf("packets metric mismatch:\n%v", err)
	}
}

func TestNetIOCollectorProviderError(t *testing.T) {
	fake := &netiotest.FakeProvider{Err: fmt.Errorf("simulated error")}
	c := collector.NewNetIOCollector(fake, nil)

	// On provider error, Collect emits no metric values — no panic, empty series.
	if err := testutil.CollectAndCompare(c, strings.NewReader(""), "ml_net_interface_bytes_total"); err != nil {
		t.Errorf("expected no metrics on error: %v", err)
	}
}

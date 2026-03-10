//go:build linux && nvlink

package collector

import (
	"fmt"
	"log"

	"github.com/NVIDIA/go-dcgm/pkg/dcgm"
	"github.com/prometheus/client_golang/prometheus"
)

// nvlinkFields lists the DCGM field IDs for per-link NVLink TX and RX bandwidth.
// These are cumulative byte counters (proper Prometheus Counters).
// Links 0–11 cover the full complement on A100 SXM.
var nvlinkTXFields = []dcgm.Short{
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L0,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L1,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L2,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L3,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L4,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L5,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L6,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L7,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L8,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L9,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L10,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_TX_L11,
}

var nvlinkRXFields = []dcgm.Short{
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L0,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L1,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L2,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L3,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L4,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L5,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L6,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L7,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L8,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L9,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L10,
	dcgm.DCGM_FI_DEV_NVLINK_BANDWIDTH_RX_L11,
}

// NVLinkCollector collects per-GPU per-link NVLink bandwidth counters via DCGM.
// Requires CGO_ENABLED=1 and dcgm-hostengine (or embedded DCGM) at runtime.
type NVLinkCollector struct {
	hostengine string
	cleanup    func()
	bytesDesc  *prometheus.Desc
}

// NewNVLinkCollector initialises DCGM and returns an NVLinkCollector.
// If DCGM initialisation fails the agent exits loudly — a failure here means
// the DCGM hostengine is absent or misconfigured.
func NewNVLinkCollector(hostengine string) *NVLinkCollector {
	var (
		cleanup func()
		err     error
	)
	if hostengine == "" {
		cleanup, err = dcgm.Init(dcgm.Embedded)
	} else {
		cleanup, err = dcgm.Init(dcgm.Standalone, hostengine)
	}
	if err != nil {
		log.Fatalf("nvlink: DCGM init failed: %v; is dcgm-hostengine running?", err)
	}

	return &NVLinkCollector{
		hostengine: hostengine,
		cleanup:    cleanup,
		bytesDesc: prometheus.NewDesc(
			"ml_nvlink_bytes_total",
			"Total bytes transferred over an NVLink link per GPU (cumulative DCGM counter).",
			[]string{"gpu_index", "link_index", "direction"}, nil,
		),
	}
}

// Close releases the DCGM connection. Call from main() via defer.
func (c *NVLinkCollector) Close() {
	if c.cleanup != nil {
		c.cleanup()
	}
}

func (c *NVLinkCollector) Name() string { return "nvlink" }

func (c *NVLinkCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.bytesDesc
}

func (c *NVLinkCollector) Collect(ch chan<- prometheus.Metric) {
	gpus, err := dcgm.GetAllDevices()
	if err != nil {
		// DCGM temporarily unavailable — emit nothing for this scrape.
		return
	}

	for _, gpuIdx := range gpus {
		gpuLabel := fmt.Sprintf("%d", gpuIdx)
		c.collectDirection(ch, gpuIdx, gpuLabel, nvlinkTXFields, "tx")
		c.collectDirection(ch, gpuIdx, gpuLabel, nvlinkRXFields, "rx")
	}
}

func (c *NVLinkCollector) collectDirection(
	ch chan<- prometheus.Metric,
	gpuIdx uint,
	gpuLabel string,
	fields []dcgm.Short,
	direction string,
) {
	values, err := dcgm.GetLatestValuesForFields(gpuIdx, fields)
	if err != nil {
		return
	}
	for linkIdx, v := range values {
		if v.Status != dcgm.ST_OK {
			continue
		}
		bytes, ok := v.Value.(int64)
		if !ok {
			continue
		}
		linkLabel := fmt.Sprintf("%d", linkIdx)
		ch <- prometheus.MustNewConstMetric(
			c.bytesDesc, prometheus.CounterValue,
			float64(bytes), gpuLabel, linkLabel, direction,
		)
	}
}

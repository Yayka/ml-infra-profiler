//go:build linux && nvlink

package collector

import (
	"fmt"
	"log"
	"time"

	"github.com/NVIDIA/go-dcgm/pkg/dcgm"
	"github.com/prometheus/client_golang/prometheus"
)

var pcieFields = []dcgm.Short{
	dcgm.DCGM_FI_PROF_PCIE_TX_BYTES,
	dcgm.DCGM_FI_PROF_PCIE_RX_BYTES,
}

var pcieFieldDirection = map[dcgm.Short]string{
	dcgm.DCGM_FI_PROF_PCIE_TX_BYTES: "tx",
	dcgm.DCGM_FI_PROF_PCIE_RX_BYTES: "rx",
}

// PCIeCollector collects per-GPU PCIe bandwidth via DCGM profiling fields
// (DCGM_FI_PROF_PCIE_TX/RX_BYTES, field IDs 1009/1010).
// Metrics are Gauges (bytes/s current rate). Shares the DCGM session
// initialised by NVLinkCollector — do not call dcgm.Init again.
type PCIeCollector struct {
	gpuGroup   dcgm.GroupHandle
	fieldGroup dcgm.FieldHandle
	bwDesc     *prometheus.Desc
}

// NewPCIeCollector creates a PCIeCollector using an already-initialised DCGM
// session. Must be called after dcgm.Init() (i.e. after NewNVLinkCollector).
func NewPCIeCollector() *PCIeCollector {
	fieldGroup, err := dcgm.FieldGroupCreate("ml_pcie_bw", pcieFields)
	if err != nil {
		log.Fatalf("pcie: FieldGroupCreate failed: %v", err)
	}

	gpuGroup := dcgm.GroupAllGPUs()

	if err := dcgm.WatchFieldsWithGroup(fieldGroup, gpuGroup); err != nil {
		log.Fatalf("pcie: WatchFieldsWithGroup failed: %v", err)
	}

	return &PCIeCollector{
		gpuGroup:   gpuGroup,
		fieldGroup: fieldGroup,
		bwDesc: prometheus.NewDesc(
			"ml_pcie_bandwidth_bytes_per_sec",
			"Current PCIe bandwidth in bytes/s per GPU (DCGM profiling field).",
			[]string{"gpu_index", "direction"}, nil,
		),
	}
}

func (c *PCIeCollector) Name() string { return "pcie" }

func (c *PCIeCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.bwDesc
}

func (c *PCIeCollector) Collect(ch chan<- prometheus.Metric) {
	values, _, err := dcgm.GetValuesSince(c.gpuGroup, c.fieldGroup, time.Time{})
	if err != nil {
		return
	}

	type valueKey struct {
		entityID uint
		fieldID  dcgm.Short
	}
	latest := make(map[valueKey]dcgm.FieldValue_v2, len(values))
	for _, v := range values {
		if v.Status != 0 {
			continue
		}
		k := valueKey{v.EntityID, v.FieldID}
		if existing, ok := latest[k]; !ok || v.TS > existing.TS {
			latest[k] = v
		}
	}

	for k, v := range latest {
		direction, ok := pcieFieldDirection[k.fieldID]
		if !ok {
			continue
		}
		ch <- prometheus.MustNewConstMetric(
			c.bwDesc, prometheus.GaugeValue,
			float64(v.Int64()),
			fmt.Sprintf("%d", k.entityID),
			direction,
		)
	}
}

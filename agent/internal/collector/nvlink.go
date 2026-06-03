//go:build linux && nvlink

package collector

import (
	"fmt"
	"log"
	"time"

	"github.com/NVIDIA/go-dcgm/pkg/dcgm"
	"github.com/prometheus/client_golang/prometheus"
)

// fieldInfo maps a DCGM field ID to a human-readable link index and direction.
type fieldInfo struct {
	linkIdx   int
	direction string
}

// nvlinkAllFields is the combined TX+RX field list passed to FieldGroupCreate.
// A100 SXM has 12 NVLink links (L0–L11).
var nvlinkAllFields []dcgm.Short

// nvlinkFieldMeta maps each field ID to its link index and direction label.
var nvlinkFieldMeta map[dcgm.Short]fieldInfo

func init() {
	txFields := []dcgm.Short{
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L0,
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L1,
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L2,
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L3,
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L4,
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L5,
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L6,
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L7,
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L8,
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L9,
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L10,
		dcgm.DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L11,
	}
	rxFields := []dcgm.Short{
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L0,
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L1,
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L2,
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L3,
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L4,
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L5,
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L6,
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L7,
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L8,
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L9,
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L10,
		dcgm.DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L11,
	}

	nvlinkFieldMeta = make(map[dcgm.Short]fieldInfo, len(txFields)+len(rxFields))
	for i, f := range txFields {
		nvlinkAllFields = append(nvlinkAllFields, f)
		nvlinkFieldMeta[f] = fieldInfo{linkIdx: i, direction: "tx"}
	}
	for i, f := range rxFields {
		nvlinkAllFields = append(nvlinkAllFields, f)
		nvlinkFieldMeta[f] = fieldInfo{linkIdx: i, direction: "rx"}
	}
}

// NVLinkCollector collects per-GPU per-link NVLink bandwidth via DCGM.
// NewNVLinkCollector returns nil if the GPU has no NVLink (e.g. PCIe A100).
// Metrics are Gauges (current KB/s) — DCGM bandwidth fields are rates, not
// cumulative counters. Requires CGO_ENABLED=1 and libdcgm.so.4 at runtime.
type NVLinkCollector struct {
	cleanup    func()
	gpuGroup   dcgm.GroupHandle
	fieldGroup dcgm.FieldHandle
	bwDesc     *prometheus.Desc
}

// NewNVLinkCollector initialises DCGM and registers NVLink bandwidth watches.
// If DCGM init fails the agent exits loudly — missing DCGM means the binary
// variant was wrong for this node.
//
// hostengine: empty → embedded mode; "host:port" → standalone TCP connection.
func NewNVLinkCollector(hostengine string) *NVLinkCollector {
	var (
		cleanup func()
		err     error
	)
	if hostengine == "" {
		cleanup, err = dcgm.Init(dcgm.Embedded)
	} else {
		// Standalone TCP: args are address and "0" (0 = not a unix socket).
		cleanup, err = dcgm.Init(dcgm.Standalone, hostengine, "0")
	}
	if err != nil {
		log.Fatalf("nvlink: DCGM init failed: %v; is dcgm-hostengine running?", err)
	}

	fieldGroup, err := dcgm.FieldGroupCreate("ml_nvlink_bw", nvlinkAllFields)
	if err != nil {
		cleanup()
		log.Printf("nvlink: FieldGroupCreate failed: %v — NVLink not available on this GPU, skipping NVLink metrics", err)
		return nil
	}

	gpuGroup := dcgm.GroupAllGPUs()

	if err := dcgm.WatchFieldsWithGroup(fieldGroup, gpuGroup); err != nil {
		cleanup()
		log.Fatalf("nvlink: WatchFieldsWithGroup failed: %v", err)
	}

	return &NVLinkCollector{
		cleanup:    cleanup,
		gpuGroup:   gpuGroup,
		fieldGroup: fieldGroup,
		bwDesc: prometheus.NewDesc(
			"ml_nvlink_bandwidth_kbps",
			"Current NVLink bandwidth in KB/s per GPU per link (DCGM rate metric).",
			[]string{"gpu_index", "link_index", "direction"}, nil,
		),
	}
}

// Close releases the DCGM connection. Defer this in main() after construction.
func (c *NVLinkCollector) Close() {
	if c.cleanup != nil {
		c.cleanup()
	}
}

func (c *NVLinkCollector) Name() string { return "nvlink" }

func (c *NVLinkCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.bwDesc
}

func (c *NVLinkCollector) Collect(ch chan<- prometheus.Metric) {
	// time.Time{} requests all samples in DCGM's buffer; we then keep only
	// the latest per (EntityID, FieldID) to emit a single current value.
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
		meta, ok := nvlinkFieldMeta[k.fieldID]
		if !ok {
			continue
		}
		ch <- prometheus.MustNewConstMetric(
			c.bwDesc, prometheus.GaugeValue,
			float64(v.Int64()),
			fmt.Sprintf("%d", k.entityID),
			fmt.Sprintf("%d", meta.linkIdx),
			meta.direction,
		)
	}
}

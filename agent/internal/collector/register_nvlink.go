//go:build linux && nvlink && !infiniband

package collector

import (
	"github.com/Yayka/ml-infra-profiler/agent/internal/config"
	"github.com/Yayka/ml-infra-profiler/agent/internal/netio"
	"github.com/prometheus/client_golang/prometheus"
)

// Register adds all active collectors to reg for a Linux + NVLink build.
// Fails loudly at startup if DCGM is unavailable.
func Register(reg *prometheus.Registry, cfg *config.Config, provider netio.NetworkStatsProvider) {
	reg.MustRegister(NewNetIOCollector(provider, cfg.Network.IncludeInterfaces))
	if c := NewNVLinkCollector(cfg.NVLink.DCGMHostengine); c != nil {
		reg.MustRegister(c)
	}
	reg.MustRegister(NewPCIeCollector())
}

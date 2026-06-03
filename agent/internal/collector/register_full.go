//go:build linux && infiniband && nvlink

package collector

import (
	"github.com/Yayka/ml-infra-profiler/agent/internal/config"
	"github.com/Yayka/ml-infra-profiler/agent/internal/netio"
	"github.com/prometheus/client_golang/prometheus"
)

// Register adds all active collectors to reg for a full Linux build
// (Ethernet + InfiniBand + NVLink). Fails loudly if either hardware
// subsystem is absent at startup.
func Register(reg *prometheus.Registry, cfg *config.Config, provider netio.NetworkStatsProvider) {
	reg.MustRegister(
		NewNetIOCollector(provider, cfg.Network.IncludeInterfaces),
		NewInfinibandCollector(cfg.Infiniband.SysfsPath),
	)
	if c := NewNVLinkCollector(cfg.NVLink.DCGMHostengine); c != nil {
		reg.MustRegister(c)
	}
	reg.MustRegister(NewPCIeCollector())
}

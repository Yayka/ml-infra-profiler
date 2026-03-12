//go:build linux && infiniband && !nvlink

package collector

import (
	"github.com/Yayka/ml-infra-profiler/agent/internal/config"
	"github.com/Yayka/ml-infra-profiler/agent/internal/netio"
	"github.com/prometheus/client_golang/prometheus"
)

// Register adds all active collectors to reg for a Linux + InfiniBand build.
// Fails loudly at startup if the IB sysfs path is absent.
func Register(reg *prometheus.Registry, cfg *config.Config, provider netio.NetworkStatsProvider) {
	reg.MustRegister(
		NewNetIOCollector(provider, cfg.Network.IncludeInterfaces),
		NewInfinibandCollector(cfg.Infiniband.SysfsPath),
	)
}

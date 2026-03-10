//go:build darwin

package collector

import (
	"github.com/Yayka/ml-infra-profiler/agent/internal/config"
	"github.com/Yayka/ml-infra-profiler/agent/internal/netio"
	"github.com/prometheus/client_golang/prometheus"
)

// Register adds all active collectors to reg for macOS (Ethernet only).
func Register(reg *prometheus.Registry, cfg *config.Config, provider netio.NetworkStatsProvider) {
	reg.MustRegister(
		NewNetIOCollector(provider, cfg.Network.IncludeInterfaces),
	)
}

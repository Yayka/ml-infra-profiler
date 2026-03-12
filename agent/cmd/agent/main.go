// Command agent runs the ml-netprof monitoring agent.
// It exposes a Prometheus /metrics endpoint with network interface counters
// collected via a platform-specific implementation selected at compile time.
package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/Yayka/ml-infra-profiler/agent/internal/collector"
	"github.com/Yayka/ml-infra-profiler/agent/internal/config"
	"github.com/Yayka/ml-infra-profiler/agent/internal/netio"
	"github.com/prometheus/client_golang/prometheus"
	promcollectors "github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	cfgPath := flag.String("config", "configs/agent_default.yaml", "path to agent config file")
	flag.Parse()

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	// netio.NewProvider() is defined in internal/netio/linux.go or darwin.go,
	// resolved at compile time by the build tag for the target OS.
	provider := netio.NewProvider()

	reg := prometheus.NewRegistry()
	reg.MustRegister(promcollectors.NewGoCollector())
	// collector.Register is defined in one of register_*.go, selected at
	// compile time by the combination of OS and feature build tags.
	collector.Register(reg, cfg, provider)

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{
		EnableOpenMetrics: true,
	}))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	log.Printf("ml-netprof agent listening on %s", cfg.Agent.ListenAddress)
	if err := http.ListenAndServe(cfg.Agent.ListenAddress, mux); err != nil {
		log.Fatalf("HTTP server: %v", err)
	}
}

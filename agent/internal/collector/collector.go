// Package collector defines the Collector interface and concrete Prometheus
// collectors for the ml-netprof monitoring agent.
package collector

import "github.com/prometheus/client_golang/prometheus"

// Collector extends prometheus.Collector with a Name() method for logging.
type Collector interface {
	prometheus.Collector
	Name() string
}

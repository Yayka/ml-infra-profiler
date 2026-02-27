// Package config loads and validates the agent's YAML configuration.
package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config holds all agent configuration.
type Config struct {
	Agent   AgentConfig   `yaml:"agent"`
	Network NetworkConfig `yaml:"network"`
}

// AgentConfig controls the HTTP listener.
type AgentConfig struct {
	ListenAddress string `yaml:"listen_address"`
}

// NetworkConfig controls which network interfaces are collected.
type NetworkConfig struct {
	// IncludeInterfaces is a list of interface name prefixes to include.
	// An empty list means all non-loopback interfaces are included.
	IncludeInterfaces []string `yaml:"include_interfaces"`
}

// DefaultConfig returns the config with default values applied.
func DefaultConfig() *Config {
	return &Config{
		Agent: AgentConfig{
			ListenAddress: ":9100",
		},
	}
}

// Load reads a YAML file at path and returns a Config with defaults applied
// for any fields not present in the file.
func Load(path string) (*Config, error) {
	cfg := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %q: %w", path, err)
	}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parse config %q: %w", path, err)
	}

	// Re-apply defaults for zero-value fields after unmarshal.
	if cfg.Agent.ListenAddress == "" {
		cfg.Agent.ListenAddress = ":9100"
	}

	return cfg, nil
}

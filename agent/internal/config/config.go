// Package config loads and validates the agent's YAML configuration.
package config

import (
	"fmt"

	"github.com/spf13/viper"
)

// Config holds all agent configuration.
type Config struct {
	Agent   AgentConfig   `mapstructure:"agent"`
	Network NetworkConfig `mapstructure:"network"`
}

// AgentConfig controls the HTTP listener.
type AgentConfig struct {
	ListenAddress string `mapstructure:"listen_address"`
}

// NetworkConfig controls which network interfaces are collected.
type NetworkConfig struct {
	// IncludeInterfaces is a list of interface name prefixes to include.
	// An empty list means all non-loopback interfaces are included.
	IncludeInterfaces []string `mapstructure:"include_interfaces"`
}

// Load reads a YAML file at path and returns a Config with defaults applied
// for any fields not present in the file.
func Load(path string) (*Config, error) {
	v := viper.New()

	v.SetDefault("agent.listen_address", ":9100")

	v.SetConfigFile(path)
	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("read config %q: %w", path, err)
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("parse config %q: %w", path, err)
	}

	return &cfg, nil
}

// Package config loads and validates the agent's YAML configuration.
package config

import (
	"fmt"

	"github.com/spf13/viper"
)

// Config holds all agent configuration.
type Config struct {
	Agent      AgentConfig      `mapstructure:"agent"`
	Network    NetworkConfig    `mapstructure:"network"`
	Infiniband InfinibandConfig `mapstructure:"infiniband"`
	NVLink     NVLinkConfig     `mapstructure:"nvlink"`
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

// InfinibandConfig controls the InfiniBand sysfs collector.
// There is no Enabled flag — the build tag (-tags infiniband) is the switch.
// If the driver is absent at startup the agent exits loudly.
type InfinibandConfig struct {
	SysfsPath string `mapstructure:"sysfs_path"`
}

// NVLinkConfig controls the DCGM NVLink collector.
// There is no Enabled flag — the build tag (-tags nvlink) is the switch.
// If DCGM is absent at startup the agent exits loudly.
type NVLinkConfig struct {
	// DCGMHostengine is the address of a running dcgm-hostengine process
	// (e.g. "localhost:5555"). Leave empty to use DCGM embedded mode.
	DCGMHostengine string `mapstructure:"dcgm_hostengine"`
}

// Load reads a YAML file at path and returns a Config with defaults applied
// for any fields not present in the file.
func Load(path string) (*Config, error) {
	v := viper.New()

	v.SetDefault("agent.listen_address", ":9100")
	v.SetDefault("infiniband.sysfs_path", "/sys/class/infiniband")
	v.SetDefault("nvlink.dcgm_hostengine", "")

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

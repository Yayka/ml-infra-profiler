// Package netiotest provides test helpers for packages that depend on netio.
package netiotest

import "github.com/Yayka/ml-infra-profiler/agent/internal/netio"

// FakeProvider is an in-memory NetworkStatsProvider for use in unit tests.
type FakeProvider struct {
	Results []netio.InterfaceStat
	Err     error
}

func (f *FakeProvider) Stats(_ []string) ([]netio.InterfaceStat, error) {
	return f.Results, f.Err
}

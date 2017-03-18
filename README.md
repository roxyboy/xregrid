# xregrid

[![codecov](https://codecov.io/gh/roxyboy/xregrid/branch/master/graph/badge.svg)](https://codecov.io/gh/roxyboy/xregrid)

xregrid is a package for aggregating and/or regridding data onto a orthogonal grid using the KDTree algorithm.

xregrid is motivated by the fact that non-orthogonal grids are becoming increasingly common in general circulation models. In order to conduct physically meaningful analysis on the model outputs and compare them with satellite observations, however, we are in need of regridding the data onto orthogonal grids. 

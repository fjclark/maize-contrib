# CHANGELOG

## Version 0.4.0
### Features
- Added Gromacs module (Lily)
- Added flexible docking with Vina
- Added additional tag handling utilities
- Added explicit prior and agent parameters to Reinvent
- Added `IsomerCollection` tagging system
- Added active learning module, subgraph, and example notebook

### Changes
- Reorganised cheminformatics

### Fixes
- Fixed relative-absolute conversion of BFEs not using reference correctly
- Fixed `MakeAbsolute` not handling disconnected subgraphs
- Fixed `SaveLibrary` not saving all conformers
- Fixed `AutoDockGPU` not detecting tarred grids as inputs
- Fixed OpenFE dumping failing on existing dump

## Version 0.3.2
### Features
- Tests can now be skipped automatically based on available config options
- Added absolute free energy conversion for `OpenRFE`

### Changes
- Updated REINVENT implementation
- Updated bundled maize

## Version 0.3.1
### Changes
- Updated bundled maize

### Fixes
- Fixed SaveCSV not using the correct tag order
- Various type fixes

## Version 0.3.0
### Features
- Several new filters
- More subgraphs
- OpenFE node
- Constraints for AutoDockGPU and Glide
- Nodes for reaction prediction
- Batched CSV saving
- Flexible docking with Vina

### Changes
- Various improvements to RMSD filtering
- Performance improvements for `Isomer` tag handling
- Allowed certain RDKit ops to timeout
- More robust Schrodinger job handling
- Integration tests
- Maize core wheel for automated builds

### Fixes
- Fix for zombie threads when using REINVENT
- Fix for installation only being possible using `-e`
- Cleaned up typing

## Version 0.2.3
### Changes
- Removed interface for REINVENT 3.2

### Fixes
- Cleanup of package structure
- Updated dependencies
- Added explicit check for Schrodinger license

## Version 0.2.2
### Features
- Added Schrodinger grid preparation tools

### Changes
- Adjusted timeout for Gypsum-DL
- Various enhancements for GLIDE

### Fixes
- Fixed `Vina` command validation for newer `Vina` versions
- Fixed parsing issues for `Vina`


## Version 0.2.1
### Features
- Improved REINVENT interface, using `ReinventEntry` and `ReinventExit`

### Changes
- Timeouts for conformer / isomer generation
- Improved logging

### Fixes
- Check for zero bytes for `Vina` output

## Version 0.2.0

Updated for public release.

## Version 0.1

Initial release.
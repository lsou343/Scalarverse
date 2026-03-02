#!/bin/bash

################
# Update AMReX #
################

# Small script to update AMReX to the latest tagged release.
# Run this periodically to make sure you're keeping up-to-date.

if [ -z "$AMREX_HOME" ]; then AMREX_HOME="../amrex"; fi

cd $AMREX_HOME

git remote update
tag=$(git describe --tags `git rev-list --tags --max-count=1`)
git checkout tags/$tag

cd -

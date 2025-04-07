#!/bin/bash

# Prune local references to remote branches that no longer exist
# Soource: https://github.com/MakieOrg/Makie.jl/issues/3025
git fetch --prune

git branch -r | grep -o compathelper.* | while read -r branch; do
    git push origin --delete "$branch"
done

git fetch --prune
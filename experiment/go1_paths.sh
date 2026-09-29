# Source from Bash in this checkout; this only sets path variables.
# The optional local file is shell configuration controlled by the operator.
GO1_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 1
if [[ -f "$GO1_PROJECT_ROOT/.go1-paths.env" ]]; then
    source "$GO1_PROJECT_ROOT/.go1-paths.env" || return 1
fi
export GO1_PROJECT_ROOT

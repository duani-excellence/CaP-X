#!/usr/bin/env bash
# Create minimal path-dep stubs so `uv sync` can resolve uv.lock entries for
# extras we do not install on the cloud server (curobo / libero / verl).
set -euo pipefail

stub_pep621() {
  local dir="$1" name="$2" version="$3" pkg="$4"
  if [[ -f "${dir}/pyproject.toml" || -f "${dir}/setup.py" || -f "${dir}/setup.cfg" ]]; then
    return 0
  fi
  mkdir -p "${dir}/${pkg}"
  printf '%s\n' \
    '[build-system]' \
    'requires = ["setuptools>=61"]' \
    'build-backend = "setuptools.build_meta"' \
    '' \
    '[project]' \
    "name = \"${name}\"" \
    "version = \"${version}\"" \
    'requires-python = ">=3.10"' \
    '' \
    '[tool.setuptools.packages.find]' \
    'include = ["'"${pkg}"'*"]' \
    > "${dir}/pyproject.toml"
  printf '# Path-dep stub — not a real install of %s\n' "${name}" > "${dir}/${pkg}/__init__.py"
  echo "[stub] ${dir} → ${name}==${version}"
}

stub_pep621 "capx/third_party/curobo" "nvidia-curobo" "0.7.8" "curobo"
stub_pep621 "capx/third_party/LIBERO-PRO" "libero" "0.1.0" "libero"
stub_pep621 "capx/third_party/libero_dependencies/robosuite" "robosuite" "1.4.0" "robosuite"
stub_pep621 "capx/third_party/verl" "verl" "0.6.1" "verl"

#!/usr/bin/env nix-shell
#!nix-shell -i bash --packages inotify-tools
#shellcheck shell=bash

# "Quick-and-Dirty" Topiary Playground

set -euo pipefail

readonly PROGNAME="$(basename "$0")"

fail() {
  local error="$*"

  cat >&2 <<-EOF
	Error: ${error}

	Usage: ${PROGNAME} (LANGUAGE | QUERY_FILE) [INPUT_SOURCE]

	LANGUAGE can be one of the supported languages (e.g., "ocaml", "rust",
	etc.); alternatively, give the path to the query file itself, as
	QUERY_FILE.

	The INPUT_SOURCE is optional. If not specified, it defaults to trying
	to find the bundled integration test input file for the given language.
	EOF

  exit 1
}

get_sample_input() {
  local language="$1"

  # Only return the first result, presuming there is one
  find tests/samples/input -name "${language}.*" \
  | head -1
}

main() {
  local query="${1-}"
  if ! [[ -e "${query}" ]]; then
    query="languages/${query}.scm"
    [[ -e "${query}" ]] || fail "Couldn't find language query file '${query}'"
  fi

  local language="$(basename --suffix=.scm "${query}")"
  local input="${2-$(get_sample_input "${language}")}"
  [[ -e "${input}" ]] || fail "Couldn't find input source file '${input}'"

  while true; do
    clear

    cat <<-EOF
		Query File    ${query}
		Input Source  ${input}
		$(printf "%${COLUMNS}s" "" | tr " " "-")
		EOF

    cargo run --quiet -- \
      --skip-idempotence \
      --query "${query}" \
      --input-file "${input}"

    # NOTE We don't wait for specific inotify events because different
    # editors have different strategies for modifying files
    inotifywait --quiet "${query}" "${input}"
  done
}

main "$@"

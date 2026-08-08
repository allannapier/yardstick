#!/usr/bin/env bash
# success_check for experiments/opencode-gemini-vs-deepseek.yaml.
#
# Lives here, in the yardstick repo, and NOT in the per-repeat workspace on
# purpose: the workspace is a clone of the blank-web scaffold, so the agent
# never sees this file and cannot target the gate instead of the task.
#
# Invoked by `ys end`/`ys run` with cwd set to the repeat's workspace, so
# every path below is relative to whatever the agent produced. Each check
# maps to one clause of the prompt, and every one of them fails on an empty
# workspace -- the property a bare `pytest -q` on an untouched clone would
# not have (an agent that did nothing would still score success).
#
# Exit 0 = task_success, non-zero = failure. Reasons go to stderr so a
# failing repeat says why in the run log rather than just "non-zero".
set -uo pipefail

fail() { echo "FAIL: $*" >&2; exit 1; }

# 1. "one page website" -- a single HTML entry point.
html=$(ls -1 ./*.html 2>/dev/null | head -1)
[ -n "$html" ] || fail "no .html file in the workspace"

# 2. "create a seperate css file" -- an actual .css file on disk, anywhere
#    (the agent may put it at ./style.css or ./css/style.css).
css=$(find . -name '*.css' -not -path './.git/*' 2>/dev/null | head -1)
[ -n "$css" ] || fail "no .css file found (prompt required a separate stylesheet)"

# 3. "keep the html page clean" -- the stylesheet must actually be wired up,
#    and the styling must not have been inlined into the HTML after all.
grep -qiE '<link[^>]+stylesheet' "$html" \
  || fail "$html does not <link> a stylesheet"
grep -qiE '<style[[:space:]>]' "$html" \
  && fail "$html contains an inline <style> block (prompt asked for a separate css file)"

# 4. The business actually named in the prompt, not a generic template.
grep -qiE "gary'?s gardens" "$html" \
  || fail "$html never mentions Gary's Gardens"

# 5. The location detail the prompt supplied, rather than invented or dropped.
grep -qi 'bonnyrigg' "$html" || fail "$html never mentions Bonnyrigg"
grep -qi 'midlothian' "$html" || fail "$html never mentions Midlothian"

echo "PASS: $html + $css"
exit 0

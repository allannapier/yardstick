# yardstick docs -- authoring contract

Read this before writing a tutorial page. It exists so all seven tutorial
pages, written by different agents, come out looking and behaving like one
site. If you deviate from this contract, the site's design system and
site.js behavior will silently break on your page.

## 1. Every page is a full standalone HTML document copied from `_template.html`

- Do not hand-roll a page from scratch. Copy `docs/_template.html` to
  `docs/tutorials/NN-your-page.html` and edit only the regions marked
  `<!-- REPLACE: ... -->`.
- The result must be a complete `<!doctype html>` document: `<html lang="en">`,
  a `<head>` with `charset`, `viewport`, `<title>`, a one-sentence
  `<meta name="description">`, the stylesheet link, and the deferred script
  tag -- plus `<body>` with the header, sidebar, `<main>`, and footer.
- Do not split the shell into includes/partials. GitHub Pages here serves
  static files with no templating, so each `.html` file must be self-contained.

## 2. Relative-path rules

- Tutorial pages live in `docs/tutorials/`. From there:
  - CSS: `../assets/site.css`
  - JS: `../assets/site.js`
  - Home: `../index.html`
  - Other tutorials: `../tutorials/NN-name.html`
- `docs/index.html` lives one directory shallower than tutorial pages:
  - CSS: `assets/site.css` (no `../`)
  - JS: `assets/site.js`
  - Tutorials: `tutorials/NN-name.html`
- Never use a path starting with `/`. The site is served from a subpath
  (`https://allannapier.github.io/yardstick/`), so an absolute path like
  `/assets/site.css` resolves to the wrong host path and 404s.
- External links (e.g. to GitHub) are the only absolute URLs allowed, and
  must be full `https://...` URLs.

## 3. The seven tutorials, in order

| # | Filename | Title |
|---|----------|-------|
| 1 | `tutorials/01-install-first-run.html` | Install and your first run |
| 2 | `tutorials/02-experiment-yaml.html` | Designing an experiment |
| 3 | `tutorials/03-live-run-claude-code.html` | Measuring a live Claude Code session |
| 4 | `tutorials/04-compare-and-report.html` | Comparing arms and reporting |
| 5 | `tutorials/05-dashboard.html` | Using the dashboard |
| 6 | `tutorials/06-metrics-reference.html` | Metrics reference |
| 7 | `tutorials/07-troubleshooting.html` | Troubleshooting |

This is also the sidebar order. Do not add, remove, rename, or reorder pages
-- if a topic doesn't fit one of these seven, it belongs inside one of them,
not in a new file.

## 4. Sidebar markup (reproduce verbatim on every tutorial page)

```html
<aside class="sidebar" id="sidebar">
  <nav class="sidebar-nav" id="sidebar-nav" aria-label="Tutorials">
    <p class="sidebar-title">yardstick docs</p>
    <ul>
      <li><a class="home-link" href="../index.html">Home</a></li>
    </ul>
    <p class="sidebar-title">Tutorials</p>
    <ul>
      <li><a href="../tutorials/01-install-first-run.html">1. Install and your first run</a></li>
      <li><a href="../tutorials/02-experiment-yaml.html">2. Designing an experiment</a></li>
      <li><a href="../tutorials/03-live-run-claude-code.html">3. Measuring a live Claude Code session</a></li>
      <li><a href="../tutorials/04-compare-and-report.html">4. Comparing arms and reporting</a></li>
      <li><a href="../tutorials/05-dashboard.html">5. Using the dashboard</a></li>
      <li><a href="../tutorials/06-metrics-reference.html">6. Metrics reference</a></li>
      <li><a href="../tutorials/07-troubleshooting.html">7. Troubleshooting</a></li>
    </ul>
  </nav>
</aside>
```

`site.js` marks the current page's link with `aria-current="page"` at
runtime by comparing `location.pathname`, so you don't have to hand-edit
this per page -- but the markup (classes, ids, href set) must match exactly
or that script silently finds nothing to mark.

## 5. Pager markup (bottom of every tutorial page)

```html
<nav class="pager" aria-label="Page navigation">
  <a class="pager-prev" href="../tutorials/01-install-first-run.html">
    <span class="pager-dir">Previous</span>
    <span>Install and your first run</span>
  </a>
  <a class="pager-next" href="../tutorials/03-live-run-claude-code.html">
    <span class="pager-dir">Next</span>
    <span>Measuring a live Claude Code session</span>
  </a>
</nav>
```

- Tutorial 1's "previous" points to `../index.html` with label `Home`.
- Tutorial 7's pager has no next link -- replace the `.pager-next` anchor
  with `<span class="pager-spacer"></span>` so the previous link doesn't
  stretch full-width.

## 6. Component class list (one-line usage each)

| Class | Usage |
|---|---|
| `.note` | `<div class="note"><p>...</p></div>` -- neutral supplementary info, blue accent. |
| `.warn` | `<div class="warn"><p>...</p></div>` -- things that can break a run or lose data, amber accent. |
| `.tip` | `<div class="tip"><p>...</p></div>` -- optional shortcuts / nice-to-know, green accent. |
| `.steps` | `<ol class="steps"><li><h4>..</h4><p>..</p></li></ol>` -- numbered walkthrough, one `<li>` per step. |
| `.code` / `.code-header` / `.code-label` | see section 7 below -- labeled code block with copy button. |
| `.table-wrap` | `<div class="table-wrap"><table>...</table></div>` -- wrap every `<table>` so wide tables scroll instead of breaking layout. |
| `.card-grid` / `.card` | `<div class="card-grid"><a class="card" href="..."><span class="card-index">01</span><span class="card-title">..</span><p class="card-desc">..</p></a></div>` -- landing-page tutorial index; avoid inside tutorial pages except for a "see also" grid. |
| `.pager` / `.pager-prev` / `.pager-next` / `.pager-spacer` | bottom-of-page prev/next navigation, see section 5. |
| `.lede` | `<p class="lede">...</p>` -- larger intro sentence directly under an `<h1>`. |
| `.btn` / `.btn-primary` / `.btn-secondary` | `<a class="btn btn-primary" href="...">Label</a>` -- call-to-action buttons (landing page hero). |
| inline `code` | plain `<code>...</code>` for flags, filenames, short commands -- no extra class needed. |

Do not invent new top-level classes for things this list already covers.
If you need something not listed here, add plain semantic HTML (`h2`-`h4`,
`p`, `ul`/`ol`, `blockquote`) styled by the existing base rules rather than
introducing new class names -- ask before extending `site.css`.

## 7. Code block pattern (and how to set the header label)

```html
<div class="code">
  <div class="code-header">
    <span class="code-label">bash</span>
  </div>
  <pre data-lang="bash"><code>ys proxy up --exp experiments/example.yaml</code></pre>
</div>
```

- The label shown in the block's header is just the text inside
  `.code-label` -- set it to whatever's useful: `bash`, `experiments/example.yaml`,
  `output`, `~/.claude/settings.json`, etc.
- `site.js` finds every `pre[data-lang]` or `pre` inside `.code`, and appends
  a "Copy" button into the nearest `.code-header` automatically. **Do not
  hand-write the copy button** -- just provide the `.code`/`.code-header`/`.code-label`
  wrapper and the `<pre data-lang="...">` block; the button appears at
  runtime and copies `pre code`'s exact text content.
- Always put code text inside `<pre><code>...</code></pre>`, not `<pre>` alone,
  and don't add leading/trailing blank lines inside `<code>` -- the copy
  button copies the text node verbatim.
- For long shell transcripts spanning multiple commands, one `.code` block
  with multiple lines (and `#` comments) is preferred over several small
  blocks.

## 8. Theme toggle and other shared chrome

- The header, exactly as in `_template.html`, includes
  `<button type="button" id="theme-toggle" aria-label="Toggle color theme">☽</button>`.
  Do not duplicate this id on a page or the toggle will only affect the first
  match.
- The mobile nav toggle (`.nav-toggle`, only relevant on pages with a
  sidebar) must keep `aria-controls="sidebar-nav"` pointing at the sidebar
  nav's `id="sidebar-nav"`.
- Do not add inline `<style>` or `<script>` blocks to a page. All styling and
  behavior lives in `assets/site.css` / `assets/site.js` so every page stays
  visually and behaviorally consistent.

## 9. Content width and prose

- Prose (`<p>`, lists) is capped at `~72ch` by `.content`'s max-width -- you
  do not need to add manual line breaks or width constraints yourself.
- Use `<h2>` for top-level sections within a tutorial, `<h3>` for
  subsections, `<h4>` sparingly (it's also used as the step title inside
  `.steps`).

## 10. What not to do

- No CDN scripts/fonts, no inline event handlers (`onclick=...`), no
  framework markup (React/Vue attributes, `class="tw-..."`, etc).
- No absolute paths (`/assets/...`, `/tutorials/...`).
- No new pages beyond the seven listed in section 3.
- No editing of `docs/assets/site.css`, `docs/assets/site.js`,
  `docs/_template.html`, or `docs/index.html` -- those are owned by the
  foundation and shared across all tutorial pages. If you find a bug or gap
  in them, flag it rather than patching it yourself, since other pages
  depend on their current behavior.

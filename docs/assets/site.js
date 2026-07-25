/*!
 * yardstick docs -- site.js
 * Vanilla JS, no dependencies. See docs/CONTRACT.md for the markup contract
 * this script relies on. Every piece below is defensive: a page missing an
 * element simply skips that feature instead of throwing.
 */
(function () {
  "use strict";

  var THEME_KEY = "ys-theme";

  /* ---- theme toggle ------------------------------------------------ */

  function getStoredTheme() {
    try {
      return localStorage.getItem(THEME_KEY);
    } catch (e) {
      return null;
    }
  }

  function storeTheme(value) {
    try {
      localStorage.setItem(THEME_KEY, value);
    } catch (e) {
      /* storage unavailable (private mode, etc) -- ignore */
    }
  }

  function systemPrefersDark() {
    return (
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
    );
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    var btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.setAttribute(
        "aria-label",
        theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
      );
      btn.textContent = theme === "dark" ? "☀" : "☽"; /* sun / moon */
    }
  }

  function initTheme() {
    var stored = getStoredTheme();
    var theme = stored === "dark" || stored === "light" ? stored : (systemPrefersDark() ? "dark" : "light");
    applyTheme(theme);

    var btn = document.getElementById("theme-toggle");
    if (!btn) return;

    btn.addEventListener("click", function () {
      var current = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
      var next = current === "dark" ? "light" : "dark";
      applyTheme(next);
      storeTheme(next);
    });
  }

  /* ---- copy buttons on code blocks ---------------------------------- */

  function findCodeBlocks() {
    var blocks = [];
    var pres = document.querySelectorAll("pre[data-lang], .code pre, pre.code");
    for (var i = 0; i < pres.length; i++) {
      if (blocks.indexOf(pres[i]) === -1) blocks.push(pres[i]);
    }
    return blocks;
  }

  function copyText(text, onDone) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard
        .writeText(text)
        .then(function () {
          onDone(true);
        })
        .catch(function () {
          onDone(fallbackCopy(text));
        });
      return;
    }
    onDone(fallbackCopy(text));
  }

  function fallbackCopy(text) {
    try {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.top = "-1000px";
      ta.style.left = "-1000px";
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch (e) {
      return false;
    }
  }

  function ensureCopyButton(pre) {
    var codeEl = pre.querySelector("code") || pre;
    var wrapper = pre.closest(".code");
    var header = wrapper ? wrapper.querySelector(".code-header") : null;
    var host = header;

    if (!host) {
      /* no header present: create a minimal inline button positioned by
         the caller's own CSS; this keeps the script useful even outside
         the standard .code/.code-header pattern. */
      if (pre.querySelector(".copy-btn")) return;
      host = document.createElement("div");
      host.className = "code-header";
      pre.parentNode.insertBefore(host, pre);
      host.appendChild(document.createTextNode(""));
    }

    if (host.querySelector(".copy-btn")) return;

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.setAttribute("aria-label", "Copy code to clipboard");

    btn.addEventListener("click", function () {
      var text = codeEl.textContent || "";
      copyText(text, function (ok) {
        btn.dataset.copied = ok ? "true" : "false";
        btn.textContent = ok ? "Copied" : "Copy failed";
        setTimeout(function () {
          btn.textContent = "Copy";
          delete btn.dataset.copied;
        }, 1600);
      });
    });

    host.appendChild(btn);
  }

  function initCopyButtons() {
    var blocks = findCodeBlocks();
    for (var i = 0; i < blocks.length; i++) {
      ensureCopyButton(blocks[i]);
    }
  }

  /* ---- sidebar: current-page highlighting ---------------------------- */

  function initCurrentPageLink() {
    var links = document.querySelectorAll(".sidebar-nav a[href]");
    if (!links.length) return;

    var here = location.pathname.replace(/\/index\.html$/, "/");

    links.forEach(function (a) {
      try {
        var target = new URL(a.getAttribute("href"), location.href).pathname.replace(
          /\/index\.html$/,
          "/"
        );
        if (target === here) {
          a.setAttribute("aria-current", "page");
        }
      } catch (e) {
        /* malformed href -- skip */
      }
    });
  }

  /* ---- mobile nav toggle ---------------------------------------------- */

  function initNavToggle() {
    var toggle = document.querySelector(".nav-toggle");
    var sidebar = document.querySelector(".sidebar");
    if (!toggle || !sidebar) return;

    sidebar.dataset.collapsed = "true";
    toggle.setAttribute("aria-expanded", "false");

    toggle.addEventListener("click", function () {
      var collapsed = sidebar.dataset.collapsed !== "false";
      sidebar.dataset.collapsed = collapsed ? "false" : "true";
      toggle.setAttribute("aria-expanded", collapsed ? "true" : "false");
    });
  }

  /* ---- boot ------------------------------------------------------------- */

  function init() {
    initTheme();
    initCopyButtons();
    initCurrentPageLink();
    initNavToggle();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

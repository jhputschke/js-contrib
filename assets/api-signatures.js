/* The Python API pages: mark the parts of each object's signature, so api.css can show the
 * module path muted and the object's name bold, as Sphinx does.  The signature is a
 * Pygments-highlighted <code> in the heading; its name is the last name token before the
 * first "(", ":" or "=" (or the last one, for a bare name).  Runs on every page load,
 * including Material's instant navigation; pages without API headings are left alone. */
(function () {
  "use strict";

  function mark(root) {
    root.querySelectorAll(".doc-heading > code.highlight").forEach(function (code) {
      if (code.dataset.apiMarked) return;
      code.dataset.apiMarked = "1";
      var spans = Array.prototype.slice.call(code.children);
      var stop = spans.length;
      for (var i = 0; i < spans.length; i++) {
        var t = spans[i].textContent;
        if (spans[i].classList.contains("p") && (t === "(" || t === ":") ||
            spans[i].classList.contains("o") && t === "=") {
          stop = i;
          break;
        }
      }
      var nameAt = -1;
      for (var j = stop - 1; j >= 0; j--) {
        if (/^(n|nc|nf|nn|nb|bp)$/.test(spans[j].className)) { nameAt = j; break; }
      }
      if (nameAt < 0) return;
      spans[nameAt].classList.add("api-name");
      for (var k = 0; k < nameAt; k++) spans[k].classList.add("api-prefix");
    });
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(function () { mark(document); });
  } else {
    document.addEventListener("DOMContentLoaded", function () { mark(document); });
  }
})();

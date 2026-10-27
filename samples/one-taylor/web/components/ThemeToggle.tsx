"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark" | "system";

function read(): Theme {
  try {
    const v = localStorage.getItem("ot-theme");
    if (v === "light" || v === "dark") return v;
  } catch {
    // private window, blocked site data — fall through to "system"
  }
  return "system";
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => setTheme(read()), []);

  function apply(next: Theme) {
    setTheme(next);
    const root = document.documentElement;
    if (next === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", next);
    try {
      if (next === "system") localStorage.removeItem("ot-theme");
      else localStorage.setItem("ot-theme", next);
    } catch {
      // The toggle still works for this page view; it just will not persist.
    }
  }

  const order: Theme[] = ["system", "light", "dark"];
  const labels: Record<Theme, string> = { system: "Auto", light: "Light", dark: "Dark" };

  return (
    <div className="row" role="group" aria-label="Colour theme" style={{ gap: 2 }}>
      {order.map((t) => (
        <button
          key={t}
          type="button"
          className={`chip chip-btn ${theme === t ? "chip-on" : ""}`}
          aria-pressed={theme === t}
          onClick={() => apply(t)}
        >
          {labels[t]}
        </button>
      ))}
    </div>
  );
}

"use client";

import { useState } from "react";

export function CopyBlock({ code }: { code: string }) {
  const [label, setLabel] = useState("Kopyala");
  return (
    <pre>
      <button
        className="copy"
        type="button"
        onClick={() => {
          void navigator.clipboard.writeText(code).then(() => {
            setLabel("Kopyalandı");
            window.setTimeout(() => setLabel("Kopyala"), 1200);
          });
        }}
      >
        {label}
      </button>
      <code>{code}</code>
    </pre>
  );
}

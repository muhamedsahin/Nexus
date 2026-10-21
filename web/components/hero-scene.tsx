"use client";

import { useEffect, useRef } from "react";

export function HeroScene() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    let frame = 0;
    let alive = true;
    const points: Array<[number, number, number]> = [];
    for (let x = -1; x <= 1; x += 1) {
      for (let y = -1; y <= 1; y += 1) {
        for (let z = -1; z <= 1; z += 1) points.push([x, y, z]);
      }
    }
    const draw = () => {
      if (!alive) return;
      const angle = frame * 0.01;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.strokeStyle = "#c6f135";
      context.lineWidth = 1.2;
      const projected = points.map(([x, y, z]) => {
        const rx = x * Math.cos(angle) - z * Math.sin(angle);
        const rz = x * Math.sin(angle) + z * Math.cos(angle);
        const scale = 48 / (2.4 + rz);
        return [90 + rx * scale * 28, 90 + y * scale * 28];
      });
      context.beginPath();
      for (let i = 0; i < projected.length; i += 1) {
        for (let j = i + 1; j < projected.length; j += 1) {
          const a = points[i];
          const b = points[j];
          const dist = Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]);
          if (dist === 2) {
            context.moveTo(projected[i][0], projected[i][1]);
            context.lineTo(projected[j][0], projected[j][1]);
          }
        }
      }
      context.stroke();
      frame += 1;
      requestAnimationFrame(draw);
    };
    draw();
    return () => {
      alive = false;
    };
  }, []);
  return <canvas ref={ref} className="scene" width={180} height={180} aria-hidden />;
}

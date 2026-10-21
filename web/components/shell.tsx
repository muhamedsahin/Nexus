"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  ["/", "Ana sayfa"],
  ["/kurulum", "Kurulum"],
  ["/cpp", "C++"],
  ["/python", "Python"],
  ["/api", "API"],
  ["/benchmark", "Benchmark"],
  ["/lisans", "Lisans"],
];

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  return (
    <div className="shell">
      <nav className="nav">
        <Link href="/" className="brand">
          NEX<span>US</span>
        </Link>
        {links.map(([href, label]) => (
          <Link key={href} href={href} className={path === href ? "item active" : "item"}>
            {label}
          </Link>
        ))}
      </nav>
      <main className="main">
        {children}
        <footer className="note">
          Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih
          Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
        </footer>
      </main>
    </div>
  );
}

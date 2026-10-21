import type { Metadata } from "next";
import "./globals.css";
import { Shell } from "../components/shell";

export const metadata: Metadata = {
  title: "Nexus dokümantasyon",
  description: "Nexus C++20 çerçevesi ve Python kabuğunun kurulumu, API'si ve ölçülmüş hızı.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="tr">
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}

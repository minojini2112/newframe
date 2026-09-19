import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ISRO PS12 — FillFrame Mission Console",
  description: "Geostationary satellite frame interpolation dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

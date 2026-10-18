import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Uni Knowledge Engine",
  description: "Ask questions about university course PDFs and get answers cited to page.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

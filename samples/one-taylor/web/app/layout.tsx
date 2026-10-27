import type { Metadata } from "next";
import "@/styles/globals.css";
import { ThemeScript } from "@/components/ThemeScript";

export const metadata: Metadata = {
  title: "One Taylor",
  description:
    "Share your life and meet people your own age. One Taylor keeps under-18s and adults in completely separate worlds.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="color-scheme" content="light dark" />
        <ThemeScript />
      </head>
      <body>{children}</body>
    </html>
  );
}

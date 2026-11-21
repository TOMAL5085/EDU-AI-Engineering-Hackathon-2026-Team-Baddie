import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "WardWatch | Dengue Early Warning System",
  description: "Next.js dashboard for ward-level dengue risk forecasting in Bangladesh using local supervised machine learning.",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased selection:bg-emerald-500 selection:text-slate-950">
        {children}
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import localFont from "next/font/local";

import "./globals.css";
import { Providers } from "./providers";

// NOTE(CLAUDE.md 예외): RootLayout은 metadata export 때문에 Server Component로 유지한다.
// 단순화 규칙("모든 페이지 'use client'")의 의도(RSC 데이터 패칭/Server Action 금지)는
// 지키며, layout은 html shell + metadata + <Providers> 마운트만 담당한다.
const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Pind",
  description: "영상 속 장소를 추출해 지도로 보여주는 서비스",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}

import "./globals.css";

export const metadata = {
  title: "DRISHTI Nowcast",
  description: "Multi-task nowcasting of severe thunderstorms, cloudbursts and flash floods at 2–6 h lead time.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}

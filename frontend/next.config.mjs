/** Static export: the dashboard reads model output from public/data, so it
 *  deploys to Vercel or GitHub Pages with no server. When the FastAPI backend
 *  lands, lib/data.js switches to live endpoints. */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || "",
};
export default nextConfig;

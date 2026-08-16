/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Standalone output untuk Docker image yang lebih kecil
  output: "standalone",

  // Proxy API ke backend — menghindari CORS di development dan production
  async rewrites() {
    // API base path configuration:
    // - Local development: http://localhost:8000
    // - Docker: http://backend:8000 (Docker network)
    // - Cloud: Production URL from environment variable
    const apiBase =
      process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
    
    // If using Docker network, use the service name
    const dockerApiBase = process.env.DOCKER_API_BASE || "http://backend:8000";
    const finalApiBase = process.env.DOCKER_MODE === "true" ? dockerApiBase : apiBase;
    
    return [
      {
        source: "/api/:path*",
        destination: `${finalApiBase}/api/:path*`,
      },
      {
        source: "/health",
        destination: `${finalApiBase}/health`,
      },
    ];
  },

  // WebSocket harus menggunakan server-level proxy (tidak bisa via rewrites).
  // Pada mode Docker, frontend akan menggunakan URL WS langsung ke window.location.host
  // yang sudah di-expose ke port 3000 dan backend di-akses via internal Docker network.
  // Tidak perlu konfigurasi tambahan di sini.
};

module.exports = nextConfig;

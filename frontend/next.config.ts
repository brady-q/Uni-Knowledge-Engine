import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // The API is reached over the network by URL, never proxied through Next's
  // rewrites: keeping the two services independent means the frontend can be
  // pointed at a deployed backend without a rebuild.
};

export default nextConfig;

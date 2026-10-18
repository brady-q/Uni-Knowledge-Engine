# Development image: runs `next dev` against the bind-mounted source.
# A multi-stage production build (standalone output) lands with the real UI.
FROM node:22-slim

ENV NEXT_TELEMETRY_DISABLED=1
WORKDIR /app

# npm ci (not npm install) so the committed lockfile is authoritative.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# .next must exist and be owned by `node` before the anonymous volume for it
# is created, or the non-root dev server cannot write its build output.
RUN mkdir -p /app/.next && chown -R node:node /app
USER node

EXPOSE 3000
CMD ["npm", "run", "dev"]

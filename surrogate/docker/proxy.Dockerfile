FROM node:22-bookworm-slim AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY index.html vite.config.ts postcss.config.js tailwind.config.js eslint.config.js ./
COPY tsconfig.json tsconfig.app.json tsconfig.node.json ./
COPY public /app/public
COPY src /app/src

RUN npm run build

FROM nginx:1.27-alpine

ARG OCI_SOURCE=""

ENV API_UPSTREAM=http://api:8000 \
    API_UPSTREAM_231=http://api:8000 \
    API_UPSTREAM_232=http://api:8000

COPY docker/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

LABEL org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.description="PhotoGuard upgraded frontend and nginx reverse proxy"

EXPOSE 80

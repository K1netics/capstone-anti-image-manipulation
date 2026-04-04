FROM node:20-bookworm-slim AS build

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine

ARG OCI_SOURCE=""

ENV API_UPSTREAM=http://api:8000

COPY proxy/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

LABEL org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.description="PhotoGuard frontend and nginx reverse proxy"

EXPOSE 80

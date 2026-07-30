FROM node:22-alpine AS build

WORKDIR /app

COPY immuna-meeting-docker/frontend/package.json immuna-meeting-docker/frontend/package-lock.json ./
RUN npm ci

COPY immuna-meeting-docker/frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine

ARG OCI_SOURCE=""

ENV API_UPSTREAM=http://api:8000 \
    API_UPSTREAM_231=http://api:8000 \
    API_UPSTREAM_232=http://api:8000

COPY upgraded/docker/default.conf.template /etc/nginx/templates/default.conf.template
COPY immuna-meeting-docker/docker-entrypoint.d/40-runtime-config.sh /docker-entrypoint.d/40-runtime-config.sh
RUN chmod +x /docker-entrypoint.d/40-runtime-config.sh
COPY --from=build /app/dist /usr/share/nginx/html

LABEL org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.description="Immuna frontend bundled with PhotoGuard upgraded backend wiring"

EXPOSE 80

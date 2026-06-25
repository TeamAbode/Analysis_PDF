# Jury Analyst Pipeline — container image for hosting
#
# Bundles the native libraries WeasyPrint needs (Pango/Cairo/gdk-pixbuf) so the
# PDF step works the same everywhere, with no per-machine setup.

FROM python:3.12-slim

# System libraries required by WeasyPrint (not installable via pip) + fonts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libpangoft2-1.0-0 \
        libgdk-pixbuf-2.0-0 \
        libcairo2 \
        libffi-dev \
        shared-mime-info \
        fonts-dejavu \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so they cache across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app.
COPY . .

# Bind to all interfaces; the hosting platform provides $PORT.
ENV HOST=0.0.0.0 \
    PORT=8765 \
    PYTHONUNBUFFERED=1

EXPOSE 8765

CMD ["python", "app.py"]

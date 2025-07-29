#!/usr/bin/env python3
"""
Clarín ▸ Scraper masivo 2025 – Páginas 1-2423
────────────────────────────────────────────────────────────
Versión *deployment* (v1.0 – reconexión automática + pooling)
• Recorre todas las páginas de índice de Economía desde la 1 hasta la 2423:
      https://www.clarin.com/economia/page/<n>
• Bloques de 500 páginas con pausa de 5 min.
• Delay aleatorio 1.2 – 1.8 s entre notas (≈ 0.6 req/s).
• Back-off 10 min ante HTTP 429/503.
• Progreso en `scraper.log` y stdout.
• PostgreSQL con reconexión automática y connection pooling.
• Extrae fecha vía (1) JSON-LD NewsArticle → (2) meta `article:published_time`
  → (3) tag <time>.  Normaliza a «YYYY-MM-DD HH:MM:SS».


"""
import argparse, json, logging, random, re, time, shutil, os
from pathlib import Path

import requests
import dateutil.parser
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# Configuración de PostgreSQL con pooling
# ----------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
BACKUP_DIR   = "backups"
Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)

# Connection pool global
connection_pool = None

# ----------------------------------------------------------------------------
# Configuración por defecto — Clarín Economía 1-2423
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    start=1,
    end=2423,
    delay_min=1.2,
    delay_max=1.8,
    block_size=500,
    pause=300,
    db=DATABASE_URL,
)

HEADERS = {
    "User-Agent": "clarin-econ-scraper/1.0 (+mailto:tu_correo@dominio.com)",
    "Accept-Language": "es-AR,es;q=0.9",
}

# ----------------------------------------------------------------------------
# PostgreSQL helpers con pooling y reconexión
# ----------------------------------------------------------------------------
def init_connection_pool(database_url: str):
    """Inicializa el pool de conexiones PostgreSQL"""
    global connection_pool
    try:
        # Neon pooler (si aplica) para mejor performance
        pooled_url = database_url.replace(".us-east-2", "-pooler.us-east-2")
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,  # min=1, max=10 conexiones
            pooled_url
        )
        logging.getLogger("scraper").info("Connection pool inicializado")
        return True
    except Exception as e:
        # Fallback a URL original
        logging.getLogger("scraper").warning(f"Pooler falló, usando URL original: {e}")
        try:
            connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,
                database_url
            )
            return True
        except Exception as e2:
            logging.getLogger("scraper").error(f"Error inicializando pool: {e2}")
            return False

def get_connection():
    """Obtiene una conexión del pool con reintentos"""
    global connection_pool
    max_retries = 3

    for attempt in range(max_retries):
        try:
            if connection_pool is None:
                if not init_connection_pool(DATABASE_URL):
                    raise Exception("No se pudo inicializar el pool")
            conn = connection_pool.getconn()
            # Ping barato
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return conn
        except Exception as e:
            logging.getLogger("scraper").warning(f"Intento {attempt+1} falló: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                if connection_pool:
                    try:
                        connection_pool.closeall()
                    except:
                        pass
                    connection_pool = None
            else:
                raise Exception(f"No se pudo obtener conexión después de {max_retries} intentos")

def return_connection(conn):
    """Devuelve una conexión al pool"""
    global connection_pool
    try:
        if connection_pool and conn:
            connection_pool.putconn(conn)
    except Exception as e:
        logging.getLogger("scraper").warning(f"Error devolviendo conexión: {e}")

def init_db():
    """Verifica/crea tabla de artículos"""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS articles (
                       id     SERIAL PRIMARY KEY,
                       url    TEXT UNIQUE,
                       title  TEXT,
                       date   TEXT,
                       author TEXT,
                       text   TEXT
                   )"""
            )
        conn.commit()
        logging.getLogger("scraper").info("Tabla articles verificada/creada")
    except Exception as e:
        logging.getLogger("scraper").error(f"Error inicializando DB: {e}")
        raise
    finally:
        if conn:
            return_connection(conn)

# ----------------------------------------------------------------------------
# Listar links de una página de índice
# ----------------------------------------------------------------------------
def list_page(page: int):
    """
    Devuelve todas las URLs de notas listadas en
    https://www.clarin.com/economia/page/<page>

    Patrón de nota: /economia/<slug>_<id>.html
    """
    url = f"https://www.clarin.com/economia/page/{page}"
    r = requests.get(url, headers=HEADERS, timeout=30)

    if r.status_code in (429, 503):
        raise RuntimeError(f"Bloqueo temporal: HTTP {r.status_code}")
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    links, seen = [], set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = "https://www.clarin.com" + href
        if (
            href.startswith("https://www.clarin.com/economia/")
            and re.search(r"_[0-9A-Za-z]{2,}\.html$", href)
            and href not in seen
        ):
            seen.add(href)
            links.append(href)

    return links

# ----------------------------------------------------------------------------
# Parseo de artículo completo
# ----------------------------------------------------------------------------
def parse_article(url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code in (429, 503):
        raise RuntimeError(f"Bloqueo temporal en nota: HTTP {r.status_code}")
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # título
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else "(sin título)"

    # fecha – JSON-LD → meta → <time>
    date = ""
    for s in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(s.string)
            items = data if isinstance(data, list) else [data]
            for it in items:
                if it.get("@type") == "NewsArticle" and it.get("datePublished"):
                    date = it["datePublished"]
                    raise StopIteration
        except StopIteration:
            break
        except Exception:
            continue

    if not date:
        meta = soup.find("meta", {"property": "article:published_time"})
        if meta and meta.get("content"):
            date = meta["content"]

    if not date:
        time_el = soup.find("time")
        if time_el and time_el.get("datetime"):
            date = time_el["datetime"]

    try:
        if date:
            date = dateutil.parser.parse(date).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass  # deja date cruda si falla

    # autor
    author_el = soup.select_one(".author, .firma, .authorName")
    author = author_el.get_text(strip=True) if author_el else ""

    # cuerpo
    selectors = [
        "article", "[itemprop='articleBody']", ".article-body",
        ".article__text", ".article__content", ".d3_rich_text",
        ".article-content", ".rich-text"
    ]
    seen, paras = set(), []
    for sel in selectors:
        for c in soup.select(sel):
            for p in c.find_all("p"):
                t = p.get_text(" ", strip=True)
                if t and t not in seen:
                    seen.add(t)
                    paras.append(t)
    if sum(len(p) for p in paras) < 400:
        for p in soup.find_all("p"):
            t = p.get_text(" ", strip=True)
            if t and t not in seen and len(t) > 40:
                seen.add(t)
                paras.append(t)

    return dict(url=url, title=title, date=date, author=author, text="\n".join(paras))

# ----------------------------------------------------------------------------
# Guardar artículo con manejo de reconexión
# ----------------------------------------------------------------------------
def save_article(art):
    conn = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO articles(url, title, date, author, text)
                       VALUES (%(url)s, %(title)s, %(date)s, %(author)s, %(text)s)
                       ON CONFLICT (url) DO NOTHING""",
                    art,
                )
            conn.commit()
            return True
        except Exception as e:
            logging.getLogger("scraper").warning(f"Error guardando (intento {attempt+1}): {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                logging.getLogger("scraper").error(f"Error guardando artículo después de {max_retries} intentos: {e}")
                return False
        finally:
            if conn:
                return_connection(conn)

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def scrape_range(cfg):
    log = logging.getLogger("scraper")

    # Inicializar pool y DB
    if not init_connection_pool(cfg.db):
        log.error("No se pudo inicializar la conexión a la base de datos")
        return
    init_db()

    current = cfg.start
    while current <= cfg.end:
        block_end = min(current + cfg.block_size - 1, cfg.end)
        log.info("=== Bloque %d – %d ===", current, block_end)

        # 1. enlaces del bloque
        links = []
        for page in range(current, block_end + 1):
            try:
                page_links = list_page(page)
                log.info("Página %d → %d enlaces", page, len(page_links))
                links.extend(page_links)
            except RuntimeError as e:
                log.warning("%s; esperando 10 min", e)
                time.sleep(600)
                continue
            except Exception as e:
                log.exception("Error p.%d: %s", page, e)
            time.sleep(1)

        # 2. Scrapeo artículos
        uniq_links = list(dict.fromkeys(links))
        log.info("%d enlaces únicos", len(uniq_links))
        for idx, url in enumerate(uniq_links, 1):
            try:
                art = parse_article(url)
                if save_article(art):
                    log.info("[%d/%d] ✔︎ %s", idx, len(uniq_links), art["title"][:60])
                else:
                    log.warning("[%d/%d] ✗ Error guardando: %s", idx, len(uniq_links), art["title"][:60])
            except RuntimeError as e:
                log.warning("%s; reintento en 10 min", e)
                time.sleep(600)
                idx -= 1
                continue
            except Exception as e:
                log.exception("Error nota %s: %s", url, e)
            time.sleep(random.uniform(cfg.delay_min, cfg.delay_max))

        # pausa entre bloques
        current = block_end + 1
        if current <= cfg.end:
            log.info("Bloque completo, pausa %d s", cfg.pause)
            time.sleep(cfg.pause)

    # Cerrar pool
    global connection_pool
    if connection_pool:
        try:
            connection_pool.closeall()
        except:
            pass

    # backup final (dump SQL)
    ts = time.strftime("%Y%m%d_%H%M%S")
    try:
        backup_cmd = f"pg_dump {DATABASE_URL} > {BACKUP_DIR}/articles_{ts}.sql"
        os.system(backup_cmd)
        log.info("Backup creado en: %s/articles_%s.sql", BACKUP_DIR, ts)
    except Exception as e:
        log.warning("No se pudo crear backup: %s", e)
    log.info("🎉 Scraping terminado. DB: PostgreSQL con pooling")

# ----------------------------------------------------------------------------
# CLI & logging setup
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Scraper masivo Clarín Economía")
    p.add_argument("--start", type=int, default=DEFAULTS["start"])
    p.add_argument("--end", type=int, default=DEFAULTS["end"])
    p.add_argument("--delay-min", type=float, default=DEFAULTS["delay_min"])
    p.add_argument("--delay-max", type=float, default=DEFAULTS["delay_max"])
    p.add_argument("--block-size", type=int, default=DEFAULTS["block_size"])
    p.add_argument("--pause", type=int, default=DEFAULTS["pause"])
    p.add_argument("--db", default=DEFAULTS["db"])
    return p.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("scraper.log"),
            logging.StreamHandler()
        ]
    )

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    scrape_range(args)

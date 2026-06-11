from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.db.session import init_db
from app.api.chat import router as chat_router
from app.api.misc import catalog_router, health_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Persistent Sales Assistant Agent with cross-session memory, "
        "real tool use, and structured self-evaluation on every response."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


app.include_router(chat_router)
app.include_router(catalog_router)
app.include_router(health_router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }
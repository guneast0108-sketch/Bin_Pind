from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.settings import settings

# ─────────────────────────────────────────────
# 1. Engine — 앱 전체에서 단 하나만 생성
# ─────────────────────────────────────────────
engine = create_async_engine(
    # postgresql:// → postgresql+asyncpg:// 로 변환 (비동기 드라이버 명시)
    settings.database_pool_url.replace("postgresql://", "postgresql+asyncpg://"),
    pool_size=5,
    max_overflow=10,
    # Supabase Pooler(Transaction mode) 호환 필수 설정
    connect_args={"statement_cache_size": 0},
)


# ─────────────────────────────────────────────
# 2. Session Factory — 요청마다 세션을 만들어내는 공장
# ─────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ─────────────────────────────────────────────
# 3. FastAPI 의존성 함수 — 라우터에서 Depends(get_db)로 사용
# ─────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """요청 시작 시 세션을 열고, 끝나면 자동으로 닫는다."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

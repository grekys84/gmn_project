import logging
from  datetime import datetime
from contextlib import asynccontextmanager
from functools import lru_cache
from dotenv import load_dotenv
import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
# import ngrok
import uvicorn

from application.services.graph_matching_service import GraphMatchingService
from domain.schemas.matching import MatchResponse


# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
PORT=5000

# if not NGROK_AUTH_TOKEN:
#     raise RuntimeError("Переменная окружения NGROK_AUTH_TOKEN не установлена")


@lru_cache(maxsize=1)
def get_graph_matching_service() -> GraphMatchingService:
    """Возвращает синглтон ``GraphMatchingService``."""
    logger.info("Инициализация GraphMatchingService...")
    service = GraphMatchingService()
    logger.info("GraphMatchingService успешно инициализирован")
    return service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Предварительная загрузка службы сопоставления графов при запуске приложения."""
    # logger.info("Запуск ngrok...")
    # ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    # public_url = ngrok.forward(addr=PORT)
    # logger.info(f"ngrok: {public_url}")
    #
    app.state.neural = get_graph_matching_service()
    # logger.info("Сервис сопоставления графов загружен")
    yield
    # ngrok.disconnect()
    # logger.info("ngrok отключён")


app = FastAPI(lifespan=lifespan)


@app.post("/match", response_model=MatchResponse)
def match_graph(
    file: UploadFile = File(...),
    service: GraphMatchingService = Depends(get_graph_matching_service),
) -> MatchResponse | dict:
    logger.info(f"Получен запрос на сопоставление: {file.filename}")

    if not file.filename.lower().endswith(".svg"):
        logger.warning(f"Файл не SVG: {file.filename}")
        raise HTTPException(400, "Только .svg файлы разрешены")


    try:
        # Синхронное чтение (FastAPI позволяет)
        contents = file.file.read()
        logger.info(f"Файл {file.filename} прочитан, размер: {len(contents)} байт")

        # Синхронный вызов сервиса
        raw_result = service.predict_bytes(contents, filename=file.filename)
        logger.info(f"Сопоставление завершено: id={raw_result['id']}, valid={raw_result['valid']}")

        return MatchResponse(
            **raw_result,
            processed_at=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file.filename}: {e}", exc_info=True)
        return {  # ← тоже dict
            "id": None,
            "similarity_percent": 0.0,
            "overlay_path": None,
            "valid": False,
            "processed_at": datetime.now(),
        }

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
